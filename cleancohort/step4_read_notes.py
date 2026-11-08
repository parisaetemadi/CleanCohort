"""Step 4 - Read doctor notes: turn free text into facts (rules, or an LLM with guardrails)."""

import json
import os
import re

import pandas as pd

from .vocab import CONDITIONS

STEP = "4_read_notes"

NEGATION_WORDS = ["denies", "denied", "no ", "negative for", "without", "ruled out", "no history of"]
FAMILY_WORDS = ["mother", "father", "sister", "brother", "family history", "fhx", "parent",
                "grandmother", "grandfather"]

SMOKING_RULES = [
    ("never",   ["never smoker", "non-smoker", "nonsmoker", "denies tobacco", "never smoked"]),
    ("former",  ["former smoker", "ex-smoker", "quit smoking", "quit 20", "stopped smoking"]),
    ("current", ["current smoker", "smokes", "active tobacco", "ppd", "pack daily"]),
]

LLM_PROMPT = """You extract facts from a DE-IDENTIFIED clinical note.

Allowed condition concepts: {concepts}

Rules:
- Only use concepts from the allowed list.
- "evidence" must be words copied EXACTLY from the note.
- status: "affirmed" (patient has it), "negated" (patient does not), or "family" (a relative has it).
- smoking status: current, former, never, or unknown.

Return ONLY JSON, like:
{{"conditions": [{{"concept": "hypertension", "evidence": "HTN", "status": "affirmed"}}],
  "smoking": {{"status": "former", "evidence": "Ex-smoker"}}}}

Note:
<<<{note}>>>"""

_PATTERNS = sorted(
    [(concept, syn, re.compile(rf"\b{re.escape(syn)}\b", re.IGNORECASE))
     for concept, info in CONDITIONS.items() for syn in info["synonyms"]],
    key=lambda x: -len(x[1]),
)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def mention_status(sentence: str) -> str:
    s = " " + sentence.lower()
    if any(w in s for w in FAMILY_WORDS):
        return "family"
    if any(" " + w in s for w in NEGATION_WORDS):
        return "negated"
    return "affirmed"


def extract_with_rules(note_text: str):
    mentions = []
    for sentence in split_sentences(note_text):
        found = set()
        for concept, _, pattern in _PATTERNS:
            m = pattern.search(sentence)
            if m and concept not in found:
                found.add(concept)
                mentions.append({"concept": concept, "evidence": m.group(0),
                                 "status": mention_status(sentence), "sentence": sentence})
    smoking = None
    low = note_text.lower()
    for status, phrases in SMOKING_RULES:
        hit = next((p for p in phrases if p in low), None)
        if hit:
            smoking = {"status": status, "evidence": hit}
            break
    return mentions, smoking


def make_anthropic_caller(model: str | None = None):
    import anthropic
    client = anthropic.Anthropic()
    model = model or os.environ.get("CLEANCOHORT_LLM_MODEL", "claude-sonnet-4-5")

    def call(prompt: str) -> str:
        msg = client.messages.create(model=model, max_tokens=800,
                                     messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text
    return call


def extract_with_llm(note_text: str, call_llm, audit=None, note_id=""):
    prompt = LLM_PROMPT.format(concepts=", ".join(CONDITIONS), note=note_text)
    raw = call_llm(prompt)
    try:
        data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    except ValueError:
        if audit:
            audit.log(STEP, "notes", note_id, "llm_output", raw[:80], None, "LLM output was not valid JSON - rejected")
        return [], None

    def evidence_ok(ev):
        return bool(ev) and ev.lower() in note_text.lower()

    mentions = []
    for c in data.get("conditions", []):
        concept, ev, status = c.get("concept"), c.get("evidence", ""), c.get("status")
        if concept not in CONDITIONS or status not in ("affirmed", "negated", "family") or not evidence_ok(ev):
            if audit:
                audit.log(STEP, "notes", note_id, "llm_condition", json.dumps(c), None,
                          "LLM answer failed guardrail (unknown concept / bad status / evidence not in note) - rejected")
            continue
        sentence = next((s for s in split_sentences(note_text) if ev.lower() in s.lower()), "")
        mentions.append({"concept": concept, "evidence": ev, "status": status, "sentence": sentence})

    smoking = None
    sm = data.get("smoking") or {}
    if sm.get("status") in ("current", "former", "never") and evidence_ok(sm.get("evidence", "")):
        smoking = {"status": sm["status"], "evidence": sm["evidence"]}
    return mentions, smoking


def read_all_notes(notes: pd.DataFrame, audit, method: str = "rules", call_llm=None):
    mention_rows, smoking_rows = [], []
    for _, n in notes.iterrows():
        if method == "llm":
            mentions, smoking = extract_with_llm(n["note_text"], call_llm, audit, n["note_id"])
        else:
            mentions, smoking = extract_with_rules(n["note_text"])
        for m in mentions:
            mention_rows.append({"study_id": n["study_id"], "note_id": n["note_id"],
                                 "date": n["note_date"], "method": method, **m})
            if m["status"] != "affirmed":
                audit.log(STEP, "notes", n["note_id"], m["concept"], m["evidence"], "not counted",
                          f"mention is {m['status']} - not the patient's condition")
        if smoking:
            smoking_rows.append({"study_id": n["study_id"], "note_id": n["note_id"], "date": n["note_date"],
                                 "smoking_status": smoking["status"], "evidence": smoking["evidence"],
                                 "method": method})
    mention_cols = ["study_id", "note_id", "date", "method", "concept", "evidence", "status", "sentence"]
    smoking_cols = ["study_id", "note_id", "date", "smoking_status", "evidence", "method"]
    return pd.DataFrame(mention_rows, columns=mention_cols), pd.DataFrame(smoking_rows, columns=smoking_cols)


# ---------------------------------------------------------------------------
# KEY POINTS
# - Notes hold facts the coded EHR often misses (see the validation table: sensitivity
#   jumps when notes are added).
# - The hard part is STATUS, not finding the word:
#     "PMH significant for HTN."  -> affirmed (patient has it)
#     "Denies HTN."               -> negated  (patient does not)
#     "Mother has HTN."           -> family   (relative's disease)
#   Counting all three would create false positives -> biased prevalence.
# - Method A, RULES (default): synonym lists + negation/family words. Fast, free,
#   predictable, offline. Misses wording that isn't in the lists.
# - Method B, LLM (optional): an AI reads the already de-identified note. It understands
#   more wording but can make things up, so GUARDRAILS check every answer:
#     * concept must be in our allowed list
#     * status must be affirmed / negated / family
#     * evidence must be copied word-for-word from the note
#   Anything else is rejected and logged.
# - Synonyms are searched longest-first so "type 2 diabetes" wins over shorter matches.
# - LLM setup: pip install anthropic, set ANTHROPIC_API_KEY, optional CLEANCOHORT_LLM_MODEL.
# - The LLM path is tested with a fake LLM (tests/test_pipeline.py), not a real API call yet.
# ---------------------------------------------------------------------------
