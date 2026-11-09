"""Step 4 - Read doctor notes: turn free text into facts (rules, or an LLM with guardrails)."""

import json
import os
import re

import pandas as pd

from .vocab import CONDITIONS, LOOKALIKES

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


def part_of_lookalike(sentence: str, start: int, end: int, concept: str) -> bool:
    """True when the matched word is inside the name of a different disease.

    "ST depression" is a heart tracing and "subcutaneous emphysema" is air under the skin.
    Both contain a word we search for, so the match is only rejected when a lookalike
    phrase actually covers it - a sentence that mentions both still keeps the real one."""
    low = sentence.lower()
    for bad in LOOKALIKES.get(concept, []):
        for hit in re.finditer(re.escape(bad), low):
            if hit.start() <= start and hit.end() >= end:
                return True
    return False


def extract_with_rules(note_text: str):
    mentions = []
    for sentence in split_sentences(note_text):
        found = set()
        for concept, _, pattern in _PATTERNS:
            m = pattern.search(sentence)
            if m and concept not in found:
                if part_of_lookalike(sentence, m.start(), m.end(), concept):
                    continue
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
    client = None
    model = model or os.environ.get("CLEANCOHORT_LLM_MODEL", "claude-haiku-5-5")

    def call(prompt: str) -> str:
        nonlocal client
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        msg = client.messages.create(model=model, max_tokens=2000,
                                     output_config={"effort": "low"},
                                     messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content if b.type == "text")
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
        used = method
        if method == "llm":
            mentions, smoking = extract_with_llm(n["note_text"], call_llm, audit, n["note_id"])
        elif method == "hybrid":
            mentions, smoking = extract_with_rules(n["note_text"])
            used = "rules"
            if not mentions and not smoking:
                audit.log(STEP, "notes", n["note_id"], "reader", "rules", "llm",
                          "rules found nothing - escalated to the paid model")
                mentions, smoking = extract_with_llm(n["note_text"], call_llm, audit, n["note_id"])
                used = "llm"
        else:
            mentions, smoking = extract_with_rules(n["note_text"])
        for m in mentions:
            mention_rows.append({"study_id": n["study_id"], "note_id": n["note_id"],
                                 "date": n["note_date"], "method": used, **m})
            if m["status"] != "affirmed":
                audit.log(STEP, "notes", n["note_id"], m["concept"], m["evidence"], "not counted",
                          f"mention is {m['status']} - not the patient's condition")
        if smoking:
            smoking_rows.append({"study_id": n["study_id"], "note_id": n["note_id"], "date": n["note_date"],
                                 "smoking_status": smoking["status"], "evidence": smoking["evidence"],
                                 "method": used})
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
# - Method C, HYBRID (the cheap way to use AI): rules read every note for free; the paid
#   model is called ONLY for notes where the rules found nothing at all. Every escalation
#   is logged, so the audit trail shows exactly what you paid for and why.
#   On the fake data this escalates 0 of 40 notes - but that is because the fake notes are
#   written from this same synonym list (see step0 CONDITION_SPELLINGS), so treat 0% as a
#   property of the test data, not a promise about real notes.
# - Method B, LLM (optional): an AI reads the already de-identified note. It understands
#   more wording but can make things up, so GUARDRAILS check every answer:
#     * concept must be in our allowed list
#     * status must be affirmed / negated / family
#     * evidence must be copied word-for-word from the note
#   Anything else is rejected and logged.
# - Synonyms are searched longest-first so "type 2 diabetes" wins over shorter matches.
# - part_of_lookalike is the precision half of the reader, and it was missing: step 5 had
#   the LOOKALIKES guard but step 4 did not, so until now the rules read "ST depression"
#   (9 real notes) and "respiratory depression" (8) as depression, and "subcutaneous
#   emphysema" (5) as COPD. Measured on 4,966 MTSamples notes, the guard now rejects 141
#   such matches.
# - It rejects a match only when a lookalike phrase COVERS it, not merely appears in the
#   same sentence, so "History of depression; ECG showed ST depression." still counts once.
# - LLM setup: pip install anthropic, set ANTHROPIC_API_KEY, optional CLEANCOHORT_LLM_MODEL.
# - Default model is claude-haiku-5-5, the cheapest current model (~20x cheaper per token
#   than Sonnet). The guardrails below re-check every answer anyway, so a smaller model
#   costs accuracy far less here than it would in an unchecked pipeline.
# - No prompt caching: the instruction block is only ~250 tokens, well under the minimum
#   cacheable prefix, so a cache_control breakpoint would silently do nothing.
# - Current models think before answering, so two things matter:
#     * effort="low" keeps thinking short - this is simple extraction, not a hard problem
#     * we join ALL text blocks instead of reading content[0], because block 0 can be a
#       thinking block; reading it would return "" and reject every note as invalid JSON
# - max_tokens is 2000 because thinking tokens share that budget with the JSON answer.
# - The LLM path is tested with a fake LLM (tests/test_pipeline.py), not a real API call yet.
# ---------------------------------------------------------------------------
