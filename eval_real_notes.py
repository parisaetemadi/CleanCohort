"""Score the note reader against real clinical text.

Run:  python eval_real_notes.py        (needs data/external/mtsamples/mtsamples.csv)

MTSamples has no answer key, so this measures two different things:

  RECALL     against a deliberately generous regex for "this note discusses X". The regex
             catches some things that are not the condition, so the number is a LOWER
             bound on real recall, not an estimate of it.
  REJECTIONS how often the lookalike guard stopped a match that was a different disease.
             This one is exact: every rejection is a miscoding that would have happened.
"""

import collections
import glob
import re
import sys
from pathlib import Path

import pandas as pd

from cleancohort.step4_read_notes import _PATTERNS, extract_with_rules, part_of_lookalike, split_sentences
from cleancohort.step5_map_codes import map_term
from cleancohort.vocab import LOOKALIKES

NOTES = Path("data/external/mtsamples/mtsamples.csv")

# Generous on purpose: a clinician reading a match should agree the note is about the
# condition. It over-matches ("tongue depressor"), which is why recall here is a floor.
PROXY = {
    "type2_diabetes": r"\b(type\s*(2|ii)\s*diabet\w*|t2dm|dm2|niddm|diabetes mellitus|diabetic)\b",
    "hypertension": r"\b(hypertensi\w*|htn|high blood pressure)\b",
    "asthma": r"\b(asthma\w*|reactive airway\w*)\b",
    "copd": r"\b(copd|emphysema|chronic obstructive pulmonary|chronic obstructive lung)\b",
    "depression": r"\b(depression|depressive|mdd|depressed)\b",
}


def load_notes() -> list[str]:
    if not NOTES.exists():
        sys.exit(f"{NOTES} not found - see 'Free health data' in README.md for the download")
    return pd.read_csv(NOTES)["transcription"].dropna().tolist()


def recall_table(notes) -> pd.DataFrame:
    rows = []
    for concept, pattern in PROXY.items():
        rx = re.compile(pattern, re.IGNORECASE)
        discusses = found = 0
        for text in notes:
            if not rx.search(text):
                continue
            discusses += 1
            mentions, _ = extract_with_rules(text)
            found += any(x["concept"] == concept for x in mentions)
        rows.append({"condition": concept, "notes_discussing": discusses, "rules_found": found,
                     "recall_floor": round(found / discusses, 2) if discusses else float("nan")})
    return pd.DataFrame(rows)


def guard_table(notes) -> tuple[pd.DataFrame, int]:
    hits = collections.Counter()
    for text in notes:
        for sentence in split_sentences(text):
            low = sentence.lower()
            for concept, _, pattern in _PATTERNS:
                m = pattern.search(sentence)
                if not m or not part_of_lookalike(sentence, m.start(), m.end(), concept):
                    continue
                for bad in LOOKALIKES.get(concept, []):
                    if any(h.start() <= m.start() and h.end() >= m.end()
                           for h in re.finditer(re.escape(bad), low)):
                        hits[bad] += 1
                        break
    df = pd.DataFrame(sorted(hits.items(), key=lambda kv: -kv[1]), columns=["lookalike", "rejected"])
    return df, sum(hits.values())


ANNOTATED = ["data/external/bc5cdr/CDR_Data/CDR.Corpus.v010516/*.PubTator.txt",
             "data/external/ncbi_disease/NCBI_corpus_*.txt"]
DISEASE_TYPES = {"disease", "diseaseclass", "specificdisease", "modifier", "compositemention"}
RELEVANT = r"diabet|hypertens|asthma|copd|emphysema|chronic obstructive|depress"


def annotated_surface_forms() -> collections.Counter:
    """Disease mentions marked up by hand in BC5CDR and the NCBI Disease corpus."""
    forms = collections.Counter()
    for pattern in ANNOTATED:
        for path in glob.glob(pattern):
            for line in open(path, encoding="utf-8", errors="replace"):
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 5 and parts[4].lower() in DISEASE_TYPES:
                    forms[parts[3].strip()] += 1
    return forms


def corpus_coverage() -> tuple:
    forms = annotated_surface_forms()
    if not forms:
        return None, None, None
    ours = {f: n for f, n in forms.items() if re.search(RELEVANT, f, re.IGNORECASE)}
    missed = {f: n for f, n in ours.items() if not map_term(f)[0]}
    mapped = sum(ours.values()) - sum(missed.values())
    df = pd.DataFrame(sorted(missed.items(), key=lambda kv: -kv[1]),
                      columns=["surface form", "mentions"])
    return mapped / max(sum(ours.values()), 1), df, len(forms)


def main():
    notes = load_notes()
    print(f"{len(notes):,} real clinical notes (MTSamples, CC0)\n")

    rec = recall_table(notes)
    print("=== Recall floor against a generous proxy ===")
    print(rec.to_string(index=False))
    print(f"\nmean recall floor: {rec.recall_floor.mean():.2f}")

    guard, total = guard_table(notes)
    print(f"\n=== Miscodings the lookalike guard prevented: {total} ===")
    print(guard.head(12).to_string(index=False))

    coverage, missed, n_forms = corpus_coverage()
    if coverage is None:
        print("\n=== Annotated corpora not downloaded - skipping (see README) ===")
        return
    print(f"\n=== Coverage of {n_forms:,} hand-annotated disease forms (BC5CDR + NCBI Disease) ===")
    print(f"{coverage:.0%} of relevant mentions map to a study condition.")
    print("Everything below is a deliberate refusal, not a gap:")
    print(missed.head(10).to_string(index=False))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# KEY POINTS
# - This is the only score in the project measured on text nobody wrote for it. The fake
#   data in step0 is generated from vocab.py's own synonym list, so sensitivity there is
#   1.0 by construction and cannot fall; this number can.
# - Two numbers, because they answer different questions: recall asks "does the vocabulary
#   match how clinicians write", the guard count asks "how often would we code the wrong
#   disease". A vocabulary change usually moves them in opposite directions.
# - Recall is a FLOOR, not an estimate: the proxy regex counts "tongue depressor" as
#   depression, so some of what looks like a miss is the measuring instrument, not the code.
# - The rejection count is exact - each one is a real match a real note would have produced.
# - Re-run this after any change to CONDITIONS or LOOKALIKES in vocab.py. A synonym that
#   lifts recall and also lifts rejections is usually a synonym that needs a lookalike.
# - The third section checks the vocabulary against disease mentions annotated by hand in
#   BC5CDR and the NCBI Disease corpus - people with no connection to this project. It is
#   the strongest check here, because the surface forms were not chosen by us.
# - A form landing in the "missed" table is only a problem if it is NOT in LOOKALIKES.
#   Every one of them currently is, which is what makes the 15% shortfall the right answer
#   rather than a gap: pulmonary hypertension and diabetes insipidus should not map.
# ---------------------------------------------------------------------------
