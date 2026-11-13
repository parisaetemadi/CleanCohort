"""Learn the affirmed/negated decision from labelled clinical text, instead of guessing it.

Run:  python train_negation.py        (needs data/external/negex/, see README)

Deciding whether a note SAYS a patient has a condition or says they do NOT is the part of
step 4 that controls false positives, and until now it was seven hand-written words. The
NegEx release ships 2,376 assertions labelled by hand on 116 real discharge summaries, so
that decision can be measured and fitted rather than argued about.

Three approaches are compared on the same held-out reports:
  1. the rules currently in step4_read_notes.py
  2. the published NegEx algorithm, driven by its own 271-trigger lexicon
  3. a logistic regression trained on the words around the concept

Reports, not rows, are split: the same report supplies many concepts, so splitting rows
would leak its wording across the divide and flatter every model.
"""

import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from cleancohort.step4_read_notes import mention_status, split_sentences

NEGEX = Path("data/external/negex")
WINDOW = 7          # tokens either side of the concept that NegEx and the model may read
TEST_FRACTION = 0.3


def load_lexicon() -> dict:
    """negex_triggers.txt lines look like:  no evidence of		[PREN]"""
    kinds = {"PREN": [], "PREP": [], "POST": [], "POSP": [], "PSEU": [], "CONJ": []}
    for line in (NEGEX / "negex_triggers.txt").read_text(errors="replace").splitlines():
        m = re.match(r"^(.*?)\s*\[(\w+)\]\s*$", line.strip())
        if m and m.group(2) in kinds:
            kinds[m.group(2)].append(m.group(1).strip().lower())
    for k in kinds:
        kinds[k].sort(key=len, reverse=True)
    return kinds


def load_annotations() -> pd.DataFrame:
    a = pd.read_csv(NEGEX / "Annotations-1-120.txt", sep="\t")
    a.columns = [c.strip() for c in a.columns]
    a = a[a["Negation"].isin(["Affirmed", "Negated"])].copy()
    a["concept"] = a["Concept"].str.strip().str.lower()
    a["negated"] = (a["Negation"] == "Negated").astype(int)
    a["context"] = [concept_window(r, c) for r, c in zip(a["Sentence"], a["concept"])]
    return a[a["context"].str.len() > 0].reset_index(drop=True)


def concept_window(report: str, concept: str) -> str:
    """The sentence holding the concept, trimmed to WINDOW tokens either side of it.

    The whole report is far too much context: the words that decide the assertion sit
    beside the concept, and feeding the rest teaches the model which report it is reading."""
    report = re.sub(r"\s+", " ", str(report))
    for sentence in split_sentences(report):
        low = sentence.lower()
        at = low.find(concept)
        if at < 0:
            continue
        before = sentence[:at].split()[-WINDOW:]
        after = sentence[at + len(concept):].split()[:WINDOW]
        return " ".join([*before, "__CONCEPT__", *after])
    return ""


def negex_predict(window: str, lex: dict) -> int:
    """The published algorithm: pseudo-negations are neutralised first, then a trigger is
    looked for on the correct side of the concept, stopping at a conjunction."""
    text = f" {window.lower()} "
    for phrase in lex["PSEU"]:
        text = text.replace(f" {phrase} ", " __pseudo__ ")

    before, _, after = text.partition("__concept__")
    for phrase in lex["CONJ"]:
        before = before.rsplit(f" {phrase} ", 1)[-1]
        after = after.split(f" {phrase} ", 1)[0]

    if any(f" {p} " in before for p in lex["PREN"] + lex["PREP"]):
        return 1
    if any(f" {p} " in after for p in lex["POST"] + lex["POSP"]):
        return 1
    return 0


def ours_predict(window: str) -> int:
    """Our own implementation, judging the concept at the position it actually sits."""
    sentence = window.replace("__CONCEPT__", "the finding")
    at = sentence.find("the finding")
    return int(mention_status(sentence, at, at + len("the finding")) == "negated")


def scores(name: str, truth, pred) -> dict:
    tp = int(((truth == 1) & (pred == 1)).sum())
    fp = int(((truth == 0) & (pred == 1)).sum())
    fn = int(((truth == 1) & (pred == 0)).sum())
    tn = int(((truth == 0) & (pred == 0)).sum())
    return {
        "approach": name,
        "accuracy": round((tp + tn) / max(len(truth), 1), 3),
        "sensitivity": round(tp / max(tp + fn, 1), 3),
        "specificity": round(tn / max(tn + fp, 1), 3),
        "PPV": round(tp / max(tp + fp, 1), 3),
        "missed_negations": fn,
        "false_negations": fp,
    }


def main():
    if not (NEGEX / "Annotations-1-120.txt").exists():
        sys.exit(f"{NEGEX} not found - see 'Free health data' in README.md")

    a = load_annotations()
    lex = load_lexicon()
    print(f"{len(a):,} labelled assertions over {a['Report No.'].nunique()} real discharge summaries")
    print(f"{a.negated.sum()} negated, {len(a) - a.negated.sum()} affirmed\n")

    reports = sorted(a["Report No."].unique())
    cut = int(len(reports) * (1 - TEST_FRACTION))
    train = a[a["Report No."].isin(reports[:cut])]
    test = a[a["Report No."].isin(reports[cut:])]
    print(f"train {len(train):,} assertions / {cut} reports   "
          f"test {len(test):,} / {len(reports) - cut} reports\n")

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0),
    )
    model.fit(train["context"], train["negated"])

    rows = [
        scores("ours (step4_read_notes)", test.negated, test.context.map(ours_predict)),
        scores("NegEx lexicon (271)", test.negated, test.context.map(lambda w: negex_predict(w, lex))),
        scores("trained classifier", test.negated, pd.Series(model.predict(test["context"]), index=test.index)),
    ]
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nWhat our rules still miss (negations called affirmed):")
    missed = test[(test.negated == 1) & (test.context.map(ours_predict) == 0)]
    for w in missed["context"].head(5):
        print("   ", w[:110])


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# KEY POINTS
# - This is the first thing in the project fitted to labels somebody else wrote. The fake
#   data cannot measure negation either: step0 writes "Denies COPD." using the same word
#   the rules look for.
# - Sensitivity here means "of the negations, how many did we catch". A missed negation is
#   the expensive error: it turns "denies chest pain" into a case and inflates prevalence.
# - Specificity matters just as much in the other direction: a false negation deletes a
#   real case. The table prints both, because one number alone can hide either failure.
# - The split is BY REPORT. Splitting rows would put sentences from the same discharge
#   summary on both sides and make every approach look better than it is.
# - Only WINDOW tokens either side of the concept are used. Given the whole report, the
#   classifier learns which report it is reading rather than how negation is written.
# - NegEx is 2001 and still the baseline every assertion paper reports against; the point
#   of a trained model here is to see whether 2,376 examples beat 271 hand-made rules.
# ---------------------------------------------------------------------------
