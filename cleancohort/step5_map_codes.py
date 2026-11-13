"""Step 5 - Map to standard codes and reshape into long tables (OMOP-style)."""

import re
from difflib import get_close_matches

import pandas as pd

from .vocab import CONDITIONS, LOOKALIKES, MEASURES, OBSERVATIONS, SYNONYM_TO_CONCEPT

STEP = "5_map_codes"
FUZZY_CUTOFF = 0.85

_BY_LENGTH = sorted(SYNONYM_TO_CONCEPT, key=len, reverse=True)


def contains_synonym(t: str):
    """Real diagnosis text wraps the condition name ("Childhood asthma"). Find a synonym
    as a whole word, longest first so "type 2 diabetes" beats "diabetes"."""
    for syn in _BY_LENGTH:
        if re.search(rf"\b{re.escape(syn)}\b", t):
            return SYNONYM_TO_CONCEPT[syn], syn
    return None, None


def map_term(term: str):
    t = " ".join(term.strip().lower().split())
    if not t:
        return None, "empty"

    concept, method, matched = None, "unmapped", ""
    if t in SYNONYM_TO_CONCEPT:
        concept, method, matched = SYNONYM_TO_CONCEPT[t], "exact", t
    else:
        found, syn = contains_synonym(t)
        if found:
            concept, method, matched = found, f"contains ('{syn}')", syn
        else:
            close = get_close_matches(t, SYNONYM_TO_CONCEPT.keys(), n=1, cutoff=FUZZY_CUTOFF)
            if close:
                concept, method, matched = SYNONYM_TO_CONCEPT[close[0]], f"fuzzy (~'{close[0]}')", close[0]

    if concept and any(bad in t and bad not in matched
                       for bad in LOOKALIKES.get(concept, [])):
        return None, "refused - lookalike disease"
    return concept, method


def build_conditions(ehr: pd.DataFrame, note_mentions: pd.DataFrame, audit):
    rows, unmapped = [], []
    for _, visit in ehr.iterrows():
        for term in visit["diagnoses_text"].split(";"):
            if not term.strip():
                continue
            concept, method = map_term(term)
            if concept is None:
                unmapped.append({"term": term.strip(), "source": "ehr", "study_id": visit["study_id"]})
                audit.log(STEP, "ehr", visit["study_id"], "diagnosis", term.strip(), None,
                          "no matching code - needs human review")
                continue
            if method.startswith("fuzzy"):
                audit.log(STEP, "ehr", visit["study_id"], "diagnosis", term.strip(), concept,
                          f"typo matched by {method} - review")
            rows.append({"study_id": visit["study_id"], "concept": concept, "source": "ehr",
                         "source_text": term.strip(), "map_method": method, "date": visit["visit_date"]})

    affirmed = note_mentions[note_mentions["status"] == "affirmed"]
    for _, m in affirmed.iterrows():
        rows.append({"study_id": m["study_id"], "concept": m["concept"], "source": "notes",
                     "source_text": m["evidence"], "map_method": f"note-{m['method']}", "date": m["date"]})

    cond = pd.DataFrame(rows, columns=["study_id", "concept", "source", "source_text", "map_method", "date"])
    cond["icd10"] = cond["concept"].map(lambda c: CONDITIONS[c]["icd10"])
    cond["label"] = cond["concept"].map(lambda c: CONDITIONS[c]["label"])

    unmapped_df = pd.DataFrame(unmapped, columns=["term", "source", "study_id"])
    unmapped_summary = (unmapped_df.groupby("term").size().rename("count").reset_index()
                        .sort_values("count", ascending=False))
    return cond, unmapped_summary


def build_measurements(ehr: pd.DataFrame, wearable: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, df, date_col, measures in [
        ("ehr", ehr, "visit_date", ["weight_kg", "glucose_mmol", "sbp", "dbp"]),
        ("wearable", wearable, "date", ["steps", "resting_hr", "sleep_hours"]),
    ]:
        for _, r in df.iterrows():
            for m in measures:
                if pd.notna(r[m]):
                    rows.append({"study_id": r["study_id"], "date": r[date_col], "measure": m,
                                 "loinc": MEASURES[m]["loinc"], "label": MEASURES[m]["label"],
                                 "value": float(r[m]), "unit": MEASURES[m]["unit"], "source": source})
    return pd.DataFrame(rows, columns=["study_id", "date", "measure", "loinc", "label",
                                       "value", "unit", "source"])


def build_observations(survey: pd.DataFrame, note_smoking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in survey.iterrows():
        for item in ["smoking_status", "self_rated_health", "activity_days"]:
            if pd.notna(r[item]):
                rows.append({"study_id": r["study_id"], "item": item, "code": OBSERVATIONS[item]["code"],
                             "value": r[item], "source": "survey"})
    for _, r in note_smoking.iterrows():
        rows.append({"study_id": r["study_id"], "item": "smoking_status",
                     "code": OBSERVATIONS["smoking_status"]["code"], "value": r["smoking_status"],
                     "source": "notes"})
    return pd.DataFrame(rows, columns=["study_id", "item", "code", "value", "source"])


# ---------------------------------------------------------------------------
# KEY POINTS
# - Matching a written diagnosis to a code, in 4 tiers:
#     1. EXACT     - the whole term is in the synonym list    -> accept
#     2. CONTAINS  - a synonym appears as a whole word inside -> accept
#     3. FUZZY     - very close to one (typo "Hypertensn")    -> accept, flag for review
#     4. NONE      - no good match -> "unmapped", a human decides
# - CONTAINS exists because real EHR text wraps the condition name: Synthea writes
#   "Childhood asthma" and "Pulmonary emphysema (disorder)". Exact matching alone found
#   0 asthma and 0 COPD cases in 300 real patients; adding this tier found 7 and 4.
# - CONTAINS is checked longest-synonym-first, so "type 2 diabetes" wins over "diabetes".
# - The risk of CONTAINS is miscoding a different disease that shares a word, so every
#   concept has a LOOKALIKES list in vocab.py: "Pulmonary hypertension" is NOT I10,
#   "Diabetes insipidus" is NOT type 2 diabetes. Those are refused, not mapped.
# - The LOOKALIKES guard runs ONCE, after whichever tier found a concept - never inside a
#   single tier. "Type 1 diabetes" is one character from "type 2 diabetes", so when the
#   guard sat inside CONTAINS the FUZZY tier matched it as a typo and coded a type 1
#   patient as type 2. Any new tier gets the guard for free; putting it inside one does not.
# - FUZZY_CUTOFF = 0.85 (0-1). Lower catches more typos but risks wrong matches.
# - Only AFFIRMED note mentions become conditions.
# - Data is reshaped into LONG tables (one row per fact), like the OMOP standard that
#   big research networks use:
#     conditions.csv    study_id | concept | ICD-10 | source | date
#     measurements.csv  study_id | measure | LOINC  | value  | unit | date
#     observations.csv  study_id | item    | code   | value  | source
# - Long tables keep every fact; step 6 then summarizes them into one row per patient.
# ---------------------------------------------------------------------------
