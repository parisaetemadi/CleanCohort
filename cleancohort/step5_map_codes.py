"""Step 5 - Map to standard codes and reshape into long tables (OMOP-style)."""

from difflib import get_close_matches

import pandas as pd

from .vocab import CONDITIONS, MEASURES, OBSERVATIONS, SYNONYM_TO_CONCEPT

STEP = "5_map_codes"
FUZZY_CUTOFF = 0.85


def map_term(term: str):
    t = " ".join(term.strip().lower().split())
    if not t:
        return None, "empty"
    if t in SYNONYM_TO_CONCEPT:
        return SYNONYM_TO_CONCEPT[t], "exact"
    close = get_close_matches(t, SYNONYM_TO_CONCEPT.keys(), n=1, cutoff=FUZZY_CUTOFF)
    if close:
        return SYNONYM_TO_CONCEPT[close[0]], f"fuzzy (~'{close[0]}')"
    return None, "unmapped"


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
    return pd.DataFrame(rows)


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
# - Matching a written diagnosis to a code, in 3 tiers:
#     1. EXACT  - in the synonym list            -> accept
#     2. FUZZY  - very close to one (a typo like "Hypertensn") -> accept, flag for review
#     3. NONE   - no good match                  -> "unmapped", a human decides
# - FUZZY_CUTOFF = 0.85 (0-1). Lower catches more typos but risks wrong matches.
# - Only AFFIRMED note mentions become conditions.
# - Data is reshaped into LONG tables (one row per fact), like the OMOP standard that
#   big research networks use:
#     conditions.csv    study_id | concept | ICD-10 | source | date
#     measurements.csv  study_id | measure | LOINC  | value  | unit | date
#     observations.csv  study_id | item    | code   | value  | source
# - Long tables keep every fact; step 6 then summarizes them into one row per patient.
# ---------------------------------------------------------------------------
