"""Step 2 - Link: figure out which records belong to the same person."""

from difflib import SequenceMatcher

import pandas as pd

STEP = "2_link"
NAME_SIMILARITY_CUTOFF = 0.85


def build_patient_index(ehr: pd.DataFrame) -> pd.DataFrame:
    def most_common(values):
        values = values.dropna()
        return values.mode().iloc[0] if len(values) else None

    return (ehr.groupby("mrn")
               .agg(name=("name", most_common), dob=("dob", most_common), sex=("sex", most_common))
               .reset_index())


def name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def link_surveys(survey: pd.DataFrame, index: pd.DataFrame, audit) -> pd.DataFrame:
    survey = survey.copy()
    survey["mrn"] = None
    survey["link_method"] = "unlinked"
    for i, s in survey.iterrows():
        exact = index[(index["name"] == s["name"]) & (index["dob"] == s["dob"])]
        if len(exact) == 1:
            survey.at[i, "mrn"] = exact.iloc[0]["mrn"]
            survey.at[i, "link_method"] = "exact name + DOB"
            continue
        same_dob = index[index["dob"] == s["dob"]].copy()
        if len(same_dob) and s["name"]:
            same_dob["score"] = same_dob["name"].map(lambda n: name_similarity(n, s["name"]))
            best = same_dob.sort_values("score", ascending=False).iloc[0]
            if best["score"] >= NAME_SIMILARITY_CUTOFF:
                survey.at[i, "mrn"] = best["mrn"]
                survey.at[i, "link_method"] = f"fuzzy name ({best['score']:.2f}) + DOB"
                audit.log(STEP, "survey", s["survey_id"], "name", s["name"], best["name"],
                          f"fuzzy-linked by name similarity {best['score']:.2f} + same DOB (review)")
                continue
        audit.log(STEP, "survey", s["survey_id"], "(whole record)", s["name"], None,
                  "could not link to any patient - excluded")
    return survey


def link_wearables(daily: pd.DataFrame, links: pd.DataFrame, index: pd.DataFrame, audit) -> pd.DataFrame:
    linked = daily.merge(links, on="device_id", how="left")
    unlinked = linked["mrn"].isna() | ~linked["mrn"].isin(index["mrn"])
    for dev in linked.loc[unlinked, "device_id"].unique():
        audit.log(STEP, "wearable", dev, "device_id", dev, None, "no consent link to a patient - excluded")
    return linked[~unlinked].copy()


def check_notes(notes: pd.DataFrame, index: pd.DataFrame, audit) -> pd.DataFrame:
    ok = notes["mrn"].isin(index["mrn"])
    for nid in notes.loc[~ok, "note_id"]:
        audit.log(STEP, "notes", nid, "mrn", None, None, "MRN not found in EHR - excluded")
    return notes[ok].copy()


def link_all(clean: dict, audit) -> dict:
    index = build_patient_index(clean["ehr"])
    survey = link_surveys(clean["survey"], index, audit)
    return {
        "index": index,
        "ehr": clean["ehr"],
        "notes": check_notes(clean["notes"], index, audit),
        "wearable_daily": link_wearables(clean["wearable_daily"], clean["device_links"], index, audit),
        "survey": survey[survey["mrn"].notna()].copy(),
        "survey_unlinked": survey[survey["mrn"].isna()].copy(),
    }


# ---------------------------------------------------------------------------
# KEY POINTS
# - Each source identifies people differently:
#     EHR + notes -> MRN (medical record number)
#     Wearables   -> device ID, plus a consent file mapping device ID -> MRN
#     Surveys     -> no ID at all, only the name + birth date the patient typed
# - The "master patient index" = one row per person, built from the EHR.
# - Survey linking ("record linkage"):
#     1. exact name + birth date
#     2. same birth date AND very similar name (catches typos, e.g. "Ava Smit")
#     3. otherwise leave it UNLINKED and report it - never guess
# - NAME_SIMILARITY_CUTOFF = 0.85: higher = stricter (fewer wrong links, more missed links).
# - Epidemiology link: wrong links cause misclassification bias; missed links can cause
#   selection bias. That's why every fuzzy link is flagged for review.
# - Only consented wearable devices are linked.
# ---------------------------------------------------------------------------
