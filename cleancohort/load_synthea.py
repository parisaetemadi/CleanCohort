"""Turn Synthea's real-shaped CSVs into CleanCohort's messy raw format.

Synthea (https://synthetichealth.github.io/synthea/) publishes free synthetic patients with
genuine code systems: SNOMED for conditions, LOINC for measurements. Download the sample CSVs,
unzip, and point this at the csv/ folder.
"""

import re
from pathlib import Path

import pandas as pd

WEIGHT_LOINC = "29463-7"
GLUCOSE_LOINC = "2339-0"
SBP_LOINC = "8480-6"
DBP_LOINC = "8462-4"
MEASURE_CODES = [WEIGHT_LOINC, GLUCOSE_LOINC, SBP_LOINC, DBP_LOINC]


def clean_name(first: str, last: str) -> str:
    """Synthea appends digits to names (Gómez206). Strip them so names look real."""
    f = re.sub(r"\d+", "", str(first))
    l = re.sub(r"\d+", "", str(last))
    return f"{f} {l}".strip()


def load_people(csv_dir: Path, limit: int | None) -> pd.DataFrame:
    p = pd.read_csv(csv_dir / "patients.csv", dtype=str)
    if limit:
        p = p.head(limit)
    return pd.DataFrame({
        "synthea_id": p["Id"],
        "mrn": [f"SYN{i:05d}" for i in range(1, len(p) + 1)],
        "name": [clean_name(f, l) for f, l in zip(p["FIRST"], p["LAST"])],
        "dob": p["BIRTHDATE"],
        "sex": p["GENDER"],
    })


def diagnoses_by_person(csv_dir: Path, people: pd.DataFrame) -> pd.Series:
    c = pd.read_csv(csv_dir / "conditions.csv", dtype=str)
    c = c[c["PATIENT"].isin(set(people["synthea_id"]))]
    return c.groupby("PATIENT")["DESCRIPTION"].apply(lambda s: "; ".join(sorted(set(s))))


def visits_by_person(csv_dir: Path, people: pd.DataFrame, visits_per_person: int) -> pd.DataFrame:
    o = pd.read_csv(csv_dir / "observations.csv", dtype=str)
    o = o[o["CODE"].isin(MEASURE_CODES) & o["PATIENT"].isin(set(people["synthea_id"]))].copy()
    o["visit_date"] = o["DATE"].str[:10]
    wide = o.pivot_table(index=["PATIENT", "visit_date"], columns="CODE",
                         values="VALUE", aggfunc="last").reset_index()
    wide = wide.sort_values("visit_date").groupby("PATIENT").tail(visits_per_person)
    return wide


def build_ehr_export(people: pd.DataFrame, visits: pd.DataFrame, dx: pd.Series) -> pd.DataFrame:
    v = visits.merge(people, left_on="PATIENT", right_on="synthea_id")
    sbp = v.get(SBP_LOINC)
    dbp = v.get(DBP_LOINC)
    bp = [f"{int(float(a))}/{int(float(b))}" if pd.notna(a) and pd.notna(b) else ""
          for a, b in zip(sbp, dbp)]
    weight = [f"{w} kg" if pd.notna(w) else "" for w in v.get(WEIGHT_LOINC)]
    glucose = [f"{g} mg/dL" if pd.notna(g) else "" for g in v.get(GLUCOSE_LOINC)]
    return pd.DataFrame({
        "MRN": v["mrn"],
        "Patient Name": v["name"],
        "DOB": v["dob"],
        "Sex": v["sex"],
        "Visit Date": v["visit_date"],
        "Diagnoses": v["synthea_id"].map(dx).fillna(""),
        "Weight": weight,
        "Glucose": glucose,
        "BP": bp,
    })


def main(csv_dir="data/external/synthea/csv", raw_dir="data/raw_synthea",
         limit: int | None = 300, visits_per_person: int = 3) -> Path:
    csv_dir, raw_dir = Path(csv_dir), Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    people = load_people(csv_dir, limit)
    dx = diagnoses_by_person(csv_dir, people)
    visits = visits_by_person(csv_dir, people, visits_per_person)
    ehr = build_ehr_export(people, visits, dx)
    ehr.to_csv(raw_dir / "ehr_export.csv", index=False)

    # Synthea has no free text, no wearables and no surveys. Write valid empty files so the
    # pipeline runs unchanged and those steps honestly report nothing to do.
    pd.DataFrame(columns=["note_id", "mrn", "note_date", "note_text"]).to_csv(
        raw_dir / "clinical_notes.csv", index=False)
    pd.DataFrame(columns=["device_id", "patient_mrn", "consent"]).to_csv(
        raw_dir / "device_links.csv", index=False)
    pd.DataFrame(columns=["full_name", "birth_date", "q1_general_health",
                          "q2_do_you_smoke", "q3_days_active_per_week"]).to_csv(
        raw_dir / "survey_responses.csv", index=False)
    (raw_dir / "wearable_daily.json").write_text("[]", encoding="utf-8")

    print(f"Synthea -> {raw_dir}/  {len(people)} patients, {len(ehr)} visit rows")
    return raw_dir


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# KEY POINTS
# - Why this matters: the bundled fake data generates notes from the same synonym list the
#   rules search (step0 CONDITION_SPELLINGS vs vocab.py), so the rules cannot miss and the
#   validation is circular. Synthea's codes were written by someone else, so they can.
# - Synthea uses real code systems: SNOMED text for conditions, LOINC for measurements
#   (29463-7 weight, 2339-0 glucose, 8480-6 / 8462-4 blood pressure).
# - Diagnoses are passed through as raw description text, NOT pre-filtered to our five
#   conditions. That is the point: step 5 then reports a real unmapped-term rate instead of
#   a flattering one.
# - Synthea has no clinical notes, no wearables and no surveys, so those files are written
#   empty. Steps 2 and 4 then have nothing to do, which is the honest result.
# - There is no truth.csv, so step 8 is skipped. Real data has no answer key - that is why
#   real studies do chart review on a sample instead.
# - limit/visits_per_person keep the demo quick; raise them for a scale test.
# ---------------------------------------------------------------------------
