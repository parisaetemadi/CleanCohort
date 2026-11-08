"""Step 3 - De-identify: remove who the patient is, keep what we need to study."""

import hashlib
import re
from datetime import date

import pandas as pd

STEP = "3_deidentify"
INDEX_DATE = date(2026, 1, 1)

PHONE = re.compile(r"\(?\b\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")
FULL_DATE = re.compile(r"\b(\d{4}[-./]\d{1,2}[-./]\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}-[A-Za-z]{3}-\d{4})\b")
MRN_LIKE = re.compile(r"\bMRN[-\s]?\d+\b", re.IGNORECASE)


def make_study_id(mrn, secret_salt: str) -> str:
    digest = hashlib.sha256(f"{secret_salt}:{mrn}".encode()).hexdigest()
    return "P" + digest[:8].upper()


def age_on(dob, on_date=INDEX_DATE):
    if dob is None or pd.isna(dob):
        return None
    return on_date.year - dob.year - ((on_date.month, on_date.day) < (dob.month, dob.day))


def scrub_note(text: str, patient_name: str):
    count = 0
    if patient_name:
        for piece in [patient_name] + [p for p in patient_name.split() if len(p) > 2]:
            text, n = re.subn(rf"\b{re.escape(piece)}\b", "[NAME]", text, flags=re.IGNORECASE)
            count += n
    for pattern, tag in [(PHONE, "[PHONE]"), (FULL_DATE, "[DATE]"), (MRN_LIKE, "[ID]")]:
        text, n = pattern.subn(tag, text)
        count += n
    text = re.sub(r"\[NAME\](\s*\[NAME\])+", "[NAME]", text)
    return text, count


def deidentify_all(linked: dict, audit, secret_salt: str = "change-me-and-keep-secret") -> dict:
    index = linked["index"].copy()
    index["study_id"] = index["mrn"].map(lambda m: make_study_id(m, secret_salt))
    crosswalk = index[["mrn", "study_id"]]
    to_id = dict(zip(index["mrn"], index["study_id"]))
    names = dict(zip(index["mrn"], index["name"]))

    person = pd.DataFrame({
        "study_id": index["study_id"],
        "sex": index["sex"],
        "age_at_index": index["dob"].map(age_on),
    })

    ehr = linked["ehr"].copy()
    ehr["study_id"] = ehr["mrn"].map(to_id)
    ehr = ehr.drop(columns=["mrn", "name", "dob"])

    notes = linked["notes"].copy()
    scrubbed = []
    for _, n in notes.iterrows():
        text, c = scrub_note(n["note_text"], names.get(n["mrn"]))
        scrubbed.append(text)
        if c:
            audit.log(STEP, "notes", n["note_id"], "note_text", f"{c} identifiers", "[removed]",
                      "scrubbed names/phones/dates/IDs from note")
    notes["note_text"] = scrubbed
    notes["study_id"] = notes["mrn"].map(to_id)
    notes = notes.drop(columns=["mrn"])

    wear = linked["wearable_daily"].copy()
    wear["study_id"] = wear["mrn"].map(to_id)
    wear = wear.drop(columns=["mrn", "device_id"])

    survey = linked["survey"].copy()
    survey["study_id"] = survey["mrn"].map(to_id)
    survey = survey.drop(columns=["mrn", "name", "dob"])

    audit.log(STEP, "all", "-", "mrn/name/dob", "identifiers", "study_id + age",
              f"replaced identifiers for {len(index)} patients")
    return {"person": person, "ehr": ehr, "notes": notes, "wearable_daily": wear,
            "survey": survey, "crosswalk": crosswalk}


# ---------------------------------------------------------------------------
# KEY POINTS
# - Researchers should almost never see names, MRNs, phone numbers or exact birth dates.
# - Each MRN becomes a STUDY ID (e.g. P3F9A21C0) made with a one-way hash + a secret
#   "salt". Same MRN + same salt -> same ID every time, but you can't reverse it.
# - The crosswalk (MRN <-> study ID) is saved separately in data/secure/. Only the data
#   custodian keeps it.
# - Birth date -> age at a fixed index date (2026-01-01).
# - Notes are scrubbed: names -> [NAME], phones -> [PHONE], full dates -> [DATE],
#   MRNs -> [ID]. Years like "quit 2015" are kept (useful, low risk).
# - This happens BEFORE any AI reads the notes, so private info never leaves the hospital.
# - This is a teaching version. Real tools use stronger, tested de-identification
#   (e.g. Philter, Presidio) and expert review.
# ---------------------------------------------------------------------------
