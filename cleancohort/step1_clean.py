"""Step 1 - Clean: make every value follow one standard format."""

import re
from datetime import datetime

import numpy as np
import pandas as pd

from .vocab import PLAUSIBLE_RANGES

STEP = "1_clean"
DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y", "%Y.%m.%d", "%Y-%m-%dT%H:%M:%SZ"]
NUMBER = r"(-?\d+(?:\.\d+)?)"
SEX_MAP = {"f": "F", "female": "F", "2": "F", "m": "M", "male": "M", "1": "M"}
HEALTH_WORDS = {"poor": 1, "fair": 2, "good": 3, "very good": 4, "v good": 4, "excellent": 5}
NUMBER_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}


def is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == ""


def parse_date(value):
    if is_missing(value):
        return None
    text = str(value).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def normalize_mrn(value):
    if is_missing(value):
        return None
    digits = re.sub(r"\D", "", str(value))
    return int(digits) if digits else None


def normalize_name(value):
    if is_missing(value):
        return None
    return " ".join(str(value).split()).title()


def normalize_sex(value):
    return SEX_MAP.get(str(value).strip().lower())


def parse_weight_kg(value):
    if is_missing(value):
        return None, None
    m = re.match(NUMBER + r"\s*(kg|lb|lbs)?$", str(value).strip().lower())
    if not m:
        return None, "could not read value"
    number, unit = float(m.group(1)), m.group(2)
    if unit in ("lb", "lbs"):
        return round(number * 0.453592, 1), "converted lb->kg"
    if unit is None:
        return number, "no unit written, assumed kg"
    return number, None


def parse_glucose_mmol(value):
    if is_missing(value):
        return None, None
    m = re.match(NUMBER + r"\s*(mmol/l|mg/dl)?$", str(value).strip().lower())
    if not m:
        return None, "could not read value"
    number, unit = float(m.group(1)), m.group(2)
    if unit == "mg/dl":
        return round(number / 18.016, 1), "converted mg/dL->mmol/L"
    if unit is None:
        if number > 35:
            return round(number / 18.016, 1), "no unit, looked like mg/dL, converted"
        return number, "no unit written, assumed mmol/L"
    return number, None


def parse_bp(value):
    if is_missing(value):
        return None, None
    m = re.match(r"(\d{2,3})\s*/\s*(\d{2,3})$", str(value).strip())
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def check_range(value, measure, audit, source, record):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    low, high = PLAUSIBLE_RANGES[measure]
    if not (low <= value <= high):
        audit.log(STEP, source, record, measure, value, None,
                  f"impossible value removed (allowed {low}-{high})")
        return np.nan
    return value


def _log_if_changed(audit, source, record, field, old, new, rule):
    if str(old).strip() != str(new):
        audit.log(STEP, source, record, field, old, new, rule)


def clean_ehr(raw: pd.DataFrame, audit) -> pd.DataFrame:
    rows = []
    for i, r in raw.iterrows():
        rec = f"ehr_row_{i}"
        mrn = normalize_mrn(r["MRN"])
        dob = parse_date(r["DOB"])
        visit = parse_date(r["Visit Date"])
        sex = normalize_sex(r["Sex"])
        weight, w_note = parse_weight_kg(r["Weight"])
        glucose, g_note = parse_glucose_mmol(r["Glucose"])
        sbp, dbp = parse_bp(r["BP"])

        _log_if_changed(audit, "ehr", rec, "mrn", r["MRN"], mrn, "standardized ID")
        _log_if_changed(audit, "ehr", rec, "dob", r["DOB"], dob, "standardized date")
        _log_if_changed(audit, "ehr", rec, "visit_date", r["Visit Date"], visit, "standardized date")
        _log_if_changed(audit, "ehr", rec, "sex", r["Sex"], sex, "standardized sex code")
        if w_note:
            audit.log(STEP, "ehr", rec, "weight", r["Weight"], weight, w_note)
        if g_note:
            audit.log(STEP, "ehr", rec, "glucose", r["Glucose"], glucose, g_note)

        rows.append({
            "mrn": mrn,
            "name": normalize_name(r["Patient Name"]),
            "dob": dob,
            "sex": sex,
            "visit_date": visit,
            "diagnoses_text": "" if is_missing(r["Diagnoses"]) else str(r["Diagnoses"]),
            "weight_kg": check_range(weight, "weight_kg", audit, "ehr", rec),
            "glucose_mmol": check_range(glucose, "glucose_mmol", audit, "ehr", rec),
            "sbp": check_range(sbp, "sbp", audit, "ehr", rec),
            "dbp": check_range(dbp, "dbp", audit, "ehr", rec),
        })
    clean = pd.DataFrame(rows)

    clean["_filled"] = clean.notna().sum(axis=1)
    clean = clean.sort_values("_filled", ascending=False)
    key = ["mrn", "visit_date", "sbp", "dbp", "diagnoses_text"]
    dup_mask = clean.duplicated(subset=key, keep="first")
    for idx in clean.index[dup_mask]:
        audit.log(STEP, "ehr", f"ehr_row_{idx}", "(whole row)", "duplicate", None, "duplicate row removed")
    clean = clean[~dup_mask].drop(columns="_filled").sort_values(["mrn", "visit_date"])
    return clean.reset_index(drop=True)


def clean_notes(raw: pd.DataFrame, audit) -> pd.DataFrame:
    out = raw.copy()
    out["mrn"] = raw["mrn"].map(normalize_mrn)
    out["note_date"] = raw["note_date"].map(parse_date)
    for i, r in raw.iterrows():
        _log_if_changed(audit, "notes", r["note_id"], "mrn", r["mrn"], out.at[i, "mrn"], "standardized ID")
    return out


def clean_wearables(daily: pd.DataFrame, links: pd.DataFrame, audit):
    links = links.copy()
    links["mrn"] = links["patient_mrn"].map(normalize_mrn)
    links = links[links["consent"].str.lower() == "yes"][["device_id", "mrn"]]

    rows = []
    for _, r in daily.iterrows():
        rec = f"{r['device_id']}@{r['timestamp_utc']}"
        rows.append({
            "device_id": r["device_id"],
            "date": parse_date(r["timestamp_utc"]),
            "steps": check_range(r["steps"], "steps", audit, "wearable", rec),
            "resting_hr": check_range(r["resting_hr"], "resting_hr", audit, "wearable", rec),
            "sleep_hours": check_range(r["sleep_hours"], "sleep_hours", audit, "wearable", rec),
        })
    cols = ["device_id", "date", "steps", "resting_hr", "sleep_hours"]
    return pd.DataFrame(rows, columns=cols), links.reset_index(drop=True)


def parse_health_word(value):
    text = re.sub(r"[^a-z ]", "", str(value).lower()).strip()
    return HEALTH_WORDS.get(text)


def parse_smoking_answer(value):
    text = str(value).strip().lower()
    if "never" in text or text == "no":
        return "never"
    if "used to" in text or "former" in text or "quit" in text:
        return "former"
    if "yes" in text or "smoke" in text or "daily" in text:
        return "current"
    return None


def parse_days(value):
    text = str(value).strip().lower()
    if text.isdigit():
        return int(text)
    return NUMBER_WORDS.get(text)


def clean_surveys(raw: pd.DataFrame, audit) -> pd.DataFrame:
    rows = []
    for i, r in raw.iterrows():
        rec = f"survey_{i}"
        health = parse_health_word(r["q1_general_health"])
        smoke = parse_smoking_answer(r["q2_do_you_smoke"])
        days = parse_days(r["q3_days_active_per_week"])
        _log_if_changed(audit, "survey", rec, "q1_general_health", r["q1_general_health"], health, "word -> score 1-5")
        _log_if_changed(audit, "survey", rec, "q2_do_you_smoke", r["q2_do_you_smoke"], smoke, "free text -> category")
        _log_if_changed(audit, "survey", rec, "q3_days_active", r["q3_days_active_per_week"], days, "text -> number")
        rows.append({
            "survey_id": rec,
            "name": normalize_name(r["full_name"]),
            "dob": parse_date(r["birth_date"]),
            "self_rated_health": health,
            "smoking_status": smoke,
            "activity_days": days,
        })
    return pd.DataFrame(rows, columns=["survey_id", "name", "dob", "self_rated_health",
                                       "smoking_status", "activity_days"])


def clean_all(raw: dict, audit) -> dict:
    daily, links = clean_wearables(raw["wearable_daily"], raw["device_links"], audit)
    return {
        "ehr": clean_ehr(raw["ehr"], audit),
        "notes": clean_notes(raw["notes"], audit),
        "wearable_daily": daily,
        "device_links": links,
        "survey": clean_surveys(raw["survey"], audit),
    }


# ---------------------------------------------------------------------------
# KEY POINTS
# - What gets fixed:
#     IDs     " mrn00012 ", "12", "MRN-00012"  -> 12
#     Dates   "03/12/1985", "12-Mar-1985"       -> 1985-03-12
#     Sex     "female", "2", "F"                -> F
#     Units   "180 lb" -> 81.6 kg;  "110 mg/dL" glucose -> 6.1 mmol/L (divide by 18)
#     Errors  "850 kg", 250,000 steps, heart rate 0 -> removed (set to empty)
#     Copies  duplicate visit rows -> removed (keeps the most complete copy)
# - No AI here on purpose: simple rules give the SAME answer every time. That's what
#   you want for units and dates.
# - Date trap: "03/04/1985" is March 4 in the US but 3 April in Canada/Europe. You must
#   know each source system's format. Our fake EHR uses US style (month/day/year).
# - Each small function does ONE job (parse_date, parse_weight_kg ...). That makes them
#   easy to test (see tests/test_pipeline.py).
# - Order matters in parse_smoking_answer: "never smoked" contains "smoke", so "never"
#   is checked first.
# - Every change goes into the audit log.
# ---------------------------------------------------------------------------
