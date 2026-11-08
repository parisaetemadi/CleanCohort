"""Step 0 - Make fake (but realistically messy) health data, plus an answer key."""

import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")

FIRST_NAMES = ["Ava", "Liam", "Noah", "Emma", "Olivia", "Ethan", "Mia", "Lucas", "Sara", "Omar",
               "Leila", "Daniel", "Grace", "Arash", "Chloe", "Ryan", "Zoe", "Amir", "Nina", "Jack"]
LAST_NAMES = ["Smith", "Chen", "Patel", "Nguyen", "Brown", "Martin", "Singh", "Lee", "Wilson", "Khan",
              "Taylor", "Garcia", "Roy", "Clark", "Young", "Lopez", "Hall", "King", "Wright", "Scott"]

CONDITION_SPELLINGS = {
    "type2_diabetes": ["T2DM", "type 2 diabetes", "Diabetes mellitus type II", "sugar diabetes", "DM2"],
    "hypertension":   ["HTN", "hypertension", "high blood pressure", "Hypertensn", "elevated BP"],
    "asthma":         ["asthma", "Asthma", "reactive airway disease"],
    "copd":           ["COPD", "chronic obstructive pulmonary disease", "emphysema"],
    "depression":     ["depression", "MDD", "major depressive disorder", "low mood disorder"],
}
BASE_RISK = {"type2_diabetes": .25, "hypertension": .35, "asthma": .12, "copd": .08, "depression": .18}
SMOKING = ["current", "former", "never"]


def messy_date(d: date, rng: random.Random) -> str:
    style = rng.choice(["iso", "us", "text", "dots"])
    if style == "iso":
        return d.strftime("%Y-%m-%d")
    if style == "us":
        return d.strftime("%m/%d/%Y")
    if style == "text":
        return d.strftime("%d-%b-%Y")
    return d.strftime("%Y.%m.%d")


def messy_mrn(n: int, rng: random.Random) -> str:
    return rng.choice([f"MRN-{n:05d}", f"{n}", f" mrn{n:05d} ", f"MRN{n:05d}"])


def make_patients(n_patients: int, rng: random.Random) -> list[dict]:
    patients, used_names = [], set()
    for i in range(1, n_patients + 1):
        while True:
            name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
            if name not in used_names:
                used_names.add(name)
                break
        p = {
            "mrn": i,
            "name": name,
            "dob": date(1945, 1, 1) + timedelta(days=rng.randint(0, 365 * 55)),
            "sex": rng.choice(["F", "M"]),
            "smoking": rng.choices(SMOKING, weights=[2, 3, 5])[0],
            "weight_kg": round(rng.uniform(50, 120), 1),
            "glucose_mmol": round(rng.uniform(4.2, 11.5), 1),
            "sbp": rng.randint(105, 170),
            "dbp": rng.randint(65, 105),
            "self_rated_health": rng.randint(1, 5),
            "activity_days": rng.randint(0, 7),
        }
        for cond in CONDITION_SPELLINGS:
            p[cond] = rng.random() < BASE_RISK[cond]
        true_conditions = [c for c in CONDITION_SPELLINGS if p[c]]
        p["notes_only_condition"] = rng.choice(true_conditions) if true_conditions and rng.random() < 0.3 else None
        p["has_wearable"] = rng.random() < 0.7
        p["did_survey"] = rng.random() < 0.8
        patients.append(p)
    return patients


def make_ehr(patients, rng) -> pd.DataFrame:
    rows = []
    for p in patients:
        for _ in range(rng.randint(1, 3)):
            visit = date(2025, 1, 1) + timedelta(days=rng.randint(0, 600))
            dx = [c for c in CONDITION_SPELLINGS if p[c] and c != p["notes_only_condition"]]
            dx_text = "; ".join(rng.choice(CONDITION_SPELLINGS[c]) for c in dx) if dx else ""
            w = p["weight_kg"] + rng.uniform(-1.5, 1.5)
            g = p["glucose_mmol"] + rng.uniform(-0.4, 0.4)
            sex = {"F": rng.choice(["F", "female", "Female", "2"]),
                   "M": rng.choice(["M", "male", "Male", "1"])}[p["sex"]]
            rows.append({
                "MRN": messy_mrn(p["mrn"], rng),
                "Patient Name": p["name"].upper() if rng.random() < .3 else p["name"],
                "DOB": messy_date(p["dob"], rng),
                "Sex": sex,
                "Visit Date": messy_date(visit, rng),
                "Diagnoses": dx_text,
                "Weight": rng.choice([f"{w:.1f} kg", f"{w * 2.20462:.0f} lb", f"{w:.1f}"]),
                "Glucose": rng.choice([f"{g:.1f} mmol/L", f"{g * 18:.0f} mg/dL"]),
                "BP": f"{p['sbp'] + rng.randint(-6, 6)}/{p['dbp'] + rng.randint(-4, 4)}",
            })
    df = pd.DataFrame(rows)
    dups = df.sample(n=max(2, len(df) // 15), random_state=rng.randint(0, 999))
    df = pd.concat([df, dups], ignore_index=True)
    df.loc[rng.randrange(len(df)), "Weight"] = "850 kg"
    return df.sample(frac=1, random_state=1).reset_index(drop=True)


def make_notes(patients, rng) -> pd.DataFrame:
    smoke_text = {
        "current": ["Current smoker, 1 ppd.", "Smokes half a pack daily.", "Active tobacco use."],
        "former":  ["Former smoker, quit 2015.", "Ex-smoker.", "Quit smoking 10 years ago."],
        "never":   ["Never smoker.", "Denies tobacco use.", "Non-smoker."],
    }
    rows = []
    for p in patients:
        age = 2026 - p["dob"].year
        sex_word = "female" if p["sex"] == "F" else "male"
        phone = f"(403) 555-{rng.randint(1000, 9999)}"
        parts = [f"{p['name']} is a {age} y/o {sex_word} seen today. Contact: {phone}."]
        for c in CONDITION_SPELLINGS:
            if p[c] and (c == p["notes_only_condition"] or rng.random() < 0.6):
                parts.append(f"PMH significant for {rng.choice(CONDITION_SPELLINGS[c])}.")
        healthy = [c for c in CONDITION_SPELLINGS if not p[c]]
        if healthy and rng.random() < 0.6:
            parts.append(f"Denies {rng.choice(CONDITION_SPELLINGS[rng.choice(healthy)])}.")
        if healthy and rng.random() < 0.5:
            parts.append(f"Mother has {rng.choice(CONDITION_SPELLINGS[rng.choice(healthy)])}.")
        parts.append(rng.choice(smoke_text[p["smoking"]]))
        parts.append("Plan: follow up in 3 months.")
        rows.append({
            "note_id": f"N{p['mrn']:04d}",
            "mrn": messy_mrn(p["mrn"], rng),
            "note_date": messy_date(date(2025, 6, 1) + timedelta(days=rng.randint(0, 300)), rng),
            "note_text": " ".join(parts),
        })
    return pd.DataFrame(rows)


def make_wearables(patients, rng):
    daily, links = [], []
    for p in patients:
        if not p["has_wearable"]:
            continue
        device_id = f"wd_{rng.randint(100000, 999999)}"
        links.append({"device_id": device_id, "patient_mrn": messy_mrn(p["mrn"], rng), "consent": "yes"})
        base_steps, base_hr = rng.randint(2000, 12000), rng.randint(55, 85)
        for d in range(rng.randint(10, 30)):
            steps = max(0, int(rng.gauss(base_steps, 1500)))
            hr = int(rng.gauss(base_hr, 3))
            sleep = round(rng.gauss(7, 1), 1)
            r = rng.random()
            if r < 0.03:
                steps = 250000
            elif r < 0.06:
                hr = 0
            elif r < 0.10:
                sleep = None
            daily.append({"device_id": device_id,
                          "timestamp_utc": f"{date(2026, 1, 1) + timedelta(days=d)}T06:00:00Z",
                          "steps": steps, "resting_hr": hr, "sleep_hours": sleep})
    return pd.DataFrame(daily), pd.DataFrame(links)


def make_surveys(patients, rng) -> pd.DataFrame:
    health_words = {1: ["Poor", "poor"], 2: ["Fair", "fair "], 3: ["Good", "good"],
                    4: ["Very good", "very good", "V. good"], 5: ["Excellent", "excellent!"]}
    smoke_words = {"current": ["yes", "Yes, daily", "I smoke"],
                   "former": ["used to", "Former", "quit"],
                   "never": ["no", "Never", "never smoked"]}
    number_words = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
    rows = []
    for p in patients:
        if not p["did_survey"]:
            continue
        name = p["name"].lower() if rng.random() < .4 else p["name"]
        if rng.random() < 0.15:
            name = name[:-1]
        days = p["activity_days"]
        rows.append({
            "full_name": f"  {name} ",
            "birth_date": messy_date(p["dob"], rng),
            "q1_general_health": rng.choice(health_words[p["self_rated_health"]]),
            "q2_do_you_smoke": rng.choice(smoke_words[p["smoking"]]),
            "q3_days_active_per_week": rng.choice([str(days), number_words[days]]),
        })
    rows.append({"full_name": "Jordan Unknown", "birth_date": "1990-01-01",
                 "q1_general_health": "Good", "q2_do_you_smoke": "no", "q3_days_active_per_week": "3"})
    return pd.DataFrame(rows)


def main(n_patients: int = 40, seed: int = 42):
    rng = random.Random(seed)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    patients = make_patients(n_patients, rng)

    make_ehr(patients, rng).to_csv(RAW_DIR / "ehr_export.csv", index=False)
    make_notes(patients, rng).to_csv(RAW_DIR / "clinical_notes.csv", index=False)
    daily, links = make_wearables(patients, rng)
    daily.to_json(RAW_DIR / "wearable_daily.json", orient="records", indent=1)
    links.to_csv(RAW_DIR / "device_links.csv", index=False)
    make_surveys(patients, rng).to_csv(RAW_DIR / "survey_responses.csv", index=False)

    truth = pd.DataFrame(patients)
    truth["dob"] = truth["dob"].astype(str)
    truth.to_csv(RAW_DIR / "truth.csv", index=False)
    print(f"Made messy data for {n_patients} fake patients in {RAW_DIR}/")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# KEY POINTS
# - Real hospital data is private, so we practise on FAKE patients.
# - The data is messy ON PURPOSE, the way real data is:
#     EHR export  -> mixed date formats, lb vs kg, mg/dL vs mmol/L, disease nicknames
#                    ("sugar diabetes"), a typo ("Hypertensn"), duplicate rows, "850 kg"
#     Notes       -> free text with names + phone numbers, negations ("Denies COPD"),
#                    family history ("Mother has HTN")
#     Wearables   -> device IDs instead of patient IDs, sensor glitches (250,000 steps)
#     Surveys     -> no patient ID (only name + birth date), typos, free-text answers,
#                    one person who isn't in the hospital records
# - ~30% of patients have one condition written ONLY in the notes, not the EHR codes.
#   This shows why reading notes matters.
# - truth.csv is the ANSWER KEY. Step 8 checks the pipeline against it, like validating
#   a measurement tool against a gold standard. Real data never has this.
# - seed=42 makes the "random" data the same every run (reproducible).
# - Run on its own:  python -m cleancohort.step0_make_fake_data
# ---------------------------------------------------------------------------
