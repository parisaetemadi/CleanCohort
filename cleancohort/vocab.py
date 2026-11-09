"""Vocabulary: turns messy words into standard medical codes (ICD-10, LOINC)."""

CONDITIONS = {
    "type2_diabetes": {
        "icd10": "E11",
        "label": "Type 2 diabetes mellitus",
        "synonyms": ["type 2 diabetes", "t2dm", "dm2", "diabetes mellitus type ii",
                     "type ii diabetes", "sugar diabetes", "diabetes", "diabetic",
                     "niddm", "non-insulin-dependent diabetes mellitus"],
    },
    "hypertension": {
        "icd10": "I10",
        "label": "Essential (primary) hypertension",
        "synonyms": ["hypertension", "htn", "high blood pressure", "elevated bp",
                     "hypertensive"],
    },
    "asthma": {
        "icd10": "J45",
        "label": "Asthma",
        "synonyms": ["asthma", "reactive airway disease", "asthmatic"],
    },
    "copd": {
        "icd10": "J44",
        "label": "Chronic obstructive pulmonary disease",
        "synonyms": ["copd", "chronic obstructive pulmonary disease", "emphysema",
                     "chronic obstructive lung disease", "chronic obstructive airway disease"],
    },
    "depression": {
        "icd10": "F32",
        "label": "Depressive episode",
        "synonyms": ["depression", "mdd", "major depressive disorder", "depressive",
                     "depressed mood"],
    },
}

SYNONYM_TO_CONCEPT = {
    syn: concept for concept, info in CONDITIONS.items() for syn in info["synonyms"]
}

# Phrases that CONTAIN a condition name but are a different disease. Everything here was
# found by searching the official FY2027 ICD-10-CM codebook for codes whose title holds one
# of our condition words but whose code is not ours (394 such codes for "diabetes" alone),
# and by counting how often each phrase really appears in 4,966 MTSamples clinical notes.
# They are not guesses: "ST depression" is in 9 of those notes and is a heart tracing, not a
# mood disorder, and before this list existed the pipeline coded it as depression.
LOOKALIKES = {
    "type2_diabetes": [
        "type 1 diabetes", "type i diabetes", "type 1 diabetic", "juvenile diabetes",
        "insulin-dependent diabetes mellitus", "iddm",
        "gestational diabetes", "pre-existing diabetes", "neonatal diabetes",
        "diabetes insipidus", "prediabetes", "pre-diabetes", "borderline diabetes",
        "chemical induced diabetes", "drug or chemical induced diabetes",
        "drug-induced diabetes", "steroid-induced diabetes", "secondary diabetes",
        "other specified diabetes", "family history of diabetes",
    ],
    "hypertension": [
        "pulmonary hypertension", "ocular hypertension", "portal hypertension",
        "intracranial hypertension", "venous hypertension", "chronic venous hypertension",
        "gestational hypertension", "pregnancy-induced hypertension",
        "pre-existing hypertension", "maternal hypertension", "renovascular hypertension",
        "secondary hypertension", "white coat hypertension",
        "family history of hypertension",
    ],
    "asthma": [],
    "copd": [
        "subcutaneous emphysema", "mediastinal emphysema", "interstitial emphysema",
        "surgical emphysema", "compensatory emphysema", "traumatic emphysema",
    ],
    "depression": [
        "st depression", "st-segment depression", "segment depression",
        "respiratory depression", "bone marrow depression", "marrow depression",
        "cns depression", "central nervous system depression",
        "neonatal cerebral depression", "screening for depression",
        "family history of depression",
    ],
}

MEASURES = {
    "weight_kg":    {"loinc": "29463-7", "label": "Body weight",              "unit": "kg"},
    "glucose_mmol": {"loinc": "14749-6", "label": "Glucose, serum/plasma",    "unit": "mmol/L"},
    "sbp":          {"loinc": "8480-6",  "label": "Systolic blood pressure",  "unit": "mm[Hg]"},
    "dbp":          {"loinc": "8462-4",  "label": "Diastolic blood pressure", "unit": "mm[Hg]"},
    "steps":        {"loinc": "55423-8", "label": "Number of steps",          "unit": "steps/day"},
    "resting_hr":   {"loinc": "40443-4", "label": "Resting heart rate",       "unit": "beats/min"},
    "sleep_hours":  {"loinc": "93832-4", "label": "Sleep duration",           "unit": "h"},
}

OBSERVATIONS = {
    "smoking_status":    {"code": "72166-2",       "label": "Tobacco smoking status"},
    "self_rated_health": {"code": "LOCAL:SRH",     "label": "Self-rated health (1=poor, 5=excellent)"},
    "activity_days":     {"code": "LOCAL:ACTDAYS", "label": "Days physically active per week"},
}

PLAUSIBLE_RANGES = {
    "weight_kg":    (25, 300),
    "glucose_mmol": (1, 40),
    "sbp":          (60, 260),
    "dbp":          (30, 160),
    "steps":        (0, 60000),
    "resting_hr":   (30, 200),
    "sleep_hours":  (0, 16),
}


# ---------------------------------------------------------------------------
# KEY POINTS
# - "HTN", "high blood pressure" and "hypertension" mean the same thing. Studies need
#   ONE standard label so everyone counts the same thing.
# - ICD-10 codes diagnoses (I10 = hypertension). LOINC codes measurements and lab tests
#   (8480-6 = systolic blood pressure).
# - Real products load the full code lists (tens of thousands of codes). This small
#   list is readable in one screen.
# - "LOCAL:" codes are our own, used where we did not pick a standard code.
# - Emphysema strictly has its own code (J43); many studies group it with COPD.
# - LOOKALIKES is the other half of the vocabulary: a list of what each word does NOT
#   mean. Every entry was found by evidence, not guessed - the official FY2027 ICD-10-CM
#   codebook was searched for codes holding one of our words but carrying a different
#   code (394 of them for "diabetes" alone), and each candidate was then counted in 4,966
#   real MTSamples notes. On those notes the list stops 141 miscodings, led by gestational
#   diabetes (30), pulmonary hypertension (28) and subcutaneous emphysema (11).
# - Adding a synonym without checking its lookalikes is how a cohort gets quietly wrong:
#   "depression" alone matched ST depression and respiratory depression, and "emphysema"
#   alone matched subcutaneous emphysema, which is air under the skin after trauma.
# - "diabetic" is in on purpose, and it widens the case definition: a patient with
#   diabetic retinopathy or diabetic renal disease is counted as a case, because those
#   are complications of the disease. The type 1, gestational and drug-induced forms are
#   still refused, so the widening does not pull in the other kinds of diabetes.
# - Deliberately left OUT, each for a reason: "elevated blood pressure" (R03.0 is a
#   reading, not a diagnosis), "dysthymia" (F34.1, a different disorder from F32), and a
#   bare "depressed" (about a fifth of its uses in real notes are "depressed fracture",
#   "depressed breath sounds" or a depressed ST segment, so "depressed mood" is in instead).
# - PLAUSIBLE_RANGES: values outside these are physically impossible -> treated as errors.
# - EXERCISE: "low mood disorder" is missing from depression on purpose. Run the
#   pipeline, find it under "Unmapped diagnosis terms" in the quality report, add it
#   here, run again, and watch depression sensitivity go up in the validation table.
# ---------------------------------------------------------------------------
