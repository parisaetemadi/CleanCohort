"""Vocabulary: turns messy words into standard medical codes (ICD-10, LOINC)."""

CONDITIONS = {
    "type2_diabetes": {
        "icd10": "E11",
        "label": "Type 2 diabetes mellitus",
        "synonyms": ["type 2 diabetes", "t2dm", "dm2", "diabetes mellitus type ii",
                     "type ii diabetes", "sugar diabetes"],
    },
    "hypertension": {
        "icd10": "I10",
        "label": "Essential (primary) hypertension",
        "synonyms": ["hypertension", "htn", "high blood pressure", "elevated bp"],
    },
    "asthma": {
        "icd10": "J45",
        "label": "Asthma",
        "synonyms": ["asthma", "reactive airway disease"],
    },
    "copd": {
        "icd10": "J44",
        "label": "Chronic obstructive pulmonary disease",
        "synonyms": ["copd", "chronic obstructive pulmonary disease", "emphysema"],
    },
    "depression": {
        "icd10": "F32",
        "label": "Depressive episode",
        "synonyms": ["depression", "mdd", "major depressive disorder"],
    },
}

SYNONYM_TO_CONCEPT = {
    syn: concept for concept, info in CONDITIONS.items() for syn in info["synonyms"]
}

# Phrases that CONTAIN a condition name but are a different disease. Checked before
# substring matching, so "Diabetes insipidus" never gets coded as type 2 diabetes and
# "Pulmonary hypertension" never gets coded as essential hypertension (I10).
LOOKALIKES = {
    "type2_diabetes": ["diabetes insipidus", "gestational diabetes", "type 1 diabetes",
                       "type i diabetes", "pre-diabetes", "prediabetes",
                       "family history of diabetes"],
    "hypertension": ["pulmonary hypertension", "ocular hypertension", "portal hypertension",
                     "intracranial hypertension", "gestational hypertension",
                     "pregnancy-induced hypertension", "family history of hypertension"],
    "asthma": [],
    "copd": [],
    "depression": ["family history of depression"],
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
# - PLAUSIBLE_RANGES: values outside these are physically impossible -> treated as errors.
# - EXERCISE: "low mood disorder" is missing from depression on purpose. Run the
#   pipeline, find it under "Unmapped diagnosis terms" in the quality report, add it
#   here, run again, and watch depression sensitivity go up in the validation table.
# ---------------------------------------------------------------------------
