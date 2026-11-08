# CleanCohort

Turns messy health data (EHR records, doctor notes, wearables, surveys) into one clean, de-identified, **study-ready** dataset, with an audit log and a quality report.

```
 Messy data                CleanCohort pipeline                       Outputs
┌──────────────┐   ┌─────────────────────────────────────┐   ┌──────────────────────┐
│ EHR export   │   │ 1 Clean      dates, units, IDs, dups │   │ study_dataset.csv    │
│ Doctor notes │──►│ 2 Link       same person across files│──►│ conditions / measure-│
│ Wearables    │   │ 3 De-identify names → study IDs      │   │ ments / observations │
│ Surveys      │   │ 4 Read notes affirmed/negated/family │   │ quality_report.html  │
└──────────────┘   │ 5 Map codes  ICD-10, LOINC           │   │ audit_log.csv        │
                   │ 6 Build      one row per patient     │   │ validation.csv       │
                   │ 7 Report     what changed + review   │   └──────────────────────┘
                   │ 8 Validate   sensitivity / PPV       │
                   └─────────────────────────────────────┘
```

## Run it

```bash
pip install -r requirements.txt
python run_demo.py            # fake data → full pipeline → results
python -m pytest -q           # run the tests
```

Then open `data/secure/quality_report.html` in your browser.

Optional: read notes with an AI model instead of rules:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...       # your key
python run_demo.py --use-llm
```

## Files

| File | What it does |
|---|---|
| `cleancohort/step0_make_fake_data.py` | Fake, messy data for 40 patients, plus an answer key (`truth.csv`) |
| `cleancohort/step1_clean.py` | Standard dates, IDs, sex, units; removes impossible values and duplicates |
| `cleancohort/step2_link.py` | Links surveys (name + DOB, fuzzy) and wearables (consent file) to patients |
| `cleancohort/step3_deidentify.py` | MRN → study ID, DOB → age, scrubs names and phone numbers from notes |
| `cleancohort/step4_read_notes.py` | Finds conditions and smoking in notes: rules, or an LLM with guardrails |
| `cleancohort/step5_map_codes.py` | Maps to ICD-10/LOINC (exact → fuzzy → unmapped) and builds long tables |
| `cleancohort/step6_build_dataset.py` | Builds one row per patient for analysis |
| `cleancohort/step7_quality_report.py` | HTML report: fixes, review list, missing data |
| `cleancohort/step8_validate.py` | Sensitivity and PPV against the answer key |
| `cleancohort/vocab.py` | Synonyms → codes, plausible value ranges |
| `cleancohort/audit.py` | Logs every change |
| `cleancohort/pipeline.py` | Runs all steps in order |

Each file ends with **KEY POINTS** bullets explaining it.

## Output folders

| Folder | Contents | Who can see it |
|---|---|---|
| `data/raw/` | Fake messy input + `truth.csv` | Practice only |
| `data/output/` | De-identified study files | Researchers |
| `data/secure/` | Crosswalk, audit log, quality report (contain identifiers) | Data custodian only |

## Current results (fake data)

| Condition | Sensitivity, EHR codes only | Sensitivity, EHR + notes | False positives |
|---|---|---|---|
| Diabetes | 0.88 | 1.00 | 0 |
| Hypertension | 0.92 | 1.00 | 0 |
| Asthma | 0.00 | 1.00 | 0 |
| COPD | 0.50 | 1.00 | 0 |
| Depression | 0.75 | 1.00 | 0 |

Reading notes finds conditions the codes miss, and handling negation and family history keeps false positives at 0.

## Exercises

1. **Unmapped term:** the report lists "low mood disorder". Add it to depression in `vocab.py` and run again.
2. **Stricter linking:** change `NAME_SIMILARITY_CUTOFF` in `step2_link.py` to 0.97. What happens to the fuzzy links?
3. **New negation:** add the sentence "No evidence of asthma." to a note. Does the rule catch it? Why?
4. **New variable:** add BMI (needs height) to the fake data and the study dataset.
5. **Date trap:** what goes wrong if a Canadian source writes dates as day/month/year?

## Limits (prototype)

- Fake data only, so it hasn't been tested on real records
- Small vocabularies; real products load full ICD-10, SNOMED and LOINC lists
- Teaching-level de-identification; real use needs tested tools and a privacy review
- The LLM path is tested with a fake LLM, not yet with a real API call
