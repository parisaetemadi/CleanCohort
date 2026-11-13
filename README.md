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

## Reading notes: three options, three prices

The pipeline runs **free and offline by default**. An AI model is optional, and when you do
want one, the hybrid mode pays for it only where the rules fall short.

| Mode | Command | API calls per 1,000 notes | Needs a key |
|---|---|---|---|
| Rules | `python run_demo.py` | 0 | no |
| **Hybrid** | `python run_demo.py --hybrid` | only notes the rules can't read | only if some escalate |
| Every note | `python run_demo.py --use-llm` | 1,000 | yes |

```bash
pip install anthropic                  # only needed for --hybrid / --use-llm
export ANTHROPIC_API_KEY=...           # from console.anthropic.com, billed per token
python run_demo.py --hybrid
```

Hybrid logs every escalation to the audit trail, so you can always see what you paid for
and why. On the bundled fake data it escalates **0 of 40 notes** and matches the rules
exactly — but those notes are written from the same synonym list the rules search
(`CONDITION_SPELLINGS` in `step0`), so that 0% is a fact about the test data, not a
promise about real records. The default model is `claude-haiku-5-5`, the cheapest current
model; override it with `CLEANCOHORT_LLM_MODEL`.

Compare the two readers head to head, including measured cost per note:

```bash
python compare_rules_vs_llm.py
```

## Run it on real data (Synthea)

```bash
curl -L -o synthea.zip https://synthetichealth.github.io/synthea-sample-data/downloads/synthea_sample_data_csv_apr2020.zip
unzip -q synthea.zip -d data/external/synthea
python run_synthea.py
```

[Synthea](https://synthetichealth.github.io/synthea/) is free synthetic patient data using
real code systems (SNOMED conditions, LOINC measurements). Nobody wrote it to match our
vocabulary, so unlike the bundled fake data it can actually catch mistakes — and it did.
Exact-string matching found **0** asthma and **0** COPD cases in 300 real patients, because
real records say `Childhood asthma`, not `asthma`. See `step5` KEY POINTS.

Synthea has no clinical notes, so there is nothing for an AI to read and nothing to pay for.

## Free health data: what is actually open

Searched for real data this pipeline could be tested and improved on. What matters for a
tool that reads notes is **free text**, and that is the scarcest thing.

| Source | Size | Licence / access | What it gives us |
|---|---|---|---|
| [MTSamples](https://huggingface.co/datasets/harishnair04/mtsamples) | 4,966 notes | CC0, no sign-up | **Real clinical free text** — the only open source of it found |
| [NegEx release](https://github.com/chapmanbe/negex) | 2,376 assertions | Open repo, no stated licence | **Labels** — affirmed/negated on 116 real discharge summaries |
| [ICD-10-CM FY2027](https://www.cms.gov/medicare/coding-billing/icd-10-codes) | 74,879 codes | Public, CMS | Official diagnosis wording, and the lookalikes |
| [BC5CDR + NCBI Disease](https://github.com/JHnlp/BioCreative-V-CDR-Corpus) | 12,850 mentions | Open | Disease mentions annotated by hand — an independent check on the vocabulary |
| [CASI](https://conservancy.umn.edu/items/6651323b-444a-479e-a41a-abca58c2e721) | 440 abbreviations | UMN Conservancy | Which clinical abbreviations are ambiguous, from 604,944 real notes |
| [BioScope](https://rgai.inf.u-szeged.hu/node/105) | 11,871 sentences | Free for research | Negation + scope, but its clinical half is anonymised to `*` |
| [Synthea](https://synthetichealth.github.io/synthea/) | ~1k patients | Open | Structured EHR with real SNOMED/LOINC codes |
| [MIMIC-IV demo](https://physionet.org/content/mimic-iv-demo/) | 100 patients | ODbL, no sign-up | Real labs and ICD codes — **no notes** |
| MIMIC-IV-Note | ~330k notes | PhysioNet credentialing + CITI course | The real prize, gated |
| n2c2 / i2b2 | task corpora | Harvard DBMI data use agreement | Annotated NLP benchmarks, gated |
| SNOMED CT | full terminology | UMLS licence / member country | Gated; ICD-10-CM is the open substitute |

Download what the pipeline uses:

```bash
mkdir -p data/external/mtsamples data/external/icd10cm
curl -L -o data/external/mtsamples/mtsamples.csv \
  https://huggingface.co/datasets/harishnair04/mtsamples/resolve/main/mtsamples.csv
curl -L -o data/external/icd10cm/icd10cm-2027.zip \
  https://www.cms.gov/files/zip/2027-code-descriptions-tabular-order.zip
unzip -q data/external/icd10cm/icd10cm-2027.zip -d data/external/icd10cm
```

MTSamples are transcription samples rather than verified records, and they carry no answer
key — so they measure whether the vocabulary matches how clinicians write, not whether a
diagnosis is correct. That is still the question the fake data cannot answer.

### What the real text changed

| | Before | After |
|---|---|---|
| Mean recall on 4,966 real notes | 0.91 | **0.93** |
| Type 2 diabetes recall | 0.86 | **0.97** |
| Miscodings caught by the lookalike guard | 0 (step 4 had none) | **141** |

Three bugs only real text could expose: `ST depression` and `respiratory depression` were
being read as the mood disorder, and `subcutaneous emphysema` — air under the skin after
trauma — as COPD. Step 5 had a lookalike guard; step 4 did not. Now both share one list,
built by searching the official codebook rather than by guessing.

### Negation, measured against labels

Deciding whether a note says a patient **has** a condition or says they **do not** is what
controls false positives, and it was seven hand-written words. The NegEx release ships
2,376 assertions labelled by hand on real discharge summaries, so it can be scored.
Held out **by report**, so no summary's wording appears on both sides:

| Approach | Accuracy | Sensitivity | PPV |
|---|---|---|---|
| Seven hand-written words | 0.969 | 0.928 | 0.941 |
| **These rules now** | **0.985** | **0.957** | **0.978** |
| NegEx's own 271 triggers | 0.981 | 0.949 | 0.970 |
| Trained classifier (TF-IDF + logistic regression) | 0.979 | 0.949 | 0.963 |

```bash
python train_negation.py     # reproduces the table
```

**The classifier lost**, and that is the useful result: negation is regular enough that a
good word list beats 1,360 training examples — and a word list can say *which phrase*
negated a finding, which the audit trail needs and a model cannot give.

Three things the old word list could not do:

| | |
|---|---|
| Scope | `Denies chest pain but has asthma.` → asthma **affirmed** |
| Pseudo-negation | `No change in his asthma.` → **affirmed**, not negated |
| Post-negation | `Asthma was ruled out.` → **negated** |

### The vocabulary, checked against strangers

BC5CDR and the NCBI Disease corpus hold 12,850 disease mentions annotated by people with
no connection to this project — the strongest available check, because they chose the
wording. **86% of the relevant mentions map to a study condition**, and every one of the
19 that does not is a deliberate refusal: pulmonary hypertension, diabetes insipidus,
respiratory depression. None is a gap.

That check found a real bug. `non-insulin-dependent diabetes mellitus` — which *is* type 2
diabetes — was being refused, because the lookalike `insulin-dependent diabetes mellitus`
is a substring of it. A lookalike sitting inside the matched term is now ignored, in both
the note reader and the code mapper.

CASI answers a different question: of our abbreviations, only `COPD` appears among the 440
most ambiguous in real notes, and its 14 senses all turn out to be long forms of the same
disease. One is a trap — `COLD` is used for chronic obstructive lung disease, and is
deliberately **not** a synonym here.

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
