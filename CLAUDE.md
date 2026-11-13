# CLAUDE.md: context for Claude Code

## About the user
- Parisa: BSc Neuroscience, applying to MSc Epidemiology / MPH. Knows some Python and R.
- **Goal: learn while building.** Explain what you change and why, in simple words.
- Keep answers brief and concise. Use tables and visuals (diagrams) when explaining.
- Don't ask permission before web searches; just search.

## Code style rules
- **Never comment line by line, or chunk by chunk.** Do not narrate the code as you write
  it. A comment inside the code earns its place only where the code cannot say the thing
  itself: a non-obvious reason, a trap, a decision someone would otherwise undo.
- **Instead, every file ends with a `KEY POINTS` bullet list** describing what the code in
  that file does and why (see any `cleancohort/step*.py` for the format). This is where the
  explaining goes. Update it whenever the file changes.
- This applies to every file, not just Python: scripts, tests and config too.
- Simple pandas, one small function per job, readable names. Avoid clever abstractions.
- Every change to data must go through `AuditLog` (`cleancohort/audit.py`).

## What this project is
CleanCohort is a
prototype of a startup idea inspired by the YC company Sutrix: turn messy health data (EHR, doctor
notes, wearables, surveys) into de-identified, study-ready datasets with an audit trail.

Why not just an LLM: privacy (no PHI to chatbots), scale, consistency, audit trail, linking,
and validation. The LLM is one part (note reading), wrapped in guardrails.

## How it works
`run_demo.py` → `cleancohort/pipeline.py` runs these steps:
0 fake data + answer key → 1 clean → 2 link → 3 de-identify → 4 read notes → 5 map codes →
6 one-row-per-patient dataset → 7 HTML quality report → 8 validate (sensitivity/PPV).

Order matters: link before de-identifying (needs names/DOB), and de-identify before reading notes
(no AI sees identifiers). `data/secure/` holds identifiable files (crosswalk, audit log, report).

## Commands
- `python run_demo.py` (rules) / `python run_demo.py --use-llm` (needs `anthropic` + `ANTHROPIC_API_KEY`)
- `python -m pytest -q` (8 tests, all passing)

## Status
- Works end to end on 40 fake patients. EHR + notes sensitivity is 1.0 for all 5 conditions;
  false positives are 0. Treat that as a ceiling, not a score: step0 writes the notes from
  the same synonym list vocab.py searches, so the rules cannot miss.
- Also runs on real-shaped data: `run_synthea.py` (300 Synthea patients, real SNOMED/LOINC)
  and 4,966 real clinical notes from MTSamples, where the mean recall floor is 0.95.
- The LLM note reader has been tested only with a fake LLM; it still needs a real API run.
  `--hybrid` calls it only for notes the rules cannot read, so the bill stays near zero.

## Free data on disk (gitignored, see README for the download commands)
- `data/external/mtsamples/` - 4,966 real clinical notes, CC0. The only open free text found.
- `data/external/icd10cm/` - official FY2027 ICD-10-CM, 74,879 codes. Source of LOOKALIKES.
- `data/external/synthea/` - ~1k synthetic patients with real code systems.
- `data/external/mimic4demo/` - 100 real patients, structured only, no notes.
- `data/external/negex/` - 2,376 hand-labelled affirmed/negated assertions on 116 real
  discharge summaries. The only LABELLED data in the project. Not vendored into the repo:
  the source has no stated licence and says research use only, so our trigger lists in
  step4 are written ourselves and merely scored against it.
- `data/external/bioscope/` - negation + scope annotations, but the clinical half ships
  anonymised to `*` and needs the 2007 CMC corpus to reconstruct. Abstracts only, so far.

## Scores worth keeping straight
- Fake data (`run_demo.py`): sensitivity 1.0 - a ceiling, not a score. Circular by design.
- Real notes (`eval_real_notes.py`): mean recall floor 0.95 over 4,966 MTSamples notes,
  and 138 miscodings stopped by the lookalike guard.
- Negation (`train_negation.py`): accuracy 0.985 on held-out labelled reports, which beat
  both NegEx's own 271 triggers and a trained classifier.

## Next steps (ideas, in rough order)
1. Do the exercises in README.md together (learning).
2. Run the LLM note reader for real and compare it with the rules (agreement table).
   MTSamples is the right test set: real text, and the rules already have a score on it.
3. ~~Synthea~~ done. Next real data is MIMIC-IV-Note, which needs PhysioNet credentialing
   plus the CITI course - worth starting early, the approval takes weeks.
4. Output to the standard OMOP CDM format.
5. Stronger de-identification (e.g., Microsoft Presidio) and a scale test (10k+ patients).
6. A simple web demo for pitching (upload CSV → see report).
