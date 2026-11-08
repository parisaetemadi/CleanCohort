# CLAUDE.md: context for Claude Code

## About the user
- Parisa: BSc Neuroscience, applying to MSc Epidemiology / MPH. Knows some Python and R.
- **Goal: learn while building.** Explain what you change and why, in simple words.
- Keep answers brief and concise. Use tables and visuals (diagrams) when explaining.
- Don't ask permission before web searches; just search.

## Code style rules
- **No comments on each line of code.** Keep the code clean.
- Put a `KEY POINTS` bullet list at the **end of each file** explaining what it does and why
  (see any `cleancohort/step*.py` for the format). Update it whenever the file changes.
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
  false positives are 0.
- The LLM note reader has been tested only with a fake LLM; it still needs a real API run.

## Next steps (ideas, in rough order)
1. Do the exercises in README.md together (learning).
2. Run the LLM note reader for real and compare it with the rules (agreement table).
3. Use real public data: Synthea (synthetic EHR), then MIMIC-IV (needs PhysioNet + CITI course).
4. Output to the standard OMOP CDM format.
5. Stronger de-identification (e.g., Microsoft Presidio) and a scale test (10k+ patients).
6. A simple web demo for pitching (upload CSV → see report).
