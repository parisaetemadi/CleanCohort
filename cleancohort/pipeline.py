"""Run every step in order, from messy raw files to a study-ready dataset."""

from pathlib import Path

import pandas as pd

from .audit import AuditLog
from .step1_clean import clean_all
from .step2_link import link_all
from .step3_deidentify import deidentify_all
from .step4_read_notes import make_anthropic_caller, read_all_notes
from .step5_map_codes import build_conditions, build_measurements, build_observations
from .step6_build_dataset import build_study_dataset
from .step7_quality_report import build_report
from .step8_validate import validate


def load_raw(raw_dir: Path) -> dict:
    return {
        "ehr": pd.read_csv(raw_dir / "ehr_export.csv", dtype=str, keep_default_na=False),
        "notes": pd.read_csv(raw_dir / "clinical_notes.csv", dtype=str),
        "wearable_daily": pd.read_json(raw_dir / "wearable_daily.json", convert_dates=False),
        "device_links": pd.read_csv(raw_dir / "device_links.csv", dtype=str),
        "survey": pd.read_csv(raw_dir / "survey_responses.csv", dtype=str),
    }


def run_pipeline(raw_dir="data/raw", out_dir="data/output", secure_dir="data/secure",
                 method="rules", call_llm=None, verbose=True) -> dict:
    raw_dir, out_dir, secure_dir = Path(raw_dir), Path(out_dir), Path(secure_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    secure_dir.mkdir(parents=True, exist_ok=True)
    say = print if verbose else (lambda *a, **k: None)
    audit = AuditLog()

    raw = load_raw(raw_dir)
    say("Step 1  cleaning ...")
    clean = clean_all(raw, audit)
    say("Step 2  linking ...")
    linked = link_all(clean, audit)
    say("Step 3  de-identifying ...")
    deid = deidentify_all(linked, audit)
    say(f"Step 4  reading notes ({method}) ...")
    if method in ("llm", "hybrid") and call_llm is None:
        call_llm = make_anthropic_caller()
    mentions, note_smoking = read_all_notes(deid["notes"], audit, method=method, call_llm=call_llm)
    say("Step 5  mapping to standard codes ...")
    conditions, unmapped = build_conditions(deid["ehr"], mentions, audit)
    measurements = build_measurements(deid["ehr"], deid["wearable_daily"])
    observations = build_observations(deid["survey"], note_smoking)
    say("Step 6  building study dataset ...")
    dataset = build_study_dataset(deid["person"], conditions, measurements, observations, audit)

    audit_df = audit.to_frame()
    stats = {
        "patients": len(dataset),
        "duplicates_removed": audit.count("duplicate row removed"),
        "unit_conversions": audit.count("converted"),
        "impossible_values": audit.count("impossible value"),
        "identifiers_scrubbed": audit.count("scrubbed"),
        "unmapped_terms": int(unmapped["count"].sum()) if not unmapped.empty else 0,
        "unlinked_records": audit.count("could not link") + audit.count("no consent link") + audit.count("MRN not found"),
        "smoking_conflicts": int(dataset["smoking_conflict"].sum()),
    }
    unlinked = linked["survey_unlinked"][["survey_id", "link_method"]]

    say("Step 7  writing outputs + quality report ...")
    dataset.to_csv(out_dir / "study_dataset.csv", index=False)
    conditions.to_csv(out_dir / "conditions.csv", index=False)
    measurements.to_csv(out_dir / "measurements.csv", index=False)
    observations.to_csv(out_dir / "observations.csv", index=False)
    mentions.to_csv(out_dir / "note_mentions.csv", index=False)
    deid["notes"].to_csv(out_dir / "notes_deidentified.csv", index=False)
    unmapped.to_csv(out_dir / "unmapped_terms.csv", index=False)
    audit_df.to_csv(secure_dir / "audit_log.csv", index=False)
    deid["crosswalk"].to_csv(secure_dir / "crosswalk_KEEP_PRIVATE.csv", index=False)
    (secure_dir / "quality_report.html").write_text(
        build_report(stats, dataset, audit_df, unmapped, mentions, unlinked), encoding="utf-8")

    result = {"dataset": dataset, "conditions": conditions, "measurements": measurements,
              "observations": observations, "mentions": mentions, "unmapped": unmapped,
              "audit": audit_df, "stats": stats, "crosswalk": deid["crosswalk"]}

    truth_path = raw_dir / "truth.csv"
    if truth_path.exists():
        say("Step 8  validating against the answer key ...")
        result["validation"] = validate(dataset, deid["crosswalk"], truth_path)
        result["validation"]["conditions"].to_csv(out_dir / "validation.csv", index=False)
    return result


# ---------------------------------------------------------------------------
# KEY POINTS
# - The order matters: link BEFORE de-identifying (linking needs names/birth dates),
#   and de-identify BEFORE reading notes (so no AI ever sees private info).
# - One AuditLog object travels through every step and collects every change.
# - De-identified outputs go to data/output/ (safe to share with researchers).
# - data/secure/ holds files that can re-identify patients: the crosswalk (MRN <-> study ID),
#   the audit log (it records original values like names) and the quality report.
#   Only the data custodian should see these.
# - method="rules" runs offline. method="llm" uses an AI model with guardrails.
# ---------------------------------------------------------------------------
