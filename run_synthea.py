"""Run the pipeline on real-shaped Synthea data instead of our own fake data."""

import argparse

import pandas as pd

from cleancohort.load_synthea import main as load_synthea
from cleancohort.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="CleanCohort on Synthea data")
    parser.add_argument("--csv-dir", default="data/external/synthea/csv")
    parser.add_argument("--patients", type=int, default=300)
    parser.add_argument("--visits", type=int, default=3)
    args = parser.parse_args()

    raw_dir = load_synthea(csv_dir=args.csv_dir, limit=args.patients, visits_per_person=args.visits)
    result = run_pipeline(raw_dir=raw_dir, out_dir="data/output_synthea",
                          secure_dir="data/secure_synthea", method="rules")

    pd.set_option("display.width", 160, "display.max_columns", 12)
    print("\n=== What happened ===")
    for k, v in result["stats"].items():
        print(f"  {k:<22} {v}")

    print("\n=== Study dataset ===")
    print(result["dataset"][["study_id", "age_at_index", "sex", "has_type2_diabetes",
                             "has_hypertension", "weight_kg_latest", "glucose_mmol_latest",
                             "completeness_pct"]].head(8).to_string(index=False))

    print("\n=== Condition prevalence found ===")
    for c in ["type2_diabetes", "hypertension", "asthma", "copd", "depression"]:
        n = int(result["dataset"][f"has_{c}"].sum())
        print(f"  {c:<16} {n:>4} / {len(result['dataset'])}  ({n / len(result['dataset']):.1%})")

    print("\n=== Top unmapped terms (real vocabulary gaps) ===")
    if result["unmapped"].empty:
        print("  none")
    else:
        print(result["unmapped"].head(12).to_string(index=False))

    print("\nNo truth.csv for real data, so step 8 is skipped - real studies chart-review a sample.")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# KEY POINTS
# - Run: python run_synthea.py   (after downloading the Synthea sample CSVs)
# - Uses method="rules": Synthea has no notes, so there is nothing for an AI to read and
#   nothing to pay for.
# - The unmapped-terms table is the most useful output here - it shows which real diagnosis
#   wordings vocab.py does not know yet, measured on data we did not write.
# - Outputs go to data/output_synthea/ and data/secure_synthea/ so the fake-data run is kept.
# ---------------------------------------------------------------------------
