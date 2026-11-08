"""Make fake data, run the full pipeline, print the results."""

import argparse

import pandas as pd

from cleancohort.pipeline import run_pipeline
from cleancohort.step0_make_fake_data import main as make_fake_data


def main():
    parser = argparse.ArgumentParser(description="CleanCohort demo")
    parser.add_argument("--use-llm", action="store_true", help="read notes with an AI model (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--patients", type=int, default=40)
    args = parser.parse_args()

    make_fake_data(n_patients=args.patients)
    result = run_pipeline(method="llm" if args.use_llm else "rules")

    pd.set_option("display.width", 160, "display.max_columns", 12)
    print("\n=== What happened ===")
    for k, v in result["stats"].items():
        print(f"  {k:<22} {v}")
    print("\n=== First rows of the study dataset ===")
    print(result["dataset"][["study_id", "age_at_index", "sex", "has_hypertension", "hypertension_source",
                             "smoking_status", "steps_mean", "completeness_pct"]].head(8).to_string(index=False))
    if "validation" in result:
        print("\n=== Accuracy vs answer key ===")
        print(result["validation"]["conditions"].to_string(index=False))
        for k, v in result["validation"]["summary"].items():
            print(f"  {k:<28} {v}")
    print("\nOpen data/secure/quality_report.html in your browser to see the full report.")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# KEY POINTS
# - Run:  python run_demo.py              (rules, works offline)
# - Run:  python run_demo.py --use-llm    (AI note reading; needs an API key)
# - Every run re-creates the fake data, so results are reproducible (fixed random seed).
# ---------------------------------------------------------------------------
