"""Step 8 - Validate against the answer key: how accurate is the pipeline?"""

import pandas as pd

from .vocab import CONDITIONS


def _rates(truth: pd.Series, found: pd.Series):
    tp = int(((truth == 1) & (found == 1)).sum())
    fp = int(((truth == 0) & (found == 1)).sum())
    fn = int(((truth == 1) & (found == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    return tp, fp, fn, sens, ppv


def validate(dataset: pd.DataFrame, crosswalk: pd.DataFrame, truth_path) -> dict:
    truth = pd.read_csv(truth_path)
    truth = truth.merge(crosswalk, on="mrn")
    d = dataset.merge(truth, on="study_id", suffixes=("", "_true"))

    rows = []
    for c in CONDITIONS:
        t = d[c].astype(int)
        ehr_only = d[f"{c}_source"].str.contains("ehr").astype(int)
        with_notes = d[f"has_{c}"]
        for label, found in [("EHR codes only", ehr_only), ("EHR + notes", with_notes)]:
            tp, fp, fn, sens, ppv = _rates(t, found)
            rows.append({"condition": c, "method": label, "true_cases": int(t.sum()),
                         "TP": tp, "FP": fp, "FN": fn,
                         "sensitivity": round(sens, 2), "PPV": round(ppv, 2)})
    conditions = pd.DataFrame(rows)

    smk = d.dropna(subset=["smoking_status"])
    smoking_acc = (smk["smoking_status"] == smk["smoking"]).mean()

    w = d.dropna(subset=["weight_kg_latest"])
    g = d.dropna(subset=["glucose_mmol_latest"])
    summary = {
        "smoking_accuracy": round(float(smoking_acc), 2),
        "weight_mean_abs_error_kg": round(float((w["weight_kg_latest"] - w["weight_kg"]).abs().mean()), 2),
        "glucose_mean_abs_error_mmol": round(float((g["glucose_mmol_latest"] - g["glucose_mmol"]).abs().mean()), 2),
        "survey_link_rate": round(float(d.loc[d["did_survey"], "self_rated_health"].notna().mean()), 2),
        "wearable_link_rate": round(float((d.loc[d["has_wearable"], "wearable_days"] > 0).mean()), 2),
    }
    return {"conditions": conditions, "summary": summary}


# ---------------------------------------------------------------------------
# KEY POINTS
# - This is classic epidemiology: compare a measurement tool to a gold standard.
# - Sensitivity = of the people who truly have the condition, how many did we find?
# - PPV (positive predictive value) = of the people we flagged, how many truly have it?
# - Compare "EHR codes only" vs "EHR + notes": reading notes should raise sensitivity
#   (finds conditions the codes missed) without lowering PPV (thanks to negation and
#   family-history handling).
# - With real data there is no answer key, so teams hand-review a random sample
#   ("chart review") to estimate these same numbers.
# - Small weight/glucose errors are expected: the fake data adds visit-to-visit noise.
# ---------------------------------------------------------------------------
