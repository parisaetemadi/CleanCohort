"""Step 6 - Build the study-ready dataset: one row per patient."""

import pandas as pd

from .vocab import CONDITIONS

ANALYSIS_COLUMNS = [
    "age_at_index", "sex", "weight_kg_latest", "glucose_mmol_latest", "sbp_mean", "dbp_mean",
    "smoking_status", "steps_mean", "resting_hr_mean", "sleep_hours_mean",
    "self_rated_health", "activity_days",
]


def condition_flags(person, conditions):
    out = person[["study_id"]].copy()
    for concept in CONDITIONS:
        rows = conditions[conditions["concept"] == concept]
        sources = rows.groupby("study_id")["source"].agg(lambda s: "+".join(sorted(set(s))))
        out[f"has_{concept}"] = out["study_id"].isin(sources.index).astype(int)
        out[f"{concept}_source"] = out["study_id"].map(sources).fillna("")
    return out


def measure_col(table, name):
    """A measure absent from the whole dataset leaves no column - return an empty one."""
    s = table.get(name)
    return pd.Series(index=table.index, dtype=float) if s is None else s


def ehr_summary(measurements):
    ehr = measurements[measurements["source"] == "ehr"].sort_values("date")
    latest = ehr.groupby(["study_id", "measure"])["value"].last().unstack()
    mean = ehr.groupby(["study_id", "measure"])["value"].mean().unstack()
    visits = ehr.groupby("study_id")["date"].nunique().rename("n_visits")
    return pd.DataFrame({
        "weight_kg_latest": measure_col(latest, "weight_kg"),
        "glucose_mmol_latest": measure_col(latest, "glucose_mmol"),
        "sbp_mean": measure_col(mean, "sbp").round(1),
        "dbp_mean": measure_col(mean, "dbp").round(1),
    }).join(visits)


def wearable_summary(measurements):
    w = measurements[measurements["source"] == "wearable"]
    mean = w.groupby(["study_id", "measure"])["value"].mean().unstack()
    days = w.groupby("study_id")["date"].nunique().rename("wearable_days")
    return pd.DataFrame({
        "steps_mean": measure_col(mean, "steps").round(0),
        "resting_hr_mean": measure_col(mean, "resting_hr").round(1),
        "sleep_hours_mean": measure_col(mean, "sleep_hours").round(1),
    }).join(days)


def smoking_summary(observations, audit):
    smk = observations[observations["item"] == "smoking_status"]
    by_source = smk.groupby(["study_id", "source"])["value"].last().unstack()
    empty = pd.Series(index=by_source.index, dtype=object)
    notes = by_source.get("notes", empty)
    survey = by_source.get("survey", empty)
    out = pd.DataFrame(index=by_source.index)
    out["smoking_status"] = notes.combine_first(survey)
    out["smoking_conflict"] = (notes.notna() & survey.notna() & (notes != survey)).astype(int)
    for sid in out.index[out["smoking_conflict"] == 1]:
        audit.log("6_build", "notes+survey", sid, "smoking_status",
                  f"survey={survey[sid]}", f"notes={notes[sid]}",
                  "sources disagree - used clinician note, flagged conflict")
    return out


def survey_summary(observations):
    s = observations[(observations["source"] == "survey") & (observations["item"] != "smoking_status")]
    return s.pivot_table(index="study_id", columns="item", values="value", aggfunc="last")


def build_study_dataset(person, conditions, measurements, observations, audit):
    data = person.set_index("study_id")
    data = data.join(condition_flags(person, conditions).set_index("study_id"))
    data = data.join(ehr_summary(measurements))
    data = data.join(wearable_summary(measurements))
    data = data.join(smoking_summary(observations, audit))
    data = data.join(survey_summary(observations))
    for col in ANALYSIS_COLUMNS:
        if col not in data:
            data[col] = pd.NA
    data["smoking_conflict"] = data["smoking_conflict"].fillna(0).astype(int)
    data["wearable_days"] = data["wearable_days"].fillna(0).astype(int)
    data["completeness_pct"] = (data[ANALYSIS_COLUMNS].notna().mean(axis=1) * 100).round(0)
    return data.reset_index()


# ---------------------------------------------------------------------------
# KEY POINTS
# - Research usually needs a "wide" table: one row per patient, one column per variable.
# - Conditions become 0/1 flags (has_hypertension = 1), plus a column saying where the
#   evidence came from: "ehr", "notes" or "ehr+notes".
# - Repeated measures are summarized: latest weight/glucose, mean blood pressure,
#   mean daily steps. Choosing "latest" vs "mean" is a study-design decision.
# - Smoking: clinician note wins over the survey when both exist. If they disagree,
#   smoking_conflict = 1 so the analyst can check (and it's in the audit log).
# - completeness_pct = how many analysis variables are filled in. Low completeness
#   can bias results (missing data is rarely random).
# ---------------------------------------------------------------------------
