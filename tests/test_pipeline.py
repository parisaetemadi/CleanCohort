import json

import pandas as pd

from cleancohort.audit import AuditLog
from cleancohort.pipeline import run_pipeline
from cleancohort.step0_make_fake_data import main as make_fake_data
from cleancohort.step1_clean import (parse_date, normalize_mrn, parse_weight_kg, parse_glucose_mmol,
                                    parse_smoking_answer, parse_health_word)
from cleancohort.step3_deidentify import scrub_note
from cleancohort.step4_read_notes import (extract_with_rules, extract_with_llm,
                                          mention_status, negation_trigger)
from cleancohort.step5_map_codes import map_term


def test_dates_in_many_formats_become_one():
    for text in ["1985-03-12", "03/12/1985", "12-Mar-1985", "1985.03.12"]:
        assert str(parse_date(text)) == "1985-03-12"


def test_ids_and_units():
    assert normalize_mrn(" mrn00012 ") == 12
    assert parse_weight_kg("180 lb")[0] == 81.6
    assert parse_glucose_mmol("110 mg/dL")[0] == 6.1


def test_survey_text():
    assert parse_smoking_answer("never smoked") == "never"
    assert parse_smoking_answer("used to") == "former"
    assert parse_smoking_answer("Yes, daily") == "current"
    assert parse_health_word("V. good") == 4


def test_note_status_affirmed_negated_family():
    mentions, smoking = extract_with_rules("PMH significant for HTN. Denies COPD. Mother has asthma. Ex-smoker.")
    status = {m["concept"]: m["status"] for m in mentions}
    assert status == {"hypertension": "affirmed", "copd": "negated", "asthma": "family"}
    assert smoking["status"] == "former"


def test_code_mapping_exact_fuzzy_unmapped():
    assert map_term("HTN") == ("hypertension", "exact")
    assert map_term("Hypertensn")[0] == "hypertension"
    assert map_term("low mood disorder") == (None, "unmapped")


def test_deidentification_removes_name_and_phone():
    text, n = scrub_note("Ava Smith is 57. Contact: (403) 555-1234. Smith agrees.", "Ava Smith")
    assert "Ava" not in text and "Smith" not in text and "555" not in text
    assert n >= 3


def test_llm_guardrails_reject_made_up_evidence():
    note = "PMH significant for HTN. Never smoker."
    fake_answer = json.dumps({
        "conditions": [
            {"concept": "hypertension", "evidence": "HTN", "status": "affirmed"},
            {"concept": "type2_diabetes", "evidence": "diabetes", "status": "affirmed"},
            {"concept": "cancer", "evidence": "HTN", "status": "affirmed"},
        ],
        "smoking": {"status": "never", "evidence": "Never smoker"},
    })
    audit = AuditLog()
    mentions, smoking = extract_with_llm(note, lambda prompt: fake_answer, audit, "N1")
    assert [m["concept"] for m in mentions] == ["hypertension"]
    assert smoking["status"] == "never"
    assert audit.count("failed guardrail") == 2


def test_full_pipeline_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_fake_data(n_patients=40)
    result = run_pipeline(verbose=False)
    data = result["dataset"]
    assert len(data) == 40
    assert data["study_id"].is_unique
    assert not {"name", "mrn", "dob"} & set(data.columns)
    notes = pd.read_csv(tmp_path / "data/output/notes_deidentified.csv")
    assert not notes["note_text"].str.contains(r"\(403\)").any()
    v = result["validation"]
    assert (v["conditions"]["FP"] == 0).all()
    with_notes = v["conditions"][v["conditions"]["method"] == "EHR + notes"]["sensitivity"].mean()
    ehr_only = v["conditions"][v["conditions"]["method"] == "EHR codes only"]["sensitivity"].mean()
    assert with_notes > ehr_only
    assert v["summary"]["wearable_link_rate"] == 1.0
    assert (tmp_path / "data/secure/quality_report.html").exists()


def test_pipeline_survives_llm_finding_no_smoking(tmp_path, monkeypatch):
    """A reader that finds no smoking status anywhere must not crash step 6."""
    monkeypatch.chdir(tmp_path)
    make_fake_data(n_patients=40)
    silent_llm = lambda prompt: '{"conditions": [], "smoking": null}'
    result = run_pipeline(method="llm", call_llm=silent_llm, verbose=False)
    data = result["dataset"]
    assert len(data) == 40
    assert data["smoking_conflict"].sum() == 0
    assert data["smoking_status"].notna().any()


def test_hybrid_only_pays_for_notes_rules_cannot_read(tmp_path, monkeypatch):
    """Hybrid must call the paid model only when the rules come up empty."""
    monkeypatch.chdir(tmp_path)
    make_fake_data(n_patients=40)
    calls = []
    def counting_llm(prompt):
        calls.append(prompt)
        return '{"conditions": [], "smoking": null}'

    hybrid = run_pipeline(method="hybrid", call_llm=counting_llm, verbose=False)
    rules = run_pipeline(method="rules", out_dir="data/o2", secure_dir="data/s2", verbose=False)

    notes = pd.read_csv(tmp_path / "data/output/notes_deidentified.csv")
    assert len(calls) < len(notes)
    escalations = int(hybrid["audit"]["rule"].str.contains("escalated", na=False).sum())
    assert escalations == len(calls)
    h = hybrid["validation"]["conditions"].query("method == 'EHR + notes'")["sensitivity"]
    r = rules["validation"]["conditions"].query("method == 'EHR + notes'")["sensitivity"]
    assert h.mean() >= r.mean()


def test_mapper_finds_condition_inside_real_diagnosis_text():
    """Real EHR text wraps the condition name; whole-string matching missed all of it."""
    assert map_term("Childhood asthma")[0] == "asthma"
    assert map_term("Pulmonary emphysema (disorder)")[0] == "copd"
    assert map_term("Major depression disorder")[0] == "depression"


def test_mapper_refuses_clinical_lookalikes():
    """Substring matching must not miscode a different disease that shares the word."""
    for term in ["Pulmonary hypertension", "Ocular hypertension", "Diabetes insipidus",
                 "Gestational diabetes", "Pre-diabetes", "Prediabetes",
                 "Family history of diabetes", "Chronic venous hypertension",
                 "Subcutaneous emphysema", "Drug-induced diabetes",
                 "Type 1 diabetes", "Type I diabetes"]:
        assert map_term(term)[0] is None, term


def test_diabetes_complications_count_as_cases():
    """A deliberate case-definition choice, not an accident.

    "Diabetic retinopathy" and "Diabetic renal disease" are complications OF diabetes, so
    a patient carrying one has the disease. Counting them widens the definition and is the
    reason "diabetic" is a synonym; the type 1 and gestational forms are still refused, so
    the widening does not pull in the other diabetes types."""
    for term in ["Diabetic renal disease (disorder)", "Diabetic retinopathy",
                 "Diabetic gastroparesis"]:
        assert map_term(term)[0] == "type2_diabetes", term
    assert map_term("Type 1 diabetic")[0] is None


def test_bare_diabetes_maps_but_typo_tier_cannot_bypass_the_guard():
    """"Type 1 diabetes" is one character from "type 2 diabetes", so the fuzzy tier will
    match it. The lookalike guard must apply to every tier, not just substring matching."""
    assert map_term("Diabetes")[0] == "type2_diabetes"
    assert map_term("Type 1 diabetes")[0] is None
    assert map_term("Hypertensn")[0] == "hypertension"


def _status(sentence, term):
    at = sentence.lower().find(term.lower())
    return mention_status(sentence, at, at + len(term))


def test_negation_respects_scope_and_pseudo_negations():
    """The three things seven hand-written words could not do."""
    # a conjunction ends the trigger's reach, so the second finding survives
    assert _status("Denies chest pain but has asthma.", "asthma") == "affirmed"
    # a pseudo-negation is not a negation
    assert _status("No change in his asthma since last visit.", "asthma") == "affirmed"
    # the trigger can follow the finding
    assert _status("Asthma was ruled out.", "Asthma") == "negated"
    # and the plain cases still work
    assert _status("Denies asthma.", "asthma") == "negated"
    assert _status("PMH significant for asthma.", "asthma") == "affirmed"


def test_negation_trigger_is_reported_for_the_audit_log():
    """A dropped mention has to be explainable, so the phrase itself comes back."""
    s = "The patient does not have asthma."
    assert negation_trigger(s, s.find("asthma"), s.find("asthma") + 6) == "does not"
    assert negation_trigger("He has asthma.", 7, 13) is None
