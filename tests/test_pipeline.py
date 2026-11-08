import json

import pandas as pd

from cleancohort.audit import AuditLog
from cleancohort.pipeline import run_pipeline
from cleancohort.step0_make_fake_data import main as make_fake_data
from cleancohort.step1_clean import (parse_date, normalize_mrn, parse_weight_kg, parse_glucose_mmol,
                                    parse_smoking_answer, parse_health_word)
from cleancohort.step3_deidentify import scrub_note
from cleancohort.step4_read_notes import extract_with_rules, extract_with_llm
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
