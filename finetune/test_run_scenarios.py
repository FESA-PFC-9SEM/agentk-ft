import json

from finetune.run_scenarios import (
    apply_multidoc_patch,
    score_instance,
    score_run,
    summarize_file,
)

CLEAN = {"findings": [], "patch": [], "new_resources": [], "notes": []}


def _finding(rule_id, doc=0, path="/x", message="m", evidence="abcd***"):
    return {"rule_id": rule_id, "severity": "high", "doc": doc, "path": path, "message": message, "evidence": evidence}


def _instance(id_, rule_id, doc=0, match_type="rule_only", match_keyword=None, category="Credenciais Expostas"):
    inst = {"id": id_, "category": category, "line": 1, "description": "d", "rule_id": rule_id, "doc": doc}
    if rule_id is not None:
        inst["match_type"] = match_type
        if match_keyword is not None:
            inst["match_keyword"] = match_keyword
    return inst


def test_apply_multidoc_patch_routes_by_doc_index():
    docs = [{"a": 1}, {"b": 1}]
    patch = [{"doc": 1, "op": "replace", "path": "/b", "value": 2}]

    result, ok = apply_multidoc_patch(docs, patch)

    assert ok is True
    assert result == [{"a": 1}, {"b": 2}]
    assert docs == [{"a": 1}, {"b": 1}]  # original left untouched


def test_apply_multidoc_patch_out_of_range_doc_fails_safely():
    docs = [{"a": 1}]
    patch = [{"doc": 5, "op": "replace", "path": "/a", "value": 2}]

    result, ok = apply_multidoc_patch(docs, patch)

    assert ok is False
    assert result is None


def test_score_instance_out_of_scope_rule_is_always_not_applicable():
    instance = _instance(1, rule_id=None)

    result = score_instance(instance, findings=[_finding("KSEC-001")], patched_findings=[])

    assert result == {"detected": False, "corrected": False, "applicable": False}


def test_score_instance_rule_only_matches_any_finding_of_that_rule():
    instance = _instance(1, rule_id="KSEC-005")
    findings = [_finding("KSEC-005", message="Container image has no pinned tag: nginx")]

    result = score_instance(instance, findings, patched_findings=[])

    assert result["applicable"] is True
    assert result["detected"] is True
    assert result["corrected"] is True  # patched_findings has no matching finding left


def test_score_instance_message_match_type_requires_keyword_in_message():
    instance = _instance(1, rule_id="KSEC-001", match_type="message", match_keyword="DB_PASSWORD")
    findings = [_finding("KSEC-001", message="Plaintext credential: value under sensitive key 'DB_PASSWORD'")]
    other_findings = [_finding("KSEC-001", message="Plaintext credential: value under sensitive key 'API_KEY'")]

    assert score_instance(instance, findings, [])["detected"] is True
    assert score_instance(instance, other_findings, [])["detected"] is False


def test_score_instance_evidence_prefix_match_type():
    instance = _instance(1, rule_id="KSEC-001", match_type="evidence_prefix", match_keyword="mysq")
    findings = [_finding("KSEC-001", evidence="mysq***")]
    other_findings = [_finding("KSEC-001", evidence="post***")]

    assert score_instance(instance, findings, [])["detected"] is True
    assert score_instance(instance, other_findings, [])["detected"] is False


def test_score_instance_not_corrected_when_finding_survives_patch():
    instance = _instance(1, rule_id="KSEC-001", match_type="message", match_keyword="DB_PASSWORD")
    findings = [_finding("KSEC-001", message="...'DB_PASSWORD'")]
    still_present = [_finding("KSEC-001", message="...'DB_PASSWORD'")]

    result = score_instance(instance, findings, still_present)

    assert result["detected"] is True
    assert result["corrected"] is False


def test_score_instance_doc_mismatch_does_not_count():
    instance = _instance(1, rule_id="KSEC-005", doc=1)
    findings = [_finding("KSEC-005", doc=0)]  # right rule, wrong document

    result = score_instance(instance, findings, [])

    assert result["detected"] is False


def test_score_run_clean_response_detects_nothing():
    docs = [{"kind": "Pod", "spec": {}}]
    instances = [_instance(1, rule_id="KSEC-005", category="Imagem sem Tag")]

    result = score_run(docs, instances, json.dumps(CLEAN))

    assert result["schema_valid"] is True
    assert result["instances"][1]["detected"] is False


def test_score_run_detects_and_corrects_image_tag():
    docs = [{"kind": "Pod", "spec": {"containers": [{"image": "nginx"}]}}]
    instances = [_instance(1, rule_id="KSEC-005", category="Imagem sem Tag")]
    response = {
        "findings": [_finding("KSEC-005", message="Container image has no pinned tag: nginx")],
        "patch": [{"doc": 0, "op": "replace", "path": "/spec/containers/0/image", "value": "nginx:1.0.0"}],
        "new_resources": [],
        "notes": [],
    }

    result = score_run(docs, instances, json.dumps(response))

    assert result["patch_applies"] is True
    assert result["instances"][1]["detected"] is True
    assert result["instances"][1]["corrected"] is True


def test_score_run_invalid_json_marks_every_instance_not_applicable_result():
    instances = [_instance(1, rule_id="KSEC-005", category="Imagem sem Tag")]

    result = score_run([{"a": 1}], instances, "not json")

    assert result["schema_valid"] is False
    assert result["instances"][1]["detected"] is False


def test_summarize_file_averages_across_runs():
    instances = [
        _instance(1, rule_id="KSEC-001", category="Credenciais Expostas"),
        _instance(2, rule_id="KSEC-005", category="Imagem sem Tag"),
        _instance(3, rule_id=None, category="Erro de Sintaxe/Config"),
    ]
    file_results = [
        {
            "schema_valid": True,
            "instances": {
                1: {"detected": True, "corrected": True, "applicable": True},
                2: {"detected": True, "corrected": False, "applicable": True},
                3: {"detected": False, "corrected": False, "applicable": False},
            },
        },
        {
            "schema_valid": True,
            "instances": {
                1: {"detected": True, "corrected": True, "applicable": True},
                2: {"detected": False, "corrected": False, "applicable": True},
                3: {"detected": False, "corrected": False, "applicable": False},
            },
        },
    ]

    summary = summarize_file(file_results, instances)

    assert summary["runs"] == 2
    assert summary["errors"] == 3
    assert summary["avg_detected"] == 1.5  # (2 + 1) / 2
    assert summary["avg_corrected"] == 1.0  # (1 + 0) / 2
    assert summary["detected_pct"] == 50.0  # 1.5 / 3
    assert summary["by_category"] == {
        "Credenciais Expostas": 1,
        "Imagem sem Tag": 1,
        "Erro de Sintaxe/Config": 1,
    }
    assert summary["schema_valid_rate"] == "2/2"


def test_score_run_uses_cross_document_detection_for_service_selectors():
    docs = [
        {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "s"}, "spec": {"selector": {"app": "wbe"}}},
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "d"},
            "spec": {
                "selector": {"matchLabels": {"app": "web"}},
                "template": {"metadata": {"labels": {"app": "web"}}, "spec": {"containers": [{"name": "c", "image": "nginx:1.25"}]}},
            },
        },
    ]
    instance = _instance(1, "KSEC-006", doc=0, category="Erro de Sintaxe/Config")
    output = dict(
        CLEAN,
        findings=[_finding("KSEC-006", doc=0, path="/spec/selector/app")],
        patch=[{"doc": 0, "op": "replace", "path": "/spec/selector/app", "value": "web"}],
    )
    result = score_run(docs, [instance], json.dumps(output))
    assert result["instances"][1] == {"detected": True, "corrected": True, "corrected_per_finding": True, "applicable": True}

    unfixed = dict(output, patch=[])
    result = score_run(docs, [instance], json.dumps(unfixed))
    assert result["instances"][1]["corrected"] is False


def test_per_finding_correction_survives_an_unrelated_bad_op():
    # 5-nginx shape: a correct fix plus one op on a path that doesn't exist.
    # Strict scoring rejects the whole patch; per-finding scoring credits the fix.
    docs = [
        {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": "p"},
            "spec": {"containers": [{"name": "c", "image": "nginx"}]},
        }
    ]
    instance = _instance(1, "KSEC-005", doc=0, category="Imagem sem Tag")
    output = dict(
        CLEAN,
        findings=[
            _finding("KSEC-005", doc=0, path="/spec/containers/0/image"),
            _finding("KSEC-006", doc=0, path="/spec/template/spec/labels/app"),
        ],
        patch=[
            {"doc": 0, "op": "replace", "path": "/spec/containers/0/image", "value": "nginx:1.25"},
            {"doc": 0, "op": "replace", "path": "/spec/template/spec/labels/app", "value": "x"},
        ],
    )
    result = score_run(docs, [instance], json.dumps(output))
    assert result["patch_applies"] is False
    assert result["instances"][1]["corrected"] is False
    assert result["instances"][1]["corrected_per_finding"] is True


def test_per_finding_correction_needs_a_matching_finding():
    docs = [{"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "p"}, "spec": {"containers": [{"name": "c", "image": "nginx"}]}}]
    instance = _instance(1, "KSEC-005", doc=0, category="Imagem sem Tag")
    output = dict(CLEAN, patch=[{"doc": 0, "op": "replace", "path": "/spec/containers/0/image", "value": "nginx:1.25"}])
    result = score_run(docs, [instance], json.dumps(output))
    # The strict score still credits a fix without a finding; per-finding follows it.
    assert result["instances"][1]["corrected"] == result["instances"][1]["corrected_per_finding"]


def test_ground_truth_matches_the_scenario_files():
    # Every listed error must be one the detectors actually find in ITS file
    # -- catches a test_cases.yaml edit that attaches errors to the wrong
    # file or names a rule that doesn't apply.
    from pathlib import Path

    import yaml

    from dataset.detect import detect_file
    from finetune.run_scenarios import _finding_matches_instance, load_test_cases

    cases = load_test_cases(Path("scenarios/test_cases.yaml"))
    for name, instances in cases.items():
        findings = [f.to_dict() for f in detect_file(list(yaml.safe_load_all(Path("scenarios", name).read_text())))]
        for inst in instances:
            assert inst["rule_id"] is not None, (name, inst["id"])
            assert any(_finding_matches_instance(f, inst) for f in findings), (name, inst["id"])
