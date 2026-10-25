import json
from pathlib import Path

import yaml

from dataset.detect import detect_file
from finetune.run_scenarios import _finding_matches_instance, analyze, apply_multidoc_patch, load_test_cases, score_run

CLEAN = {"findings": [], "patch": [], "new_resources": [], "notes": []}


def _finding(rule_id, doc=0, path="/x", message="m", evidence="abcd***"):
    return {"rule_id": rule_id, "severity": "high", "doc": doc, "path": path, "message": message, "evidence": evidence}


def _instance(id_, rule_id, doc=0, match_type="rule_only", match_keyword=None, category="Credenciais Expostas"):
    inst = {"id": id_, "category": category, "line": 1, "description": "d", "rule_id": rule_id, "doc": doc, "match_type": match_type}
    if match_keyword is not None:
        inst["match_keyword"] = match_keyword
    return inst


def _pod(image="nginx"):
    return {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "p"}, "spec": {"containers": [{"name": "c", "image": image}]}}


def test_apply_multidoc_patch_routes_by_doc_index():
    docs = [{"a": 1}, {"b": 1}]
    result, ok = apply_multidoc_patch(docs, [{"doc": 1, "op": "replace", "path": "/b", "value": 2}])
    assert ok is True
    assert result == [{"a": 1}, {"b": 2}]
    assert docs == [{"a": 1}, {"b": 1}]  # original left untouched


def test_apply_multidoc_patch_out_of_range_doc_fails_safely():
    result, ok = apply_multidoc_patch([{"a": 1}], [{"doc": 5, "op": "replace", "path": "/a", "value": 2}])
    assert ok is False and result is None


def test_matching_rules():
    assert _finding_matches_instance(_finding("KSEC-001"), _instance(1, "KSEC-001"))
    assert not _finding_matches_instance(_finding("KSEC-005"), _instance(1, "KSEC-001"))
    assert not _finding_matches_instance(_finding("KSEC-001", doc=1), _instance(1, "KSEC-001", doc=0))
    by_message = _instance(1, "KSEC-001", match_type="message", match_keyword="DB_PASSWORD")
    assert _finding_matches_instance(_finding("KSEC-001", message="key 'DB_PASSWORD'"), by_message)
    assert not _finding_matches_instance(_finding("KSEC-001", message="key 'API_KEY'"), by_message)
    by_evidence = _instance(1, "KSEC-001", match_type="evidence_prefix", match_keyword="mysq")
    assert _finding_matches_instance(_finding("KSEC-001", evidence="mysq***"), by_evidence)
    assert not _finding_matches_instance(_finding("KSEC-001", evidence="post***"), by_evidence)


def test_clean_answer_detects_nothing():
    result = score_run([_pod()], [_instance(1, "KSEC-005")], json.dumps(CLEAN))
    assert result["instances"][1] == {"detected": False, "corrected": False}


def test_detected_and_corrected():
    instance = _instance(1, "KSEC-005", category="Imagem sem Tag")
    output = dict(
        CLEAN,
        findings=[_finding("KSEC-005", path="/spec/containers/0/image")],
        patch=[{"doc": 0, "op": "replace", "path": "/spec/containers/0/image", "value": "nginx:1.25"}],
    )
    result = score_run([_pod()], [instance], json.dumps(output))
    assert result["instances"][1] == {"detected": True, "corrected": True}
    assert result["right"] == 1 and result["false_positives"] == 0


def test_detected_but_fix_does_not_remove_the_error():
    instance = _instance(1, "KSEC-005")
    output = dict(
        CLEAN,
        findings=[_finding("KSEC-005", path="/spec/containers/0/image")],
        patch=[{"doc": 0, "op": "replace", "path": "/spec/containers/0/image", "value": "nginx:latest"}],
    )
    result = score_run([_pod()], [instance], json.dumps(output))
    assert result["instances"][1] == {"detected": True, "corrected": False}


def test_an_unrelated_bad_op_does_not_cancel_a_correct_fix():
    # 5-nginx shape: one fix is right, another op points at a path that
    # doesn't exist. Each fix is applied on its own.
    instance = _instance(1, "KSEC-005")
    output = dict(
        CLEAN,
        findings=[_finding("KSEC-005", path="/spec/containers/0/image"), _finding("KSEC-006", path="/spec/template/spec/labels/app")],
        patch=[
            {"doc": 0, "op": "replace", "path": "/spec/containers/0/image", "value": "nginx:1.25"},
            {"doc": 0, "op": "replace", "path": "/spec/template/spec/labels/app", "value": "x"},
        ],
    )
    result = score_run([_pod()], [instance], json.dumps(output))
    assert result["instances"][1]["corrected"] is True
    assert result["false_positives"] == 1  # the invented KSEC-006 finding


def test_invalid_answer_scores_nothing():
    result = score_run([_pod()], [_instance(1, "KSEC-005")], "not json")
    assert result["schema_valid"] is False
    assert result["instances"][1] == {"detected": False, "corrected": False}


def test_cross_document_service_selector_is_detected_and_corrected():
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
    output = dict(
        CLEAN,
        findings=[_finding("KSEC-006", doc=0, path="/spec/selector/app")],
        patch=[{"doc": 0, "op": "replace", "path": "/spec/selector/app", "value": "web"}],
    )
    result = score_run(docs, [_instance(1, "KSEC-006", doc=0)], json.dumps(output))
    assert result["instances"][1] == {"detected": True, "corrected": True}


def test_analyze_metrics_on_the_real_scenarios():
    # A perfect answer per scenario (the detectors' own findings): every
    # metric is 100%.
    cases = load_test_cases(Path("scenarios/test_cases.yaml"))
    scored = {}
    for name, instances in cases.items():
        docs = list(yaml.safe_load_all(Path("scenarios", name).read_text()))
        findings = [f.to_dict() for f in detect_file(docs)]
        scored[name] = score_run(docs, instances, json.dumps(dict(CLEAN, findings=findings)))
    analysis = analyze(scored, cases)
    assert analysis["totals"]["detected"] == analysis["totals"]["errors"] == 36
    assert analysis["metrics"]["Precision"] == 1.0
    assert analysis["metrics"]["Recall"] == 1.0
    assert analysis["metrics"]["Accuracy"] == 1.0


def test_ground_truth_matches_the_scenario_files():
    # Every listed error must be one the detectors actually find in ITS file
    # -- catches a test_cases.yaml edit that attaches errors to the wrong
    # file or names a rule that doesn't apply.
    cases = load_test_cases(Path("scenarios/test_cases.yaml"))
    for name, instances in cases.items():
        findings = [f.to_dict() for f in detect_file(list(yaml.safe_load_all(Path("scenarios", name).read_text())))]
        for inst in instances:
            assert inst["rule_id"] is not None, (name, inst["id"])
            assert any(_finding_matches_instance(f, inst) for f in findings), (name, inst["id"])
