import json

from finetune.evaluate import (
    aggregate_results,
    evaluate_example,
    parse_model_output,
    strip_fences,
)

CLEAN_RESPONSE = {"findings": [], "patch": [], "new_resources": [], "notes": []}


def _finding(rule_id, path="/spec/containers/0/image", doc=0):
    return {
        "rule_id": rule_id,
        "severity": "medium",
        "doc": doc,
        "path": path,
        "message": "x",
        "evidence": "abcd***",
    }


def _pod():
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": "p"},
        "spec": {"containers": [{"name": "app", "image": "myapp:latest"}]},
    }


# ---------------------------------------------------------------------------
# strip_fences / parse_model_output
# ---------------------------------------------------------------------------


def test_strip_fences_removes_markdown_json_block():
    text = '```json\n{"findings": []}\n```'
    assert strip_fences(text) == '{"findings": []}'


def test_parse_model_output_valid_clean_response():
    text = json.dumps(CLEAN_RESPONSE)
    response, errors = parse_model_output(text)
    assert response == CLEAN_RESPONSE
    assert errors == []


def test_parse_model_output_invalid_json():
    response, errors = parse_model_output("not json at all")
    assert response is None
    assert "invalid JSON" in errors[0]


def test_parse_model_output_valid_json_but_fails_schema():
    text = json.dumps({"findings": [], "patch": []})  # missing new_resources/notes
    response, errors = parse_model_output(text)
    assert response is None
    assert errors  # non-empty


def test_parse_model_output_strips_fences_first():
    text = "```json\n" + json.dumps(CLEAN_RESPONSE) + "\n```"
    response, errors = parse_model_output(text)
    assert response == CLEAN_RESPONSE
    assert errors == []


# ---------------------------------------------------------------------------
# evaluate_example
# ---------------------------------------------------------------------------


def test_evaluate_example_perfect_clean_prediction():
    result = evaluate_example(_pod(), CLEAN_RESPONSE, json.dumps(CLEAN_RESPONSE))
    assert result["schema_valid"]
    assert result["expected_rules"] == []
    assert result["predicted_rules"] == []
    assert result["patch_applies"]
    assert result["patch_correct"]


def test_evaluate_example_model_hallucinates_a_finding_on_clean_input():
    doc = _pod()
    bad_response = {
        "findings": [_finding("KSEC-005")],
        "patch": [{"doc": 0, "op": "replace", "path": "/spec/containers/0/image", "value": "myapp:1.0.0"}],
        "new_resources": [],
        "notes": [],
    }
    result = evaluate_example(doc, CLEAN_RESPONSE, json.dumps(bad_response))
    assert result["schema_valid"]
    assert result["expected_rules"] == []
    assert result["predicted_rules"] == ["KSEC-005"]
    # patch still applies syntactically, but doesn't match ground truth (no-op)
    assert result["patch_applies"]
    assert not result["patch_correct"]


def test_evaluate_example_model_matches_ground_truth_patch_exactly():
    doc = _pod()
    expected = {
        "findings": [_finding("KSEC-005")],
        "patch": [{"doc": 0, "op": "replace", "path": "/spec/containers/0/image", "value": "myapp:1.0.0"}],
        "new_resources": [],
        "notes": [],
    }
    result = evaluate_example(doc, expected, json.dumps(expected))
    assert result["patch_correct"]
    assert result["predicted_rules"] == ["KSEC-005"]


def test_evaluate_example_model_output_not_valid_json():
    result = evaluate_example(_pod(), CLEAN_RESPONSE, "I think this manifest looks fine!")
    assert not result["schema_valid"]
    assert result["predicted_rules"] == []
    assert not result["patch_applies"]
    assert not result["patch_correct"]


def test_evaluate_example_patch_that_does_not_apply():
    doc = _pod()
    bad_response = {
        "findings": [_finding("KSEC-005")],
        "patch": [{"doc": 0, "op": "replace", "path": "/spec/containers/5/image", "value": "x"}],  # bad index
        "new_resources": [],
        "notes": [],
    }
    result = evaluate_example(doc, CLEAN_RESPONSE, json.dumps(bad_response))
    assert result["schema_valid"]
    assert not result["patch_applies"]
    assert not result["patch_correct"]


def test_evaluate_example_new_resources_presence_mismatch():
    doc = _pod()
    expected = {
        "findings": [_finding("KSEC-001")],
        "patch": [],
        "new_resources": ["apiVersion: v1\nkind: Secret\n"],
        "notes": [],
    }
    model_response = {"findings": [_finding("KSEC-001")], "patch": [], "new_resources": [], "notes": []}
    result = evaluate_example(doc, expected, json.dumps(model_response))
    assert not result["new_resources_presence_correct"]


# ---------------------------------------------------------------------------
# aggregate_results
# ---------------------------------------------------------------------------


def test_aggregate_results_empty():
    assert aggregate_results([]) == {"total": 0}


def test_aggregate_results_perfect_run():
    results = [evaluate_example(_pod(), CLEAN_RESPONSE, json.dumps(CLEAN_RESPONSE)) for _ in range(5)]
    summary = aggregate_results(results)
    assert summary["total"] == 5
    assert summary["schema_valid_rate"] == 1.0
    assert summary["patch_correct_rate"] == 1.0
    assert summary["clean_examples"] == 5
    assert summary["clean_correctly_identified_rate"] == 1.0
    for rule_metrics in summary["per_rule"].values():
        assert rule_metrics["tp"] == 0
        assert rule_metrics["fp"] == 0


def test_aggregate_results_per_rule_precision_recall():
    doc = _pod()
    expected_005 = {
        "findings": [_finding("KSEC-005")],
        "patch": [{"doc": 0, "op": "replace", "path": "/spec/containers/0/image", "value": "myapp:1.0.0"}],
        "new_resources": [],
        "notes": [],
    }
    # 1 true positive (correctly predicts KSEC-005)
    r1 = evaluate_example(doc, expected_005, json.dumps(expected_005))
    # 1 false positive (predicts KSEC-005 on a clean example)
    r2 = evaluate_example(doc, CLEAN_RESPONSE, json.dumps(expected_005))
    # 1 false negative (misses KSEC-005, predicts clean)
    r3 = evaluate_example(doc, expected_005, json.dumps(CLEAN_RESPONSE))
    # 1 true negative (correctly predicts clean)
    r4 = evaluate_example(doc, CLEAN_RESPONSE, json.dumps(CLEAN_RESPONSE))

    summary = aggregate_results([r1, r2, r3, r4])
    m = summary["per_rule"]["KSEC-005"]
    assert m["tp"] == 1
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["tn"] == 1
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5


def test_aggregate_results_schema_invalid_examples_excluded_from_patch_rates():
    doc = _pod()
    valid = evaluate_example(doc, CLEAN_RESPONSE, json.dumps(CLEAN_RESPONSE))
    invalid = evaluate_example(doc, CLEAN_RESPONSE, "not json")
    summary = aggregate_results([valid, invalid])
    assert summary["schema_valid_rate"] == 0.5
    # patch_correct_rate is computed only over the schema-valid subset (1 example)
    assert summary["patch_correct_rate"] == 1.0
