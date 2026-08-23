import json

from finetune.run_scenarios import apply_multidoc_patch, score_run, summarize_file

CLEAN = {"findings": [], "patch": [], "new_resources": [], "notes": []}


def _finding(rule_id, doc=0, path="/x"):
    return {"rule_id": rule_id, "severity": "high", "doc": doc, "path": path, "message": "m", "evidence": "abcd***"}


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


def test_score_run_detects_missing_image_tag_and_correct_patch():
    docs = [{"kind": "Pod", "spec": {"containers": [{"image": "nginx"}]}}]
    gt = {"missing_image_tag": True, "exposed_credentials": False}
    response = {
        "findings": [_finding("KSEC-005")],
        "patch": [{"doc": 0, "op": "replace", "path": "/spec/containers/0/image", "value": "nginx:1.0.0"}],
        "new_resources": [],
        "notes": [],
    }

    result = score_run(docs, gt, json.dumps(response))

    assert result["schema_valid"] is True
    assert result["detected_image_tag"] is True
    assert result["image_tag_correct_detection"] is True
    assert result["image_tag_patch_fixed"] is True
    assert result["credentials_patch_fixed"] is None  # not expected, so not scored


def test_score_run_flags_patch_that_does_not_actually_fix_the_issue():
    docs = [{"kind": "Pod", "spec": {"containers": [{"image": "nginx"}]}}]
    gt = {"missing_image_tag": True, "exposed_credentials": False}
    response = {
        "findings": [_finding("KSEC-005")],
        "patch": [{"doc": 0, "op": "replace", "path": "/spec/containers/0/image", "value": "nginx:latest"}],
        "new_resources": [],
        "notes": [],
    }

    result = score_run(docs, gt, json.dumps(response))

    assert result["image_tag_patch_fixed"] is False  # still unpinned after "fix"


def test_score_run_invalid_json_yields_all_nones():
    result = score_run([{"a": 1}], {"missing_image_tag": False, "exposed_credentials": False}, "not json")

    assert result["schema_valid"] is False
    assert result["detected_image_tag"] is None
    assert result["patch_applies"] is None


def test_score_run_clean_response_on_clean_ground_truth():
    docs = [{"a": 1}]
    gt = {"missing_image_tag": False, "exposed_credentials": False}

    result = score_run(docs, gt, json.dumps(CLEAN))

    assert result["detected_image_tag"] is False
    assert result["image_tag_correct_detection"] is True
    assert result["image_tag_patch_fixed"] is None  # nothing to fix
    assert result["credentials_patch_fixed"] is None


def test_summarize_file_reports_fractions():
    gt = {"missing_image_tag": True, "exposed_credentials": True, "typos": ["a typo"]}
    results = [
        {
            "schema_valid": True,
            "image_tag_correct_detection": True,
            "image_tag_patch_fixed": True,
            "credentials_correct_detection": False,
            "credentials_patch_fixed": False,
            "notes": "",
        },
        {
            "schema_valid": True,
            "image_tag_correct_detection": True,
            "image_tag_patch_fixed": True,
            "credentials_correct_detection": True,
            "credentials_patch_fixed": True,
            "notes": "saw a typo",
        },
    ]

    summary = summarize_file(results, gt)

    assert summary["runs"] == 2
    assert summary["schema_valid_rate"] == "2/2"
    assert summary["image_tag_patch_fix_rate"] == "2/2"
    assert summary["credentials_detection_rate"] == "1/2"
    assert summary["runs_with_notes"] == 1
    assert summary["expected_typos"] == "a typo"
