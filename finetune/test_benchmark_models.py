import json

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("statsmodels")

from finetune.benchmark_models import mcnemar_decisions, mcnemar_test, mean_std_ci, quality, wilson


def _s(index, expected, predicted):
    return {"index": index, "expected": expected, "predicted": predicted}


def test_quality_exact_match_and_macro_vs_weighted():
    samples = [
        _s(0, ["KSEC-001"], ["KSEC-001"]),
        _s(1, ["KSEC-001", "KSEC-005"], ["KSEC-001"]),  # misses KSEC-005
        _s(2, ["KSEC-001"], ["KSEC-001"]),
        _s(3, [], []),
    ]
    q = quality(samples)
    assert q["accuracy"] == 0.75
    assert q["labels"] == ["KSEC-001", "KSEC-005"]  # only rules present in the samples
    assert q["macro"]["recall"] == pytest.approx((1.0 + 0.0) / 2)
    assert q["weighted"]["recall"] == pytest.approx((1.0 * 3 + 0.0 * 1) / 4)
    (tn, fp), (fn, tp) = q["confusion"]["KSEC-005"].tolist()
    assert (tp, fp, fn, tn) == (0, 0, 1, 3)


def test_mcnemar_counts_discordant_pairs():
    a = [_s(i, ["KSEC-001"], ["KSEC-001"]) for i in range(10)]
    b = [_s(i, ["KSEC-001"], ["KSEC-001"] if i < 4 else []) for i in range(10)]
    r = mcnemar_test(a, b)
    assert (r["both_correct"], r["only_first"], r["only_second"], r["both_wrong"]) == (4, 6, 0, 0)
    assert r["method"] == "exact binomial" and 0 < r["p_value"] < 0.05
    d = mcnemar_decisions(a, b)
    assert d["n"] == 10 * 12 and d["only_first"] == 6


def test_intervals():
    low, high = wilson(9, 10)
    assert low < 0.9 < high <= 1.0
    m, s, lo, hi = mean_std_ci([1.0, 2.0, 3.0])
    assert m == 2.0 and lo < 2.0 < hi


def _split_flag_row():
    import yaml

    from dataset.detect import detect_file

    doc = {
        "apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": "web"},
        "spec": {"selector": {"matchLabels": {"app": "web"}}, "template": {"metadata": {"labels": {"app": "web"}}, "spec": {"containers": [{
            "name": "web", "image": "nginx:1.25.3", "args": ["serve", "--db-password", "Zq8vT3xK9pLm2", "--port", "80"],
            "resources": {"limits": {"cpu": "1", "memory": "1Gi"}},
            "securityContext": {"runAsNonRoot": True, "allowPrivilegeEscalation": False, "readOnlyRootFilesystem": True},
        }]}}},
    }
    findings = [f.to_dict() for f in detect_file([doc])]
    assert [f["path"] for f in findings] == ["/spec/template/spec/containers/0/args/2"]
    # The fix removes the flag, then the value that shifted into its place.
    patch = [{"doc": 0, "op": "remove", "path": "/spec/template/spec/containers/0/args/1"}] * 2
    reference = {"findings": findings, "patch": patch, "new_resources": [], "notes": []}
    row = {"_index": 7, "messages": [{"role": "system", "content": ""}, {"role": "user", "content": yaml.safe_dump(doc)},
                                     {"role": "assistant", "content": json.dumps(reference)}]}
    return row, reference


def test_score_sample_counts_a_split_flag_fix_as_corrected():
    from finetune.benchmark_models import score_sample

    row, reference = _split_flag_row()
    result = score_sample(row, json.dumps(reference))
    assert result["errors"] == [{"id": "7:0", "rule": "KSEC-001", "detected": True, "corrected": True}]
    assert (result["right"], result["false_positives"]) == (1, 0)


def test_score_sample_detected_without_a_fix_and_false_positives():
    from finetune.benchmark_models import score_sample

    row, reference = _split_flag_row()
    wrong = dict(reference["findings"][0], rule_id="KSEC-002", path="/spec/template/spec/hostNetwork")
    answer = {**reference, "findings": reference["findings"] + [wrong], "patch": []}
    result = score_sample(row, json.dumps(answer))
    assert result["errors"][0]["detected"] and not result["errors"][0]["corrected"]
    assert (result["right"], result["false_positives"]) == (1, 1)
    assert score_sample(row, "not json")["errors"][0] == {"id": "7:0", "rule": "KSEC-001", "detected": False, "corrected": False}


def test_mcnemar_errors_pairs_by_error_id():
    from finetune.benchmark_models import detection_correction, mcnemar_errors

    a = [{"index": 0, "errors": [{"id": f"0:{i}", "detected": True, "corrected": True} for i in range(6)], "right": 6, "false_positives": 0}]
    b = [{"index": 0, "errors": [{"id": f"0:{i}", "detected": True, "corrected": i < 2} for i in range(6)], "right": 6, "false_positives": 2}]
    r = mcnemar_errors(a, b, "corrected")
    assert (r["n"], r["both_correct"], r["only_first"], r["only_second"]) == (6, 2, 4, 0)
    dc = detection_correction(b)
    assert (dc["detection rate"], dc["correction rate"], dc["precision"]) == (1.0, pytest.approx(2 / 6), 0.75)
