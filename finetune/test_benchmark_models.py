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
