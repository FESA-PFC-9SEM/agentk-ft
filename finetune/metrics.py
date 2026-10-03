"""
Standard classification metrics shared by finetune/evaluate.py (held-out
test set) and finetune/scenario_summary.py (hand-written scenarios).

Two levels, because "a prediction" means two different things here:

- Finding level: each reported finding vs each real error. TP = a finding
  that matches a real error, FP = a finding that matches none, FN = a real
  error nobody reported. There is no finite set of "non-errors" to count as
  true negatives, so this level has precision/recall/F1 but no accuracy.
- Example x rule level: one binary question per (manifest, rule) -- "does
  this manifest have a KSEC-00X problem?" -- so TP/FP/FN/TN all exist and
  accuracy is defined. Reported per rule plus micro (pooled counts) and macro
  (mean of per-rule scores, over rules that occur at all) averages.
"""

from __future__ import annotations


def _ratio(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2 * precision * recall, precision + recall) if precision and recall else (0.0 if tp + fp + fn else None)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def confusion(tp: int, fp: int, fn: int, tn: int) -> dict:
    return {**prf(tp, fp, fn), "tn": tn, "accuracy": _ratio(tp + tn, tp + fp + fn + tn)}


def rule_confusions(pairs, rule_ids) -> dict:
    """pairs: iterable of (actual rule set, predicted rule set), one per
    example. Returns per-rule confusion matrices plus micro and macro
    averages over the rules that occur (actually or predicted) at all."""
    counts = {rule: [0, 0, 0, 0] for rule in rule_ids}  # tp, fp, fn, tn
    for actual, predicted in pairs:
        for rule in rule_ids:
            a, p = rule in actual, rule in predicted
            counts[rule][0 if a and p else 1 if p else 2 if a else 3] += 1
    per_rule = {rule: confusion(*c) for rule, c in counts.items()}
    occurring = [rule for rule, c in counts.items() if c[0] + c[1] + c[2]]
    micro = confusion(*(sum(counts[r][i] for r in occurring) for i in range(4)))
    macro = {}
    for key in ("precision", "recall", "f1", "accuracy"):
        values = [per_rule[r][key] for r in occurring if per_rule[r][key] is not None]
        macro[key] = round(sum(values) / len(values), 4) if values else None
    return {"per_rule": per_rule, "micro": micro, "macro": macro}


def paths_related(a: str, b: str) -> bool:
    """Same JSON Pointer, or one an ancestor of the other: models (and fixes)
    often point at `.../env/1` for a problem detected at `.../env/1/value`."""
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")
