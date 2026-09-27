"""
Distribution report for a built dataset (the {train,val,test}.jsonl written by
build.py). diagnostic.json only records quotas and totals; this reads the
examples back and counts what the model will actually see, per split:
positives/negatives, findings and examples per rule, severities, defects per
example, patch ops, new_resources, resource kinds, documents per example (and
which rules land in multi-document files) and input sizes.

Usage:
    python -m dataset.stats dataset/output-multi-defect
    python -m dataset.stats dataset/output --json stats.json
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

import yaml

from dataset.schema import RULE_IDS

SPLITS = ("train", "val", "test")


def _documents(manifest: str) -> int:
    try:
        return sum(1 for d in yaml.safe_load_all(manifest) if isinstance(d, dict))
    except yaml.YAMLError:
        return 1


def _kinds(manifest: str) -> list[str]:
    try:
        docs = list(yaml.safe_load_all(manifest))
    except yaml.YAMLError:
        return ["<unparseable>"]
    return [d.get("kind") or "<none>" for d in docs if isinstance(d, dict)]


def split_stats(path: Path) -> dict:
    examples = 0
    negatives = 0
    findings_per_rule = collections.Counter()
    examples_per_rule = collections.Counter()
    severities = collections.Counter()
    defects_per_example = collections.Counter()
    patch_ops = collections.Counter()
    with_new_resources = 0
    kinds = collections.Counter()
    docs_per_example = collections.Counter()
    multi_doc_findings_per_rule = collections.Counter()
    nonzero_doc_index_findings = 0
    input_chars = []

    with path.open() as f:
        for line in f:
            messages = json.loads(line)["messages"]
            manifest = next(m["content"] for m in messages if m["role"] == "user")
            response = json.loads(next(m["content"] for m in messages if m["role"] == "assistant"))
            examples += 1
            input_chars.append(len(manifest))
            kinds.update(_kinds(manifest))
            n_docs = _documents(manifest)
            docs_per_example[n_docs] += 1

            findings = response["findings"]
            if not findings:
                negatives += 1
            defects_per_example[len(findings)] += 1
            for finding in findings:
                findings_per_rule[finding["rule_id"]] += 1
                severities[finding["severity"]] += 1
                if n_docs > 1:
                    multi_doc_findings_per_rule[finding["rule_id"]] += 1
                if finding.get("doc", 0) != 0:
                    nonzero_doc_index_findings += 1
            examples_per_rule.update({finding["rule_id"] for finding in findings})
            patch_ops.update(op["op"] for op in response["patch"])
            if response["new_resources"]:
                with_new_resources += 1

    return {
        "examples": examples,
        "positives": examples - negatives,
        "negatives": negatives,
        "findings_per_rule": {r: findings_per_rule[r] for r in sorted(RULE_IDS | set(findings_per_rule))},
        "examples_per_rule": {r: examples_per_rule[r] for r in sorted(RULE_IDS | set(examples_per_rule))},
        "severities": dict(severities.most_common()),
        "findings_per_example": dict(sorted(defects_per_example.items())),
        "patch_ops": dict(patch_ops.most_common()),
        "examples_with_new_resources": with_new_resources,
        "kinds": dict(kinds.most_common()),
        "docs_per_example": dict(sorted(docs_per_example.items())),
        "multi_doc_findings_per_rule": {
            r: multi_doc_findings_per_rule[r] for r in sorted(RULE_IDS | set(multi_doc_findings_per_rule))
        },
        "findings_with_nonzero_doc_index": nonzero_doc_index_findings,
        "input_chars": {
            "min": min(input_chars, default=0),
            "median": int(statistics.median(input_chars)) if input_chars else 0,
            "p95": int(statistics.quantiles(input_chars, n=20)[-1]) if len(input_chars) > 1 else 0,
            "max": max(input_chars, default=0),
        },
    }


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:5.1f}%" if total else "    -"


def render(stats: dict[str, dict], top_kinds: int = 15) -> str:
    splits = list(stats)
    width = 10
    out = []

    def row(label, values):
        out.append(f"{label:<24}" + "".join(f"{v:>{width + 8}}" for v in values))

    def section(title):
        out.append("")
        out.append(title)
        row("", splits)

    section("== Overview")
    for key in ("examples", "positives", "negatives", "examples_with_new_resources"):
        row(key, [f"{s[key]} {_pct(s[key], s['examples'])}" if key != "examples" else str(s[key]) for s in stats.values()])

    section("== Examples containing each rule (% of positives)")
    for rule in next(iter(stats.values()))["examples_per_rule"]:
        row(rule, [f"{s['examples_per_rule'].get(rule, 0)} {_pct(s['examples_per_rule'].get(rule, 0), s['positives'])}" for s in stats.values()])

    section("== Findings per rule (% of findings)")
    for rule in next(iter(stats.values()))["findings_per_rule"]:
        row(rule, [
            f"{s['findings_per_rule'].get(rule, 0)} {_pct(s['findings_per_rule'].get(rule, 0), sum(s['findings_per_rule'].values()))}"
            for s in stats.values()
        ])

    section("== Multi-document findings per rule (% of that rule's findings)")
    for rule in next(iter(stats.values()))["multi_doc_findings_per_rule"]:
        row(rule, [
            f"{s['multi_doc_findings_per_rule'].get(rule, 0)} "
            f"{_pct(s['multi_doc_findings_per_rule'].get(rule, 0), s['findings_per_rule'].get(rule, 0))}"
            for s in stats.values()
        ])
    row("findings in doc > 0", [str(s["findings_with_nonzero_doc_index"]) for s in stats.values()])

    for title, key in (
        ("== Documents per example", "docs_per_example"),
        ("== Findings per example", "findings_per_example"),
        ("== Severities", "severities"),
        ("== Patch ops", "patch_ops"),
    ):
        section(title)
        labels = sorted({str(k) for s in stats.values() for k in s[key]}, key=lambda k: (len(k), k))
        for label in labels:
            values = []
            for s in stats.values():
                counts = {str(k): v for k, v in s[key].items()}
                values.append(f"{counts.get(label, 0)} {_pct(counts.get(label, 0), sum(counts.values()))}")
            row(label, values)

    section(f"== Resource kinds (top {top_kinds} by first split)")
    first = next(iter(stats.values()))
    for kind in list(first["kinds"])[:top_kinds]:
        row(kind, [f"{s['kinds'].get(kind, 0)} {_pct(s['kinds'].get(kind, 0), sum(s['kinds'].values()))}" for s in stats.values()])

    section("== Input size (chars)")
    for key in ("min", "median", "p95", "max"):
        row(key, [str(s["input_chars"][key]) for s in stats.values()])

    return "\n".join(out)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", nargs="?", default="dataset/output")
    parser.add_argument("--json", default=None, help="also write the raw numbers to this file")
    args = parser.parse_args(argv)

    root = Path(args.dataset_dir)
    stats = {split: split_stats(root / f"{split}.jsonl") for split in SPLITS if (root / f"{split}.jsonl").exists()}
    if not stats:
        raise SystemExit(f"no {{train,val,test}}.jsonl found in {root}")
    print(render(stats))
    if args.json:
        Path(args.json).write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
