"""
Runs a fine-tuned checkpoint against every scenarios/*.yaml file, several
times each, and scores it against scenarios/test_cases.yaml -- a per-instance
ground truth (one row per individual error, not per file) matching the
"2 - Test Cases" sheet from the project's test-case design spreadsheet:
every instance is categorized as one of exactly "Credenciais Expostas",
"Imagem sem Tag", or "Erro de Sintaxe/Config", with a line number and
description kept verbatim from that sheet.

Ground truth is hand-maintained, not derived from a detector -- these files
are hand-written scenario manifests, not dataset/build.py output, so there's
no canonical hardened form to diff against. Each instance additionally
records what this project's own rule taxonomy can say about it: a `rule_id`
(or null if none of this project's rules cover it -- an honest scope boundary,
not a bug) plus enough to identify that SPECIFIC instance among possibly
several findings of the same rule in one file (see test_cases.yaml's header
comment for the matching strategy). "Corrected" is checked automatically by
reusing the exact detectors (dataset/detect.py) this project already uses to
label training data, applied to the model's own patch: if no finding
matching that instance survives after applying the patch, it's fixed.

Sampling (do_sample=True) is used instead of evaluate.py's greedy decoding
specifically so that running a file --runs N times can produce different
outputs to measure reliability, not just reproducibility.

Usage:
    finetune/.venv/bin/python -m finetune.run_scenarios --adapter finetune/output/lora_adapter
    finetune/.venv/bin/python -m finetune.run_scenarios --adapter finetune/output/lora_adapter --runs 5 --temperature 0.7
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import yaml

from dataset.detect import detect_file
from dataset.schema import SYSTEM_PROMPT
from finetune.evaluate import apply_multidoc_patch, parse_model_output

CATEGORIES = ("Credenciais Expostas", "Imagem sem Tag", "Erro de Sintaxe/Config")

def load_test_cases(path: Path) -> dict[str, list[dict]]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_scenario_files(scenarios_dir: Path) -> list[Path]:
    return sorted(
        (p for p in scenarios_dir.glob("*.yaml") if p.name not in ("ground_truth.yaml", "test_cases.yaml")),
        key=lambda p: p.name,
    )


def _finding_matches_instance(finding: dict, instance: dict) -> bool:
    if finding["rule_id"] != instance["rule_id"]:
        return False
    if instance.get("doc") is not None and finding.get("doc") != instance["doc"]:
        return False
    match_type = instance.get("match_type", "rule_only")
    if match_type == "rule_only":
        return True
    keyword = instance["match_keyword"].lower()
    if match_type == "message":
        return keyword in finding.get("message", "").lower()
    if match_type == "evidence_prefix":
        return finding.get("evidence", "").lower().startswith(keyword)
    raise ValueError(f"unknown match_type: {match_type!r}")


def score_instance(instance: dict, findings: list[dict], patched_findings: list[dict] | None) -> dict:
    """Scores one ground-truth instance against one generation's findings
    (before the patch) and, if the patch applied, the findings remaining
    after it. rule_id: null instances (outside this project's rules) are always "not applicable" -- reported as never detected/
    corrected, honestly reflecting the taxonomy's current scope rather than
    silently excluding them."""
    if instance["rule_id"] is None:
        return {"detected": False, "corrected": False, "applicable": False}

    detected = any(_finding_matches_instance(f, instance) for f in findings)
    corrected = None
    if patched_findings is not None:
        corrected = not any(_finding_matches_instance(f, instance) for f in patched_findings)
    return {"detected": detected, "corrected": bool(corrected), "applicable": True}


def score_run(docs: list[dict], instances: list[dict], raw_output: str) -> dict:
    """Pure scoring for one generation -- no model/network involved."""
    response, errors = parse_model_output(raw_output)
    result = {
        "schema_valid": response is not None,
        "schema_errors": "; ".join(errors),
        "patch_applies": None,
        "notes": "",
        "instances": {inst["id"]: {"detected": False, "corrected": False, "applicable": False} for inst in instances},
        "raw_output": raw_output,
    }
    if response is None:
        return result

    findings = response.get("findings", [])
    result["notes"] = " | ".join(response.get("notes", []))

    patched_docs, applied_ok = apply_multidoc_patch(docs, response.get("patch", []))
    result["patch_applies"] = applied_ok
    patched_findings = None
    if applied_ok:
        patched_findings = [finding.to_dict() for finding in detect_file(patched_docs)]

    for instance in instances:
        result["instances"][instance["id"]] = score_instance(instance, findings, patched_findings)

    return result


def summarize_file(file_results: list[dict], instances: list[dict]) -> dict:
    """Per-file Detecção/Corrigido numbers, averaged across the sampled runs
    (see module docstring: --runs N sampled generations, not a single
    snapshot) -- matches the "2 - Test Cases" sheet's per-file shape
    (Arquivo | Erros | Detectado/Corrigido | Não detectado/corrigido | % OK)
    while still reflecting run-to-run variance instead of collapsing it."""
    n_runs = len(file_results)
    n_instances = len(instances)

    detected_per_run = [sum(1 for v in r["instances"].values() if v["detected"]) for r in file_results]
    corrected_per_run = [sum(1 for v in r["instances"].values() if v["corrected"]) for r in file_results]
    avg_detected = sum(detected_per_run) / n_runs if n_runs else 0.0
    avg_corrected = sum(corrected_per_run) / n_runs if n_runs else 0.0

    by_category = collections.Counter(inst["category"] for inst in instances)

    return {
        "runs": n_runs,
        "errors": n_instances,
        "avg_detected": round(avg_detected, 2),
        "avg_not_detected": round(n_instances - avg_detected, 2),
        "detected_pct": round(avg_detected / n_instances * 100, 1) if n_instances else None,
        "avg_corrected": round(avg_corrected, 2),
        "avg_not_corrected": round(n_instances - avg_corrected, 2),
        "corrected_pct": round(avg_corrected / n_instances * 100, 1) if n_instances else None,
        "by_category": dict(by_category),
        "schema_valid_rate": f"{sum(1 for r in file_results if r['schema_valid'])}/{n_runs}",
    }


def write_excel(
    rows: list[dict],
    summaries: dict[str, dict],
    all_instances: dict[str, list[dict]],
    adapter: str,
    output_path: Path,
) -> None:
    from openpyxl import Workbook
    from openpyxl.utils import get_column_letter

    wb = Workbook()

    def add_rate_sheet(title: str, count_key: str, complement_key: str, pct_key: str, label: str, complement_label: str) -> None:
        ws = wb.create_sheet(title) if wb.sheetnames != ["Sheet"] else wb.active
        ws.title = title
        ws.append(["Modelo:", adapter])
        ws.append([])
        ws.append(["Arquivo", "Erros", label, complement_label, "% OK"])
        total_errors = total_count = 0
        for file_name, summary in summaries.items():
            ws.append(
                [file_name, summary["errors"], summary[count_key], summary[complement_key], summary[pct_key]]
            )
            total_errors += summary["errors"]
            total_count += summary[count_key]
        total_pct = round(total_count / total_errors * 100, 1) if total_errors else None
        ws.append(["Total", total_errors, round(total_count, 2), round(total_errors - total_count, 2), total_pct])
        for i, width in enumerate([24, 8, 12, 16, 8], start=1):
            ws.column_dimensions[get_column_letter(i)].width = width

    add_rate_sheet("Detecção", "avg_detected", "avg_not_detected", "detected_pct", "Detectado", "Não detectado")
    add_rate_sheet("Corrigido", "avg_corrected", "avg_not_corrected", "corrected_pct", "Corrigido", "Não corrigido")

    cat_ws = wb.create_sheet("Categorias")
    cat_ws.append(["Arquivo", *CATEGORIES, "Total", "schema_valid_rate"])
    totals = collections.Counter()
    for file_name, summary in summaries.items():
        by_cat = summary["by_category"]
        row = [file_name] + [by_cat.get(c, 0) for c in CATEGORIES] + [summary["errors"], summary["schema_valid_rate"]]
        cat_ws.append(row)
        for c in CATEGORIES:
            totals[c] += by_cat.get(c, 0)
    cat_ws.append(["Total", *(totals[c] for c in CATEGORIES), sum(totals.values()), ""])
    for i, width in enumerate([24, 20, 16, 22, 8, 16], start=1):
        cat_ws.column_dimensions[get_column_letter(i)].width = width

    gt_ws = wb.create_sheet("Ground Truth")
    gt_ws.append(["#", "Arquivo", "Tipo de Erro", "Linha", "Descrição do Erro", "rule_id", "note"])
    for file_name, instances in all_instances.items():
        for inst in instances:
            gt_ws.append(
                [
                    inst["id"],
                    file_name,
                    inst["category"],
                    str(inst.get("line", "")),
                    inst["description"],
                    inst["rule_id"] or "(fora do escopo)",
                    inst.get("note", ""),
                ]
            )
    for i, width in enumerate([5, 22, 20, 8, 55, 16, 55], start=1):
        gt_ws.column_dimensions[get_column_letter(i)].width = width

    runs_ws = wb.create_sheet("Runs")
    runs_ws.append(["file", "run", "schema_valid", "patch_applies", "detected", "corrected", "notes", "schema_errors", "raw_output"])
    for row in rows:
        detected_ids = [str(i) for i, v in row["instances"].items() if v["detected"]]
        corrected_ids = [str(i) for i, v in row["instances"].items() if v["corrected"]]
        runs_ws.append(
            [
                row["file"],
                row["run"],
                row["schema_valid"],
                row["patch_applies"],
                ", ".join(detected_ids),
                ", ".join(corrected_ids),
                row["notes"],
                row["schema_errors"],
                row["raw_output"],
            ]
        )
    for i, width in enumerate([22, 5, 10, 12, 20, 20, 30, 30, 60], start=1):
        runs_ws.column_dimensions[get_column_letter(i)].width = width

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)


def run(args: argparse.Namespace) -> None:
    from finetune.evaluate import generate_response_text, load_model

    scenarios_dir = Path(args.scenarios_dir)
    test_cases = load_test_cases(Path(args.test_cases))
    files = load_scenario_files(scenarios_dir)

    model, tokenizer = load_model(args.adapter, args.max_seq_length)

    rows = []
    summaries = {}
    for path in files:
        if path.name not in test_cases:
            print(f"WARNING: no test cases entry for {path.name}, skipping")
            continue
        instances = test_cases[path.name]
        raw_text = path.read_text(encoding="utf-8")
        docs = list(yaml.safe_load_all(raw_text))

        file_results = []
        for run_idx in range(1, args.runs + 1):
            output_text = generate_response_text(
                model,
                tokenizer,
                SYSTEM_PROMPT,
                raw_text,
                args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
            )
            result = score_run(docs, instances, output_text)
            file_results.append(result)
            rows.append({"file": path.name, "run": run_idx, **result})
            n_detected = sum(1 for v in result["instances"].values() if v["detected"])
            print(
                f"[{path.name} run {run_idx}/{args.runs}] "
                f"schema_valid={result['schema_valid']} "
                f"detected={n_detected}/{len(instances)}"
            )

        summaries[path.name] = summarize_file(file_results, instances)

    output_path = Path(args.output)
    write_excel(rows, summaries, test_cases, args.adapter, output_path)
    print(f"\nWritten to {output_path}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True, help="path to a saved LoRA adapter or checkpoint-N dir")
    parser.add_argument("--scenarios-dir", default="scenarios")
    parser.add_argument("--test-cases", default="scenarios/test_cases.yaml")
    parser.add_argument("--output", default="finetune/output/scenarios_results.xlsx")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1280)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
