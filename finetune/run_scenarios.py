"""
Runs a fine-tuned checkpoint against every scenarios/*.yaml file, several
times each, and records whether it detects/correctly patches the two issues
this project has a rule for -- missing image tag (KSEC-005) and exposed
credentials (KSEC-001) -- plus whatever it says (via "notes") about the
semantic typos/bugs listed in scenarios/ground_truth.yaml, which have no
rule_id in this project's taxonomy and so can't be scored automatically.

Ground truth (scenarios/ground_truth.yaml) is hand-maintained, not derived
from a detector -- these files are hand-written scenario manifests, not
dataset/build.py output, so there's no canonical hardened form to diff
against. "Patch fixed the issue" is still checked automatically, by reusing
the exact detectors (dataset/detect.py) this project already uses to label
training data, applied to the model's own patch: if the flagged rule no
longer fires after applying the model's patch, the patch is correct.

Sampling (do_sample=True) is used instead of evaluate.py's greedy decoding
specifically so that running a file --runs N times can produce different
outputs to measure reliability, not just reproducibility.

Usage:
    finetune/.venv/bin/python -m finetune.run_scenarios --adapter finetune/output/lora_adapter
    finetune/.venv/bin/python -m finetune.run_scenarios --adapter finetune/output/lora_adapter --runs 5 --temperature 0.7
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import jsonpatch
import yaml

from dataset.detect import detect_ksec001, detect_ksec005
from dataset.schema import SYSTEM_PROMPT
from finetune.evaluate import parse_model_output


def load_ground_truth(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_scenario_files(scenarios_dir: Path) -> list[Path]:
    return sorted(
        (p for p in scenarios_dir.glob("*.yaml") if p.name != "ground_truth.yaml"),
        key=lambda p: p.name,
    )


def apply_multidoc_patch(docs: list[dict], patch: list[dict]) -> tuple[list[dict] | None, bool]:
    """Same idea as finetune/evaluate.py's _apply_patch_safe, but doc-aware:
    scenario files are multi-document, so patch ops (each carrying a "doc"
    index per dataset/schema.py) must be grouped and applied per-document."""
    docs = copy.deepcopy(docs)
    if not patch:
        return docs, True
    by_doc: dict[int, list[dict]] = {}
    for op in patch:
        doc_idx = op.get("doc", 0)
        by_doc.setdefault(doc_idx, []).append({k: v for k, v in op.items() if k != "doc"})
    try:
        for doc_idx, ops in by_doc.items():
            if not isinstance(doc_idx, int) or doc_idx < 0 or doc_idx >= len(docs):
                return None, False
            docs[doc_idx] = jsonpatch.apply_patch(docs[doc_idx], ops)
        return docs, True
    except Exception:
        return None, False


def score_run(docs: list[dict], ground_truth: dict, raw_output: str) -> dict:
    """Pure scoring for one generation -- no model/network involved."""
    response, errors = parse_model_output(raw_output)
    result = {
        "schema_valid": response is not None,
        "schema_errors": "; ".join(errors),
        "predicted_rules": [],
        "notes": "",
        "detected_image_tag": None,
        "image_tag_correct_detection": None,
        "image_tag_patch_fixed": None,
        "detected_credentials": None,
        "credentials_correct_detection": None,
        "credentials_patch_fixed": None,
        "patch_applies": None,
        "raw_output": raw_output,
    }
    if response is None:
        return result

    predicted_rules = {f["rule_id"] for f in response.get("findings", [])}
    result["predicted_rules"] = sorted(predicted_rules)
    result["notes"] = " | ".join(response.get("notes", []))

    detected_image_tag = "KSEC-005" in predicted_rules
    detected_credentials = "KSEC-001" in predicted_rules
    result["detected_image_tag"] = detected_image_tag
    result["detected_credentials"] = detected_credentials
    result["image_tag_correct_detection"] = detected_image_tag == bool(ground_truth["missing_image_tag"])
    result["credentials_correct_detection"] = detected_credentials == bool(ground_truth["exposed_credentials"])

    patched_docs, applied_ok = apply_multidoc_patch(docs, response.get("patch", []))
    result["patch_applies"] = applied_ok
    if applied_ok:
        still_has_image_tag_issue = any(detect_ksec005(d, i) for i, d in enumerate(patched_docs))
        still_has_credential_issue = any(detect_ksec001(d, i) for i, d in enumerate(patched_docs))
        if ground_truth["missing_image_tag"]:
            result["image_tag_patch_fixed"] = detected_image_tag and not still_has_image_tag_issue
        if ground_truth["exposed_credentials"]:
            result["credentials_patch_fixed"] = detected_credentials and not still_has_credential_issue

    return result


def summarize_file(file_results: list[dict], ground_truth: dict) -> dict:
    n = len(file_results)

    def rate(key: str, relevant_only: bool = False) -> str:
        values = [r[key] for r in file_results if not relevant_only or r[key] is not None]
        if not values:
            return "N/A"
        return f"{sum(1 for v in values if v)}/{len(values)}"

    return {
        "runs": n,
        "schema_valid_rate": rate("schema_valid"),
        "image_tag_detection_rate": rate("image_tag_correct_detection"),
        "image_tag_patch_fix_rate": rate("image_tag_patch_fixed", relevant_only=True),
        "credentials_detection_rate": rate("credentials_correct_detection"),
        "credentials_patch_fix_rate": rate("credentials_patch_fixed", relevant_only=True),
        "expected_typos": " | ".join(ground_truth.get("typos", [])) or "(none)",
        "runs_with_notes": sum(1 for r in file_results if r["notes"]),
    }


def write_excel(rows: list[dict], summaries: dict[str, dict], output_path: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.utils import get_column_letter

    wb = Workbook()

    summary_ws = wb.active
    summary_ws.title = "Summary"
    summary_headers = [
        "file",
        "runs",
        "schema_valid_rate",
        "image_tag_detection_rate",
        "image_tag_patch_fix_rate",
        "credentials_detection_rate",
        "credentials_patch_fix_rate",
        "runs_with_notes",
        "expected_typos (manual review)",
    ]
    summary_ws.append(summary_headers)
    for file_name, summary in summaries.items():
        summary_ws.append(
            [
                file_name,
                summary["runs"],
                summary["schema_valid_rate"],
                summary["image_tag_detection_rate"],
                summary["image_tag_patch_fix_rate"],
                summary["credentials_detection_rate"],
                summary["credentials_patch_fix_rate"],
                summary["runs_with_notes"],
                summary["expected_typos"],
            ]
        )

    runs_ws = wb.create_sheet("Runs")
    run_headers = [
        "file",
        "run",
        "schema_valid",
        "detected_image_tag",
        "image_tag_correct_detection",
        "image_tag_patch_fixed",
        "detected_credentials",
        "credentials_correct_detection",
        "credentials_patch_fixed",
        "patch_applies",
        "predicted_rules",
        "notes",
        "schema_errors",
        "raw_output",
    ]
    runs_ws.append(run_headers)
    for row in rows:
        runs_ws.append(
            [
                row["file"],
                row["run"],
                row["schema_valid"],
                row["detected_image_tag"],
                row["image_tag_correct_detection"],
                row["image_tag_patch_fixed"],
                row["detected_credentials"],
                row["credentials_correct_detection"],
                row["credentials_patch_fixed"],
                row["patch_applies"],
                ", ".join(row["predicted_rules"]),
                row["notes"],
                row["schema_errors"],
                row["raw_output"],
            ]
        )

    for ws, widths in ((summary_ws, [22, 6, 14, 18, 18, 18, 18, 12, 50]), (runs_ws, [22, 5, 10, 12, 16, 16, 14, 18, 16, 12, 24, 40, 30, 60])):
        for i, width in enumerate(widths, start=1):
            ws.column_dimensions[get_column_letter(i)].width = width

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)


def run(args: argparse.Namespace) -> None:
    from finetune.evaluate import generate_response_text, load_model

    scenarios_dir = Path(args.scenarios_dir)
    ground_truth = load_ground_truth(Path(args.ground_truth))
    files = load_scenario_files(scenarios_dir)

    model, tokenizer = load_model(args.adapter, args.max_seq_length)

    rows = []
    summaries = {}
    for path in files:
        if path.name not in ground_truth:
            print(f"WARNING: no ground truth entry for {path.name}, skipping")
            continue
        gt = ground_truth[path.name]
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
            result = score_run(docs, gt, output_text)
            file_results.append(result)
            rows.append({"file": path.name, "run": run_idx, **result})
            print(
                f"[{path.name} run {run_idx}/{args.runs}] "
                f"schema_valid={result['schema_valid']} "
                f"image_tag={result['image_tag_correct_detection']} "
                f"credentials={result['credentials_correct_detection']}"
            )

        summaries[path.name] = summarize_file(file_results, gt)

    output_path = Path(args.output)
    write_excel(rows, summaries, output_path)
    print(f"\nWritten to {output_path}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True, help="path to a saved LoRA adapter or checkpoint-N dir")
    parser.add_argument("--scenarios-dir", default="scenarios")
    parser.add_argument("--ground-truth", default="scenarios/ground_truth.yaml")
    parser.add_argument("--output", default="finetune/output/scenarios_results.xlsx")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
