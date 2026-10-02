"""
Per-scenario summary table for one evaluated model, re-scored from the raw
outputs finetune/run_scenarios.py saved (no model needed): model, scenario,
no. errors, no. detected/undetected, no. corrected/uncorrected -- one sheet
per evaluation setting plus a Summary sheet.

Settings, from whichever result files exist in the run folder:
  - Sampled (scenarios_results.xlsx: temperature 0.7, N runs averaged) or
    Greedy (scenarios_results_greedy.xlsx: temperature 0, 1 run);
  - strict correction (the whole patch applied all-or-nothing) or per-finding
    correction (each finding's own ops applied on their own);
  - with --lenient, the same again ignoring schema errors: findings and patch
    are scored even when the response breaks the output contract (e.g. a
    zero-shot base model writing "hardc***" evidence or "containers[0]"
    paths), to separate "didn't find it" from "found it but broke the format".

Usage:
    python -m finetune.scenario_summary runs/<run>
    python -m finetune.scenario_summary runs/zero-shot_<model> --lenient
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import yaml

from finetune import run_scenarios
from finetune.evaluate import strip_fences
from finetune.run_scenarios import load_test_cases, score_run

HEADER = [
    "model",
    "scenario",
    "no. errors",
    "no. detected errors",
    "no. undetected errors",
    "no. corrected errors",
    "no. uncorrected errors",
]
RESULT_FILES = {
    "Sampled": ("scenarios_results.xlsx", "temperature 0.7, runs averaged"),
    "Greedy": ("scenarios_results_greedy.xlsx", "greedy decoding (deterministic), 1 run"),
}
CORRECTION_KEYS = {
    "strict": ("corrected", "patch applied all-or-nothing"),
    "per finding": ("corrected_per_finding", "each finding's patch ops applied on their own"),
}


def lenient_parse(text: str) -> tuple[dict | None, list[str]]:
    """Any JSON object in the response, schema errors ignored; the four
    top-level lists default to empty and malformed entries are dropped."""
    try:
        obj = json.loads(strip_fences(text))
    except (json.JSONDecodeError, TypeError):
        return None, ["invalid JSON"]
    if not isinstance(obj, dict):
        return None, ["not a JSON object"]
    for key in ("findings", "patch", "new_resources", "notes"):
        if not isinstance(obj.get(key), list):
            obj[key] = []
    obj["findings"] = [
        {**f, "message": str(f.get("message", "")), "evidence": str(f.get("evidence", ""))}
        for f in obj["findings"]
        if isinstance(f, dict) and "rule_id" in f
    ]
    obj["patch"] = [op for op in obj["patch"] if isinstance(op, dict) and "path" in op and "op" in op]
    return obj, []


def score_file(results_path: Path, cases: dict, lenient: bool) -> dict[str, dict]:
    """{scenario file: {"runs", "detected", "corrected", "corrected_per_finding"}} summed over runs."""
    from openpyxl import load_workbook

    rows = list(load_workbook(results_path, read_only=True)["Runs"].iter_rows(values_only=True))
    header = rows[0]
    strict_parser = run_scenarios.parse_model_output
    if lenient:
        run_scenarios.parse_model_output = lenient_parse
    try:
        totals: dict[str, dict] = collections.defaultdict(collections.Counter)
        for values in rows[1:]:
            row = dict(zip(header, values))
            name = row["file"]
            if name not in cases:
                continue
            docs = list(yaml.safe_load_all((Path("scenarios") / name).read_text(encoding="utf-8")))
            result = score_run(docs, cases[name], row["raw_output"] or "")
            t = totals[name]
            t["runs"] += 1
            for key in ("detected", "corrected", "corrected_per_finding"):
                t[key] += sum(bool(v.get(key)) for v in result["instances"].values())
        return totals
    finally:
        run_scenarios.parse_model_output = strict_parser


def build_workbook(run_dir: Path, lenient: bool, test_cases: Path = Path("scenarios/test_cases.yaml")):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    cases = load_test_cases(test_cases)
    model = run_dir.name
    wb = Workbook()
    summary = wb.active
    summary.title = "Summary"
    summary.append(["model", "setting", *HEADER[2:], "% detected", "% corrected"])

    def style(ws, widths):
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(wrap_text=True, vertical="center")
            cell.fill = PatternFill("solid", fgColor="DDEBF7")
        for col, width in zip("ABCDEFGHI", widths):
            ws.column_dimensions[col].width = width
        ws.freeze_panes = "A2"

    parsings = [False, True] if lenient else [False]
    lines = []
    for decoding, (file_name, decoding_desc) in RESULT_FILES.items():
        path = run_dir / file_name
        if not path.exists():
            continue
        for use_lenient in parsings:
            totals = score_file(path, cases, use_lenient)
            for correction, (key, correction_desc) in CORRECTION_KEYS.items():
                title = f"{decoding} - {correction}" + (" (lenient)" if use_lenient else "")
                ws = wb.create_sheet(title[:31])
                ws.append(HEADER)
                n_all = det_all = cor_all = 0.0
                for name in sorted(cases, key=lambda n: int(n.split("-")[0])):
                    n = len(cases[name])
                    t = totals.get(name, collections.Counter(runs=1))
                    runs = t["runs"] or 1
                    det, cor = round(t["detected"] / runs, 1), round(t[key] / runs, 1)
                    ws.append([model, name, n, det, round(n - det, 1), cor, round(n - cor, 1)])
                    n_all += n
                    det_all += det
                    cor_all += cor
                ws.append(
                    [model, "TOTAL", n_all, round(det_all, 1), round(n_all - det_all, 1), round(cor_all, 1), round(n_all - cor_all, 1)]
                )
                for cell in ws[ws.max_row]:
                    cell.font = Font(bold=True)
                ws.append([])
                note = f"Setting: {decoding_desc}; {correction_desc}."
                if use_lenient:
                    note += " Lenient: schema errors ignored, findings/patch scored as given."
                ws.append([note + " No. errors includes the out-of-scope errors (always undetected/uncorrected)."])
                style(ws, (48, 22, 10, 12, 12, 12, 12))
                summary.append(
                    [
                        model,
                        title,
                        n_all,
                        round(det_all, 1),
                        round(n_all - det_all, 1),
                        round(cor_all, 1),
                        round(n_all - cor_all, 1),
                        round(100 * det_all / n_all, 1),
                        round(100 * cor_all / n_all, 1),
                    ]
                )
                lines.append(f"{title:34} detected {100 * det_all / n_all:5.1f}%   corrected {100 * cor_all / n_all:5.1f}%")
    style(summary, (48, 34, 10, 12, 12, 12, 12, 11, 11))
    return wb, lines


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="folder holding scenarios_results.xlsx and/or scenarios_results_greedy.xlsx")
    parser.add_argument("--lenient", action="store_true", help="also score ignoring schema errors")
    parser.add_argument("--output", default=None, help="default: <run_dir>/scenario_summary.xlsx")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    wb, lines = build_workbook(run_dir, args.lenient)
    output = Path(args.output) if args.output else run_dir / "scenario_summary.xlsx"
    wb.save(output)
    print("\n".join(lines))
    print(f"\nWritten to {output}")


if __name__ == "__main__":
    main()
