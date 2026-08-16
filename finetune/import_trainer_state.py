"""
Recovers training metrics from a checkpoint's trainer_state.json, for runs
where finetune/metrics_logger.py's JsonlMetricsLogger wasn't active from the
start (e.g. it was added mid-run, or the run predates it). The HF Trainer
always writes trainer_state.json into every checkpoint-N/ directory,
regardless of --report-to or any custom callback -- its log_history field
has the exact same shape (step, epoch, loss, eval_loss, learning_rate,
grad_norm) as what JsonlMetricsLogger records, so it drops straight into
finetune/plot_metrics.py once converted to JSONL.

Usage:
    python -m finetune.import_trainer_state --trainer-state finetune/output/checkpoint-1003/trainer_state.json --output finetune/output/metrics.jsonl
    python -m finetune.plot_metrics --metrics-file finetune/output/metrics.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def import_log_history(trainer_state_path: Path) -> list[dict]:
    data = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    return data.get("log_history", [])


def write_metrics_jsonl(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> dict:
    trainer_state_path = Path(args.trainer_state)
    output_path = Path(args.output)

    records = import_log_history(trainer_state_path)
    write_metrics_jsonl(records, output_path)

    summary = {
        "source": str(trainer_state_path),
        "records_imported": len(records),
        "output": str(output_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trainer-state", required=True, help="path to a checkpoint's trainer_state.json")
    parser.add_argument("--output", default="finetune/output/metrics.jsonl")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
