"""
Plots training curves from a metrics.jsonl file written by
finetune/train_unsloth.py (via finetune/metrics_logger.py). No GPU or
training stack needed -- run this from the project's main .venv (matplotlib
is already a core dependency), not finetune/.venv.

Usage:
    python -m finetune.plot_metrics --metrics-file finetune/output/metrics.jsonl
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from finetune.metrics_logger import load_metrics


def split_series(records: list[dict]) -> dict[str, list[tuple[float, float]]]:
    """Groups metric records into named (step, value) series. Training-step
    logs carry 'loss'; eval logs carry 'eval_loss'; both may carry
    'learning_rate' and 'grad_norm'."""
    series: dict[str, list[tuple[float, float]]] = {}
    for record in records:
        step = record.get("step")
        if step is None:
            continue
        for key in ("loss", "eval_loss", "learning_rate", "grad_norm"):
            if key in record and record[key] is not None:
                series.setdefault(key, []).append((step, record[key]))
    return series


def write_csv(series: dict[str, list[tuple[float, float]]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "step", "value"])
        for name, points in series.items():
            for step, value in points:
                writer.writerow([name, step, value])


def write_png(series: dict[str, list[tuple[float, float]]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    layout = {
        "loss": axes[0][0],
        "eval_loss": axes[0][1],
        "learning_rate": axes[1][0],
        "grad_norm": axes[1][1],
    }
    titles = {
        "loss": "Training loss",
        "eval_loss": "Eval loss",
        "learning_rate": "Learning rate",
        "grad_norm": "Gradient norm",
    }

    for name, ax in layout.items():
        points = series.get(name, [])
        if not points:
            ax.set_title(f"{titles[name]} (no data)")
            continue
        steps, values = zip(*points)
        ax.plot(steps, values, marker="o", markersize=2)
        ax.set_title(titles[name])
        ax.set_xlabel("step")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot(args: argparse.Namespace) -> dict:
    metrics_path = Path(args.metrics_file)
    records = load_metrics(metrics_path)
    if not records:
        print(f"No records found in {metrics_path}")
        return {"total_records": 0}

    series = split_series(records)

    output_dir = Path(args.output_dir) if args.output_dir else metrics_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(series, output_dir / "metrics.csv")
    write_png(series, output_dir / "metrics.png")

    summary = {"total_records": len(records), "series": {k: len(v) for k, v in series.items()}}
    print(summary)
    print(f"CSV and PNG written to {output_dir}/")
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics-file", default="finetune/output/metrics.jsonl")
    parser.add_argument("--output-dir", default=None, help="default: same directory as --metrics-file")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    plot(args)


if __name__ == "__main__":
    main()
