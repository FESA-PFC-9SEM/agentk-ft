"""
Persists every metric the HF Trainer logs (train loss per logging_steps,
eval_loss per eval_steps, learning_rate, etc.) to a plain JSONL file, so
training curves can be plotted later (finetune/plot_metrics.py) without
needing TensorBoard or a Weights & Biases account -- consistent with this
project's local-only, no-cloud-service constraint everywhere else.

With report_to="none" (what finetune/train_unsloth.py uses), the Trainer's
default behavior is to print each metrics dict to stdout and discard it.
This callback is the only thing that persists it anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

from transformers import TrainerCallback


class JsonlMetricsLogger(TrainerCallback):
    """One JSON object per line, written on every Trainer.log() call (both
    training-step logs and evaluation logs land here, distinguished by which
    keys are present -- eval logs have 'eval_loss', training logs have
    'loss'). Appends, so --resume continues the same history file."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        record = dict(logs)
        record["step"] = state.global_step
        record["epoch"] = logs.get("epoch", state.epoch)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_metrics(path: Path | str) -> list[dict]:
    """Reads a metrics.jsonl file back into a list of dicts. Used by both
    plot_metrics.py and tests."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
