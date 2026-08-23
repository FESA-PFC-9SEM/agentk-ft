"""Generates consistent, self-describing run folder names for
finetune/train_unsloth.py's default --output-dir, so training runs land in
runs/<name>/ automatically instead of needing a manual copy/rename after
each run (see README.md's "Training run history" section)."""

from __future__ import annotations

from datetime import datetime


def slugify_model(model: str) -> str:
    name = model.rsplit("/", 1)[-1]
    return name.replace(".", "_").lower()


def generate_run_name(strategy: str, model: str, when: datetime | None = None) -> str:
    timestamp = (when or datetime.now()).strftime("%Y%m%d-%H%M")
    return f"{timestamp}_{strategy}_{slugify_model(model)}"
