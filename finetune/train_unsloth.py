"""
Fine-tunes a Qwen2.5-Coder model with Unsloth QLoRA on the dataset exported
by finetune/export_for_unsloth.py. Wraps the standard Unsloth + TRL
SFTTrainer recipe with this project's agreed hyperparameters as defaults
(see README.md's fine-tuning section).

Requires a dedicated Python 3.11 venv (this stack breaks on the project's
main Python 3.14 venv -- see finetune/requirements.txt for why) with
finetune/requirements.txt installed:
    uv python install 3.11
    uv venv --python 3.11 finetune/.venv
    uv pip install --python finetune/.venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu126
    uv pip install --python finetune/.venv/bin/python -r finetune/requirements.txt

Usage:
    python -m finetune.train_unsloth
    python -m finetune.train_unsloth --model unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit --batch-size 4 --grad-accum 4 --max-seq-length 4096
"""

from __future__ import annotations
import json
import unsloth
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

from finetune.metrics_logger import JsonlMetricsLogger

import argparse
from pathlib import Path


def load_jsonl_as_dataset(path: Path) -> Dataset:
    """Reads a .jsonl file into a datasets.Dataset via Dataset.from_list,
    bypassing datasets.load_dataset's dill-based cache fingerprinting --
    that path breaks on newer Python (e.g. 3.13+/3.14) where dill's patched
    Pickler is incompatible with the stdlib's _batch_setitems signature
    (TypeError: Pickler._batch_setitems() takes 2 positional arguments but
    3 were given). from_list stays in-process and avoids it entirely."""
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


# Hardware-dependent knobs only -- --r/--lora-alpha, --lr, --epochs etc. are
# task/data decisions, not hardware ones, so they stay as plain argparse
# defaults below and aren't part of a preset. --model IS part of the preset:
# every batch/seq-length number below was sized assuming the 7B model:
# swapping in --model without --preset (or a smaller/larger model on the
# same preset) invalidates them.
PRESETS = {
    # RTX 4060 Ti, 8GB VRAM: batch=1 + heavy grad-accum + capped seq length
    # + defensive eval settings, all forced by tight VRAM (see the OOM
    # history in README.md's fine-tuning section).
    "4060ti": {
        "model": "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        "max_seq_length": 4096,
        "batch_size": 4,
        "grad_accum": 4,
        "eval_batch_size": 1,
        "eval_accumulation_steps": 1,
        "eval_steps": 100,
        "save_steps": 100,
    },
    # NVIDIA L4, 24GB VRAM: far less VRAM-constrained than the 4060 Ti, but
    # noticeably less compute than an A100 (Ada Lovelace, no NVLink, roughly
    # RTX 4090-class throughput) -- same effective batch (32) as the a100
    # preset via more accumulation, but a smaller per-device batch, since a
    # slower GPU gains less from a large single batch and 24GB has less
    # margin for an allocation spike than 40-80GB.
    "l4": {
        "model": "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        "max_seq_length": 4096,
        "batch_size": 8,
        "grad_accum": 4,
        "eval_batch_size": 4,
        "eval_accumulation_steps": 2,
        "eval_steps": 50,
        "save_steps": 50,
    },
    # A100 (40GB or 80GB): no longer VRAM-constrained -- bigger batches,
    # less accumulation, less defensive eval offloading, more frequent
    # checkpoints since total steps roughly halve at the larger effective
    # batch size.
    "a100": {
        "model": "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        "max_seq_length": 4096,
        "batch_size": 16,
        "grad_accum": 2,
        "eval_batch_size": 8,
        "eval_accumulation_steps": 4,
        "eval_steps": 50,
        "save_steps": 50,
    },
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None)
    parser.add_argument("--data-dir", default="dataset/output/unsloth")
    parser.add_argument("--output-dir", default="finetune/output")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="4060ti",
        help="hardware preset for the knobs in PRESETS; any of those flags passed explicitly overrides the preset's value",
    )
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-accumulation-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--eval-limit", type=int, default=None, help="cap eval_dataset to this many rows (default: use all)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from the latest checkpoint-* in --output-dir (e.g. after an OOM crash) instead of starting fresh",
    )
    parser.add_argument(
        "--metrics-file",
        default="finetune/output/metrics.jsonl",
        help="where to append training/eval metrics as JSONL (default: <output-dir>/metrics.jsonl)",
    )
    parser.add_argument("--logging-steps", type=int, default=10, help="how often to log train loss/lr")

    args = parser.parse_args(argv)
    preset = PRESETS[args.preset]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def main(argv=None) -> None:
    args = parse_args(argv)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    data_dir = Path(args.data_dir)
    train_dataset = load_jsonl_as_dataset(data_dir / "train.jsonl")
    eval_dataset = load_jsonl_as_dataset(data_dir / "val.jsonl")
    if args.eval_limit is not None:
        eval_dataset = eval_dataset.select(range(min(args.eval_limit, len(eval_dataset))))

    metrics_path = Path(args.metrics_file) if args.metrics_file else Path(args.output_dir) / "metrics.jsonl"
    metrics_logger = JsonlMetricsLogger(metrics_path)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        callbacks=[metrics_logger],
        args=SFTConfig(
            max_steps=args.max_steps,
            output_dir=args.output_dir,
            logging_steps=args.logging_steps,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            eval_accumulation_steps=args.eval_accumulation_steps,
            # We only need eval_loss (for load_best_model_at_end below), never
            # raw logits -- skips materializing+fp32-converting a 7B-vocab
            # logits tensor per eval batch, which is what OOM'd on 8GB VRAM.
            prediction_loss_only=True,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            optim="adamw_8bit",
            weight_decay=0.01,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            seed=args.seed,
            report_to="none",
        ),
    )

    # Qwen2.5's ChatML markers -- masks loss to the assistant turn only, per
    # this project's agreed training params (see README.md).
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    trainer.train(resume_from_checkpoint=args.resume)

    adapter_dir = Path(args.output_dir) / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"LoRA adapter saved to {adapter_dir}")
    print(f"Metrics logged to {metrics_path} -- plot with: python -m finetune.plot_metrics --metrics-file {metrics_path}")


if __name__ == "__main__":
    main()
