"""
Fine-tunes a small Qwen2.5-Coder model with Unsloth QLoRA to turn a
natural-language cluster request into a single kubectl command, on the data
produced by query_training/clean.py + query_training/export.py.

This is a self-contained script for the CLI-agent task -- it shares no code
with the manifest-security trainer in finetune/. It does reuse the same
Python 3.11 venv and dependency stack (finetune/requirements.txt), since the
Unsloth/TRL/torch install is identical:

    uv python install 3.11
    uv venv --python 3.11 finetune/.venv
    uv pip install --python finetune/.venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu126
    uv pip install --python finetune/.venv/bin/python -r finetune/requirements.txt

Usage:
    finetune/.venv/bin/python -m query_training.train
    finetune/.venv/bin/python -m query_training.train --model unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit --epochs 3
"""

from __future__ import annotations

import unsloth  # noqa: F401  -- must import before transformers/trl for the patches to take

import argparse
import json
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only


class JsonlMetricsLogger(TrainerCallback):
    """Persists every Trainer.log() payload as one JSON object per line, so
    the loss curve survives report_to='none'. Appends, so --resume extends
    the same file."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        record = dict(logs)
        record["step"] = state.global_step
        record["epoch"] = logs.get("epoch", state.epoch)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def slugify_model(model: str) -> str:
    return model.rsplit("/", 1)[-1].replace(".", "_").lower()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit")
    parser.add_argument("--data-dir", default="query_training/output/unsloth")
    parser.add_argument("--output-dir", default=None, help="defaults to query_training/runs/<timestamp>_cli_<model>/")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--resume", action="store_true", help="resume from the latest checkpoint-* in --output-dir")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        args.output_dir = str(Path("query_training/runs") / f"{stamp}_cli_{slugify_model(args.model)}")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_info.json").write_text(
        json.dumps(
            {
                "task": "cli-agent",
                "model": args.model,
                "data_dir": args.data_dir,
                "r": args.r,
                "lora_alpha": args.lora_alpha,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.lr,
                "epochs": args.epochs,
                "max_seq_length": args.max_seq_length,
                "seed": args.seed,
                "started_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Run folder: {output_dir}")

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
    train_dataset = load_dataset("json", data_files=str(data_dir / "train.jsonl"), split="train")
    eval_dataset = load_dataset("json", data_files=str(data_dir / "val.jsonl"), split="train")

    metrics_path = output_dir / "metrics.jsonl"
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        callbacks=[JsonlMetricsLogger(metrics_path)],
        args=SFTConfig(
            max_steps=args.max_steps,
            output_dir=args.output_dir,
            logging_steps=args.logging_steps,
            dataset_num_proc=1,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
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

    # Qwen2.5 ChatML markers -- mask loss to the assistant turn only.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    trainer.train(resume_from_checkpoint=args.resume)

    adapter_dir = output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"LoRA adapter saved to {adapter_dir}")
    print(f"Metrics logged to {metrics_path}")
    print(f"  plot:     python -m query_training.plot_metrics --metrics-file {metrics_path}")
    print(f"  evaluate: finetune/.venv/bin/python -m query_training.evaluate --adapter {adapter_dir}")


if __name__ == "__main__":
    main()
