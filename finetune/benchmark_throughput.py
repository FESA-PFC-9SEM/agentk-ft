"""
Benchmarks training throughput (samples/sec, steps/sec, peak VRAM) across
combinations of --batch-size and --grad-accum, to find the sweet spot on the
current GPU for this project's model/dataset. Each combo runs a short burst
of real training steps (no eval, no checkpointing, no load_best_model_at_end
-- none of that is representative of pure throughput and just adds time), on
the actual training data so sequence-length distribution matches a real run.
A combo that hits CUDA OOM is recorded and skipped, not fatal to the sweep.

The model is loaded once and reused across every combo -- LoRA weights drift
a little between short bursts, but that doesn't matter here: only step time
is measured, not training quality.

Results are always written as JSON -- to --output if given, otherwise to an
auto-generated finetune/output/throughput/<timestamp>_<model-slug>.json.

Usage:
    finetune/.venv/bin/python -m finetune.benchmark_throughput \\
        --model unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit \\
        --data-dir dataset/output-multi-defect/unsloth \\
        --combos 4x8 8x4 16x2 2x16 \\
        --steps 20

    # fixed effective batch (32), sweeping how it's split
    finetune/.venv/bin/python -m finetune.benchmark_throughput \\
        --model unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit \\
        --combos 2x16 4x8 8x4 16x2 32x1 --steps 20
"""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path

from finetune.run_naming import slugify_model


def parse_combo(s: str) -> tuple[int, int]:
    """Parses a "batch_sizeXgrad_accum" string, e.g. "8x4" -> (8, 4)."""
    b, g = s.lower().split("x")
    return int(b), int(g)


def format_row(r: dict) -> str:
    if r["oom"]:
        return f"{r['batch_size']:>6} {r['grad_accum']:>6} {r['effective_batch']:>5} {'OOM':>12}"
    return (
        f"{r['batch_size']:>6} {r['grad_accum']:>6} {r['effective_batch']:>5} "
        f"{r['samples_per_second']:>10.3f} {r['steps_per_second']:>8.3f} {r['peak_vram_gb']:>10.2f}GB"
    )


def run(args: argparse.Namespace) -> list[dict]:
    # unsloth must be imported before trl/transformers/peft -- it patches
    # them at import time, and importing trl first (as this used to do)
    # triggers Unsloth's own "should be imported before" warning and skips
    # some of its speed optimizations. Same ordering as train_unsloth.py.
    import unsloth  # noqa: F401
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    print(f"Loading model {args.model}...")
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
        random_state=42,
    )

    train_dataset = load_dataset("json", data_files=str(Path(args.data_dir) / "train.jsonl"), split="train")

    combos = [parse_combo(c) for c in args.combos]
    results = []

    for batch_size, grad_accum in combos:
        effective = batch_size * grad_accum
        print(f"\n=== batch_size={batch_size} grad_accum={grad_accum} (effective={effective}) ===")
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        try:
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                dataset_text_field="text",
                max_seq_length=args.max_seq_length,
                packing=False,
                args=SFTConfig(
                    max_steps=args.steps,
                    output_dir="/tmp/finetune_throughput_benchmark",
                    logging_steps=max(1, args.steps // 4),
                    dataset_num_proc=1,
                    per_device_train_batch_size=batch_size,
                    gradient_accumulation_steps=grad_accum,
                    eval_strategy="no",
                    save_strategy="no",
                    report_to="none",
                    seed=42,
                ),
            )
            trainer = train_on_responses_only(
                trainer,
                instruction_part="<|im_start|>user\n",
                response_part="<|im_start|>assistant\n",
            )
            train_output = trainer.train()
            metrics = train_output.metrics
            peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
            result = {
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "effective_batch": effective,
                "samples_per_second": metrics.get("train_samples_per_second"),
                "steps_per_second": metrics.get("train_steps_per_second"),
                "peak_vram_gb": round(peak_vram_gb, 2),
                "oom": False,
            }
            del trainer
        except torch.OutOfMemoryError:
            print("  OOM")
            result = {
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "effective_batch": effective,
                "samples_per_second": None,
                "steps_per_second": None,
                "peak_vram_gb": None,
                "oom": True,
            }

        results.append(result)
        print("  " + format_row(result))
        torch.cuda.empty_cache()
        gc.collect()

    print("\n=== Summary ===")
    print(f"{'batch':>6} {'accum':>6} {'eff':>5} {'samples/s':>10} {'steps/s':>8} {'peak VRAM':>12}")
    for r in results:
        print(format_row(r))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nWritten to {output_path}")

    return results


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", default="dataset/output-multi-defect/unsloth")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument(
        "--combos",
        nargs="+",
        required=True,
        help='batch_sizeXgrad_accum pairs to benchmark, e.g. --combos 4x8 8x4 16x2',
    )
    parser.add_argument("--steps", type=int, default=20, help="training steps per combo (short burst)")
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--output",
        default=None,
        help="where to write results as JSON (default: auto-generated under finetune/output/throughput/)",
    )

    args = parser.parse_args(argv)
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        args.output = str(Path("finetune/output/throughput") / f"{timestamp}_{slugify_model(args.model)}.json")
    return args


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
