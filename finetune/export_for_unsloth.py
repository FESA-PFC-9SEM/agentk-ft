"""
Prepares dataset/output/{train,val,test}.jsonl for fine-tuning with Unsloth
(or any TRL-style SFTTrainer workflow). Reads the {"messages": [system,
user, assistant]} records dataset/build.py already emits and:

1. Renders each example through the target model's own chat template
   (tokenizer.apply_chat_template), so train-time formatting is guaranteed
   identical to what inference will actually see -- the single biggest
   real-world way this kind of SFT setup silently underperforms.
2. Counts real tokens for that rendering and DROPS (never truncates)
   examples over --max-seq-length. Truncating would cut into the assistant
   turn's JSON and corrupt the label; dropping is the only safe option.
3. Writes {"messages", "text", "num_tokens"} records, so you can use either
   the pre-rendered "text" field directly (dataset_text_field="text") or
   apply the chat template yourself from "messages" if you need something
   like Unsloth's train_on_responses_only.

The corpus has a long tail of huge CustomResourceDefinition manifests (no
pod spec, so no mutator ever touches them -- they only ever end up as
oversized clean negatives). --max-seq-length defaults to 4096, which in a
20k-example real run dropped ~1.3% of examples while keeping the
distribution's p95 comfortably (see the printed diagnostic for your data).

Usage:
    python -m finetune.export_for_unsloth
    python -m finetune.export_for_unsloth --tokenizer unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit --max-seq-length 4096
    python -m finetune.export_for_unsloth --char-approx   # offline, no tokenizer download
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_TOKENIZER = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
DEFAULT_MAX_SEQ_LENGTH = 4096
CHARS_PER_TOKEN_ESTIMATE = 4  # fallback only -- a real tokenizer is always more accurate


def load_tokenizer(name: str):
    from transformers import AutoTokenizer  # imported lazily: not a hard dependency of the core pipeline

    return AutoTokenizer.from_pretrained(name)


def render_and_count(messages: list[dict], tokenizer) -> tuple[str, int]:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    num_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return text, num_tokens


def char_approx_render_and_count(messages: list[dict]) -> tuple[str, int]:
    text = "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)
    return text, max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


def percentiles(values: list[int], ps: tuple[float, ...] = (0.5, 0.75, 0.9, 0.95, 0.99, 1.0)) -> dict[str, int]:
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)
    return {f"p{p * 100:.0f}": ordered[min(int(n * p), n - 1)] for p in ps}


def process_split(
    input_path: Path,
    output_path: Path,
    tokenizer,
    max_seq_length: int,
    use_char_approx: bool,
) -> dict:
    kept = 0
    dropped = 0
    kept_lengths: list[int] = []
    dropped_lengths: list[int] = []

    with input_path.open("r", encoding="utf-8") as fh, output_path.open("w", encoding="utf-8") as out:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            messages = example["messages"]

            if use_char_approx:
                text, num_tokens = char_approx_render_and_count(messages)
            else:
                text, num_tokens = render_and_count(messages, tokenizer)

            if num_tokens > max_seq_length:
                dropped += 1
                dropped_lengths.append(num_tokens)
                continue

            kept += 1
            kept_lengths.append(num_tokens)
            record = {"messages": messages, "text": text, "num_tokens": num_tokens}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "kept": kept,
        "dropped": dropped,
        "kept_token_percentiles": percentiles(kept_lengths),
        "dropped_token_percentiles": percentiles(dropped_lengths),
    }


def export(args: argparse.Namespace) -> dict:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    use_char_approx = args.char_approx
    if not use_char_approx:
        try:
            print(f"Loading tokenizer '{args.tokenizer}'...", file=sys.stderr)
            tokenizer = load_tokenizer(args.tokenizer)
        except Exception as e:
            print(
                f"[WARNING] could not load tokenizer '{args.tokenizer}' ({e}); "
                "falling back to a chars/4 token estimate. Install `transformers` "
                "and ensure network access to Hugging Face for accurate counts, "
                "or pass --char-approx to silence this.",
                file=sys.stderr,
            )
            use_char_approx = True

    summary: dict = {
        "max_seq_length": args.max_seq_length,
        "token_counting_method": "chars/4 approximation" if use_char_approx else args.tokenizer,
    }
    for split in ("train", "val", "test"):
        input_path = input_dir / f"{split}.jsonl"
        if not input_path.exists():
            continue
        output_path = output_dir / f"{split}.jsonl"
        print(f"Processing {split}...", file=sys.stderr)
        stats = process_split(input_path, output_path, tokenizer, args.max_seq_length, use_char_approx)
        summary[split] = stats
        print(json.dumps({split: stats}, indent=2, ensure_ascii=False), file=sys.stderr)

    diag_path = output_dir / "export_diagnostic.json"
    diag_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone. Filtered splits + export_diagnostic.json written to {output_dir}/", file=sys.stderr)
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", default="dataset/output")
    parser.add_argument("--output-dir", default="dataset/output/unsloth")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument(
        "--char-approx",
        action="store_true",
        help="skip loading a real tokenizer; estimate tokens as chars/4 (offline, less accurate)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    export(args)


if __name__ == "__main__":
    main()
