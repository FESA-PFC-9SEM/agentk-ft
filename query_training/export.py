"""
Renders query_training/output/{train,val,test}.jsonl (the {"messages": [...]}
records clean.py writes) into the format query_training/train.py consumes:

  {"messages": [...], "text": "<full chat-template rendering>", "num_tokens": N}

Why a separate step rather than doing it inside the trainer:
  1. The "text" field is rendered through the *target model's own* chat
     template (tokenizer.apply_chat_template), so what the trainer sees is
     byte-identical to what inference will see -- the usual silent
     underperformance trap in this kind of SFT.
  2. Real token counts let us DROP (never truncate) the handful of
     over-length rows. Truncating would cut into the assistant turn and
     corrupt the label.

CLI commands are short, so --max-seq-length defaults to 1024 and in practice
nothing is dropped; the knob is here for safety and for odd corpora.

Usage:
    python -m query_training.export
    python -m query_training.export --tokenizer unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit
    python -m query_training.export --char-approx     # offline, no tokenizer download
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_TOKENIZER = "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit"
DEFAULT_MAX_SEQ_LENGTH = 1024
CHARS_PER_TOKEN_ESTIMATE = 4  # fallback only


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def render_and_count(messages: list[dict], tokenizer) -> tuple[str, int]:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    num_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return text, num_tokens


def char_approx_render_and_count(messages: list[dict]) -> tuple[str, int]:
    text = "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)
    return text, max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


def percentiles(values: list[int], ps=(0.5, 0.9, 0.95, 0.99, 1.0)) -> dict[str, int]:
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)
    return {f"p{p * 100:.0f}": ordered[min(int(n * p), n - 1)] for p in ps}


def process_split(input_path: Path, output_path: Path, tokenizer, max_seq_length: int, use_char_approx: bool) -> dict:
    kept = dropped = 0
    kept_lengths: list[int] = []

    with input_path.open("r", encoding="utf-8") as fh, output_path.open("w", encoding="utf-8") as out:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            messages = json.loads(line)["messages"]
            if use_char_approx:
                text, num_tokens = char_approx_render_and_count(messages)
            else:
                text, num_tokens = render_and_count(messages, tokenizer)
            if num_tokens > max_seq_length:
                dropped += 1
                continue
            kept += 1
            kept_lengths.append(num_tokens)
            out.write(json.dumps({"messages": messages, "text": text, "num_tokens": num_tokens}, ensure_ascii=False) + "\n")

    return {"kept": kept, "dropped": dropped, "kept_token_percentiles": percentiles(kept_lengths)}


def export(args: argparse.Namespace) -> dict:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    if not args.char_approx:
        # A failed tokenizer load is a real problem, not a reason to silently
        # fall back to chars/4 -- that rendering has no real <|im_start|>assistant
        # marker, which train.py's train_on_responses_only needs. Fail loud.
        print(f"Loading tokenizer '{args.tokenizer}'...", file=sys.stderr)
        tokenizer = load_tokenizer(args.tokenizer)

    summary: dict = {
        "max_seq_length": args.max_seq_length,
        "token_counting_method": "chars/4 approximation" if args.char_approx else args.tokenizer,
    }
    for split in ("train", "val", "test"):
        input_path = input_dir / f"{split}.jsonl"
        if not input_path.exists():
            continue
        stats = process_split(input_path, output_dir / f"{split}.jsonl", tokenizer, args.max_seq_length, args.char_approx)
        summary[split] = stats
        print(json.dumps({split: stats}, indent=2, ensure_ascii=False), file=sys.stderr)

    (output_dir / "export_diagnostic.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone. Written to {output_dir}/", file=sys.stderr)
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", default="query_training/output")
    parser.add_argument("--output-dir", default="query_training/output/unsloth")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--char-approx", action="store_true", help="estimate tokens as chars/4 (offline, less accurate)")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    export(parse_args(argv))


if __name__ == "__main__":
    main()
