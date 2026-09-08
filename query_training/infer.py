"""
Runs a single inference pass with a fine-tuned CLI-agent checkpoint against
one request, for quick manual inspection. For metrics across the whole test
split use query_training/evaluate.py -- this is the "let me look at one" tool.

Three ways to give it a request:
    --instruction "list all pods in kube-system"   a request string
    --stdin                                        pipe the request in
    --test-file X.jsonl --index N                  pull the Nth example from a
                                                   split -- also prints the
                                                   expected command and a
                                                   pass/fail verdict

Usage:
    finetune/.venv/bin/python -m query_training.infer --adapter query_training/runs/<run>/lora_adapter --instruction "restart the nginx deployment"
    echo "get the logs of the api pod" | finetune/.venv/bin/python -m query_training.infer --adapter <path> --stdin
    finetune/.venv/bin/python -m query_training.infer --adapter <path> --test-file query_training/output/test.jsonl --index 7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from query_training.evaluate import clean_prediction, generate, load_model, score_example
from query_training.schema import PLAN_COMMAND_SYSTEM_PROMPT, SYSTEM_PROMPT


def load_request_and_expected(args: argparse.Namespace) -> tuple[str, str, str | None]:
    """Returns (system_prompt, instruction, expected_assistant_turn_or_None).
    For --test-file the system prompt comes from the data; for a free-form
    request it's picked by --target-format (must match the adapter's training)."""
    if args.test_file:
        with Path(args.test_file).open("r", encoding="utf-8") as fh:
            lines = [line for line in fh if line.strip()]
        if args.index >= len(lines):
            raise IndexError(f"--index {args.index} out of range (file has {len(lines)} examples)")
        messages = json.loads(lines[args.index])["messages"]
        return messages[0]["content"], messages[1]["content"], messages[2]["content"]
    system = PLAN_COMMAND_SYSTEM_PROMPT if args.target_format == "plan-command" else SYSTEM_PROMPT
    if args.stdin:
        return system, sys.stdin.read().strip(), None
    if args.instruction:
        return system, args.instruction, None
    raise ValueError("one of --instruction, --stdin, or --test-file/--index is required")


def run(args: argparse.Namespace) -> dict:
    system, instruction, expected = load_request_and_expected(args)

    model, tokenizer = load_model(args.adapter, args.max_seq_length)
    raw = generate(model, tokenizer, instruction, args.max_new_tokens, system=system)
    predicted = clean_prediction(raw)

    print("=" * 80)
    print("REQUEST")
    print("=" * 80)
    print(instruction)
    print()
    print("=" * 80)
    print("MODEL OUTPUT (raw)")
    print("=" * 80)
    print(raw)
    print()
    print("PREDICTED COMMAND:", predicted)

    verdict = None
    if expected is not None:
        verdict = score_example(expected, raw)
        print()
        print("=" * 80)
        print("EXPECTED")
        print("=" * 80)
        print(expected)
        print()
        print("=" * 80)
        print("VERDICT")
        print("=" * 80)
        print(json.dumps(verdict, indent=2, ensure_ascii=False))

    return {"instruction": instruction, "raw_output": raw, "predicted": predicted, "expected": expected, "verdict": verdict}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True, help="path to a saved LoRA adapter or checkpoint-N dir")
    parser.add_argument("--instruction", default=None, help="the request string")
    parser.add_argument("--stdin", action="store_true", help="read the request from stdin")
    parser.add_argument("--test-file", default=None, help="pull an example from this split .jsonl instead")
    parser.add_argument("--index", type=int, default=0, help="which example, with --test-file")
    parser.add_argument(
        "--target-format",
        choices=["command", "plan-command"],
        default="command",
        help="must match the adapter's training; ignored with --test-file (taken from the data)",
    )
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
