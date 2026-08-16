"""
Runs a single inference pass with a fine-tuned checkpoint against one
manifest, for quick manual testing/inspection. For metrics across the whole
test set, use finetune/evaluate.py instead -- this is the "let me look at
one example" tool.

Three ways to give it a manifest:
    --manifest path/to/file.yaml       a YAML file on disk
    --stdin                            pipe YAML in
    --test-file X.jsonl --index N      pull the Nth example from an existing
                                        dataset split -- also prints ground
                                        truth and a pass/fail verdict, since
                                        the label is already known

Usage:
    finetune/.venv/bin/python -m finetune.infer --adapter finetune/output/lora_adapter --manifest pod.yaml
    cat pod.yaml | finetune/.venv/bin/python -m finetune.infer --adapter finetune/output/lora_adapter --stdin
    finetune/.venv/bin/python -m finetune.infer --adapter finetune/output/lora_adapter --test-file dataset/output/test.jsonl --index 7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from dataset.schema import SYSTEM_PROMPT
from finetune.evaluate import evaluate_example, generate_response_text, load_model, parse_model_output


def load_manifest_and_ground_truth(args: argparse.Namespace) -> tuple[str, dict | None]:
    """Returns (manifest_yaml_text, ground_truth_response_or_None)."""
    if args.test_file:
        with Path(args.test_file).open("r", encoding="utf-8") as fh:
            lines = [line for line in fh if line.strip()]
        if args.index >= len(lines):
            raise IndexError(f"--index {args.index} out of range (file has {len(lines)} examples)")
        example = json.loads(lines[args.index])
        messages = example["messages"]
        user_content = messages[1]["content"]
        ground_truth = json.loads(messages[2]["content"])
        return user_content, ground_truth

    if args.stdin:
        return sys.stdin.read(), None

    if args.manifest:
        return Path(args.manifest).read_text(encoding="utf-8"), None

    raise ValueError("one of --manifest, --stdin, or --test-file/--index is required")


def run(args: argparse.Namespace) -> dict:
    manifest_text, ground_truth = load_manifest_and_ground_truth(args)

    model, tokenizer = load_model(args.adapter, args.max_seq_length)
    output_text = generate_response_text(model, tokenizer, SYSTEM_PROMPT, manifest_text, args.max_new_tokens)

    print("=" * 80)
    print("PROMPT (manifest)")
    print("=" * 80)
    print(manifest_text.rstrip())

    print()
    print("=" * 80)
    print("MODEL OUTPUT (raw)")
    print("=" * 80)
    print(output_text)

    response, errors = parse_model_output(output_text)
    print()
    if response is not None:
        print("Parsed OK:")
        print(json.dumps(response, indent=2, ensure_ascii=False))
    else:
        print(f"FAILED to parse/validate: {errors}")

    result = None
    if ground_truth is not None:
        input_doc = yaml.safe_load(manifest_text)
        result = evaluate_example(input_doc, ground_truth, output_text)

        print()
        print("=" * 80)
        print("GROUND TRUTH")
        print("=" * 80)
        print(json.dumps(ground_truth, indent=2, ensure_ascii=False))

        print()
        print("=" * 80)
        print("VERDICT")
        print("=" * 80)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    return {"output_text": output_text, "response": response, "ground_truth": ground_truth, "verdict": result}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True, help="path to a saved LoRA adapter or checkpoint-N dir")
    parser.add_argument("--manifest", default=None, help="path to a YAML manifest file")
    parser.add_argument("--stdin", action="store_true", help="read the manifest from stdin")
    parser.add_argument("--test-file", default=None, help="pull an example from this dataset .jsonl instead")
    parser.add_argument("--index", type=int, default=0, help="which example, with --test-file")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
