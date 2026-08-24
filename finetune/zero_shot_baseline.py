"""
Zero-shot baseline: sends this project's exact SYSTEM_PROMPT (the same
contract the fine-tuned model is trained on) plus a manifest to a base
(non-fine-tuned) model via Ollama, and scores the response with the same
pure-Python logic finetune/evaluate.py uses -- so the fine-tuned adapter and
an untuned base model are judged by identical rules. Useful for answering
"does the bigger base model already do this without any fine-tuning at
all?" without loading a second copy of the training stack.

Usage:
    ollama serve &   # if not already running
    .venv/bin/python -m finetune.zero_shot_baseline --model qwen2.5-coder:14b --manifest sql.yaml
    .venv/bin/python -m finetune.zero_shot_baseline --model qwen2.5-coder:14b --test-file dataset/output/test.jsonl --index 7
"""

from __future__ import annotations

import argparse
import json

import httpx
import yaml

from dataset.schema import SYSTEM_PROMPT
from finetune.evaluate import evaluate_example, parse_model_output

OLLAMA_URL = "http://localhost:11434"


def load_manifest_and_ground_truth(args: argparse.Namespace) -> tuple[str, dict | None]:
    sources = [args.manifest, args.test_file]
    if sum(s is not None for s in sources) != 1:
        raise ValueError("pass exactly one of --manifest or --test-file")

    if args.manifest:
        return open(args.manifest, encoding="utf-8").read(), None

    with open(args.test_file, encoding="utf-8") as fh:
        lines = [line for line in fh if line.strip()]
    example = json.loads(lines[args.index])
    messages = example["messages"]
    return messages[1]["content"], json.loads(messages[2]["content"])


def query_ollama(model: str, system: str, user: str, timeout: float = 180.0) -> str:
    resp = httpx.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def run(args: argparse.Namespace) -> None:
    manifest_text, ground_truth = load_manifest_and_ground_truth(args)

    print("=" * 80)
    print(f"MODEL (zero-shot, no fine-tuning): {args.model}")
    print("=" * 80)
    print(manifest_text)

    output_text = query_ollama(args.model, SYSTEM_PROMPT, manifest_text)

    print("=" * 80)
    print("MODEL OUTPUT (raw)")
    print("=" * 80)
    print(output_text)

    response, errors = parse_model_output(output_text)
    if response is None:
        print("\nSchema validation FAILED:")
        for e in errors:
            print(" -", e)
        return

    print("\nParsed OK:")
    print(json.dumps(response, indent=2, ensure_ascii=False))

    if ground_truth is not None:
        input_doc = yaml.safe_load(manifest_text)
        result = evaluate_example(input_doc, ground_truth, output_text)
        print("\n" + "=" * 80)
        print("GROUND TRUTH")
        print("=" * 80)
        print(json.dumps(ground_truth, indent=2, ensure_ascii=False))
        print("\n" + "=" * 80)
        print("VERDICT")
        print("=" * 80)
        print(json.dumps(result, indent=2, ensure_ascii=False))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen2.5-coder:14b", help="Ollama model tag")
    parser.add_argument("--manifest", default=None, help="path to a raw YAML manifest file")
    parser.add_argument("--test-file", default=None, help="dataset .jsonl file to pull an example + ground truth from")
    parser.add_argument("--index", type=int, default=0, help="line index into --test-file")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
