"""
Evaluates a fine-tuned CLI-agent checkpoint against query_training/output/test.jsonl.

The target is a shell command string, so the metrics are string/AST-level,
not the JSON-schema metrics the manifest-security task uses:

  - exact_match     : identical after whitespace normalisation
  - normalized_match : identical after canonicalisation (verb kept in place,
                       remaining argv tokens sorted) -- forgives flag order
  - valid_kubectl    : output parses and starts `kubectl <known-verb>`
  - verb_match       : predicted first verb == expected first verb

The scoring half (parse/canonicalise/score_example/aggregate) is pure Python
+ stdlib and is unit-tested from the project's main venv. run() needs the
training stack (finetune/.venv).

Usage:
    finetune/.venv/bin/python -m query_training.evaluate --adapter query_training/runs/<run>/lora_adapter
    finetune/.venv/bin/python -m query_training.evaluate --adapter <path> --limit 200
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path

from query_training.schema import KUBECTL_VERBS, SYSTEM_PROMPT, extract_command

_FENCE_RE = re.compile(r"^```(?:[a-zA-Z]*)?\s*|\s*```$")
_WS_RE = re.compile(r"\s+")


def clean_prediction(raw: str) -> str:
    """Take the model's raw generation down to a single candidate command.

    Handles both target formats: a {"plan","command"} JSON object (extract
    .command) or a bare command line. Then strips markdown fences and a
    leading `$ ` prompt and keeps the first non-empty line."""
    text = extract_command(raw.strip())
    text = _FENCE_RE.sub("", text).strip()
    for line in text.splitlines():
        line = line.strip()
        if line:
            return _WS_RE.sub(" ", line).lstrip("$ ").strip()
    return ""


def parse_argv(command: str) -> list[str] | None:
    try:
        return shlex.split(command)
    except ValueError:
        return None


def command_verb(command: str) -> str | None:
    argv = parse_argv(command)
    if not argv or len(argv) < 2 or argv[0] != "kubectl":
        return None
    return argv[1]


def canonicalise(command: str) -> str:
    """`kubectl <verb>` kept in order, every remaining token sorted. Lets
    `kubectl get pods -n foo -o wide` and `kubectl get -o wide -n foo pods`
    compare equal without a full kubectl flag grammar."""
    argv = parse_argv(command)
    if not argv:
        return _WS_RE.sub(" ", command.strip())
    head, tail = argv[:2], sorted(argv[2:])
    return " ".join(head + tail)


def score_example(expected: str, raw_prediction: str) -> dict:
    """`expected` is the dataset's assistant turn -- a bare command or a
    {"plan","command"} JSON string; either way it's reduced to the command."""
    predicted = clean_prediction(raw_prediction)
    expected = extract_command(expected.strip())
    exp_norm = _WS_RE.sub(" ", expected.strip())
    exp_verb = command_verb(expected)
    pred_verb = command_verb(predicted)
    return {
        "predicted": predicted,
        "exact_match": predicted == exp_norm,
        "normalized_match": canonicalise(predicted) == canonicalise(expected),
        "valid_kubectl": pred_verb in KUBECTL_VERBS,
        "verb_match": pred_verb is not None and pred_verb == exp_verb,
    }


def aggregate(results: list[dict]) -> dict:
    total = len(results)
    if not total:
        return {"total": 0}
    keys = ("exact_match", "normalized_match", "valid_kubectl", "verb_match")
    summary = {"total": total}
    for k in keys:
        summary[f"{k}_rate"] = round(sum(bool(r[k]) for r in results) / total, 4)
    return summary


def load_examples(path: Path, limit: int | None) -> list[tuple[str, str, str]]:
    """Returns (system_prompt, instruction, expected_assistant_turn) triples --
    the system prompt is taken from the data so eval matches whichever
    --target-format the split was built with."""
    out: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            messages = json.loads(line)["messages"]
            out.append((messages[0]["content"], messages[1]["content"], messages[2]["content"]))
            if limit is not None and len(out) >= limit:
                break
    return out


def load_model(adapter_path: str, max_seq_length: int):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path, max_seq_length=max_seq_length, load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate(model, tokenizer, instruction: str, max_new_tokens: int, system: str = SYSTEM_PROMPT) -> str:
    import torch

    messages = [{"role": "system", "content": system}, {"role": "user", "content": instruction}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def run(args: argparse.Namespace) -> dict:
    model, tokenizer = load_model(args.adapter, args.max_seq_length)
    examples = load_examples(Path(args.test_file), args.limit)

    results, details = [], []
    for i, (system, instruction, expected) in enumerate(examples):
        raw = generate(model, tokenizer, instruction, args.max_new_tokens, system=system)
        result = score_example(expected, raw)
        results.append(result)
        details.append({"index": i, "instruction": instruction, "expected": expected, "raw_output": raw, **result})
        if (i + 1) % max(1, args.log_every) == 0:
            rate = sum(r["normalized_match"] for r in results) / len(results)
            print(f"[{i + 1}/{len(examples)}] normalized_match so far: {rate:.3f}")

    summary = aggregate(results)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "eval_details.jsonl").open("w", encoding="utf-8") as fh:
        for record in details:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWritten to {out_dir}/eval_summary.json and eval_details.jsonl")
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True, help="path to a saved LoRA adapter or checkpoint-N dir")
    parser.add_argument("--test-file", default="query_training/output/test.jsonl")
    parser.add_argument("--output-dir", default="query_training/output/eval")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
