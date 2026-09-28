"""
Evaluates a fine-tuned checkpoint against dataset/output/test.jsonl, reusing
this project's own ground-truth machinery instead of inventing new metrics:

- dataset/schema.py::validate_response() for schema validity.
- the exact same jsonpatch round-trip check dataset/build.py uses when
  generating the dataset (apply_patch(mutated_doc, patch) == canonical),
  now checking the MODEL's own patch against ground truth instead of a
  mutator's.
- per-rule precision/recall/F1, since findings are a multi-label prediction
  over the active KSEC rules (dataset/schema.py::RULE_IDS).

Loss alone (finetune/plot_metrics.py) can't tell you any of this -- see the
README.md's fine-tuning section for why that matters here specifically
(~35% of examples share one identical "clean" target, which pulls average
loss down without proving the model distinguishes dirty from clean).

The scoring logic (parse_model_output/evaluate_example/aggregate_results)
has no heavy dependencies and is unit-tested from the project's main venv.
Actually running inference (run()) needs the training stack -- finetune/.venv.

Usage:
    finetune/.venv/bin/python -m finetune.evaluate --adapter finetune/output/lora_adapter
    finetune/.venv/bin/python -m finetune.evaluate --adapter finetune/output/checkpoint-1003 --limit 200
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

import jsonpatch
import yaml

from dataset.schema import RULE_IDS, validate_response

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$", re.MULTILINE)


def strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()


def parse_model_output(raw_text: str) -> tuple[dict | None, list[str]]:
    """Parses the model's raw generated text into a response dict. Returns
    (response, errors): response is None whenever the text isn't even valid
    JSON, or fails schema validation -- errors explains which."""
    cleaned = strip_fences(raw_text)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return None, [f"invalid JSON: {e}"]
    errors = validate_response(obj)
    return (obj if not errors else None), errors


def parse_manifest(text: str) -> list:
    """Every YAML document in a (possibly multi-document) manifest, in
    order -- the indices findings' and patch ops' "doc" fields refer to."""
    return list(yaml.safe_load_all(text))


def apply_multidoc_patch(docs: list, patch: list[dict]) -> tuple[list | None, bool]:
    """Applies patch ops to a (possibly multi-document) manifest: each op
    carries a "doc" index per dataset/schema.py, so ops are grouped and
    applied per document, exactly like dataset/build.py's round-trip checks.
    Returns (patched docs, True), or (None, False) if any op fails -- the
    patch is model-generated, so a bad path or doc index is an expected
    outcome for a wrong prediction, not a bug."""
    docs = copy.deepcopy(docs)
    if not patch:
        return docs, True
    by_doc: dict[int, list[dict]] = {}
    for op in patch:
        doc_idx = op.get("doc", 0)
        by_doc.setdefault(doc_idx, []).append({k: v for k, v in op.items() if k != "doc"})
    try:
        for doc_idx, ops in by_doc.items():
            if not isinstance(doc_idx, int) or doc_idx < 0 or doc_idx >= len(docs):
                return None, False
            docs[doc_idx] = jsonpatch.apply_patch(docs[doc_idx], ops)
        return docs, True
    except Exception:
        return None, False


def evaluate_example(input_docs: dict | list, expected_response: dict, model_output_text: str) -> dict:
    """Pure scoring for one example -- no model/network involved. Compares
    the model's parsed response against this example's ground truth.
    `input_docs` is the manifest's document list (parse_manifest), or a
    single document dict."""
    docs = input_docs if isinstance(input_docs, list) else [input_docs]
    model_response, schema_errors = parse_model_output(model_output_text)
    expected_rules = {f["rule_id"] for f in expected_response.get("findings", [])}

    result = {
        "schema_valid": model_response is not None,
        "schema_errors": schema_errors,
        "expected_rules": sorted(expected_rules),
        "predicted_rules": [],
        "patch_applies": False,
        "patch_correct": False,
        "new_resources_presence_correct": False,
    }
    if model_response is None:
        return result

    predicted_rules = {f["rule_id"] for f in model_response.get("findings", [])}
    result["predicted_rules"] = sorted(predicted_rules)

    expected_result, _ = apply_multidoc_patch(docs, expected_response.get("patch", []))
    model_result, applied_ok = apply_multidoc_patch(docs, model_response.get("patch", []))
    result["patch_applies"] = applied_ok
    if applied_ok:
        result["patch_correct"] = model_result == expected_result

    expected_has_resources = bool(expected_response.get("new_resources"))
    predicted_has_resources = bool(model_response.get("new_resources"))
    result["new_resources_presence_correct"] = expected_has_resources == predicted_has_resources

    return result


def _rule_confusion(results: list[dict], rule_id: str) -> dict:
    tp = fp = fn = tn = 0
    for r in results:
        expected = rule_id in r["expected_rules"]
        predicted = r["schema_valid"] and rule_id in r["predicted_rules"]
        if expected and predicted:
            tp += 1
        elif predicted and not expected:
            fp += 1
        elif expected and not predicted:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
    }


def aggregate_results(results: list[dict]) -> dict:
    total = len(results)
    if total == 0:
        return {"total": 0}

    schema_valid_results = [r for r in results if r["schema_valid"]]
    schema_valid = len(schema_valid_results)
    patch_applies = sum(1 for r in schema_valid_results if r["patch_applies"])
    patch_correct = sum(1 for r in schema_valid_results if r["patch_correct"])
    new_resources_correct = sum(1 for r in schema_valid_results if r["new_resources_presence_correct"])

    per_rule = {rule_id: _rule_confusion(results, rule_id) for rule_id in sorted(RULE_IDS)}

    clean_examples = [r for r in results if not r["expected_rules"]]
    clean_correct = sum(1 for r in clean_examples if r["schema_valid"] and not r["predicted_rules"])

    return {
        "total": total,
        "schema_valid_rate": round(schema_valid / total, 4),
        "patch_applies_rate": round(patch_applies / schema_valid, 4) if schema_valid else None,
        "patch_correct_rate": round(patch_correct / schema_valid, 4) if schema_valid else None,
        "new_resources_presence_correct_rate": (
            round(new_resources_correct / schema_valid, 4) if schema_valid else None
        ),
        "clean_examples": len(clean_examples),
        "clean_correctly_identified_rate": (
            round(clean_correct / len(clean_examples), 4) if clean_examples else None
        ),
        "per_rule": per_rule,
    }


def load_test_examples(path: Path, limit: int | None) -> list[dict]:
    examples = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
            if limit is not None and len(examples) >= limit:
                break
    return examples


def load_model(adapter_path: str, max_seq_length: int):
    # Imported lazily: only run() needs the training stack. Everything above
    # is pure Python + jsonpatch/yaml and testable from the main venv.
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_response_text(
    model,
    tokenizer,
    system: str,
    user: str,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float | None = None,
) -> str:
    import torch

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, pad_token_id=tokenizer.eos_token_id)
    if do_sample and temperature is not None:
        gen_kwargs["temperature"] = temperature
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def prompt_fits(tokenizer, system: str, user: str, max_seq_length: int, max_new_tokens: int) -> bool:
    """Whether the prompt leaves room for max_new_tokens within max_seq_length.
    The raw test split isn't length-filtered (only export_for_unsloth.py
    drops over-long examples, from what the model trains on), so it holds a
    long tail of huge manifests -- up to ~34k tokens on the v3 split, past
    even the base model's 32k context -- that the model was never trained
    on and generate() rejects outright."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) + max_new_tokens <= max_seq_length


def run(args: argparse.Namespace) -> dict:
    model, tokenizer = load_model(args.adapter, args.max_seq_length)
    examples = load_test_examples(Path(args.test_file), args.limit)

    results = []
    details = []
    skipped_too_long = 0
    for i, example in enumerate(examples):
        messages = example["messages"]
        system, user, assistant = (messages[0]["content"], messages[1]["content"], messages[2]["content"])
        if not prompt_fits(tokenizer, system, user, args.max_seq_length, args.max_new_tokens):
            skipped_too_long += 1
            continue
        expected_response = json.loads(assistant)
        input_docs = parse_manifest(user)

        output_text = generate_response_text(model, tokenizer, system, user, args.max_new_tokens)
        result = evaluate_example(input_docs, expected_response, output_text)
        results.append(result)
        details.append({"index": i, "raw_output": output_text, **result})

        if (i + 1) % max(1, args.log_every) == 0:
            running_rate = sum(r["schema_valid"] for r in results) / len(results)
            print(f"[{i + 1}/{len(examples)}] schema_valid_rate so far: {running_rate:.3f}", flush=True)

    summary = aggregate_results(results)
    summary["skipped_too_long"] = skipped_too_long
    if skipped_too_long:
        print(
            f"Skipped {skipped_too_long} example(s) whose prompt + --max-new-tokens exceeds "
            f"--max-seq-length {args.max_seq_length}.",
            flush=True,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "eval_details.jsonl").open("w", encoding="utf-8") as fh:
        for record in details:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWritten to {output_dir}/eval_summary.json and eval_details.jsonl")
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True, help="path to a saved LoRA adapter or checkpoint-N dir")
    parser.add_argument("--test-file", default="dataset/output/test.jsonl")
    parser.add_argument("--output-dir", default="finetune/output/eval")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1280)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
