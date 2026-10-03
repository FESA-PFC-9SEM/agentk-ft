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

from dataset.schema import RULE_IDS, mask_evidence, validate_response
from finetune.metrics import confusion, paths_related, prf, rule_confusions

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
    enforce_evidence_mask(obj)
    errors = validate_response(obj)
    return (obj if not errors else None), errors


def enforce_evidence_mask(obj) -> None:
    """Re-masks every finding's evidence to its first 4 characters + "***",
    in place. Masking is a mechanical safety rule (a secret must never be
    shown in full), so it's enforced in code instead of trusted to the model:
    the v5 adapter got the rule right but miscounted characters on 12 of 50
    sampled scenario answers ("sk-12***", "mypass***"), and one such field
    invalidated an otherwise fully correct response."""
    if not isinstance(obj, dict) or not isinstance(obj.get("findings"), list):
        return
    for finding in obj["findings"]:
        if isinstance(finding, dict) and isinstance(finding.get("evidence"), str) and finding["evidence"]:
            value = finding["evidence"]
            if value.endswith("***"):
                value = value[:-3]
            finding["evidence"] = mask_evidence(value) if value else finding["evidence"]


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


def patch_ops_for_finding(finding: dict, patch_ops: list[dict]) -> list[dict]:
    """Which patch ops implement a given finding's fix, so a UI can offer
    "apply just this finding". A patch op relates to a finding if they're in
    the same doc and the op's path is the finding's path or an ancestor of it
    -- mutators sometimes replace a whole list element (e.g. `.../env/1`) to
    fix something a finding reports more specifically (`.../env/1/value`)."""
    doc = finding.get("doc", 0)
    fpath = finding.get("path", "")
    return [
        op
        for op in patch_ops
        if op.get("doc", 0) == doc and (fpath == op.get("path", "") or fpath.startswith(op.get("path", "") + "/"))
    ]


def match_findings(expected: list[dict], predicted: list[dict]) -> dict:
    """Finding-level matching: each predicted finding pairs with at most one
    expected finding of the same rule, in the same document, at the same
    JSON Pointer or a parent/child of it. Returns overall and per-rule
    TP/FP/FN counts."""
    unmatched = list(expected)
    by_rule: dict[str, list[int]] = {}
    tp = fp = 0
    for finding in predicted:
        rule = finding.get("rule_id")
        hit = next(
            (
                e
                for e in unmatched
                if e.get("rule_id") == rule
                and e.get("doc", 0) == finding.get("doc", 0)
                and paths_related(str(e.get("path", "")), str(finding.get("path", "")))
            ),
            None,
        )
        counts = by_rule.setdefault(rule, [0, 0, 0])
        if hit is not None:
            unmatched.remove(hit)
            tp += 1
            counts[0] += 1
        else:
            fp += 1
            counts[1] += 1
    for e in unmatched:
        by_rule.setdefault(e.get("rule_id"), [0, 0, 0])[2] += 1
    return {"finding_tp": tp, "finding_fp": fp, "finding_fn": len(unmatched), "finding_counts_by_rule": by_rule}


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
        # An invalid response reports nothing: every expected finding is missed.
        **match_findings(expected_response.get("findings", []), []),
        "patch_applies": False,
        "patch_correct": False,
        "new_resources_presence_correct": False,
    }
    if model_response is None:
        return result

    predicted_rules = {f["rule_id"] for f in model_response.get("findings", [])}
    result["predicted_rules"] = sorted(predicted_rules)
    result.update(match_findings(expected_response.get("findings", []), model_response.get("findings", [])))

    expected_result, _ = apply_multidoc_patch(docs, expected_response.get("patch", []))
    model_result, applied_ok = apply_multidoc_patch(docs, model_response.get("patch", []))
    result["patch_applies"] = applied_ok
    if applied_ok:
        result["patch_correct"] = model_result == expected_result

    expected_has_resources = bool(expected_response.get("new_resources"))
    predicted_has_resources = bool(model_response.get("new_resources"))
    result["new_resources_presence_correct"] = expected_has_resources == predicted_has_resources

    return result


def aggregate_results(results: list[dict]) -> dict:
    total = len(results)
    if total == 0:
        return {"total": 0}

    schema_valid_results = [r for r in results if r["schema_valid"]]
    schema_valid = len(schema_valid_results)
    patch_applies = sum(1 for r in schema_valid_results if r["patch_applies"])
    patch_correct = sum(1 for r in schema_valid_results if r["patch_correct"])
    new_resources_correct = sum(1 for r in schema_valid_results if r["new_resources_presence_correct"])

    # Example x rule: "does this example have a KSEC-00X problem?" -- an
    # invalid response predicts nothing.
    rule_level = rule_confusions(
        ((set(r["expected_rules"]), set(r["predicted_rules"]) if r["schema_valid"] else set()) for r in results),
        sorted(RULE_IDS),
    )

    # Finding level: precision/recall/F1 of the individual findings.
    finding_by_rule: dict[str, list[int]] = {}
    for r in results:
        for rule, counts in r.get("finding_counts_by_rule", {}).items():
            acc = finding_by_rule.setdefault(rule, [0, 0, 0])
            for i in range(3):
                acc[i] += counts[i]
    findings = {
        "overall": prf(
            sum(r.get("finding_tp", 0) for r in results),
            sum(r.get("finding_fp", 0) for r in results),
            sum(r.get("finding_fn", 0) for r in results),
        ),
        "per_rule": {rule: prf(*finding_by_rule[rule]) for rule in sorted(finding_by_rule)},
        "exact_finding_set_rate": round(
            sum(1 for r in results if r.get("finding_fp", 0) == 0 and r.get("finding_fn", 0) == 0) / total, 4
        ),
    }

    # Clean vs dirty: "does this example have any problem at all?"
    tp = fp = fn = tn = 0
    for r in results:
        actual, predicted = bool(r["expected_rules"]), bool(r["schema_valid"] and r["predicted_rules"])
        tp += actual and predicted
        fp += predicted and not actual
        fn += actual and not predicted
        tn += not actual and not predicted

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
        "clean_vs_dirty": confusion(tp, fp, fn, tn),
        "findings": findings,
        "per_rule": rule_level["per_rule"],
        "per_rule_micro": rule_level["micro"],
        "per_rule_macro": rule_level["macro"],
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
