"""
Benchmarks fine-tuned adapters on the held-out test set: quality, speed,
resource usage and statistical comparison.

Two steps, because Unsloth can't release a model's GPU memory within one
process -- each model is measured in its own process:

  # 1. once per model (GPU): generate + measure, saves per-sample results
  finetune/.venv/bin/python -m finetune.benchmark_models run \\
      --adapter runs/<run>/lora_adapter --label 7B --out runs/benchmark

  # 2. once (CPU): tables, per-class reports, confusion matrices, statistics
  finetune/.venv/bin/python -m finetune.benchmark_models report --out runs/benchmark

Data (--source):
  - scenarios: the 10 hand-written scenarios/*.yaml files. Reference rules
    per file = the rules in scenarios/test_cases.yaml plus any real defect
    the project's detectors confirm in that file (the same reference
    finetune/run_scenarios.py scores against). Small: wide intervals.
  - test: the held-out v6 test set (seeded random subset, --samples).

Task framing: each test sample is a manifest whose answer lists 0..n
problems, each tagged with one of 12 rules -> multi-label classification
over the rules. Per sample, "expected" is the set of rules in the reference
answer, "predicted" the set in the model's answer (an invalid answer
predicts nothing).

Measurement: one sample at a time (batch size 1), greedy decoding, up to
--max-new-tokens. A streamer timestamps the first generated token (TTFT) and
counts every generated token; a background thread samples CPU, RAM and VRAM
every --sample-interval seconds while the model generates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import threading
import time
from pathlib import Path

RULES = [f"KSEC-{i:03d}" for i in range(1, 13)]
DEFAULT_TEST_FILE = "dataset/output-multi-defect-v6/test.jsonl"


# ---------------------------------------------------------------------------
# Step 1: run one model
# ---------------------------------------------------------------------------


def scenario_samples() -> list[dict]:
    """The scenario files as samples: the user turn is the file, the
    expected rules come from the ground truth plus the detectors."""
    import yaml

    from dataset.detect import detect_file
    from dataset.schema import SYSTEM_PROMPT
    from finetune.run_scenarios import load_scenario_files, load_test_cases

    cases = load_test_cases(Path("scenarios/test_cases.yaml"))
    rows = []
    for index, path in enumerate(load_scenario_files(Path("scenarios"), cases)):
        text = path.read_text(encoding="utf-8")
        expected = {i["rule_id"] for i in cases[path.name]} | {f.rule_id for f in detect_file(list(yaml.safe_load_all(text)))}
        rows.append({"_index": index, "_name": path.name, "_expected": sorted(expected),
                     "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]})
    return rows


def select_samples(test_file: Path, tokenizer, n: int | None, seed: int, max_seq_length: int, max_new_tokens: int) -> list[dict]:
    """The same samples for every model: those whose prompt + answer budget
    fit the context window, then a seeded random subset of size n."""
    from finetune.evaluate import prompt_fits

    rows = [json.loads(line) for line in test_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    for index, row in enumerate(rows):
        row["_index"] = index
    fitting = [r for r in rows if prompt_fits(tokenizer, r["messages"][0]["content"], r["messages"][1]["content"], max_seq_length, max_new_tokens)]
    if n is not None and n < len(fitting):
        fitting = sorted(random.Random(seed).sample(fitting, n), key=lambda r: r["_index"])
    return fitting


class TimingStreamer:
    """transformers streamer that timestamps the first generated token and
    counts generated tokens. generate() first passes the prompt (ignored),
    then each new token -- including the end-of-turn token that stops it."""

    def __init__(self):
        self.first_token_time = None
        self.generated_tokens = 0
        self._prompt_seen = False

    def put(self, value) -> None:
        if not self._prompt_seen:
            self._prompt_seen = True
            return
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
        self.generated_tokens += int(value.numel())

    def end(self) -> None:
        pass


class ResourceSampler(threading.Thread):
    """Samples this process's CPU % and RSS, and GPU memory, every interval.
    VRAM is this process's own usage when NVML lists per-process memory,
    otherwise the device's used memory (noted in the results)."""

    def __init__(self, interval: float):
        super().__init__(daemon=True)
        import psutil
        import pynvml

        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.process.cpu_percent(None)  # prime the counter
        pynvml.nvmlInit()
        self.pynvml = pynvml
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.samples: list[dict] = []
        self.vram_source = "device"
        self._halt = threading.Event()

    def vram_gb(self) -> float:
        try:
            for proc in self.pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle):
                if proc.pid == os.getpid() and proc.usedGpuMemory:
                    self.vram_source = "process"
                    return proc.usedGpuMemory / 1e9
        except self.pynvml.NVMLError:
            pass
        return self.pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / 1e9

    def run(self) -> None:
        while not self._halt.wait(self.interval):
            self.samples.append(
                {"cpu_percent": self.process.cpu_percent(None), "ram_gb": self.process.memory_info().rss / 1e9, "vram_gb": self.vram_gb()}
            )

    def stop(self) -> None:
        self._halt.set()
        self.join()


def run(args: argparse.Namespace) -> None:
    import torch

    from finetune.evaluate import load_model, parse_model_output

    out_dir = Path(args.out) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    load_start = time.perf_counter()
    model, tokenizer = load_model(args.adapter, args.max_seq_length)
    load_seconds = time.perf_counter() - load_start
    if args.source == "scenarios":
        samples = scenario_samples()
    else:
        samples = select_samples(Path(args.test_file), tokenizer, args.samples, args.seed, args.max_seq_length, args.max_new_tokens)
    print(f"[{args.label}] {len(samples)} samples, model loaded in {load_seconds:.0f}s", flush=True)

    # Warm-up (CUDA kernels, caches) -- not measured.
    warm = tokenizer.apply_chat_template(samples[0]["messages"][:2], tokenize=False, add_generation_prompt=True)
    with torch.no_grad():
        model.generate(**tokenizer(warm, return_tensors="pt").to(model.device), max_new_tokens=16, do_sample=False)

    sampler = ResourceSampler(args.sample_interval)
    sampler.start()
    results = []
    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as fh:
        for i, row in enumerate(samples):
            prompt = tokenizer.apply_chat_template(row["messages"][:2], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            streamer = TimingStreamer()
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False, streamer=streamer, pad_token_id=tokenizer.eos_token_id
                )
            torch.cuda.synchronize()
            end = time.perf_counter()
            text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            parsed, errors = parse_model_output(text)
            expected = row.get("_expected") or sorted({f["rule_id"] for f in json.loads(row["messages"][2]["content"])["findings"]})
            predicted = sorted({f["rule_id"] for f in parsed["findings"]}) if parsed else []
            record = {
                "index": row["_index"],
                "name": row.get("_name", str(row["_index"])),
                "expected": expected,
                "predicted": predicted,
                "valid": parsed is not None,
                "prompt_tokens": int(inputs["input_ids"].shape[1]),
                "generated_tokens": streamer.generated_tokens,
                "latency_s": end - start,
                "ttft_s": (streamer.first_token_time - start) if streamer.first_token_time else None,
            }
            results.append(record)
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            if (i + 1) % 25 == 0:
                done = results
                print(
                    f"[{args.label}] {i + 1}/{len(samples)} | mean latency {statistics.mean(r['latency_s'] for r in done):.2f}s | "
                    f"exact match {sum(r['expected'] == r['predicted'] for r in done) / len(done):.3f}",
                    flush=True,
                )
    sampler.stop()

    import psutil

    meta = {
        "label": args.label,
        "adapter": args.adapter,
        "source": args.source,
        "test_file": "scenarios/*.yaml" if args.source == "scenarios" else args.test_file,
        "samples": len(results),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "model_load_seconds": load_seconds,
        "gpu": sampler.pynvml.nvmlDeviceGetName(sampler.handle),
        "gpu_total_gb": sampler.pynvml.nvmlDeviceGetMemoryInfo(sampler.handle).total / 1e9,
        "cpu_cores": psutil.cpu_count(logical=True),
        "vram_source": sampler.vram_source,
        "torch_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "resource_samples": sampler.samples,
    }
    (out_dir / "run.json").write_text(json.dumps(meta, indent=1))
    print(f"[{args.label}] done -> {out_dir}", flush=True)


# ---------------------------------------------------------------------------
# Step 2: report
# ---------------------------------------------------------------------------


def load_runs(out: Path) -> dict:
    runs = {}
    for meta_path in sorted(out.glob("*/run.json")):
        meta = json.loads(meta_path.read_text())
        samples = [json.loads(line) for line in (meta_path.parent / "samples.jsonl").read_text().splitlines() if line.strip()]
        runs[meta["label"]] = {"meta": meta, "samples": samples}
    if not runs:
        raise SystemExit(f"no runs in {out} -- run step 1 first")
    return runs


def indicator(samples: list[dict], key: str):
    import numpy as np

    return np.array([[rule in s[key] for rule in RULES] for s in samples], dtype=int)


def quality(samples: list[dict]) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, multilabel_confusion_matrix, precision_recall_fscore_support

    y_true, y_pred = indicator(samples, "expected"), indicator(samples, "predicted")
    # Macro/weighted over the rules that occur in the evaluated samples
    # (a rule absent from the sample has no defined recall).
    labels = [i for i, _ in enumerate(RULES) if y_true[:, i].sum() > 0]
    out = {"accuracy": accuracy_score(y_true, y_pred), "labels": [RULES[i] for i in labels]}
    for average in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=average, zero_division=0)
        out[average] = {"precision": p, "recall": r, "f1": f}
    out["report"] = classification_report(y_true, y_pred, labels=labels, target_names=[RULES[i] for i in labels], output_dict=True, zero_division=0)
    out["confusion"] = {RULES[i]: m for i, m in zip(labels, multilabel_confusion_matrix(y_true, y_pred, labels=labels))}
    return out


def macro_f1(samples: list[dict]) -> float:
    return quality(samples)["macro"]["f1"]


def bootstrap(samples: list[dict], statistic, iterations: int, seed: int = 0) -> tuple[float, float, float, float]:
    """(mean, std, CI95 low, CI95 high) of `statistic` over bootstrap
    resamples of whole samples."""
    rng = random.Random(seed)
    values = sorted(statistic([samples[rng.randrange(len(samples))] for _ in samples]) for _ in range(iterations))
    return statistics.mean(values), statistics.stdev(values), values[int(0.025 * iterations)], values[int(0.975 * iterations) - 1]


def mean_std_ci(values: list[float]) -> tuple[float, float, float, float]:
    """Mean, std and a t-based CI95 of the mean."""
    from scipy import stats

    m, s = statistics.mean(values), statistics.stdev(values)
    half = stats.t.ppf(0.975, len(values) - 1) * s / math.sqrt(len(values))
    return m, s, m - half, m + half


def wilson(successes: int, total: int) -> tuple[float, float]:
    from statsmodels.stats.proportion import proportion_confint

    return proportion_confint(successes, total, alpha=0.05, method="wilson")


def mcnemar_test(a: list[dict], b: list[dict]) -> dict:
    """McNemar's test on per-sample exact-match correctness of two models
    over the same samples. Exact binomial version when there are few
    discordant pairs, chi-square with continuity correction otherwise."""
    from statsmodels.stats.contingency_tables import mcnemar

    by_index = {s["index"]: s for s in b}
    pairs = [(s["expected"] == s["predicted"], by_index[s["index"]]["expected"] == by_index[s["index"]]["predicted"]) for s in a if s["index"] in by_index]
    both = sum(x and y for x, y in pairs)
    only_a = sum(x and not y for x, y in pairs)
    only_b = sum(y and not x for x, y in pairs)
    neither = sum(not x and not y for x, y in pairs)
    exact = only_a + only_b < 25
    result = mcnemar([[both, only_a], [only_b, neither]], exact=exact, correction=True)
    return {"n": len(pairs), "both_correct": both, "only_first": only_a, "only_second": only_b, "both_wrong": neither,
            "method": "exact binomial" if exact else "chi-square (continuity corrected)", "statistic": float(result.statistic), "p_value": float(result.pvalue)}


def mcnemar_decisions(a: list[dict], b: list[dict]) -> dict:
    """McNemar's test on every (sample, rule) yes/no decision -- many more
    paired observations than per-sample exact match (10 x 12 = 120 on the
    scenarios), so it can detect smaller differences."""
    from statsmodels.stats.contingency_tables import mcnemar

    by_index = {s["index"]: s for s in b}
    pairs = []
    for s in a:
        other = by_index.get(s["index"])
        if other is None:
            continue
        for rule in RULES:
            truth = rule in s["expected"]
            pairs.append(((rule in s["predicted"]) == truth, (rule in other["predicted"]) == truth))
    both = sum(x and y for x, y in pairs)
    only_a = sum(x and not y for x, y in pairs)
    only_b = sum(y and not x for x, y in pairs)
    neither = sum(not x and not y for x, y in pairs)
    exact = only_a + only_b < 25
    result = mcnemar([[both, only_a], [only_b, neither]], exact=exact, correction=True)
    return {"n": len(pairs), "both_correct": both, "only_first": only_a, "only_second": only_b, "both_wrong": neither,
            "method": "exact binomial" if exact else "chi-square (continuity corrected)", "statistic": float(result.statistic), "p_value": float(result.pvalue)}


def resources(meta: dict) -> dict:
    samples = meta["resource_samples"]
    cores = meta["cpu_cores"]
    out = {}
    for key in ("cpu_percent", "ram_gb", "vram_gb"):
        values = [s[key] for s in samples]
        out[key] = {"mean": statistics.mean(values), "peak": max(values)}
    out["cpu_percent_of_machine"] = {k: v / cores for k, v in out["cpu_percent"].items()}
    return out


def report(args: argparse.Namespace) -> None:
    import pandas as pd

    out = Path(args.out)
    runs = load_runs(out)
    order = sorted(runs, key=lambda label: float(label.rstrip("Bb")) if label.rstrip("Bb").replace(".", "", 1).isdigit() else 0, reverse=True)
    common = set.intersection(*(set(s["index"] for s in runs[l]["samples"]) for l in order))

    table1, table2, table3, mcnemar_rows, per_class, confusion_rows = [], [], [], [], [], []
    for label in order:
        samples = [s for s in runs[label]["samples"] if s["index"] in common]
        meta = runs[label]["meta"]
        q = quality(samples)
        total_tokens = sum(s["generated_tokens"] for s in samples)
        total_time = sum(s["latency_s"] for s in samples)
        table1.append({
            "model": label, "samples": len(samples), "valid answers": sum(s["valid"] for s in samples) / len(samples),
            "accuracy (exact match)": q["accuracy"],
            "precision (macro)": q["macro"]["precision"], "precision (weighted)": q["weighted"]["precision"],
            "recall (macro)": q["macro"]["recall"], "recall (weighted)": q["weighted"]["recall"],
            "F1 (macro)": q["macro"]["f1"], "F1 (weighted)": q["weighted"]["f1"],
            "mean latency per sample (s)": statistics.mean(s["latency_s"] for s in samples),
            "throughput (generated tokens/s)": total_tokens / total_time,
        })
        r = resources(meta)
        ttft = [s["ttft_s"] for s in samples if s["ttft_s"] is not None]
        table2.append({
            "model": label,
            "CPU % mean (process; 100% = 1 core)": r["cpu_percent"]["mean"], "CPU % peak": r["cpu_percent"]["peak"],
            f"CPU % of machine mean ({meta['cpu_cores']} threads)": r["cpu_percent_of_machine"]["mean"],
            "RAM GB mean (process RSS)": r["ram_gb"]["mean"], "RAM GB peak": r["ram_gb"]["peak"],
            f"VRAM GB mean ({meta['vram_source']})": r["vram_gb"]["mean"], "VRAM GB peak": r["vram_gb"]["peak"],
            "TTFT s mean": statistics.mean(ttft), "TTFT s std": statistics.stdev(ttft),
        })
        correct = [s["expected"] == s["predicted"] for s in samples]
        acc_low, acc_high = wilson(sum(correct), len(correct))
        f1_mean, f1_std, f1_low, f1_high = bootstrap(samples, macro_f1, args.bootstrap)
        lat = mean_std_ci([s["latency_s"] for s in samples])
        tps = mean_std_ci([s["generated_tokens"] / s["latency_s"] for s in samples])
        tt = mean_std_ci(ttft)
        for metric, (m, sd, lo, hi) in (
            ("accuracy (exact match)", (statistics.mean(correct), statistics.stdev(correct), acc_low, acc_high)),
            ("F1 (macro), bootstrap", (f1_mean, f1_std, f1_low, f1_high)),
            ("latency per sample (s)", lat),
            ("throughput per sample (tokens/s)", tps),
            ("TTFT (s)", tt),
        ):
            table3.append({"model": label, "metric": metric, "mean": m, "std": sd, "CI95 low": lo, "CI95 high": hi})
        for rule, values in q["report"].items():
            if isinstance(values, dict):
                per_class.append({"model": label, "class": rule, **values})
        for rule, matrix in q["confusion"].items():
            (tn, fp), (fn, tp) = matrix.tolist()
            confusion_rows.append({"model": label, "rule": rule, "TP": tp, "FP": fp, "FN": fn, "TN": tn})
    for i, first in enumerate(order):
        for second in order[i + 1:]:
            mcnemar_rows.append({"comparison": f"{first} vs {second}", "unit": "sample (exact match)",
                                 **mcnemar_test(runs[first]["samples"], runs[second]["samples"])})
            mcnemar_rows.append({"comparison": f"{first} vs {second}", "unit": "sample x rule decision",
                                 **mcnemar_decisions(runs[first]["samples"], runs[second]["samples"])})

    xlsx = out / "benchmark_results.xlsx"
    with pd.ExcelWriter(xlsx) as writer:
        pd.DataFrame(table1).to_excel(writer, sheet_name="Table 1 - quality & speed", index=False)
        pd.DataFrame(table2).to_excel(writer, sheet_name="Table 2 - resources", index=False)
        pd.DataFrame(table3).to_excel(writer, sheet_name="Table 3 - statistics", index=False)
        pd.DataFrame(mcnemar_rows).to_excel(writer, sheet_name="Table 3 - McNemar", index=False)
        pd.DataFrame(per_class).to_excel(writer, sheet_name="Per-class report", index=False)
        pd.DataFrame(confusion_rows).to_excel(writer, sheet_name="Confusion matrices", index=False)
        for label in order:
            pd.DataFrame(runs[label]["samples"]).to_excel(writer, sheet_name=f"Samples {label}"[:31], index=False)
        for ws in writer.book.worksheets:
            for column in ws.columns:
                ws.column_dimensions[column[0].column_letter].width = max(12, min(48, max(len(str(c.value or "")) for c in column) + 2))

    write_markdown(out, order, runs, table1, table2, table3, mcnemar_rows, len(common))
    print(f"Written to {xlsx} and {out / 'report.md'}")


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def write_markdown(out, order, runs, table1, table2, table3, mcnemar_rows, n) -> None:
    meta0 = runs[order[0]]["meta"]
    source = (
        f"The {n} hand-written scenario files (`scenarios/*.yaml`); reference rules per file = `scenarios/test_cases.yaml` plus real "
        "defects the project's detectors confirm in the file. **Small sample: read the confidence intervals.**"
        if meta0.get("source") == "scenarios"
        else f"Held-out test set `{meta0['test_file']}` -- {n} samples (seed {meta0['seed']}; the same samples for every model)."
    )
    lines = [
        "# Model benchmark: fine-tuned Qwen2.5-Coder manifest auditors",
        "",
        source + f" Evaluated on an {meta0['gpu']} ({meta0['gpu_total_gb']:.1f} GB): greedy decoding, batch size 1, "
        f"up to {meta0['max_new_tokens']} new tokens.",
        "",
        "**Task framing.** Each sample is a Kubernetes manifest whose reference answer lists 0..n problems, each tagged with one of 12 "
        "rules (KSEC-001..012): multi-label classification. *Accuracy* is exact match (the predicted set of rules equals the reference "
        "set). *Macro* averages weigh every rule equally; *weighted* averages weigh rules by how often they occur. Both are computed over "
        "the rules present in the evaluated samples. An answer that isn't valid JSON per the output schema predicts no rule.",
        "",
        "## Table 1 - Quality and speed",
        "",
        "| Model | Accuracy | Precision macro / weighted | Recall macro / weighted | F1 macro / weighted | Mean latency (s) | Throughput (tok/s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in table1:
        lines.append(
            f"| {r['model']} | {_pct(r['accuracy (exact match)'])} | {_pct(r['precision (macro)'])} / {_pct(r['precision (weighted)'])} | "
            f"{_pct(r['recall (macro)'])} / {_pct(r['recall (weighted)'])} | {_pct(r['F1 (macro)'])} / {_pct(r['F1 (weighted)'])} | "
            f"{r['mean latency per sample (s)']:.2f} | {r['throughput (generated tokens/s)']:.1f} |"
        )
    lines += [
        "",
        "*Throughput* counts generated tokens only: every token `generate()` produced after the prompt, including the end-of-turn token "
        "that stops it, excluding prompt tokens; total generated tokens / total generation time (prefill included). *Latency* is the wall "
        "time of one `generate()` call (prefill + decoding), GPU synchronized. Per-class reports and confusion matrices are in "
        "`benchmark_results.xlsx`.",
        "",
        "## Table 2 - Resource usage (during inference)",
        "",
        "| Model | CPU % mean / peak | RAM GB mean / peak | VRAM GB mean / peak | TTFT (s) mean ± std |",
        "|---|---|---|---|---|",
    ]
    for r in table2:
        keys = list(r)
        lines.append(
            f"| {r['model']} | {r[keys[1]]:.0f} / {r[keys[2]]:.0f} | {r[keys[4]]:.2f} / {r[keys[5]]:.2f} | {r[keys[6]]:.2f} / {r[keys[7]]:.2f} | "
            f"{r['TTFT s mean']:.3f} ± {r['TTFT s std']:.3f} |"
        )
    lines += [
        "",
        f"Sampled every {0.2:.1f} s while generating. CPU % is the inference process's own usage via psutil (100% = one logical core; the "
        f"machine has {meta0['cpu_cores']}). RAM is the process's resident memory. VRAM is via NVML ({meta0['vram_source']}-level). TTFT is "
        "measured with a streamer that timestamps the first generated token.",
        "",
        "## Table 3 - Statistical analysis",
        "",
        "| Model | Metric | Mean ± std | CI95 |",
        "|---|---|---|---|",
    ]
    for r in table3:
        fmt = (lambda x: _pct(x)) if "accuracy" in r["metric"] or "F1" in r["metric"] else (lambda x: f"{x:.3f}")
        lines.append(f"| {r['model']} | {r['metric']} | {fmt(r['mean'])} ± {fmt(r['std'])} | [{fmt(r['CI95 low'])}, {fmt(r['CI95 high'])}] |")
    lines += [
        "",
        "Accuracy: per-sample correctness (0/1), Wilson CI95. Macro F1: 2,000 bootstrap resamples of the samples (mean, std, percentile "
        "CI95). Latency, throughput and TTFT: per-sample values, t-distribution CI95 of the mean.",
        "",
        "**McNemar's test** (paired, same samples; exact binomial when there are fewer than 25 discordant pairs). Two units: per sample "
        "(exact match right/wrong) and per sample x rule decision (each of the 12 rules judged present/absent correctly or not -- more "
        "observations, more power):",
        "",
        "| Comparison | Unit | n | Both right | Only first right | Only second right | Both wrong | Method | p-value |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in mcnemar_rows:
        lines.append(
            f"| {r['comparison']} | {r['unit']} | {r['n']} | {r['both_correct']} | {r['only_first']} | {r['only_second']} | {r['both_wrong']} | {r['method']} | "
            f"{r['p_value']:.4f}{' (significant at 0.05)' if r['p_value'] < 0.05 else ''} |"
        )
    (out / "report.md").write_text("\n".join(lines) + "\n")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    r = sub.add_parser("run", help="benchmark one model (GPU)")
    r.add_argument("--adapter", required=True)
    r.add_argument("--label", required=True, help="e.g. 7B")
    r.add_argument("--out", default="runs/benchmark")
    r.add_argument("--source", choices=["scenarios", "test"], default="test")
    r.add_argument("--test-file", default=DEFAULT_TEST_FILE)
    r.add_argument("--samples", type=int, default=500, help="seeded random subset of the test set; 0 = all")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--max-seq-length", type=int, default=4096)
    r.add_argument("--max-new-tokens", type=int, default=1280)
    r.add_argument("--sample-interval", type=float, default=0.2)
    p = sub.add_parser("report", help="tables + statistics from the saved runs (CPU)")
    p.add_argument("--out", default="runs/benchmark")
    p.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.step == "run" and args.samples == 0:
        args.samples = None
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    run(args) if args.step == "run" else report(args)


if __name__ == "__main__":
    main()
