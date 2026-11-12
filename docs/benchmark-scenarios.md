# Model benchmark: Qwen2.5-Coder manifest auditors

The 10 hand-written scenario files (`scenarios/*.yaml`); reference rules per file = `scenarios/test_cases.yaml` plus real defects the project's detectors confirm in the file. **Small sample: read the confidence intervals.** Evaluated on an NVIDIA GeForce RTX 4060 Ti (8.6 GB): greedy decoding, batch size 1, up to 1280 new tokens.

**Task framing.** Each sample is a Kubernetes manifest whose reference answer lists 0..n problems, each tagged with one of 12 rules (KSEC-001..012): multi-label classification. *Accuracy* is exact match (the predicted set of rules equals the reference set). *Macro* averages weigh every rule equally; *weighted* averages weigh rules by how often they occur. Both are computed over the rules present in the evaluated samples. An answer that isn't valid JSON per the output schema predicts no rule.

## Table 1 - Quality and speed

| Model | Accuracy | Precision macro / weighted | Recall macro / weighted | F1 macro / weighted | Mean latency (s) | Throughput (tok/s) |
|---|---|---|---|---|---|---|
| 7B | 90.0% | 100.0% / 100.0% | 98.3% / 96.6% | 99.1% / 98.2% | 16.94 | 30.5 |
| 3B | 90.0% | 100.0% / 100.0% | 95.8% / 96.6% | 97.6% / 98.0% | 9.51 | 51.6 |
| 1.5B | 60.0% | 100.0% / 100.0% | 75.4% / 75.9% | 83.3% / 84.6% | 4.97 | 70.1 |
| 7B base | 0.0% | 66.7% / 58.6% | 42.5% / 37.9% | 49.3% / 44.2% | 11.35 | 33.9 |

*Throughput* counts generated tokens only: every token `generate()` produced after the prompt, including the end-of-turn token that stops it, excluding prompt tokens; total generated tokens / total generation time (prefill included). *Latency* is the wall time of one `generate()` call (prefill + decoding), GPU synchronized. Per-class reports and confusion matrices are in `benchmark_results.xlsx`.

## Table 1b - Detection and correction (per individual error)

| Model | Errors | Detected | Undetected | Corrected | Uncorrected | Detection rate (recall) | Correction rate | Precision (findings) | F1 (findings) |
|---|---|---|---|---|---|---|---|---|---|
| 7B | 36 | 32 | 4 | 32 | 4 | 88.9% | 88.9% | 94.7% | 91.7% |
| 3B | 36 | 31 | 5 | 31 | 5 | 86.1% | 86.1% | 100.0% | 92.5% |
| 1.5B | 36 | 21 | 15 | 21 | 15 | 58.3% | 58.3% | 96.0% | 72.6% |
| 7B base | 36 | 3 | 33 | 1 | 35 | 8.3% | 2.8% | 56.2% | 14.5% |

Errors are the ground-truth instances in `scenarios/test_cases.yaml`. An error is *detected* when a reported finding matches it (same rule and document, and the same location), and *corrected* when applying only that finding's patch operations makes the project's own detectors (`dataset/detect.py`) stop flagging it. *Precision* is the share of reported findings that are right: they match an error, or a real defect the detectors confirm. Per-sample and per-rule counts are in `benchmark_results.xlsx`.

## Table 2 - Resource usage (during inference)

| Model | CPU % mean / peak | RAM GB mean / peak | VRAM GB mean / peak | TTFT (s) mean ± std |
|---|---|---|---|---|
| 7B | 100 / 110 | 2.16 / 2.16 | 6.58 / 6.58 | 0.563 ± 0.044 |
| 3B | 100 / 105 | 2.52 / 2.52 | 2.80 / 2.80 | 0.249 ± 0.028 |
| 1.5B | 100 / 105 | 2.52 / 2.52 | 1.75 / 1.75 | 0.126 ± 0.010 |
| 7B base | 100 / 105 | 2.48 / 2.48 | 6.26 / 6.27 | 0.500 ± 0.039 |

Sampled every 0.2 s while generating. CPU % is the inference process's own usage via psutil (100% = one logical core; the machine has 20). RAM is the process's resident memory. VRAM is via NVML (process-level). TTFT is measured with a streamer that timestamps the first generated token.

## Table 3 - Statistical analysis

| Model | Metric | Mean ± std | CI95 |
|---|---|---|---|
| 7B | accuracy (exact match) | 90.0% ± 31.6% | [59.6%, 98.2%] |
| 7B | F1 (macro), bootstrap | 98.9% ± 1.1% | [96.3%, 100.0%] |
| 7B | detection rate (per error), bootstrap | 88.9% ± 4.2% | [80.6%, 97.1%] |
| 7B | correction rate (per error), bootstrap | 88.9% ± 4.2% | [80.6%, 97.1%] |
| 7B | latency per sample (s) | 16.944 ± 6.602 | [12.221, 21.667] |
| 7B | throughput per sample (tokens/s) | 30.444 ± 0.935 | [29.775, 31.113] |
| 7B | TTFT (s) | 0.563 ± 0.044 | [0.531, 0.594] |
| 3B | accuracy (exact match) | 90.0% ± 31.6% | [59.6%, 98.2%] |
| 3B | F1 (macro), bootstrap | 97.1% ± 3.4% | [88.0%, 100.0%] |
| 3B | detection rate (per error), bootstrap | 86.3% ± 5.5% | [75.6%, 96.7%] |
| 3B | correction rate (per error), bootstrap | 86.3% ± 5.5% | [75.6%, 96.7%] |
| 3B | latency per sample (s) | 9.513 ± 4.283 | [6.449, 12.577] |
| 3B | throughput per sample (tokens/s) | 52.046 ± 2.183 | [50.485, 53.607] |
| 3B | TTFT (s) | 0.249 ± 0.028 | [0.229, 0.269] |
| 1.5B | accuracy (exact match) | 60.0% ± 51.6% | [31.3%, 83.2%] |
| 1.5B | F1 (macro), bootstrap | 79.1% ± 8.8% | [59.2%, 94.1%] |
| 1.5B | detection rate (per error), bootstrap | 58.2% ± 9.8% | [38.7%, 76.5%] |
| 1.5B | correction rate (per error), bootstrap | 58.2% ± 9.8% | [38.7%, 76.5%] |
| 1.5B | latency per sample (s) | 4.967 ± 2.936 | [2.866, 7.067] |
| 1.5B | throughput per sample (tokens/s) | 66.441 ± 8.251 | [60.538, 72.343] |
| 1.5B | TTFT (s) | 0.126 ± 0.010 | [0.119, 0.133] |
| 7B base | accuracy (exact match) | 0.0% ± 0.0% | [0.0%, 27.8%] |
| 7B base | F1 (macro), bootstrap | 42.0% ± 13.0% | [17.8%, 61.6%] |
| 7B base | detection rate (per error), bootstrap | 8.7% ± 4.8% | [0.0%, 19.4%] |
| 7B base | correction rate (per error), bootstrap | 2.8% ± 2.7% | [0.0%, 8.8%] |
| 7B base | latency per sample (s) | 11.347 ± 2.288 | [9.710, 12.983] |
| 7B base | throughput per sample (tokens/s) | 33.885 ± 0.907 | [33.237, 34.534] |
| 7B base | TTFT (s) | 0.500 ± 0.039 | [0.472, 0.528] |

Accuracy: per-sample correctness (0/1), Wilson CI95. Macro F1, detection rate and correction rate: 2,000 bootstrap resamples of the samples (mean, std, percentile CI95; whole samples are resampled because errors in the same file aren't independent). Latency, throughput and TTFT: per-sample values, t-distribution CI95 of the mean.

**McNemar's test** (paired, same samples; exact binomial when there are fewer than 25 discordant pairs). Four units: per sample (exact match right/wrong), per sample x rule decision (each of the 12 rules judged present/absent correctly or not -- more observations, more power), and per individual error (detected or not; corrected or not):

| Comparison | Unit | n | Both right | Only first right | Only second right | Both wrong | Method | p-value |
|---|---|---|---|---|---|---|---|---|
| 7B vs 3B | sample (exact match) | 10 | 8 | 1 | 1 | 0 | exact binomial | 1.0000 |
| 7B vs 3B | sample x rule decision | 120 | 118 | 1 | 1 | 0 | exact binomial | 1.0000 |
| 7B vs 3B | error (detected) | 36 | 28 | 4 | 3 | 1 | exact binomial | 1.0000 |
| 7B vs 3B | error (corrected) | 36 | 28 | 4 | 3 | 1 | exact binomial | 1.0000 |
| 7B vs 1.5B | sample (exact match) | 10 | 5 | 4 | 1 | 0 | exact binomial | 0.3750 |
| 7B vs 1.5B | sample x rule decision | 120 | 112 | 7 | 1 | 0 | exact binomial | 0.0703 |
| 7B vs 1.5B | error (detected) | 36 | 19 | 13 | 2 | 2 | exact binomial | 0.0074 (significant at 0.05) |
| 7B vs 1.5B | error (corrected) | 36 | 19 | 13 | 2 | 2 | exact binomial | 0.0074 (significant at 0.05) |
| 7B vs 7B base | sample (exact match) | 10 | 0 | 9 | 0 | 1 | exact binomial | 0.0039 (significant at 0.05) |
| 7B vs 7B base | sample x rule decision | 120 | 100 | 19 | 1 | 0 | exact binomial | 0.0000 (significant at 0.05) |
| 7B vs 7B base | error (detected) | 36 | 2 | 30 | 1 | 3 | chi-square (continuity corrected) | 0.0000 (significant at 0.05) |
| 7B vs 7B base | error (corrected) | 36 | 1 | 31 | 0 | 4 | chi-square (continuity corrected) | 0.0000 (significant at 0.05) |
| 3B vs 1.5B | sample (exact match) | 10 | 6 | 3 | 0 | 1 | exact binomial | 0.2500 |
| 3B vs 1.5B | sample x rule decision | 120 | 113 | 6 | 0 | 1 | exact binomial | 0.0312 (significant at 0.05) |
| 3B vs 1.5B | error (detected) | 36 | 20 | 11 | 1 | 4 | exact binomial | 0.0063 (significant at 0.05) |
| 3B vs 1.5B | error (corrected) | 36 | 20 | 11 | 1 | 4 | exact binomial | 0.0063 (significant at 0.05) |
| 3B vs 7B base | sample (exact match) | 10 | 0 | 9 | 0 | 1 | exact binomial | 0.0039 (significant at 0.05) |
| 3B vs 7B base | sample x rule decision | 120 | 101 | 18 | 0 | 1 | exact binomial | 0.0000 (significant at 0.05) |
| 3B vs 7B base | error (detected) | 36 | 3 | 28 | 0 | 5 | chi-square (continuity corrected) | 0.0000 (significant at 0.05) |
| 3B vs 7B base | error (corrected) | 36 | 1 | 30 | 0 | 5 | chi-square (continuity corrected) | 0.0000 (significant at 0.05) |
| 1.5B vs 7B base | sample (exact match) | 10 | 0 | 6 | 0 | 4 | exact binomial | 0.0312 (significant at 0.05) |
| 1.5B vs 7B base | sample x rule decision | 120 | 99 | 14 | 2 | 5 | exact binomial | 0.0042 (significant at 0.05) |
| 1.5B vs 7B base | error (detected) | 36 | 3 | 18 | 0 | 15 | exact binomial | 0.0000 (significant at 0.05) |
| 1.5B vs 7B base | error (corrected) | 36 | 1 | 20 | 0 | 15 | exact binomial | 0.0000 (significant at 0.05) |
