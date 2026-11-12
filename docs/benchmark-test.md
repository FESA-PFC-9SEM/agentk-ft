# Model benchmark: Qwen2.5-Coder manifest auditors

Held-out test set `dataset/output-multi-defect-v6/test.jsonl` -- 500 samples (seed 42; the same samples for every model). Evaluated on an NVIDIA GeForce RTX 4060 Ti (8.6 GB): greedy decoding, batch size 1, up to 1280 new tokens.

**Task framing.** Each sample is a Kubernetes manifest whose reference answer lists 0..n problems, each tagged with one of 12 rules (KSEC-001..012): multi-label classification. *Accuracy* is exact match (the predicted set of rules equals the reference set). *Macro* averages weigh every rule equally; *weighted* averages weigh rules by how often they occur. Both are computed over the rules present in the evaluated samples. An answer that isn't valid JSON per the output schema predicts no rule.

## Table 1 - Quality and speed

| Model | Accuracy | Precision macro / weighted | Recall macro / weighted | F1 macro / weighted | Mean latency (s) | Throughput (tok/s) |
|---|---|---|---|---|---|---|
| 7B | 96.8% | 98.3% / 99.1% | 93.4% / 98.9% | 94.8% / 98.9% | 9.49 | 33.1 |
| 3B | 92.0% | 89.1% / 97.8% | 81.8% / 96.1% | 84.9% / 96.8% | 5.96 | 51.9 |
| 1.5B | 88.2% | 86.2% / 97.3% | 74.9% / 93.3% | 78.9% / 94.9% | 4.56 | 66.2 |

*Throughput* counts generated tokens only: every token `generate()` produced after the prompt, including the end-of-turn token that stops it, excluding prompt tokens; total generated tokens / total generation time (prefill included). *Latency* is the wall time of one `generate()` call (prefill + decoding), GPU synchronized. Per-class reports and confusion matrices are in `benchmark_results.xlsx`.

## Table 1b - Detection and correction (per individual error)

| Model | Errors | Detected | Undetected | Corrected | Uncorrected | Detection rate (recall) | Correction rate | Precision (findings) | F1 (findings) |
|---|---|---|---|---|---|---|---|---|---|
| 7B | 1172 | 1149 | 23 | 1144 | 28 | 98.0% | 97.6% | 98.5% | 98.3% |
| 3B | 1172 | 1119 | 53 | 1112 | 60 | 95.5% | 94.9% | 98.1% | 96.8% |
| 1.5B | 1172 | 1070 | 102 | 1066 | 106 | 91.3% | 91.0% | 96.0% | 93.6% |

Errors are the findings in each sample's reference answer. An error is *detected* when a reported finding matches it (same rule and document, and the same location), and *corrected* when applying only that finding's patch operations makes the project's own detectors (`dataset/detect.py`) stop flagging it. *Precision* is the share of reported findings that are right: they match an error, or a real defect the detectors confirm. Per-sample and per-rule counts are in `benchmark_results.xlsx`.

## Table 2 - Resource usage (during inference)

| Model | CPU % mean / peak | RAM GB mean / peak | VRAM GB mean / peak | TTFT (s) mean ± std |
|---|---|---|---|---|
| 7B | 100 / 110 | 2.53 / 2.53 | 6.81 / 6.84 | 0.522 ± 0.099 |
| 3B | 100 / 110 | 2.53 / 2.53 | 2.94 / 2.95 | 0.250 ± 0.049 |
| 1.5B | 100 / 112 | 2.53 / 2.53 | 1.86 / 1.87 | 0.129 ± 0.025 |

Sampled every 0.2 s while generating. CPU % is the inference process's own usage via psutil (100% = one logical core; the machine has 20). RAM is the process's resident memory. VRAM is via NVML (process-level). TTFT is measured with a streamer that timestamps the first generated token.

## Table 3 - Statistical analysis

| Model | Metric | Mean ± std | CI95 |
|---|---|---|---|
| 7B | accuracy (exact match) | 96.8% ± 17.6% | [94.9%, 98.0%] |
| 7B | F1 (macro), bootstrap | 94.6% ± 1.8% | [90.8%, 97.7%] |
| 7B | detection rate (per error), bootstrap | 98.0% ± 0.5% | [97.0%, 98.9%] |
| 7B | correction rate (per error), bootstrap | 97.6% ± 0.5% | [96.5%, 98.5%] |
| 7B | latency per sample (s) | 9.490 ± 9.245 | [8.678, 10.302] |
| 7B | throughput per sample (tokens/s) | 28.514 ± 6.784 | [27.918, 29.110] |
| 7B | TTFT (s) | 0.522 ± 0.099 | [0.513, 0.531] |
| 3B | accuracy (exact match) | 92.0% ± 27.2% | [89.3%, 94.1%] |
| 3B | F1 (macro), bootstrap | 84.8% ± 1.3% | [82.1%, 87.0%] |
| 3B | detection rate (per error), bootstrap | 95.5% ± 0.6% | [94.2%, 96.7%] |
| 3B | correction rate (per error), bootstrap | 94.9% ± 0.7% | [93.5%, 96.2%] |
| 3B | latency per sample (s) | 5.960 ± 5.833 | [5.448, 6.473] |
| 3B | throughput per sample (tokens/s) | 45.750 ± 9.044 | [44.955, 46.544] |
| 3B | TTFT (s) | 0.250 ± 0.049 | [0.246, 0.254] |
| 1.5B | accuracy (exact match) | 88.2% ± 32.3% | [85.1%, 90.7%] |
| 1.5B | F1 (macro), bootstrap | 78.7% ± 1.7% | [75.1%, 81.9%] |
| 1.5B | detection rate (per error), bootstrap | 91.3% ± 1.0% | [89.3%, 93.2%] |
| 1.5B | correction rate (per error), bootstrap | 91.0% ± 1.0% | [88.9%, 92.8%] |
| 1.5B | latency per sample (s) | 4.559 ± 4.485 | [4.165, 4.954] |
| 1.5B | throughput per sample (tokens/s) | 60.179 ± 8.657 | [59.418, 60.939] |
| 1.5B | TTFT (s) | 0.129 ± 0.025 | [0.126, 0.131] |

Accuracy: per-sample correctness (0/1), Wilson CI95. Macro F1, detection rate and correction rate: 2,000 bootstrap resamples of the samples (mean, std, percentile CI95; whole samples are resampled because errors in the same file aren't independent). Latency, throughput and TTFT: per-sample values, t-distribution CI95 of the mean.

**McNemar's test** (paired, same samples; exact binomial when there are fewer than 25 discordant pairs). Four units: per sample (exact match right/wrong), per sample x rule decision (each of the 12 rules judged present/absent correctly or not -- more observations, more power), and per individual error (detected or not; corrected or not):

| Comparison | Unit | n | Both right | Only first right | Only second right | Both wrong | Method | p-value |
|---|---|---|---|---|---|---|---|---|
| 7B vs 3B | sample (exact match) | 500 | 456 | 28 | 4 | 12 | chi-square (continuity corrected) | 0.0000 (significant at 0.05) |
| 7B vs 3B | sample x rule decision | 6000 | 5950 | 32 | 6 | 12 | chi-square (continuity corrected) | 0.0001 (significant at 0.05) |
| 7B vs 3B | error (detected) | 1172 | 1109 | 40 | 10 | 13 | chi-square (continuity corrected) | 0.0000 (significant at 0.05) |
| 7B vs 3B | error (corrected) | 1172 | 1099 | 45 | 13 | 15 | chi-square (continuity corrected) | 0.0000 (significant at 0.05) |
| 7B vs 1.5B | sample (exact match) | 500 | 439 | 45 | 2 | 14 | chi-square (continuity corrected) | 0.0000 (significant at 0.05) |
| 7B vs 1.5B | sample x rule decision | 6000 | 5927 | 55 | 6 | 12 | chi-square (continuity corrected) | 0.0000 (significant at 0.05) |
| 7B vs 1.5B | error (detected) | 1172 | 1065 | 84 | 5 | 18 | chi-square (continuity corrected) | 0.0000 (significant at 0.05) |
| 7B vs 1.5B | error (corrected) | 1172 | 1058 | 86 | 8 | 20 | chi-square (continuity corrected) | 0.0000 (significant at 0.05) |
| 3B vs 1.5B | sample (exact match) | 500 | 436 | 24 | 5 | 35 | chi-square (continuity corrected) | 0.0008 (significant at 0.05) |
| 3B vs 1.5B | sample x rule decision | 6000 | 5923 | 33 | 10 | 34 | chi-square (continuity corrected) | 0.0008 (significant at 0.05) |
| 3B vs 1.5B | error (detected) | 1172 | 1058 | 61 | 12 | 41 | chi-square (continuity corrected) | 0.0000 (significant at 0.05) |
| 3B vs 1.5B | error (corrected) | 1172 | 1048 | 64 | 18 | 42 | chi-square (continuity corrected) | 0.0000 (significant at 0.05) |
