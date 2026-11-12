# Training: time and environment

Fine-tuning of the three Qwen2.5-Coder models (QLoRA, Unsloth) on the v6 dataset.

## Training time

| Model | Run | Total time | Of which evaluation (8 passes) | Pure training | Final eval loss |
|---|---|---|---|---|---|
| 7B | `20261003-0240_multi-defect_qwen2_5-coder-7b-instruct-bnb-4bit` | **10 h 10 min** (36,633 s) | 2 h 29 min (8,970 s) | ≈ 7 h 41 min | 0.0137 |
| 3B | `20261025-2302_multi-defect_qwen2_5-coder-3b-instruct-bnb-4bit` | **5 h 28 min** (19,662 s) | 1 h 22 min (4,900 s) | ≈ 4 h 06 min | 0.0156 |
| 1.5B | `20261025-2225_multi-defect_qwen2_5-coder-1_5b-instruct-bnb-4bit` | **3 h 08 min** (11,290 s) | 49 min (2,962 s) | ≈ 2 h 19 min | 0.0175 |

*Total time* is the Trainer's `train_runtime` (from each run's `metrics.jsonl`). It includes the evaluation passes run during
training (every 50 steps and at the end). *Pure training* is the total minus the summed `eval_runtime`.

## Estimated cost

Assuming about **US$ 1.00 per hour** for the VM below, billed on the total training time:

| Model | Hours | Estimated cost |
|---|---|---|
| 7B | 10.18 h | ≈ US$ 10.18 |
| 3B | 5.46 h | ≈ US$ 5.46 |
| 1.5B | 3.14 h | ≈ US$ 3.14 |
| **Total** | **18.77 h** | **≈ US$ 18.77** |

These figures cover the training runs only. VM time spent on setup, building and exporting the dataset, and idle periods
isn't included.

> **Note:** the 1.5B run started on 2026-10-25 at 22:25 and the 3B run at 23:02, while 1.5B was still training. If both ran on
> the same VM, they shared the GPU for about 2.5 h and their times are inflated relative to the 7B run.

## Training configuration (same for all three models)

| Setting | Value |
|---|---|
| Dataset | `dataset/output-multi-defect-v6` — 11,894 training examples, ≈ 18.9M tokens |
| Epochs / steps | 1 epoch = 372 optimizer steps |
| Batch | 8 per device × 4 gradient accumulation = effective batch 32 |
| LoRA | r = 16, α = 32 |
| Learning rate | 2e-4 |
| Max sequence length | 4,096 tokens |
| Base model quantization | 4-bit (bitsandbytes, `unsloth/Qwen2.5-Coder-*-Instruct-bnb-4bit`) |
| Evaluation / checkpoints | every 50 steps (`l4` preset; eval batch 4) |
| Seed | 42 |

## Training VM

From `scripts/gcp_vm.sh`:

| | |
|---|---|
| Provider / zone | Google Cloud, `us-east1-b` |
| Machine type | `g2-standard-8`: 8 vCPUs, 32 GB RAM |
| GPU | 1 × NVIDIA L4, 24 GB |
| Image | Deep Learning VM `common-cu129-ubuntu-2404-nvidia-580` (Ubuntu 24.04, CUDA 12.9, NVIDIA driver 580) |
| Disk | 100 GB `pd-balanced` boot disk (kept when the VM is deleted) |
| Billing model | Standard on-demand (not spot); stops during host maintenance |

## Software

Training stack (`finetune/requirements.txt`, Python 3.11), the same on the VM and locally:

| Package | Version |
|---|---|
| Unsloth | **2026.8.18** (unsloth_zoo 2026.8.12) |
| Transformers | 5.5.0 |
| TRL | 0.24.0 |
| PEFT | 0.20.0 |
| bitsandbytes | 0.50.1 |
| PyTorch | 2.13.0 (CUDA 12.6 build) |

The most recent Unsloth release on PyPI as of 2026-11-08 is **2026.11.3**. The models were trained with 2026.8.18.

The benchmarks ([benchmark-scenarios.md](benchmark-scenarios.md), [benchmark-test.md](benchmark-test.md)) ran on a different machine: a local workstation with an
NVIDIA GeForce RTX 4060 Ti (8.6 GB).
