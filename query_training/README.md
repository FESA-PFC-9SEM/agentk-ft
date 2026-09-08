# `query_training/` — kubectl-query fine-tuning

A **self-contained** pipeline that fine-tunes a small language model to turn a
natural-language cluster request into a single `kubectl` command:

```
"Show all pods in the kube-system namespace"  ->  kubectl get pods -n kube-system
```

This is a **separate task** from the manifest-security auditor in `dataset/` +
`finetune/`. Nothing here imports from those packages; the only shared thing is
the training venv (`finetune/.venv`), because the Unsloth/TRL/torch install is
identical.

## Why a CLI command and not MCP tool calls

The target is a `kubectl` string, not a structured tool call against a specific
Kubernetes MCP server, because:

- The source dataset already *is* NL→`kubectl` — no relabelling.
- `kubectl ...` is dense in every base model's pretraining; a 1.5–7B model
  reproduces it far more reliably than schema-conformant tool-call JSON.
- The system prompt stays tiny (no per-tool schemas eating the context window).
- Evaluation is a string/AST comparison, not a structural diff.
- The agent stays portable — it works against any cluster with a kubeconfig,
  and execution guardrails (a read-only verb allow-list) are enforced
  independently of the model, which is a firmer guarantee than trusting the
  model to only pick read tools.

If a deployment later mandates a particular MCP server, translate the generated
`kubectl` command to that server's API as a thin post-hoc layer rather than
retraining.

## Data sources

All under `corpus/`, all members of the same `dereklck` kubectl-query family
(same objectives, same synthetic resource-name generator). Per-source loaders
live in `sources.py`; each provides a `source` provenance file.

| Source (`--sources` name) | Rows | What it adds | Default? |
|---|---|---|---|
| `kubectl-command-csv` (`kubernetes-kubectl-command-dataset/train.csv`) | 19,661 | `question` → `command`, **plus `chain_of_thought` for every row** (drives `--target-format plan-command`), plus `flags` / `syntax` metadata | **yes** |
| `devops-kubectl` (`devops-kubectl-v1/*.parquet`) | 34,535 | more concrete phrasings; command parsed out of a ```` ```bash ```` fence; **no CoT** | opt-in |
| `cli_queries_1/` | 19,661 | nothing — `output` is byte-identical to `kubectl-command-csv`'s `command`. Not a registered source. | never |

`clean.py` then:

| Step | Effect (default run) |
|---|---|
| drop non-`kubectl` (`KUBE_EDITOR=… kubectl …`, prose) | `not_kubectl` ≈ 240 |
| drop chained / multi-line commands (`a && b`, `\n`, pipes) — quote-aware, so `nginx -g "daemon off;"` survives | `not_single_command` ≈ 70 |
| `--target-format plan-command` only: drop rows with no CoT | `no_cot` |
| **cross-source exact de-dup** on `(instruction, command)` | `exact_duplicate` ≈ 19k when `devops-kubectl` is on |
| `--on-conflict first`: one command per distinct instruction | `conflict_minority` ≈ 1k |
| **group split** by normalised command — no command (or its phrasings, from any source) straddles train/val/test | — |

Default run (`kubectl-command-csv` only) keeps **~18.4k** rows (93%); zero
command overlap across splits. `devops-kubectl` added on top nets ~13.5k more.
See `output/clean_diagnostic.json` for per-source counts, the full drop
breakdown, and the verb histogram after any run.

## Target formats

`--target-format` (in `clean.py` / `pipeline.sh`'s `TARGET_FORMAT`):

- **`command`** (default) — assistant turn is the bare command string.
- **`plan-command`** — assistant turn is `{"plan": "...", "command": "..."}`.
  The short plan gives a small model a few tokens to reason before committing;
  `evaluate.py` / `infer.py` auto-detect it and score `.command`. Needs a CoT,
  so it only works with `kubectl-command-csv`.

## Usage

```bash
# data only (no GPU, main .venv):
.venv/bin/python -m query_training.clean            # -> output/{train,val,test}.jsonl
.venv/bin/python -m query_training.export           # -> output/unsloth/  (renders chat template, counts tokens)

# or both at once:
query_training/pipeline.sh

# train + evaluate (needs finetune/.venv + a GPU):
finetune/.venv/bin/python -m query_training.train
finetune/.venv/bin/python -m query_training.evaluate --adapter query_training/runs/<run>/lora_adapter

# plot the loss curves (main .venv, no GPU):
python -m query_training.plot_metrics --metrics-file query_training/runs/<run>/metrics.jsonl

# eyeball a single prediction (finetune/.venv):
finetune/.venv/bin/python -m query_training.infer --adapter query_training/runs/<run>/lora_adapter \
    --instruction "restart the nginx deployment"
finetune/.venv/bin/python -m query_training.infer --adapter query_training/runs/<run>/lora_adapter \
    --test-file query_training/output/test.jsonl --index 7

# or the whole thing (clean -> export -> train -> plot -> evaluate):
query_training/pipeline.sh --train
```

Useful flags:

- `--sources kubectl-command-csv,devops-kubectl` — add the extra corpus.
- `--target-format plan-command` — train the `{"plan","command"}` target.
- `--read-only` — keep only read-verb commands (`get`, `describe`, `logs`,
  `top`, …), for an agent that must not mutate the cluster.
- `--on-conflict {first,most-common,drop,keep-all}`.
- `python -m query_training.export --char-approx` — skip the tokenizer download
  (offline; less accurate, and the rendered `text` won't carry the real ChatML
  markers, so only use it for inspection, never for a real train run).
- `python -m query_training.train --model unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit --epochs 3`.
- `python -m query_training.infer --target-format plan-command --instruction "..."` —
  must match how the adapter was trained (ignored with `--test-file`).

## Files

| File | Role |
|---|---|
| `schema.py` | both system prompts, valid `kubectl` verb set, `build_messages` / `extract_command` |
| `sources.py` | per-source loaders → `Record(instruction, command, cot, source)` stream |
| `clean.py` | sources → cleaned, cross-source-deduped, conflict-resolved, **group-split** `.jsonl` |
| `export.py` | `.jsonl` → `{"messages", "text", "num_tokens"}` rendered through the model's real chat template; drops over-length rows |
| `train.py` | Unsloth QLoRA SFT, loss masked to the assistant turn; runs land in `runs/<timestamp>_cli_<model>/` |
| `evaluate.py` | `exact_match`, `normalized_match` (flag-order-insensitive), `valid_kubectl`, `verb_match`; format-agnostic |
| `plot_metrics.py` | `metrics.jsonl` → `metrics.png` + `metrics.csv` (loss / eval-loss / LR / grad-norm); main `.venv`, no GPU |
| `infer.py` | one request → raw output + parsed command (+ verdict when pulled from a split); `finetune/.venv` |
| `test_*.py` | unit tests for the pure logic — run from the main `.venv` (`pytest query_training/`) |

## Metrics

`evaluate.py` reports four rates over `output/test.jsonl`:

- **exact_match** — identical after whitespace normalisation.
- **normalized_match** — identical after canonicalisation (`kubectl <verb>` kept
  in place, remaining argv tokens sorted), so flag reordering is forgiven. This
  is the headline number.
- **valid_kubectl** — output parses and starts `kubectl <known-verb>`.
- **verb_match** — predicted first verb equals the expected one.

## Limitations

- Single-turn only: request → one command. A real agent loop (command →
  observation → next command → final answer) needs multi-step trajectory data,
  which these datasets do not contain and this pipeline does not synthesise.
- **No coverage of compositional queries.** Across every source: `--field-selector`
  = 0, set-based selectors = 0, `--sort-by` ≈ 9, `--all-namespaces` ≈ 2. Adding
  `devops-kubectl` does not change this — it resamples the same distribution.
  Requests like "pods across all namespaces not Running/Succeeded, sorted by
  restart count" need a separate synthetic generator (the CSV's `flags` column
  is the raw material for one).
- Residual label noise: some rows are generic (`question == objective`) or have
  a malformed command (`kubectl get deployments.v1.0.0.api-v2`), inherited from
  the templated construction.
- Verb distribution is skewed toward `port-forward`/`create`/`set`/`config` and
  thin on `describe`/`top`/`explain`.
- The CoT (`plan-command`) is shallow templated text ("1. use the get
  subcommand 2. specify the name X") — enough to teach a plan step, not real
  compositional reasoning.
