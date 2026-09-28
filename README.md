# Kubernetes Manifest Security Dataset Pipeline

Dataset-generation pipeline for an undergraduate capstone project: a small
language model, fine-tuned elsewhere, that reads a Kubernetes manifest and
returns a single JSON object describing security/configuration findings and
an RFC 6902 patch that fixes them. This repository produces the training
dataset only — no training happens here.

## Table of contents

- [Core design principle](#core-design-principle)
- [The task the trained model performs](#the-task-the-trained-model-performs)
- [Detection rules](#detection-rules)
- [Response schema](#response-schema)
- [Architecture](#architecture)
  - [Part A — `dataset/`](#part-a--dataset)
  - [Part B — `generation/`](#part-b--generation)
- [Pipeline flow](#pipeline-flow)
- [Dataset generation strategies](#dataset-generation-strategies)
- [Real-world testing](#real-world-testing)
- [Benchmarking training throughput](#benchmarking-training-throughput)
- [Setup](#setup)
- [Usage](#usage)
- [Testing](#testing)
- [Design decisions and rationale](#design-decisions-and-rationale)
- [Known limitations](#known-limitations)
- [Project structure](#project-structure)

---

## Core design principle

**Labels are never written by hand and never produced by a model.**

The pipeline takes a manifest proven clean, normalizes it into a canonical
hardened form, then *programmatically* injects exactly one defect. Because
the injection code knows precisely what it changed, the `findings` and
`patch` fields are **derived from the mutation itself** — never hand-authored,
never guessed by an LLM. This gives perfect ground truth at zero labeling
cost.

```
clean manifest → normalize() → canonical form → mutate_ksecNNN() → ┬─ mutated_doc (training input)
                                                                    ├─ findings   (derived via detect_ksecNNN on mutated_doc)
                                                                    └─ patch      (the exact inverse of the injection)
```

Every mutator is required to satisfy one invariant, checked for **100% of
the dataset** (not sampled) in `dataset/build.py`:

```
apply_patch(mutated_doc, patch) == canonical_doc
```

If a mutator's patch doesn't reconstruct the canonical form exactly, the
build fails loudly. This is the cheapest correctness guarantee available,
and any failure is a bug in a mutator, not a data-quality nuisance to be
tolerated.

A local LLM (Part B) is used **only** to generate additional clean,
realistic input manifests — it never sees the defect and never produces a
label. Labels always come from Part A's mutation code, whether the base
manifest came from the real corpus or from the local model.

---

## The task the trained model performs

**Input:** a Kubernetes manifest file (possibly multi-document, YAML
documents separated by `---`).

**Output:** a single JSON object, no prose, no markdown fences:

```json
{
  "findings": [
    {"rule_id": "KSEC-001", "severity": "critical", "doc": 0,
     "path": "/spec/containers/0/env/0/value",
     "message": "...", "evidence": "Tr0u***"}
  ],
  "patch": [{"doc": 0, "op": "replace", "path": "...", "value": {}}],
  "new_resources": ["<complete YAML for resources that must be created>"],
  "notes": []
}
```

A clean manifest has all four arrays empty.

---

## Detection rules

| Rule | Detects | Typical fix | Status |
|---|---|---|---|
| **KSEC-001** | Plaintext credential — password, token, API key, license key, connection string, private key. Two injection shapes: as an env var `{name, value}` pair, or in `command`/`args` — a `--password=...` flag, a basic-auth URL, or a flag and its value as separate list elements (`-dbpwd`, `123456789`). | Env variant: externalize to a `Secret` + `secretKeyRef`. Command variant: remove the offending arg(s). | **active** |
| **KSEC-002** | Insecure `securityContext` — `privileged: true`, `runAsUser: 0`, `allowPrivilegeEscalation: true`, or added `capabilities`. | Remove/revert the offending field. | **active** |
| **KSEC-003** | Host access — `hostNetwork`/`hostPID`/`hostIPC: true`, or a `hostPath` volume mounting a sensitive host path (`/`, `/etc`, `/var/run/docker.sock`, `/proc`, `/root`, `/var/lib/kubelet`, `/boot`, `/sys`, `/home`). | Remove the field, or remove the volume + its `volumeMount`. | **active** |
| **KSEC-004** | Permissive RBAC — a wildcard `"*"` in `apiGroups`/`resources`/`verbs` on a `Role`/`ClusterRole`, or a `RoleBinding`/`ClusterRoleBinding` granting `cluster-admin`. | Revert the wildcard or the binding's `roleRef.name`. | **active** |
| **KSEC-005** | Unpinned container image — missing tag or `:latest` (digest-pinned images are not flagged). | Revert to the original pinned tag. | **active** |
| **KSEC-006** | Selector/label mismatch — a workload's `spec.selector.matchLabels` (a `ReplicationController`'s flat `spec.selector`) doesn't match its own pod template labels, **or**, across documents of one file, a `Service`'s `spec.selector` matches no workload's pod labels while a workload with the same label keys is right there. Breaks routing and discovery. | Revert the selector label value (on the Service, for the cross-document form). | **active** |
| **KSEC-007** | Probe port mismatch — a `livenessProbe`/`readinessProbe`/`startupProbe` (`httpGet` or `tcpSocket`) targets a port not declared in the container's `ports` (checked for both numeric and named ports). | Revert the probe's port. | **active** |
| **KSEC-008** | `resources.requests` exceeds `resources.limits` for `cpu` or `memory` — passes schema-only validation (`kubeconform`) but is rejected by the Kubernetes API at admission time. | Revert the request value. | **active** |
| **KSEC-009** | Dangling volume reference — a `volumeMount.name` doesn't match any declared `volumes[].name` (or, on a `StatefulSet`, any `volumeClaimTemplates[].metadata.name`), generated via a single human-plausible character edit (transpose/delete/duplicate) of a real volume name. | Revert to the correct name. | **active** |
| **KSEC-010** | Probe protocol mismatch — an `httpGet` liveness/readiness/startup probe on a container whose image is a non-HTTP server (`postgres`, `mysql`, `mariadb`, `redis`, `mongo`, `memcached`, incl. vendor rebuilds like `bitnami/postgresql`). The probe can never succeed: liveness/startup → restart loop (`high`), readiness → never Ready (`medium`). | Replace `httpGet` with `tcpSocket` on the same port (or restore the original `exec` check). | **active** |
| **KSEC-011** | Missing required env var for the image — a database server container without a variable its entrypoint requires. Covers official `postgres`/`mysql`/`mariadb`/`percona` (+ `percona/percona-server`), Bitnami `postgresql`/`mysql`/`mariadb`/`redis`/`mongodb` (password or `ALLOW_EMPTY_PASSWORD`; replicas skipped), and Microsoft SQL Server (`mcr.microsoft.com/mssql/server`, `azure-sql-edge`: `ACCEPT_EULA` **and** an SA password — two independent findings). Accepted alternatives such as `*_FILE` satisfy it. The container exits at startup. | Passwords: add the variable from a `Secret` (`secretKeyRef`) + a placeholder `Secret` in `new_resources` — the same shape as KSEC-001's externalization. `ACCEPT_EULA`: add `value: "Y"`. | **active** |

Severities are assigned per finding sub-case (e.g. `privileged: true` is
`critical`, `allowPrivilegeEscalation: true` is `medium`) — see
`dataset/detect.py` for the exact mapping.

KSEC-001, KSEC-002 and KSEC-005 are "security" rules in the strict sense.
KSEC-006 through KSEC-011 are semantic/configuration-correctness checks
added later, sharing the same rule-ID numbering and response schema by
design decision (see [Design decisions](#design-decisions-and-rationale)).
KSEC-010 and KSEC-011 are the first *context-dependent* rules: whether a
field is wrong, and what the right value is, depends on which application
the image is — see
[Why KSEC-010/011 key on image identity](#design-decisions-and-rationale).

### Active vs. disabled rules

**All 11 rules currently generate training examples.** The rule set the
pipeline trains on is defined by two registries: `dataset/schema.py`'s
`RULES` dict and `dataset/mutate.py`'s `MUTATORS`. Disabling a rule from
generation — without deleting its tested detector/mutator — means removing
its entry from both:

- `SYSTEM_PROMPT` is generated dynamically from `RULES`, so it lists only
  the registered rules — the model is never told to detect something it was
  never shown a labeled example of.
- `dataset/build.py` derives its rule set, quotas, and mutation pool from
  `MUTATORS`, so no other file requires editing.

KSEC-003, KSEC-004 and KSEC-009 were disabled this way for a while and have
since been re-enabled (KSEC-009 after fixing its `StatefulSet`
`volumeClaimTemplates` false positive — see
[Known limitations](#known-limitations)).

---

## Response schema

Defined once in `dataset/schema.py` and imported everywhere else — nothing
duplicates this contract.

- `SYSTEM_PROMPT` — the exact prompt the trained model is given. Its rule
  list is generated dynamically from the `RULES` dict, so the prompt and the
  taxonomy can never drift apart.
- `RULES` — `{rule_id: description}`.
- `Finding`, `PatchOp`, `Response` — dataclasses with `to_dict()`.
- `validate_response(obj) -> list[str]` — schema validator; returns a list
  of errors (empty = valid), never raises.
- `mask_evidence(value) -> str` — first 4 characters + `"***"`. **A secret
  value must never appear in full in any output field.** (The one
  intentional exception: the *training input* — the mutated manifest text
  itself — legitimately contains a full synthetic fake credential, because
  that's the pattern the model needs to learn to recognize. Only output
  fields, i.e. `evidence` and `new_resources`, are always masked.)
- `escape_json_pointer_token(token) -> str` — RFC 6901 escaping (`~`→`~0`,
  `/`→`~1`). Needed whenever a free-form key (not a fixed field name or a
  list index) is embedded in a JSON Pointer path — e.g. Kubernetes label
  keys, which routinely contain `/` (`app.kubernetes.io/name`).

---

## Architecture

### Part A — `dataset/`

The critical path: turns the real corpus into `dataset.jsonl`.

| File | Responsibility |
|---|---|
| `schema.py` | System prompt, rule taxonomy, response dataclasses, validator. |
| `scanning.py` | Real-secret detection: sensitive-key regex, Shannon entropy, connection-string/PEM patterns, placeholder allowlist, and a dedicated scan of `command`/`args` for CLI-embedded credentials. Correctly resolves the Kubernetes `{name: X, value: Y}` env-var pattern (this is the single most important correctness property in the project — see `test_env_name_value_pair`). |
| `k8s.py` | Shared, read-only Kubernetes navigation helpers: `get_pod_spec` (resolves the PodSpec location per kind, including the 4-levels-deep CronJob case), `iter_containers`, `get_selector_match_labels`, `get_template_labels`, `get_pod_labels`/`get_service_selector` (for the cross-document Service check), `get_container_ports`, `parse_quantity` (Kubernetes resource-quantity parser), image tag helpers, sensitive-hostpath check. |
| `detect.py` | One read-only detector per rule (`detect_ksec001`..`detect_ksec011`), plus `detect_structural` (002-005), `detect_semantic` (006-011), `detect_all` (one document) and `detect_file` (a whole multi-document file: every document's findings tagged with its index, plus the cross-document KSEC-006 Service check). Used for filtering dirty corpus docs, mutator preconditions, post-normalize assertions and scenario scoring. |
| `dedup.py` | Structural deduplication — reduces a document to a "skeleton" (strips names/namespaces/labels/annotations/selectors, collapses leaf values to placeholders) and hashes it. Two manifests differing only in naming collide. |
| `normalize.py` | Produces the canonical hardened form — the gold target. Deterministic, idempotent. Only fixes rules 002-005 forward (see rationale below); KSEC-001 docs are dropped rather than fixed. |
| `mutate.py` | One mutator per rule (`mutate_ksec001`..`mutate_ksec011`). `MUTATORS` — the registry `build.py` and `multi_mutate.py` actually read from — wraps each one in a guard that rejects any mutation changing another rule's findings (see [composition](#design-decisions-and-rationale)). Each mutator takes a canonical doc + `random.Random` and returns a `MutationResult(mutated_doc, canonical, findings, patch, new_resources)` or `None` if not applicable. `mutate_ksec001` additionally accepts an optional `candidate_names` override (see below). |
| `stats.py` | Distribution report for a built dataset: per split, positives/negatives, findings and examples per rule, severities, findings and documents per example, which rules land in multi-document files, patch ops, kinds and input sizes (`python -m dataset.stats <dir>`). |
| `build.py` | Orchestrates the whole pipeline: load → assemble multi-document sibling bundles (before dedup) → filter → dedup → drop dirty (harvesting credential key names along the way) → normalize → mutate with per-rule quotas → 100% round-trip check → write `train/val/test.jsonl`, split by source repository. Catches a mutator's internal `AssertionError` per document/rule rather than crashing the whole run on one anomalous document (see rationale below). |
| `view.py` | Utility to extract manifests from any `.jsonl` (dataset or generation output) into individual `.yaml` files for manual inspection — no JSON archaeology required. |

### Part B — `generation/`

Local LLM generates **clean input manifests only** — never labels. Exists to
fill a gap in the real corpus: "hard negative" material (clean manifests
that *look* suspicious). It also supports an `rbac` generation mode, built
to address RBAC scarcity for KSEC-004 — not run by `pipeline.sh`, because
the real corpus turned out to have enough RBAC for KSEC-004 on its own
(2,802 `ClusterRole`s, 935 `Role`s, 158 bindings; 3,895 documents KSEC-004
can mutate). Run it manually if you want more RBAC variety.

| File | Responsibility |
|---|---|
| `SETUP.md` | Runtime choice (Ollama, justified against llama.cpp server), model choice (Qwen2.5-Coder-7B-Instruct, Q4_K_M), step-by-step setup. |
| `check_env.py` | Verifies GPU/driver/VRAM, Ollama service, and (if pulled) that the model responds. Downloads nothing. |
| `seeds.py` | Combinatorial seed sampler — domain, naming convention, stack, resource kind (RBAC deliberately oversampled for `--mode rbac`), namespace convention, labels/annotations, YAML style, comment language, single/multi-doc. All randomness goes through a seeded `random.Random`. |
| `generate.py` | Resumable, async-batched calls to Ollama. Modes `base` / `rbac` / `hard-negative` (`pipeline.sh` only runs `base` and `hard-negative` by default — see note above). Strips markdown fences, retries on malformed YAML, records seed + manifest per line. |
| `curate.py` | Filters generated manifests: valid YAML with `apiVersion`+`kind`, passes `kubeconform`, contains no real secret (reused from `scanning.py`), not a structural duplicate (reused from `dedup.py`). Hard-negative mode inverts the secret check: must trip a *naive* detector but pass the real one. |
| `report.py` | Diversity diagnostics — kind/namespace/image/container-count distributions, structural-uniqueness ratio, repeated n-grams, CSV + PNG. Catches generation collapse before spending GPU time at scale. |

---

## Pipeline flow

```
                          ┌─────────────────────────┐
                          │   generation/generate.py │  (local LLM, Ollama)
                          │   modes: base/hard-      │
                          │   negative (+rbac, unused│
                          │   by pipeline.sh for now)│
                          └────────────┬─────────────┘
                                       │  generation/output/{mode}.jsonl
                          ┌────────────▼─────────────┐
                          │   generation/curate.py    │  filters bad output
                          └────────────┬─────────────┘
                                       │  generation/output/{mode}.curated.jsonl
                                       │
   corpus/*.parquet                   │
   (real manifests)                   │
        │                             │
        ▼                             ▼
┌───────────────────────────────────────────────┐
│                dataset/build.py                 │
│  load → filter Helm/invalid → dedup structurally │
│  → drop real-secret docs → normalize (canonical) │
│  → mutate per rule (quotas) → 100% round-trip    │
│  check → split by repo → write train/val/test    │
└───────────────────────┬───────────────────────┘
                         ▼
              dataset/output/{train,val,test}.jsonl
```

`pipeline.sh` chains generate → curate → report → `dataset.build` in one
command (`--smoke` for a small end-to-end validation run, no args for the
full-scale run).

---

## Dataset generation strategies

Real-world testing of the first trained adapter (`sql.yaml`, then a batch of
hand-written scenarios in `scenarios/`) surfaced a generalization gap: the
model reliably caught a manifest's *one* problem, but missed a *second*
simultaneous one (e.g. it found one plaintext credential but not a second in
the same manifest). Root cause: every training example up to that point had
**exactly one** injected defect — `dataset/mutate.py`'s per-rule mutators are
each called once per example — so the model plausibly learned "at most one
finding" as an implicit prior rather than actually searching the whole
manifest.

Rather than replace the original pipeline, this repo keeps **both**
strategies side by side so results can be compared directly in the TCC
write-up:

| | **single-defect** (baseline) | **multi-defect** |
|---|---|---|
| Findings per positive example | exactly 1 | 1–8, mostly 2–4 (`--min-defects`/`--max-defects`) |
| Mutation logic | `dataset/mutate.py` (`MUTATORS`) | `dataset/multi_mutate.py`, composing the same `MUTATORS` |
| Dataset output | `dataset/output/` (default) | `dataset/output-multi-defect/` |
| Training run folder | `runs/<timestamp>_single-defect_<model>/` | `runs/<timestamp>_multi-defect_<model>/` |

`dataset/multi_mutate.py` doesn't reimplement any rule: it starts from the
same clean canonical document, applies 1–8 of `mutate.py`'s existing
per-rule mutators to it *in sequence* (each one seeing the previous step's
already-mutated document), chains their inverse patches together, and
re-derives the final finding set by re-running the detectors
(`dataset/detect.py::detect_all`) against the fully mutated document — see
that module's docstring for why this composition is safe (each rule injects
into a disjoint structural area, so injecting rule B never dirties rule A's
already-injected field).

**Which rules and how many (v4).** The v3 dataset drew rules uniformly and
2–4 defects per example. Tested on `scenarios/`, the model trained on it
regressed on plain credentials: with 11 rules sharing the budget, KSEC-001
fell from 74% to 50% of positives (KSEC-005 from 69% to 50%), and probing the
model showed it now called `9-storm.yaml` clean with 96% confidence despite
a plaintext password and an unpinned image. It also stopped at ~4 findings
on `8-newrelic.yaml`, which has 9 — it had never seen more than 4. Three
changes, all in `dataset/multi_mutate.py`/`build.py` defaults:
`RULE_WEIGHTS` makes KSEC-001 and KSEC-005 3× as likely to be picked (back to
~66% of positives each, every other rule still present); defect counts run
1–8 with counts above 4 sampled at 0.4× weight; and `--min-defects 1` lets a
file with a single defect be a positive too, as it often is in practice.
Answers now reach ~1,030 tokens at 8 findings, so every inference entry
point (`evaluate`, `infer`, `run_scenarios`, the demos) defaults to
`--max-new-tokens 1280`; the 4,096-token export drops ~1% of examples.

**The same rule can fire more than once in one example** (`MAX_REPEATS_PER_RULE`,
currently 2) — e.g. two separate plaintext credentials in one manifest, not
just two different rule types. This closes the *actual* original gap: a
real manifest (`scenarios/3-mysql.yaml`) has two passwords, and the first
version of the multi-defect dataset could still only ever compose *different*
rules together, never the same one twice. Each repeat attempt is wrapped in
its own `try/except AssertionError`, so a rule that's already exhausted its
eligible targets in a document (e.g. only one container left to flag) is
skipped like any other inapplicable attempt rather than aborting the whole
example.

**`dataset/mutate.py::_fake_secret_value()` produces format-diverse values**,
not just fixed-length alphanumeric strings. A model trained only on
`letters+digits, always 20 chars` learned to recognize that exact shape
rather than the general concept "this is a plaintext credential" — confirmed
by a real miss (`mypassowrd 123`, an unquoted, typo'd, space-containing real
value) that a differently-shaped *fake* password during manual testing
didn't trip up. The generator now draws from: random alphanumeric (variable
length), word+digits+symbol (`Summer2024!`), keyboard walks (`qwerty123`),
well-known weak-password shapes (`Passw0rd!`), a "messy" style reusing the
existing typo helper (`sunshien 3`), and full connection strings
(`postgres://user:pass@host:port/db`) — the last of these closing a second
gap, where a plain env value shaped like a connection string was essentially
unrepresented in training despite `scanning.py`'s `CONN_STRING_RE` already
being able to catch it independent of the key name. None of these are drawn
from a real leaked-password corpus (e.g. rockyou.txt) — deliberately: real
breach data isn't necessary here, only format *diversity* is, which is a much
smaller and uncomplicated thing to synthesize from scratch. A `cli_safe`
flag keeps the CLI/URL-embedded injection variant free of characters
(spaces, `@`, quotes) that would break the regex captures `scanning.py` uses
to find that specific shape.

Running the multi-defect variant through the whole pipeline, end to end:

```bash
# 1. generate the dataset (same corpus, different mutation strategy)
python -m dataset.build --strategy multi-defect --total 2000

# 2. export for Unsloth
finetune/.venv/bin/python -m finetune.export_for_unsloth \
    --input-dir dataset/output-multi-defect \
    --output-dir dataset/output-multi-defect/unsloth

# 3. train -- --strategy only labels the auto-generated runs/ folder name,
#    it does not change training logic; --data-dir is what actually matters
finetune/.venv/bin/python -m finetune.train_unsloth \
    --strategy multi-defect \
    --data-dir dataset/output-multi-defect/unsloth \
    --preset l4

# 4. evaluate against its own held-out test set, and against scenarios/
finetune/.venv/bin/python -m finetune.evaluate \
    --adapter runs/<generated-name>/lora_adapter \
    --test-file dataset/output-multi-defect/test.jsonl \
    --output-dir runs/<generated-name>/eval
finetune/.venv/bin/python -m finetune.run_scenarios \
    --adapter runs/<generated-name>/lora_adapter \
    --output runs/<generated-name>/scenarios_results.xlsx
```

The single-defect commands are identical minus `--strategy multi-defect`
and the `-multi-defect` path suffixes — see [Usage](#usage) below.

### Training run history

`finetune/train_unsloth.py --output-dir` defaults to an auto-generated
`runs/<YYYYMMDD-HHMM>_<strategy>_<model-slug>/` folder instead of a fixed
path, so every run gets its own timestamped, strategy-labeled directory
without a manual copy/rename step. Each run folder ends up self-describing:
`run_info.json` (strategy, model, preset, hyperparameters — written at
start), `checkpoint-*/`, `lora_adapter/`, `metrics.{jsonl,csv,png}`, and,
once you run the commands above against it, `eval/` and
`scenarios_results.xlsx`. `runs/` is gitignored (multi-GB checkpoints) —
keep whatever you want in the TCC write-up (e.g. `run_info.json`,
`metrics.png`, `eval_summary.json`) copied out separately. Note
`--strategy` here only labels the run folder name; it doesn't control which
dataset gets loaded (`--data-dir` does) — the CLI prints a warning if the
two look inconsistent (e.g. `--strategy multi-defect` with a `--data-dir`
that doesn't mention it), since this has silently produced a mislabeled run
before.

---

## Real-world testing

`dataset/output*/test.jsonl` measures in-distribution recall against the
same synthetic mutation pipeline that generated training data — useful, but
it can't catch a model that's overfit to that pipeline's own surface
patterns. `scenarios/` is a separate, hand-written set of 10 real-world
Kubernetes manifests (`1-orion.yaml` … `10-mongodb.yaml`, sourced from public
examples) used specifically to catch that.

**`scenarios/test_cases.yaml`** is the ground truth: 40 individual error
instances across the 10 files (not one row per file), each categorized as
exactly one of `Credenciais Expostas` / `Imagem sem Tag` / `Erro de
Sintaxe/Config`, with a line number and description. Every instance also
records what this project's own rule taxonomy can say about it — a `rule_id`
(or `null` if no rule covers it) plus enough to identify that *specific*
instance among possibly several findings of the same rule in one file. 33 of
the 40 are in scope, and the detectors catch all 33: the four Service
selector mismatches are KSEC-006's cross-document form, `-dbpwd 123456789` is
KSEC-001's split-flag form. The other 7 are deliberately `rule_id: null` —
usernames (not secrets on their own), a container name that doesn't match
its image, a nonexistent binary, mistyped paths and an invalid `volumeID`:
none can be labeled programmatically without guessing intent. That's
reported as an honest scope boundary, not a failing test.

**`finetune/run_scenarios.py`** runs a checkpoint against every scenario file
several times (sampled, `temperature>0`, so repeated runs can actually
differ) and scores it against `test_cases.yaml`, producing an Excel report
with `Detecção` and `Corrigido` sheets (`Arquivo | Erros | Detectado/Corrigido
| Não detectado/corrigido | % OK`, averaged across the sampled runs) plus a
`Categorias` breakdown and the raw ground truth for reference. "Corrected"
is checked automatically by running `dataset/detect.py`'s `detect_file`
(every rule, including cross-document checks) on the manifest after the
model's own patch — if no finding matching that instance survives, it's
fixed. Findings the ground truth doesn't list (e.g. KSEC-003 on
`8-newrelic.yaml`'s host-level monitoring agent) are accepted as extras:
they don't count against the score. Re-scored this way, the
`20260907-1545` Qwen2.5-Coder-7B multi-defect model (trained before
multi-document examples existed) detects 74% and corrects 53% of the
in-scope instances across 5 sampled runs; most misses are the
cross-document selector fixes and the files whose defects sit in document 1.

**`finetune/zero_shot_baseline.py`** sends the exact same `SYSTEM_PROMPT` to
a *non-fine-tuned* base model via Ollama and scores the response with the
same logic `evaluate.py` uses, to answer "does the bigger base model already
do this without any training?" A real comparison run (Qwen2.5-Coder-14B,
zero-shot, against `sql.yaml`) found it correctly identified both real
issues with sound reasoning, but produced malformed JSON (an invalid escape
inside a generated YAML string) — a genuine base-model-vs-fine-tuned
tradeoff worth stating explicitly: better raw semantic understanding, but
unreliable structured output, which is exactly what fine-tuning on a fixed
schema is supposed to fix.

```bash
finetune/.venv/bin/python -m finetune.run_scenarios --adapter runs/<name>/lora_adapter
finetune/.venv/bin/python -m finetune.zero_shot_baseline --model qwen2.5-coder:14b --manifest scenarios/3-mysql.yaml
```

---

## Benchmarking training throughput

`finetune/benchmark_throughput.py` finds the `--batch-size`/`--grad-accum`
sweet spot for a given model/GPU by running short bursts of real training
steps (no eval, no checkpointing — neither is representative of pure
throughput) across whatever combos you give it, on the actual training data
so sequence-length padding matches a real run. Reports samples/sec,
steps/sec, tokens/sec (computed from `include_num_input_tokens_seen`, since
the installed `transformers` version tracks the token count but doesn't
itself divide it into a final rate), and peak VRAM per combo — a combo that
hits CUDA OOM is recorded and skipped rather than crashing the sweep.

```bash
finetune/.venv/bin/python -m finetune.benchmark_throughput \
    --model unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit \
    --data-dir dataset/output-multi-defect/unsloth \
    --combos 1x32 2x16 4x8 8x4 16x2 \
    --steps 20
```

Results are written as JSON, auto-named the same way `runs/` folders are
(`finetune/output/throughput/<timestamp>_<model-slug>.json`) unless
`--output` is given explicitly.

---

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Part B additionally needs Ollama and kubeconform (see
[`generation/SETUP.md`](generation/SETUP.md) for full detail and rationale):

```bash
ollama pull qwen2.5-coder:7b-instruct-q4_K_M   # ~4.7GB — model weights are never pulled automatically
GOBIN=$(go env GOPATH)/bin go install github.com/yannh/kubeconform/cmd/kubeconform@latest
export PATH="$PATH:$(go env GOPATH)/bin"
```

Fine-tuning (`finetune/`) needs its own environment — a dedicated Python
3.11 venv, separate from the one above (see `finetune/requirements.txt` for
why). On a fresh training VM, one command does the whole setup, including
pulling the compressed dataset back in:

```bash
./finetune/setup_vm.sh
# different CUDA version than the default (cu126)?
CUDA_INDEX=https://download.pytorch.org/whl/cu128 ./finetune/setup_vm.sh
```

---

## Usage

### Full pipeline

```bash
./pipeline.sh --smoke      # small validation run
./pipeline.sh              # full-scale run
```

### Individual stages

```bash
# Part A only, against the real corpus
python -m dataset.build --limit 1000 --total 300

# Part B: generate, curate, report
python -m generation.generate --mode base -n 500
python -m generation.generate --mode hard-negative -n 200
python -m generation.curate --mode base
python -m generation.curate --mode hard-negative
python -m generation.report --input generation/output/hard-negative.curated.jsonl

# --mode rbac exists but isn't run by pipeline.sh (the real corpus already
# has enough RBAC for KSEC-004) -- optional, for more RBAC variety
python -m generation.generate --mode rbac -n 200

# merge synthetic + real corpus
python -m dataset.build --synthetic-dir generation/output --total 5000
```

### Inspecting results

```bash
# extract every manifest to its own .yaml file
python -m dataset.view dataset/output/train.jsonl --out /tmp/yamls --sidecar

# only examples for one rule
python -m dataset.view dataset/output/train.jsonl --rule-id KSEC-006 --stdout --limit 5
```

All `generate.py` runs are resumable — an interrupted or re-run command
fills in only what's missing, keyed by a deterministic per-index seed
(`--seed * 1_000_003 + index`), never duplicating work.

### Compressing dataset files for git

A full-scale `train.jsonl` runs tens to hundreds of MB —
`dataset/output/unsloth/train.jsonl` alone hit ~130MB in one run, over
**GitHub's 100MiB hard file-size limit**. `dataset/output/`,
`dataset/output/unsloth/`, and `generation/output/`'s raw `.jsonl` files are
gitignored for exactly this reason; only their gzipped `.jsonl.gz`
counterparts get committed.

```bash
# before committing: compress every dataset/generation .jsonl to .jsonl.gz
./scripts/compress_dataset.sh

# after cloning/pulling on a new machine (e.g. a training VM): inflate them back
./scripts/decompress_dataset.sh
```

Measured on the ~130MB `dataset/output/unsloth/train.jsonl`: gzip took it to
~7MB (~18.6x) — comfortably clear of the limit. Round-trip integrity is
byte-exact (verified via checksum, not just "it decompresses without error").
`finetune/setup_vm.sh` (below) calls `decompress_dataset.sh` automatically as
its last step, so a fresh training VM doesn't need this run by hand.

---

## Testing

```bash
export PATH="$PATH:$(go env GOPATH)/bin"   # for the two kubeconform integration tests
.venv/bin/python -m pytest dataset/ generation/ -q
```

930 tests, covering:
- Unit tests per detector/mutator (including the
  critical `{name, value}` env-var case, and RFC 6901 escaping for
  slash-containing label keys).
- Property-style round-trip tests (`apply_patch(mutated, patch) ==
  canonical`) across hand-written fixtures and a 10-seed × ~1,500-document
  fuzz against the real corpus for every implemented mutator, active or not
  (0 errors).
- A guard test asserting every name in `FAKE_SECRET_VAR_NAMES` is actually
  detectable by `SENSITIVE_KEY_RE` — catches the class of bug where a pool
  name relies entirely on the probabilistic entropy fallback.
- `build.py` resilience: a mutator raising an internal `AssertionError`
  (real or injected via monkeypatch) must not crash the whole run.
- `curate.py` behavior for both normal and hard-negative modes.
- `report.py` collapse detection.

---

## Design decisions and rationale

**Why absence of a hardening field isn't a finding.** Rules 002/003 only
flag *explicitly* insecure configuration (`privileged: true`, `runAsUser:
0`, `hostNetwork: true`, …), never the *absence* of a hardening field (no
`securityContext` block at all, no explicit `allowPrivilegeEscalation:
false`). If absence were flagged, the overwhelming majority of the real
corpus would be "dirty" before mutation even happens, making the ~35%
clean-negative target unreachable. This also matches how real security
scanners behave in practice.

**Why `runAsUser: 0` is always flagged, with no "is it necessary" logic.**
Whether root is "needed" depends on runtime facts a static manifest can't
capture. Every mainstream posture (Kubernetes Pod Security Standards, CIS
Benchmark, NSA/CISA hardening guide) flags it unconditionally, because
almost every apparent justification has a narrower fix that doesn't need
full root — `NET_BIND_SERVICE` for privileged ports, `fsGroup` or an
init-container for volume permissions. A finding is a statement of fact, not
a verdict; whether the risk is accepted is a governance decision downstream
of the scanner, not inside it.

**Why `normalize.py` fixes rules 002-005 forward but drops KSEC-001 docs.**
`normalize.py`'s job is to deterministically harden a document into the gold
canonical form, regardless of its original state — this maximizes usable
corpus volume (a manifest with `privileged: true` doesn't get discarded, it
gets fixed). Real secrets are the one exception: dropping the whole document
is strictly safer than "fixing" it, since even briefly holding a real leaked
value in memory to externalize it is exactly the kind of transient exposure
the "never write a full secret" constraint guards against.

**Why KSEC-006..011 use a soft `return None` precondition instead of a hard
`assert`.** Rules 001-005 are guaranteed clean by construction — either
`normalize.py` actively fixes them, or the document was already dropped —
so an `assert` firing there indicates a genuine pipeline bug worth crashing
on. Rules 006-011 have no such guarantee: a real corpus document might
already exhibit a selector mismatch, a bad probe port, or a genuine typo "in
the wild" (e.g. an example YAML that was never actually applied). The
mutators for these rules check their own precondition and skip (return
`None`) rather than assert, so a messy real-world document doesn't crash the
whole build — it's just not used as base material for that rule.

**Why the split is grouped by repository, never random.** The real corpus
has many near-duplicate forks of the same manifest across different repos.
A random split would leak near-identical examples across train/val/test,
inflating apparent accuracy. `build.py` buckets by a deterministic hash of
`max_stars_repo_name` instead. Synthetic examples get a unique fake
"repository" per item, since they're already deduplicated and carry no fork
risk.

**Why the round-trip check is not sampled.** `patch` is the cheapest
correctness signal available for this dataset — if `apply_patch(mutated,
patch) != canonical`, the corresponding training example teaches the model
a wrong fix. Checking all of them costs nothing at this scale and catches
mutator bugs immediately rather than shipping a dataset with a silent
labeling defect.

**Why KSEC-001's synthetic secret appears in full in the training input but
never in outputs.** The model needs to see the actual vulnerable pattern to
learn to detect it, so the mutated *input* manifest legitimately contains a
full (synthetic, never real) fake credential. Every *output* field —
`evidence` (masked to 4 chars + `***`) and the `new_resources` Secret's
value (replaced with a placeholder) — never carries the value in full. This
keeps the model's habit consistent with how a real security tool should
behave, even though the constraint technically only needs to protect real
secrets.

**Why command/args credential detection needed its own scanner
(`find_cli_embedded_secrets`).** The generic per-leaf scanner deliberately
excludes any value containing a space, `/`, or `:` — otherwise it would
flag ordinary CLI flags (`--timeout=60s`) and URLs as false positives (an
earlier, more naive version of the entropy heuristic did exactly this and
had to be walked back — see `scanning.py`'s docstring). That exclusion makes
it blind by design to a credential embedded inside a longer command string.
A separate, narrowly-scoped scanner looks specifically inside
`command`/`args` for known-bad substrings: CLI flags (`--password=`),
basic-auth URLs (`user:pass@host`), and bearer tokens.

**Why RFC 6901 escaping matters here specifically.** Every other rule's
JSON Pointer path is built from fixed field names or list indices — always
safe. KSEC-006 is the first rule to put a genuinely free-form key (a label
key) into a path, and Kubernetes label keys routinely contain `/`
(`app.kubernetes.io/name`), which is the JSON Pointer path separator. Left
unescaped, this silently corrupts the patch. `escape_json_pointer_token`
fixes it; a real-corpus fuzz run is what surfaced the bug in the first
place.

**Why disabling a rule is a two-registry change.** Disabling a
rule from *generation* while keeping it in the codebase is deliberately a
two-registry change (`RULES` in `schema.py`, `MUTATORS` in `mutate.py`), not
a config flag or a code deletion. A flag would tempt silently toggling
behavior per-run without updating the prompt; deleting the code would throw
away tested, working detectors/mutators for a decision that may well be
temporary. Because `SYSTEM_PROMPT` is generated from `RULES` and `build.py`
derives its rule set from `MUTATORS`, removing an entry from both is
sufficient — no other file needs to change, and the prompt never claims to
check something the model was never shown.

**Why the KSEC-001 env-var name pool was expanded and partly corpus-
harvested.** The *detection* logic (`SENSITIVE_KEY_RE` in `scanning.py`) is
a general regex, but a fine-tuned model only learns from what it's shown.
The original pool had 8 fixed `SCREAMING_SNAKE_CASE` names; a small model
trained on only those risks memorizing 8 literal strings instead of
inducing the general "credential-shaped identifier" rule, and would likely
miss real names like `MYSQL_ROOT_PASSWORD` or kebab-case `admin-password`
it never saw. Two fixes, both in the codebase now: the curated pool grew to
52 names spanning `SCREAMING_SNAKE_CASE`/`kebab-case`/`camelCase`; and
`dataset/build.py` harvests the *key names* (never the values) from every
real corpus document dropped for containing an actual secret, unioning them
into a much larger, organically diverse pool passed to `mutate_ksec001` via
its `candidate_names` parameter. In one run against ~5,000 corpus rows this
took the pool from 8 to 184 names and produced 68 distinct injected
identifiers in the emitted dataset. A guard test
(`test_every_pool_name_is_actually_detectable_by_scanning`) asserts every
curated name actually matches `SENSITIVE_KEY_RE` — four originally didn't
(`ENCRYPTION_KEY`, `GCP_SERVICE_ACCOUNT_KEY`, `SIGNING_KEY`,
`encryptionKey`, since the regex matches `api_key`/`access_key`/
`private_key` but not a bare `key`), which caused a small number of
probabilistic round-trip failures during fuzzing before being renamed to
regex-matching equivalents (`ENCRYPTION_SECRET_KEY`,
`GCP_SERVICE_ACCOUNT_CREDENTIALS`, `SIGNING_SECRET`,
`encryptionSecretKey`).

**Why `build.py` catches a mutator's `AssertionError` instead of letting it
crash the run.** Discovered via a real failure: a synthetic hard-negative
manifest from the local LLM contained a malformed image reference
(`repo:sha256:<hash>` instead of the valid `repo@sha256:<hash>` digest
syntax) that defeated `mutate_ksec005`'s "strip the tag" branch — the
leftover `sha256` substring still looked like a valid pinned tag, so the
assertion that the mutation always produces a finding failed. `kubeconform`
doesn't catch this at curation time because `image` is an opaque string in
the schema, not a validated reference. Two fixes: `mutate_ksec005` itself
now verifies the stripped image actually looks unpinned and falls back to
an explicit `:latest` deterministically rather than depending on which
random branch fired; and, as defense in depth, `build.py`'s mutation
pool-building loop catches `AssertionError` per document/rule (counted in
`diagnostic.json`'s `mutation_precondition_failures`) so one anomalous
document — from this or any other similarly narrow edge case in any rule —
can't take down a multi-thousand-document production run.

**Why the multi-defect strategy composes mutators instead of writing new
ones.** Once a canonical document is confirmed fully clean
(`detect_all(doc)` empty across every active rule — a stricter check than
the single-defect pipeline runs per-rule, needed here because a multi-defect
example asserts a *complete* finding set, not just one), the active rules
are chained — apply mutator A to the clean doc, then mutator B to A's
already-mutated output, etc. This reuses every rule's existing
injection/detection logic; the composition itself only chains inverse
patches in reverse mutation order and re-derives the combined finding set
via `detect_all` on the final document (intermediate JSON Pointer paths
aren't guaranteed to still be valid after a later mutation touches a
sibling field), asserting one finding per applied mutation.

The first six rules touched disjoint structural areas (env vars,
`securityContext`, image tag, selector labels, probe ports, resource
quantities), so chaining them was safe by construction. KSEC-010 and
KSEC-011 broke that: KSEC-010 shares probes with KSEC-007, and KSEC-011
shares the env list with KSEC-001. The invariant every mutator now keeps
instead is **an injection must never create, erase or hide another rule's
finding**. It is enforced twice. First, each mutator avoids the overlaps it
is known to have:

- KSEC-010 only rewrites a probe whose port KSEC-007 considers consistent,
  and never changes the port — so it can't create or hide a KSEC-007
  finding in either order.
- KSEC-001's env variant never injects a variable KSEC-011 accepts (a
  plaintext `POSTGRES_PASSWORD` would silently satisfy — and erase — a
  KSEC-011 finding), and never appends to an env list with a pending
  KSEC-011 removal (KSEC-011 restores by index insertion, which would shift
  KSEC-001's entry out from under its own `replace` patch).
- KSEC-001's command variant refuses an injection that changes whether
  KSEC-011 applies to the container (a `curl ...` first arg makes a
  postgres server look like a client job, which KSEC-011 skips).

Second, because the overlaps kept turning up in places nobody predicted,
every entry in `MUTATORS` is wrapped by `_preserving_other_rules`, which
runs `detect_all` on the mutated document and on the mutator's round-trip
target and returns `None` (inapplicable) if any *other* rule's finding
count differs. That catches what the per-mutator guards miss, e.g.:

- KSEC-009 typoing a replica's data-volume mount breaks KSEC-011's
  "data directory is pre-populated" exemption, so a KSEC-011 finding
  appears out of nowhere (seen in 183 of 600 compositions on a replica
  fixture before the wrapper).
- KSEC-006 appending its `-xNNN` suffix to a selector value occasionally
  yields a string random-looking enough for KSEC-001's entropy check to flag
  as a credential — an
  unlabeled finding in 113 of 22,420 real-corpus KSEC-006 single-defect
  mutations. This one predates the wrapper and was silently producing
  mislabeled examples.

On the real corpus the wrapper rejects only those 113 KSEC-006 and 2
KSEC-009 mutations; every other rule is unaffected.
`test_multi_mutate.py::test_new_rules_compose_with_the_others` and
`test_mutate.py::test_registered_mutators_never_change_another_rules_findings`
exercise this, and a 3,000-composition stress run across
postgres/mysql/mariadb/redis/replica/generic documents with all 11 rules
enabled finds zero round-trip or finding-count failures.

**Why KSEC-010/011 key on image identity.** Whether an `httpGet` probe or a
missing env var is a bug depends on what the application *is* — the same
field is fine on one image and fatal on another. Both rules read that
context from the manifest itself (the image name) through small lookup
tables in `dataset/k8s.py`, so labels stay programmatic — no cluster access,
no other documents, no human judgment. The two lookups differ on purpose:
`NON_HTTP_IMAGE_PORTS` matches by image basename, so vendor rebuilds
(`bitnami/postgresql`) count too — the wire protocol doesn't change with the
packager; `IMAGE_ENV_CONTRACTS` matches exact image identities instead
(Docker Official Images, `bitnami/*` and `percona/percona-server` on Docker
Hub, SQL Server on `mcr.microsoft.com`), because each packager names its
variables differently (`bitnami/postgresql` reads `POSTGRESQL_PASSWORD` —
and, as an alias, the official `POSTGRES_PASSWORD` its Helm chart sets).
Mirrors and look-alikes (`localhost:32000/percona`, mcr's `oss/bitnami`
mirror, `registry.corp/mssql`) don't match: the contract can't be confirmed
from the name. Each contract lists one or more requirements — SQL Server has
two, `ACCEPT_EULA` (a literal setting, fixed as `value: "Y"`) and the SA
password — plus a replica-role variable where the packager has one: a
Bitnami replica (`*_REPLICATION_MODE=slave`, `MONGODB_REPLICA_SET_MODE=secondary`)
reads the primary's password from a different variable and is skipped.
KSEC-011 also skips containers whose requirement
can't be checked from the manifest: init containers (`pg_isready`-style
waiters), a `command` override or client-style args (`psql`/`pg_dump`
jobs), `envFrom` (the variable may come from a ConfigMap/Secret that isn't
visible), `imagePullPolicy: Never` (a locally built image that merely
shares the official name), and an init container writing to the volume
behind the server's data directory. The official entrypoints only demand
a password when initializing an *empty* data directory (Bitnami validates
on every start, and SQL Server's EULA check always runs, so this exemption
doesn't apply to those), and the replica-cloning
pattern (xtrabackup `clone-mysql`, kubegres `setup-replica-data-directory`)
copies an existing database in first. The last two guards came from
hand-checking the detector's 38 hits on the real corpus: 7 were those two
patterns (false positives); the remaining 31 are genuine, e.g. a postgres
container configured with `PG_USER`/`PG_PASS` instead of the variable
names the image actually reads. Extending the tables to Bitnami, Percona
and SQL Server added 4 hits, all genuine on inspection (e.g. a `percona:5.7`
with flags-only args and no env; an SQL Server container with neither
variable). Run-as-root was considered as a
context-dependent rule and rejected: whether an image can run as non-root,
and as which UID, depends on the image's internals and volume ownership,
which the manifest doesn't reveal — its labels would sometimes be wrong.

**Why the fixes are "fixed forward".** Like KSEC-001's env variant, both new
mutators may return a round-trip target that differs from the corpus
document. KSEC-010 on a container with no probe at all adds a `tcpSocket`
probe to the target, so the fix taught is always "use `tcpSocket`", never
"delete the probe". KSEC-011 restores a `valueFrom`/`*_FILE`/random-password
entry verbatim, but replaces a plaintext placeholder or an insecure
`trust`/allow-empty setting with a `secretKeyRef` + placeholder `Secret`, so
the model is never trained to re-add a plaintext or insecure setting.

**Why documents with a pre-existing semantic finding are dropped.**
`normalize.py` only guarantees rules 001-005 are clean. A real corpus
document can already violate a semantic rule — measured on the full
corpus: 735 of 55,747 canonical documents did (886 KSEC-007 findings, 184
KSEC-009, 52 KSEC-006, 35 KSEC-011, 9 KSEC-008, 4 KSEC-010). Kept, such a document
becomes either a "clean" negative with a real, unlabeled defect, or a
single-defect example whose label misses its second defect — both teach
the model to ignore that defect. `build.py` now drops them after
normalization (`dropped_preexisting_semantic_finding` in
`diagnostic.json`). The multi-defect strategy already rejected them.

**Why multi-document examples come from sibling files.** The model's
input is a whole manifest file, often several documents long (every
scenario file is), yet the corpus stores exactly one document per parquet
row: 268,596 rows, 268,596 documents, and only 22 repository paths occur
twice — real multi-document files were split upstream. A model trained only
on single documents answered `doc: 0` for everything and never learned the
cross-document selector fix; on `10-mongodb.yaml` (defects in document 1) it
returned no findings at all in 3 of 5 runs. The pairing that matters most is
already in the corpus as *sibling files*: 16,526 repository directories hold
both a workload and a Service, and 19,950 of those Services select exactly
one workload there. `build.py::find_sibling_bundles` turns each such pair
into a `[workload, Service]` file in random order (both orders appear in the
scenarios), adding one unrelated sibling (ConfigMap, HPA, ...) 30% of the
time so defects don't always sit at index 0/1. Two details matter:
bundling runs on records *before* structural dedup, because dedup strips
labels and selectors and collapses nearly every Service into a handful of
skeletons; and every member goes through the same secret/normalize/
pre-existing-finding gauntlet as a single document, plus `detect_file` on
the whole bundle, before it's used. Bundles are then deduplicated on their
members' skeletons. `--multi-doc-ratio` (default 0.3) sets their share of
positives and negatives; `0` reproduces the single-document dataset exactly
(bundles draw from their own `random.Random(seed + 1)`).

**Why the cross-document KSEC-006 fix edits the Service.** The four scenario
mismatches (`orionlds`/`orionld`, `sellenium-hub`/`selenium-hub`, ...) are
typos in whichever side was written second, but the model can't know which.
The Service is the side whose only job is to point at the workload's pods,
so the fix always makes it match them; `mutate_ksec006_service` injects the
same shapes the scenarios show (a one-character typo, or a dropped/extra
name segment) and the inverse patch is a `replace` on the Service. The
detector only fires when some workload in the file carries every selector
key — otherwise the Service most likely targets a workload in another file.

**Why the scanner changes for the scenarios.** Three scenario files exposed
detector bugs that corrupted both scoring and training labels, fixed
together: `SCRAM-SHA-256` and `vm.max_map_count=262144` were flagged as
random secrets (all-caps hyphenated identifiers are now exempt like
lowercase ones, and a `name=value` string is judged as the value under that
name — `DB_PASSWORD=hunter2` is still caught); `NEW_RELIC_LICENSE_KEY` wasn't
a sensitive key name; and `ReplicationController` wasn't a pod-template kind,
so every container rule skipped it (1,004 corpus documents). The new
split-flag form (`-dbpwd`, `123456789`) needed care on the other side:
`--secret` and `--*-token` flags very often take a resource *name*
(`--secret webhook-certs`, etcd's `--initial-cluster-token skydns-etcd`), so a
hyphenated lowercase name after those isn't flagged, and a value containing
spaces is treated as prose. Net effect on the deduplicated corpus: 5,793
documents dropped as dirty instead of 6,017 (224 more usable), 39 genuine
split-flag credentials newly caught.

---

## Known limitations

- **KSEC-004 never appears in multi-defect examples.** It is the only
  rule that applies to RBAC documents, and it can only fire once per
  document (its precondition is a clean RBAC doc), so an RBAC doc never
  reaches the 2-defect minimum. KSEC-004 is learned from single-defect
  examples only.
- **KSEC-009 used to misfire on `StatefulSet`s.** A `volumeMount` naming a
  `volumeClaimTemplates` entry is valid, but the detector only looked at
  `volumes` — 817 of the 957 KSEC-009 hits on the real corpus were that
  false positive. `k8s.declared_volume_names` now includes claim templates;
  184 genuine hits remain (dropped as pre-existing findings).
- **KSEC-006..011 don't have a `normalize.py` hardening guarantee.** Unlike
  001-005, the real corpus isn't actively fixed forward for these rules —
  documents already exhibiting the bug are simply skipped as mutation base
  material for that rule. Extending `normalize.py` to also fix these forward
  would recover more usable corpus volume.
- **KSEC-010/011 only know the images in their lookup tables.** A
  database under a custom image name, a private mirror of an official image
  (`registry.corp/postgres`), or an image outside the tables isn't checked.
  KSEC-011 additionally doesn't cover images with a mandatory setting that
  isn't a single env var (`elasticsearch` 8's discovery config), images
  outside its three families (`couchdb`, `quay.io/bitnami/*`), checks
  presence only (`ACCEPT_EULA: "N"` counts as set), and can't see a data directory whose `mountPath` is an unrendered
  template placeholder (e.g. kubegres fixtures' `toBeReplaced`).
- **KSEC-011 has the smallest mutation pool of any active rule** — 295
  documents on the full corpus (KSEC-010: 1,137; the Bitnami/Percona/SQL
  Server extension contributed 36 of the 295), since it needs a known
  database server container that already sets its required variable. At the default `--total 2000` that's plenty (~160 per rule); at
  much larger totals it caps and `_resolve_quotas` redistributes the rest to
  other rules, so KSEC-011 ends up underrepresented.
- **KSEC-006's Service check only sees one file.** A Service whose workload
  lives in another file is never checked, and one that shares no label key
  with any workload in its file is assumed to target such a workload (not
  flagged). The flip side: a file that bundles a Service with an *unrelated*
  workload using the same label keys gets flagged. Measured at directory
  level on the corpus (sibling files, not one file), the check would fire
  on 1,426 of 23,683 Services, mostly for exactly that reason — which is why
  the training bundles are only ever built from matched pairs.
- **`Job`/`CronJob` are excluded from KSEC-006** — their selector is
  normally auto-populated/immutable rather than hand-written, so a mismatch
  there isn't the same class of human error.
- **KSEC-008's quantity parser** covers the common Kubernetes suffixes
  (`m`, `k`/`M`/`G`/`T`/`P`/`E`, `Ki`/`Mi`/`Gi`/`Ti`/`Pi`/`Ei`) but not
  exponential notation (`1e2`), which is valid but rare in practice.
- **"Typo" injection (KSEC-009) is a single-edit-
  distance corruption**, not a model of realistic human typing errors (no
  keyboard-adjacency weighting, no common misspelling dictionary). It's a
  deliberately simple, fully automatable proxy for "looks right but isn't."
- **Harvested KSEC-001 key names aren't restricted to env-var-shaped
  leaves.** `build.py` harvests any leaf key `scanning.py` matched on inside
  a dropped document — usually a realistic env var name, but occasionally
  something else entirely (an annotation key, a filename like
  `config.toml`). These get reinjected as fake env var names too. Harmless
  to correctness (the round-trip and finding are still accurate) but a
  minor realism/precision blemish in a small fraction of KSEC-001 examples.
  Restricting the harvest to keys that co-occur with a sibling `value` field
  (i.e. only the `{name, value}` env-var shape) would tighten this.
- **Multi-document examples are assembled, not observed.** The corpus has no
  multi-document files (see
  [Why multi-document examples come from sibling files](#design-decisions-and-rationale)),
  so every multi-document example is a workload + Service (+ sometimes one
  other sibling) pairing from one repository directory. Files mixing
  several workloads, or unrelated resources in a different order, are only
  represented by the scenarios, not by training data.
- **`scenarios/test_cases.yaml`'s "Erro de Sintaxe/Config" category (9 of its
  40 instances) is entirely outside the current 6-rule taxonomy** — typos,
  an invalid `volumeID`, a nonexistent command binary, and all 4
  cross-document selector mismatches. `finetune/run_scenarios.py` reports
  this honestly (expected ~0% detected today) rather than silently excluding
  it, so the report shows the taxonomy's actual scope, not an inflated
  score.

---

## Project structure

```
.
├── corpus/                       # real Kubernetes manifests (parquet shards, input only)
├── dataset/                      # Part A — corpus → dataset.jsonl
│   ├── schema.py                 # system prompt, taxonomy, dataclasses, validator
│   ├── scanning.py               # real-secret detection
│   ├── k8s.py                    # shared Kubernetes navigation helpers
│   ├── detect.py                 # one detector per rule
│   ├── dedup.py                  # structural deduplication
│   ├── normalize.py              # canonical hardening
│   ├── mutate.py                 # one mutator per rule (single-defect strategy)
│   ├── multi_mutate.py           # composes mutate.py's mutators (multi-defect strategy)
│   ├── build.py                  # pipeline orchestration (--strategy single-defect|multi-defect)
│   ├── view.py                   # extract manifests from .jsonl for inspection
│   ├── test_*.py                 # unit + round-trip tests
│   ├── output/                   # generated (single-defect, default): train/val/test.jsonl, diagnostic.json
│   └── output-multi-defect/      # generated (multi-defect strategy) -- see "Dataset generation strategies"
├── generation/                   # Part B — synthetic manifest generation
│   ├── SETUP.md                  # runtime/model choice and setup steps
│   ├── check_env.py              # GPU/Ollama/model readiness check
│   ├── seeds.py                  # combinatorial seed sampler
│   ├── generate.py               # resumable batched generation via Ollama
│   ├── curate.py                 # filters generated manifests
│   ├── report.py                 # diversity diagnostics
│   ├── test_*.py                 # unit tests
│   └── output/                   # generated: {mode}.jsonl, {mode}.curated.jsonl, report-*/
├── finetune/                     # fine-tuning: export, train, evaluate, infer -- own venv (Python 3.11)
│   ├── export_for_unsloth.py     # renders the real chat template, drops oversized examples
│   ├── train_unsloth.py          # Unsloth QLoRA training, hardware presets, auto-named runs/ folders
│   ├── run_naming.py             # runs/<timestamp>_<strategy>_<model-slug>/ naming convention
│   ├── metrics_logger.py         # TrainerCallback -> metrics.jsonl
│   ├── plot_metrics.py           # metrics.jsonl -> metrics.{csv,png} (runs from main .venv)
│   ├── import_trainer_state.py   # recovers metrics from a checkpoint's trainer_state.json
│   ├── evaluate.py               # scores a checkpoint against dataset/output*/test.jsonl
│   ├── infer.py                  # single-prompt interactive testing CLI
│   ├── run_scenarios.py          # scores a checkpoint against scenarios/test_cases.yaml
│   ├── zero_shot_baseline.py     # same scoring, against a non-fine-tuned base model via Ollama
│   ├── benchmark_throughput.py   # batch-size/grad-accum throughput sweep
│   ├── setup_vm.sh               # one-command training VM bootstrap
│   ├── requirements.txt          # separate, heavier stack (torch/unsloth/trl/...)
│   ├── test_*.py                 # unit tests (pure-logic ones run from main .venv)
│   └── output/                   # legacy fixed output dir; runs/ is now the default
├── scenarios/                    # 10 hand-written real-world manifests, for out-of-distribution testing
│   ├── test_cases.yaml           # per-instance ground truth (40 rows) -- see "Real-world testing"
│   └── *.yaml                    # 1-orion.yaml ... 10-mongodb.yaml
├── runs/                         # auto-generated training run history (gitignored)
├── scripts/                      # compress/decompress_dataset.sh, VM setup helpers
├── pipeline.sh                   # chains generation → curation → report → dataset.build
├── requirements.txt
└── README.md                     # this file
```
