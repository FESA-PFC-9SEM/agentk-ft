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
| **KSEC-001** | Plaintext credential — password, token, API key, connection string, private key. Two injection shapes: as an env var `{name, value}` pair, or embedded in a `command`/`args` CLI flag or basic-auth URL. | Env variant: externalize to a `Secret` + `secretKeyRef`. Command variant: remove the offending arg. | **active** |
| **KSEC-002** | Insecure `securityContext` — `privileged: true`, `runAsUser: 0`, `allowPrivilegeEscalation: true`, or added `capabilities`. | Remove/revert the offending field. | **active** |
| **KSEC-003** | Host access — `hostNetwork`/`hostPID`/`hostIPC: true`, or a `hostPath` volume mounting a sensitive host path (`/`, `/etc`, `/var/run/docker.sock`, `/proc`, `/root`, `/var/lib/kubelet`, `/boot`, `/sys`, `/home`). | Remove the field, or remove the volume + its `volumeMount`. | *disabled* |
| **KSEC-004** | Permissive RBAC — a wildcard `"*"` in `apiGroups`/`resources`/`verbs` on a `Role`/`ClusterRole`, or a `RoleBinding`/`ClusterRoleBinding` granting `cluster-admin`. | Revert the wildcard or the binding's `roleRef.name`. | *disabled* |
| **KSEC-005** | Unpinned container image — missing tag or `:latest` (digest-pinned images are not flagged). | Revert to the original pinned tag. | **active** |
| **KSEC-006** | Selector/label mismatch — `spec.selector.matchLabels` doesn't match the pod template's own labels, breaking Service/Deployment routing and discovery. | Revert the selector label value. | **active** |
| **KSEC-007** | Probe port mismatch — a `livenessProbe`/`readinessProbe`/`startupProbe` (`httpGet` or `tcpSocket`) targets a port not declared in the container's `ports` (checked for both numeric and named ports). | Revert the probe's port. | **active** |
| **KSEC-008** | `resources.requests` exceeds `resources.limits` for `cpu` or `memory` — passes schema-only validation (`kubeconform`) but is rejected by the Kubernetes API at admission time. | Revert the request value. | **active** |
| **KSEC-009** | Dangling volume reference — a `volumeMount.name` doesn't match any declared `volumes[].name`, generated via a single human-plausible character edit (transpose/delete/duplicate) of a real volume name. | Revert to the correct name. | *disabled* |

Severities are assigned per finding sub-case (e.g. `privileged: true` is
`critical`, `allowPrivilegeEscalation: true` is `medium`) — see
`dataset/detect.py` for the exact mapping.

KSEC-001, KSEC-002 and KSEC-005 are "security" rules in the strict sense.
KSEC-006 through KSEC-009 are semantic/configuration-correctness checks
added later, sharing the same rule-ID numbering and response schema by
design decision (see [Design decisions](#design-decisions-and-rationale)).

### Active vs. disabled rules

**Only 6 rules currently generate training examples: KSEC-001, 002, 005,
006, 007, 008.** KSEC-003, KSEC-004 and KSEC-009 are fully implemented —
`detect_ksec003/004/009` and `mutate_ksec003/004/009` exist and are unit-
and round-trip-tested exactly like the active rules — but deliberately
excluded from `dataset/schema.py`'s `RULES` dict and `dataset/mutate.py`'s
`MUTATORS` registry. Two consequences of that specific mechanism, both
intentional:

- `SYSTEM_PROMPT` is generated dynamically from `RULES`, so it lists only
  the 6 active rules — the model is never told to detect something it was
  never shown a labeled example of.
- `dataset/build.py` derives its rule set, quotas, and mutation pool from
  `MUTATORS`, so disabling a rule here is the only change needed; no other
  file requires editing.

Re-enabling a disabled rule means adding its entry back to both `RULES` and
`MUTATORS` — a small, explicit code change, not a config flag.

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
| `k8s.py` | Shared, read-only Kubernetes navigation helpers: `get_pod_spec` (resolves the PodSpec location per kind, including the 4-levels-deep CronJob case), `iter_containers`, `get_selector_match_labels`, `get_template_labels`, `get_container_ports`, `parse_quantity` (Kubernetes resource-quantity parser), image tag helpers, sensitive-hostpath check. |
| `detect.py` | One read-only detector per rule (`detect_ksec001`..`detect_ksec009` — all 9 exist, regardless of active/disabled status), plus `detect_structural` (002-005), `detect_semantic` (006-009), and `detect_all`. Used in three places: filtering dirty corpus docs, mutator preconditions, and post-normalize assertions. |
| `dedup.py` | Structural deduplication — reduces a document to a "skeleton" (strips names/namespaces/labels/annotations/selectors, collapses leaf values to placeholders) and hashes it. Two manifests differing only in naming collide. |
| `normalize.py` | Produces the canonical hardened form — the gold target. Deterministic, idempotent. Only fixes rules 002-005 forward (see rationale below); KSEC-001 docs are dropped rather than fixed. |
| `mutate.py` | One mutator per rule (`mutate_ksec001`..`mutate_ksec009` — all 9 implemented), but `MUTATORS` — the registry `build.py` actually reads from — only lists the 6 active ones (see [Active vs. disabled rules](#active-vs-disabled-rules)). Each mutator takes a canonical doc + `random.Random` and returns a `MutationResult(mutated_doc, canonical, findings, patch, new_resources)` or `None` if not applicable. `mutate_ksec001` additionally accepts an optional `candidate_names` override (see below). |
| `build.py` | Orchestrates the whole pipeline: load → filter → dedup → drop dirty (harvesting credential key names along the way) → normalize → mutate with per-rule quotas → 100% round-trip check → write `train/val/test.jsonl`, split by source repository. Catches a mutator's internal `AssertionError` per document/rule rather than crashing the whole run on one anomalous document (see rationale below). |
| `view.py` | Utility to extract manifests from any `.jsonl` (dataset or generation output) into individual `.yaml` files for manual inspection — no JSON archaeology required. |

### Part B — `generation/`

Local LLM generates **clean input manifests only** — never labels. Exists to
fill a gap in the real corpus: "hard negative" material (clean manifests
that *look* suspicious). It also supports an `rbac` generation mode, built
to address RBAC scarcity for KSEC-004 — currently unused by `pipeline.sh`
since KSEC-004 is disabled (see
[Active vs. disabled rules](#active-vs-disabled-rules)); run it manually if
you re-enable that rule.

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

# --mode rbac exists but isn't run by pipeline.sh (KSEC-004 is disabled --
# only useful if you re-enable it, see "Active vs. disabled rules")
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

### Compressing dataset files

A full-scale `train.jsonl` runs tens to hundreds of MB — worth compressing
for storage or transfer to wherever fine-tuning happens. `-k` keeps the
original file alongside the compressed one (compression is not destructive).

```bash
# compress (gzip, keeps the .jsonl around too)
gzip -k dataset/output/*.jsonl

# decompress (also keeps the .jsonl.gz around)
gzip -dk dataset/output/*.jsonl.gz
```

`zstd` is a faster alternative with a comparable or better ratio, if
available (`which zstd`):

```bash
# compress
zstd -k dataset/output/train.jsonl -o dataset/output/train.jsonl.zst

# decompress
zstd -dk dataset/output/train.jsonl.zst -o dataset/output/train.jsonl
```

Measured on this project's `val.jsonl`: gzip took it from 11 MB to 947 KB
(~11.6x).

---

## Testing

```bash
export PATH="$PATH:$(go env GOPATH)/bin"   # for the two kubeconform integration tests
.venv/bin/python -m pytest dataset/ generation/ -q
```

367 tests, covering:
- Unit tests per detector/mutator, active and disabled alike (including the
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

**Why KSEC-006..009 use a soft `return None` precondition instead of a hard
`assert`.** Rules 001-005 are guaranteed clean by construction — either
`normalize.py` actively fixes them, or the document was already dropped —
so an `assert` firing there indicates a genuine pipeline bug worth crashing
on. Rules 006-009 have no such guarantee: a real corpus document might
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

**Why KSEC-003/004/009 are implemented but not registered.** Disabling a
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

---

## Known limitations

- **KSEC-003, KSEC-004 and KSEC-009 currently generate no training
  examples.** Fully implemented and tested, but excluded from `RULES`/
  `MUTATORS` — see [Active vs. disabled rules](#active-vs-disabled-rules).
- **KSEC-006..009 don't have a `normalize.py` hardening guarantee.** Unlike
  001-005, the real corpus isn't actively fixed forward for these rules —
  documents already exhibiting the bug are simply skipped as mutation base
  material for that rule. Extending `normalize.py` to also fix these forward
  would recover more usable corpus volume. (Only 006/007/008 matter for this
  today, since 009 is disabled.)
- **KSEC-006 only checks single-document selector/template consistency.**
  A `Service`'s `spec.selector` targeting a `Deployment` in a *different*
  document isn't checked — only workload controllers where the selector and
  pod template live in the same resource (`Deployment`, `StatefulSet`,
  `DaemonSet`, `ReplicaSet`).
- **`Job`/`CronJob` are excluded from KSEC-006** — their selector is
  normally auto-populated/immutable rather than hand-written, so a mismatch
  there isn't the same class of human error.
- **KSEC-008's quantity parser** covers the common Kubernetes suffixes
  (`m`, `k`/`M`/`G`/`T`/`P`/`E`, `Ki`/`Mi`/`Gi`/`Ti`/`Pi`/`Ei`) but not
  exponential notation (`1e2`), which is valid but rare in practice.
- **"Typo" injection (KSEC-009, currently disabled) is a single-edit-
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
- **`--mode rbac` generation would currently be pointless.** It exists to
  feed KSEC-004, which is disabled — `pipeline.sh` no longer runs it by
  default for exactly this reason; it's still available to run manually if
  KSEC-004 gets re-enabled.

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
│   ├── mutate.py                 # one mutator per rule
│   ├── build.py                  # pipeline orchestration
│   ├── view.py                   # extract manifests from .jsonl for inspection
│   ├── test_*.py                 # unit + round-trip tests
│   └── output/                   # generated: train.jsonl, val.jsonl, test.jsonl, diagnostic.json
├── generation/                   # Part B — synthetic manifest generation
│   ├── SETUP.md                  # runtime/model choice and setup steps
│   ├── check_env.py              # GPU/Ollama/model readiness check
│   ├── seeds.py                  # combinatorial seed sampler
│   ├── generate.py               # resumable batched generation via Ollama
│   ├── curate.py                 # filters generated manifests
│   ├── report.py                 # diversity diagnostics
│   ├── test_*.py                 # unit tests
│   └── output/                   # generated: {mode}.jsonl, {mode}.curated.jsonl, report-*/
├── pipeline.sh                   # chains generation → curation → report → dataset.build
├── requirements.txt
└── README.md                     # this file
```
