# `demo/` — Kubernetes cluster agent, live demo UIs

Two demo UIs, both fronting a real minikube cluster:

- `streamlit_app.py` (Streamlit): the **cluster agent** — a chat where every
  message goes to the tool-using agent (`agent.py`, below). It lists
  workloads, audits them with the fine-tuned model, answers questions with
  read-only `kubectl`, and applies a fix only after you confirm it. A
  manifest pasted into the chat box is audited directly, with a selectable
  diff of the proposed fixes.
- `app.py` (Flask): manifest-security only — fetch a live resource's YAML (or
  load a hand-crafted `scenarios/*.yaml`, or paste your own), send it to the
  fine-tuned model, see the findings + fix.

| | `streamlit_app.py` (Streamlit) | `app.py` (Flask) |
|---|---|---|
| Style | native chat UI, each tool step shown live | single static page |
| What it does | plain-English requests → agent with tools; pasted manifest → audit | audit a fetched, sample or pasted manifest |
| Changing the cluster | `apply_fix` after a server-side dry run, only on your confirmation | no (preview only) |
| Adapter | pick from the sidebar dropdown, 🔄 Reload without restarting | fixed at startup via `--adapter` |
| Run | `finetune/.venv/bin/streamlit run demo/streamlit_app.py` | `finetune/.venv/bin/python -m demo.app --adapter <path>` |

Both share all model-loading, kubectl and patch logic via **`core.py`**.
Both are **demo tools**, not services: a single model instance, dev servers.
They only *consume* adapters trained by `dataset/` + `finetune/`. Run the
Streamlit app through `run_secure.sh` (below) for login, HTTPS and the audit
trail. Run on its own, and with Flask, there's none of those.

## Secured demo: Keycloak login, HTTPS, isolated agent, audit trail, reset on start

```bash
minikube start
finetune/.venv/bin/python deploy/setup_secure_demo.py   # once (re-runnable)
finetune/.venv/bin/python demo/reset_cluster.py --snapshot   # once: save today's workloads as the baseline
demo/run_secure.sh                                       # every demo: reset, then https://localhost:8501
```

- **Login (Keycloak, OIDC).** Keycloak runs in minikube, in its own
  `keycloak` namespace, at `https://<minikube ip>:30443` (realm `k8s-agent`).
  The app uses Streamlit's built-in OIDC login (`st.login`, with PKCE);
  `setup_secure_demo.py` prints the two demo users' passwords and the admin
  login. Roles come from the ID token's `roles` claim:

  | Role | Can |
  |---|---|
  | `viewer` | chat, list workloads, audit (cluster or pasted manifests) |
  | `operator` | all of the above + confirm changes to the cluster |

  A viewer's `apply_fix` is refused by the agent itself, and the Confirm
  button is disabled.
- **HTTPS.** A local certificate authority (`deploy/.secrets/ca.crt`) signs
  Keycloak's and the app's certificates. The app process trusts it
  (`SSL_CERT_FILE`); your browser warns once unless you import `ca.crt`.
- **The agent can't touch Keycloak.** It runs `kubectl` as its own service
  account (`deploy/agent-rbac.yaml`, kubeconfig in
  `deploy/.secrets/agent.kubeconfig`): read workloads cluster-wide (never
  Secrets), change them only in the managed namespaces (`default`, `demo`,
  `prod`; `AGENT_NAMESPACES` overrides), never delete. The API server
  enforces this: the setup script verifies with `kubectl auth can-i`. On
  top, the agent's tools refuse and hide the `keycloak`, `agent-system` and
  `audit` namespaces.
- **Audit trail (PostgreSQL).** Every change you confirm or decline is
  recorded by `audit_log.py` in PostgreSQL, which runs in minikube in the
  `audit` namespace at `<minikube ip>:30432` over TLS. A record keeps:
  - who decided, with their roles, and when;
  - the model and the request;
  - the manifests before and after, and the diff;
  - findings, patch operations and created resources.

  The guarantees:
  - **Append-only.** The app's role can only insert and read, and triggers
    block edits and deletes for everyone.
  - **No secrets.** Credentials are masked, and Secrets are stored as key
    names only.
  - **Nothing unrecorded.** While the database is unreachable, Confirm is
    disabled and nothing is applied.

  To see the history, use the sidebar's **Show change history**, or run
  `finetune/.venv/bin/python -m demo.audit_log [--show ID]`. Schema and
  entity diagram: [docs/audit-database.md](../docs/audit-database.md).
- **Reset on start.** `run_secure.sh` first restores the managed
  namespaces to `demo/baseline/` (`demo/reset_cluster.py`): the baseline's
  objects are deleted and recreated, and everything the agent created
  (labelled `app.kubernetes.io/managed-by=cluster-agent`, e.g. Secrets from
  credential fixes) is removed. `--no-reset` skips it; `--snapshot` saves a
  new baseline. Keycloak, the agent's namespace and the audit database
  are never touched, so the change history survives resets.
- **Secrets stay local.** Keys, passwords, the client secret, the agent
  kubeconfig and the database URL live in `deploy/.secrets/` and
  `.streamlit/secrets.toml`, both git-ignored. `DEMO_AUTH=off` runs the app without login, for development.

## The agent (`agent.py`) — the default chat mode

**🤖 Agent** turns a plain-English request into a sequence of tool calls:
"check the storm deployment in demo and fix it" → audit it → prepare the
fix → validate it with a server-side dry run → ask you to confirm → apply →
re-audit. Each step appears live in the chat (expand it to see the tool's
output). When a fix is ready, the agent explains the proposed changes, and a
box below the chat shows the findings, the diff and **✅ Confirm and apply /
✖️ Decline**. You can keep asking about it ("why change the image?", "which
problem is worst?") -- the planner sees a plain-language description of every
change (never the old plaintext value), and the fix stays pending until you
click Confirm or Decline; a second fix can't be proposed meanwhile.

The model the sidebar selects by default is `DEMO_ADAPTER`
(`run_secure.sh` sets it to the v6 7B run), not simply the newest run, so
copying a new run into `runs/` never silently changes the demo's model.

- **No extra model.** The *planner* is the base Qwen2.5-Coder-7B-Instruct
  underneath the fine-tuned adapter (adapter temporarily disabled, like 💬
  Chat), using its native tool calling; the *auditor* is the same model with
  the adapter on, exposed as the `audit_manifest` tool. One model in GPU
  memory, switching is instant.
- **Tools:** `list_resources`, `get_manifest`, `audit_manifest`,
  `run_kubectl` (read-only verbs only; Secret contents never printed), and
  `apply_fix` — the only way to change the cluster: dry-run first, then it
  pauses until you confirm. New resources (e.g. the Secret a credential fix
  needs) are created first, and Confirm stays disabled while they still
  contain a placeholder value.
- **Guardrails for a 7B planner** (each added after an end-to-end run showed
  the failure): tool calls written as a ```json block are accepted; an
  answer that announces an action ("Let me check…") or tells you to run a
  `kubectl` command is retried with the reply pre-filled with `<tool_call>`;
  a fix you asked for is called by the loop itself once an audit has one
  ready; a declined fix is never re-proposed in the same turn; a new message
  while a change is waiting declines it and moves on; at most 8 steps.
- **Latency** on an RTX 4060 Ti: ~2–6 s per planning step, ~9–10 s per
  audit; "check X and fix it" reaches the confirmation in ~15 s.

The loop is tested without a GPU or cluster (scripted planner, fake kubectl)
in `test_agent.py`.

## How it works (`core.py`)

- The base model + LoRA adapter are loaded **once**, via
  `finetune.evaluate.load_model` — the exact call `finetune/evaluate.py`
  itself uses. Unsloth resolves the base model from the adapter's
  `adapter_config.json`.
- `list_resources` / `fetch_resource_yaml` shell out to `kubectl` against
  whatever cluster the current context points at (typically minikube).
  Fetched YAML has cluster-assigned noise stripped (`status`,
  `managedFields`, `resourceVersion`, …) so it looks like the spec-only YAML
  the model was trained on.
- Resource kinds are limited to what `dataset/k8s.py::get_pod_spec` actually
  extracts a PodSpec from (Pod/Deployment/StatefulSet/DaemonSet/ReplicaSet/
  Job/CronJob). Service and ReplicationController are excluded on purpose —
  the model was never trained on either, so surfacing one would look like a
  false negative instead of the honest scope boundary it is.
- `apply_patch_to_manifest` applies **any subset** of the model's patch ops
  with `jsonpatch` — the same round-trip check `dataset/build.py` uses at
  generation time. It catches any exception the underlying patch/pointer
  libraries raise (not just `jsonpatch.JsonPatchException` — see "Two bugs
  found by testing" below) and reports `(None, error)` instead of letting a
  bad model prediction crash the UI. `patch_ops_for_finding` maps each
  finding to the patch op(s) that fix it (by doc + path-is-ancestor-or-equal,
  since a mutator sometimes replaces a whole list element to fix something
  reported one level deeper, e.g. `.../env/1` fixing `.../env/1/value`) —
  this is what lets the Streamlit UI offer "apply just this finding".
- `apply_yaml_to_cluster` runs `kubectl apply -f -`, used by the Streamlit UI
  to push a fixed manifest, or a model-generated `new_resources` entry (e.g.
  a `Secret`), back to the live cluster.
- `list_adapters` only returns `runs/*/lora_adapter` whose `run_info.json`
  has `strategy` in `{single-defect, multi-defect}` — `runs/` is shared with
  other training tasks in this repo (e.g. `monitoring_training`), which use a
  different system prompt/output schema entirely and would just produce
  garbage here.
- `classify_intent`: a chat message that parses as YAML with a top-level
  `kind` → `"manifest"` (audited directly); anything else goes to the agent.

## Run it

```bash
# 1. start (or use an existing) minikube cluster
minikube start

# 2. optional but recommended: seed it with the hand-crafted scenario
#    manifests (each has a known planted issue -- see scenarios/test_cases.yaml)
#    so the "live cluster" picker has real, flawed workloads instead of an
#    empty/system-only cluster
./demo/seed_minikube.sh

# 3. install the extra dependencies into the SAME venv finetune/*.py uses --
#    torch/unsloth/transformers/jsonpatch/pyyaml are already there
finetune/.venv/bin/python -m pip install -r demo/requirements.txt

# 4a. Flask UI -- adapter fixed at startup (loads in ~20-30s on a 7B model)
finetune/.venv/bin/python -m demo.app \
    --adapter runs/20260907-1545_multi-defect_qwen2_5-coder-7b-instruct-bnb-4bit/lora_adapter
# open http://127.0.0.1:5000

# 4b. OR the interactive Streamlit chat UI -- pick the adapter from a sidebar dropdown
finetune/.venv/bin/streamlit run demo/streamlit_app.py
# opens automatically, usually http://localhost:8501
```

Verified end to end on this machine (RTX 4060 Ti, 8GB), against real
minikube, with both real models loaded together (~40s startup, ~500MB-6GB
VRAM used, comfortably under 8GB):
- Flask: fetched the live `demo/fiware-orionld` Deployment → analyzed →
  got the two `KSEC-001` credential findings + `KSEC-005` unpinned-image →
  a patch externalizing both credentials into a new `Secret` → correctly
  patched manifest.
- Streamlit chat, manifest side: same fetch/send, then **unchecked one of
  the two `KSEC-001` findings** and confirmed the diff and "Apply fix +
  re-analyze" only carried the *other* two fixes through into the next turn,
  leaving the unchecked one's plaintext value untouched; clicked **Push to
  cluster** and confirmed with a direct `kubectl get` that the live
  Deployment's image and env were actually updated; edited a `new_resources`
  Secret's placeholder value and applied it, confirmed the real Secret
  existed in the cluster with the edited value (base64-decoded correctly);
  pasted `scenarios/3-mysql.yaml` directly into the chat box and got the
  `MYSQL_ROOT_PASSWORD` (`KSEC-001`) and unpinned-image (`KSEC-005`) findings
  back as a normal turn.
- Streamlit chat, cluster-agent side: typed "list all pods in the demo
  namespace" → got `kubectl get pods --namespace=demo` **and its real
  output** (the actual pod list); typed "create a namespace called
  demo-video-test" → got `kubectl create namespace demo-video-test`, ran it,
  and confirmed independently with `kubectl get namespace` that it was
  really created.

## Flags / config

```bash
finetune/.venv/bin/python -m demo.app --adapter <path> \
    --max-seq-length 4096 --max-new-tokens 512 --host 127.0.0.1 --port 5000
```

`--host 0.0.0.0` to show the Flask UI from another device on the same network
(e.g. a projector) — still an unauthenticated dev server, trusted networks only.

The Streamlit UI has no CLI flags: the adapter is picked in the sidebar
(`max_seq_length` is fixed at 4096, as in training).

## Files

| File | Role |
|---|---|
| `core.py` | shared model-loading, kubectl, and patch-application logic — used by both UIs |
| `agent.py` | the tool-using agent: planner + tools + confirmation-gated `apply_fix` (Streamlit 🤖 Agent mode) |
| `app.py` | Flask UI: chat-style, preview-only |
| `streamlit_app.py` | Streamlit UI: the agent chat (live tool steps, confirm/decline changes) + direct audit of a pasted manifest |
| `static/index.html` | the Flask UI's entire frontend — vanilla HTML/CSS/JS, no build step, no CDN (works offline) |
| `seed_minikube.sh` | applies `scenarios/*.yaml` into a `demo` namespace so there's real, flawed content to fetch |
| `requirements.txt` | the dependencies (`flask`, `streamlit`, `Authlib`, `psycopg`) not already in `finetune/requirements.txt` |
| `audit_log.py` | PostgreSQL audit trail: records every decided change (redacted), reads history; also a CLI |
| `test_core.py` | unit tests for `core.py`'s pure patch-matching/application logic — run from either venv (`pytest demo/`) |
| `run_secure.sh` | starts the secured demo: reset, then HTTPS + Keycloak login, kubectl as the agent's service account |
| `reset_cluster.py` | restores the managed namespaces to `demo/baseline/` (`--snapshot` saves a new one) |
| `baseline/` | the saved default state of the demo workloads |
| `test_agent.py` | unit tests for the agent loop, tool parsing and guardrails, with a scripted planner and a fake cluster |
| `test_audit_log.py` | redaction, what's recorded for each outcome, fail-closed behaviour; plus an opt-in test against a real PostgreSQL (`AUDIT_TEST_DATABASE_URL`) |

## Two bugs found by testing, not by writing

**1. A silently-inert widget update.** An earlier version of `streamlit_app.py`
had a persistent manifest text box (`st.text_area(..., key="manifest_box")`)
and wrote fixes to a *different* session key (`manifest_text`). Streamlit
raised nothing — it just never updated the box. The current chat design
sidesteps the whole class of bug: sending a manifest appends to a plain list
(`st.session_state.messages`), which has none of `st.text_area`'s
"can't write to an already-instantiated widget's key" restriction. Caught
originally by `streamlit.testing.v1.AppTest` driving the real click sequence
— a page-load-only smoke test would have missed it.

**2. An uncaught exception from a bad model prediction.** Re-sending a
partially-fixed manifest for a second analysis pass, the model attempted a
`KSEC-006` (selector mismatch) fix across the Deployment/Service documents in
`1-orion.yaml` — a case `dataset/detect.py`'s single-document-only
`detect_ksec006` was never trained to patch correctly (documented in the main
README's known limitations). Its patch pointed at `.../selector/matchLabels`
on the Service document, whose `spec.selector` is actually a flat map with no
`matchLabels` wrapper — `jsonpointer.JsonPointerException`, a type
`core.apply_patch_to_manifest`'s `except jsonpatch.JsonPatchException` didn't
catch, crashing the whole Streamlit session. Fixed by catching `Exception`
broadly around the `jsonpatch.apply_patch` call — deliberate, not sloppy: the
patch is model output, so *some* malformed/inapplicable shape is an expected
outcome, not a bug, and `finetune/evaluate.py`'s own `_apply_patch_safe` uses
the same broad catch for the same reason. Regression-tested in `test_core.py`.

## Limitations (by design — these are demos)

- Single global model instance: concurrent requests/sessions queue up, not
  parallelize.
- No rate limiting. Without `run_secure.sh` (or with `DEMO_AUTH=off`) there's
  no login, no HTTPS and no audit trail, and `kubectl` uses your own
  context. That's fine for your own minikube, but don't expose it beyond a
  trusted demo network.
- The model only performs its one trained task (manifest → JSON audit) — the
  Flask UI is chat-*styled* and the Streamlit UI is a real chat widget, but
  neither is a general chatbot: every turn you send is treated as a manifest.
- `kubectl`/minikube calls have short timeouts (5-6s) and fail into an error
  message rather than hanging — if the cluster is genuinely slow rather than
  down, the picker may show a stale error until you retry.
- When a model-generated fix externalizes a credential into a `new_resources`
  Secret, the Deployment's patch and the Secret are two separate pieces the
  UI applies separately — pushing the Deployment fix without also applying
  the Secret leaves it referencing a Secret that doesn't exist yet (real
  Kubernetes behavior, not a bug in the demo: apply the Secret too).
- The model can propose a patch that doesn't apply (see "Two bugs found by
  testing" above) — shown as an inline error in that turn rather than
  crashing, but it means "Apply fix + re-analyze" doesn't always have a next
  turn to show.
