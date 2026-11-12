"""
Streamlit chat UI for the Kubernetes cluster agent (demo/agent.py).

Every message goes to the agent: it plans with the base model, uses tools
(list workloads, fetch and audit manifests with the fine-tuned model,
read-only kubectl) and applies a fix only after you confirm it here. A
manifest pasted into the chat box is audited directly instead, with a
selectable diff of the proposed fixes.

Login is Keycloak (OIDC, Streamlit's st.login) when .streamlit/secrets.toml
has an [auth] section (deploy/setup_secure_demo.py writes it; start the app
with demo/run_secure.sh for HTTPS). Keycloak role "viewer" can chat, list
and audit; "operator" can also confirm changes to the cluster. DEMO_AUTH=off
runs without login, for local development only.

Every change the user confirms or declines is recorded in the PostgreSQL
audit trail (demo/audit_log.py) when [audit] database_url is configured --
and while that database is unreachable, nothing is applied. The sidebar's
"Show change history" lists what was recorded.

The model is loaded once, at startup, and stays resident: Unsloth doesn't
release GPU memory for a model once loaded in this process, so switching
adapters means 🔄 Reload (or restarting the app).

Requires the SAME venv finetune/*.py uses (torch/unsloth/transformers) plus
streamlit -- see demo/requirements.txt.

Usage:
    finetune/.venv/bin/python -m pip install -r demo/requirements.txt
    finetune/.venv/bin/streamlit run demo/streamlit_app.py
"""

from __future__ import annotations

import difflib
import os
import sys
from pathlib import Path

# `streamlit run` sets sys.path[0] to this file's own directory (demo/), not
# the repo root -- so `from demo import core` below fails with "No module
# named 'demo'" unless the repo root is also on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import streamlit.components.v1 as components

from demo import agent as agent_mod
from demo import audit_log as audit_mod
from demo import core

st.set_page_config(page_title="Kubernetes Cluster Agent", page_icon="🛡️", layout="wide")

SEVERITY_ICON = {"critical": "🟥", "high": "🟧", "medium": "🟨", "low": "🟦"}
GREETING = "Hi! I'm your Kubernetes cluster agent."
TOOL_LABELS = {
    "list_resources": "Listing workloads",
    "get_manifest": "Fetching manifest",
    "audit_manifest": "Auditing with the fine-tuned model",
    "apply_fix": "Preparing the fix (server-side dry run)",
    "run_kubectl": "Running kubectl",
}


@st.cache_resource(show_spinner=False)
def get_model(adapter_path: str, max_seq_length: int):
    return core.load_model(adapter_path, max_seq_length)


def init_state() -> None:
    st.session_state.setdefault("messages", [{"role": "assistant", "kind": "greeting", "content": GREETING}])


init_state()


def auth_configured() -> bool:
    try:
        return "auth" in st.secrets
    except Exception:  # no secrets file at all
        return False


def require_login() -> list[str]:
    """Stops the page until the user is logged in through Keycloak; returns
    their realm roles (the "roles" claim the realm's mapper adds)."""
    if os.environ.get("DEMO_AUTH", "on") == "off":
        return ["viewer", "operator"]
    if not auth_configured():
        st.error("Login isn't configured. Run `finetune/.venv/bin/python deploy/setup_secure_demo.py`, then `demo/run_secure.sh`.")
        st.stop()
    if not st.user.is_logged_in:
        st.title("Kubernetes Cluster Agent")
        st.write("Sign in to talk to the agent.")
        st.button("🔐 Log in with Keycloak", on_click=st.login, type="primary")
        st.stop()
    roles = st.user.get("roles") or []
    return [roles] if isinstance(roles, str) else list(roles)


roles = require_login()
is_operator = "operator" in roles


@st.cache_resource(show_spinner=False)
def get_audit_log(url: str):
    return audit_mod.AuditLog(url)


def current_audit_log():
    """The audit trail, or None when it isn't configured (no login setup)."""
    url = audit_mod.database_url()
    return get_audit_log(url) if url else None


def current_actor() -> audit_mod.Actor:
    if os.environ.get("DEMO_AUTH", "on") == "off":
        return audit_mod.Actor("local-dev", os.environ.get("USER", "local-dev"), None, tuple(roles))
    return audit_mod.Actor(
        str(st.user.get("sub") or "?"),
        str(st.user.get("preferred_username") or st.user.get("name") or "?"),
        st.user.get("email"),
        tuple(roles),
    )


def get_agent(model, tokenizer, adapter_key: str, max_seq_length: int):
    """One agent per session (its conversation and pending change live in
    it), rebuilt when the loaded model changes."""
    if st.session_state.get("agent_key") != adapter_key or "agent" not in st.session_state:
        st.session_state.agent = agent_mod.build_agent(model, tokenizer, max_seq_length)
        st.session_state.agent_key = adapter_key
    # Checked again on every request: the role and identity come from the
    # login, not from anything the chat can change.
    toolbox = st.session_state.agent.toolbox
    toolbox.can_apply = is_operator
    toolbox.audit_log, toolbox.actor, toolbox.adapter = current_audit_log(), current_actor(), adapter_key
    return st.session_state.agent


def _describe_call(name: str, arguments: dict) -> str:
    target = "/".join(str(arguments[k]) for k in ("namespace", "kind", "name") if arguments.get(k))
    detail = arguments.get("command") or target
    return f"{TOOL_LABELS.get(name, name)}" + (f" — `{detail}`" if detail else "")


def run_agent_turn(events) -> None:
    """Consumes an agent event stream, showing each step live, and appends
    the turn to the chat history."""
    steps, final = [], None
    with st.status("Agent working...", expanded=True) as status:
        for event in events:
            if event.type == "tool_call":
                st.write("🔧 " + _describe_call(event.name, event.arguments))
                steps.append({"tool": event.name, "arguments": event.arguments, "result": None})
            elif event.type in ("tool_result", "confirm"):
                if not steps or steps[-1]["result"] is not None:
                    steps.append({"tool": event.name, "arguments": {}, "result": None})
                steps[-1]["result"] = event.text
            elif event.type == "final":
                final = event.text
        status.update(label="Done" if final is not None else "Waiting for your confirmation", state="complete", expanded=False)
    agent = st.session_state.agent
    st.session_state.messages.append(
        {"role": "assistant", "kind": "agent", "steps": steps, "final": final}
    )
    st.rerun()


def send_agent_message(agent, text: str) -> None:
    st.session_state.messages.append({"role": "user", "kind": "chat", "content": text})
    run_agent_turn(agent.ask(text))


def send_manifest(model, tokenizer, manifest_text: str, source_label: str) -> None:
    """Audits a manifest directly with the fine-tuned model."""
    st.session_state.messages.append({"role": "user", "kind": "manifest", "content": manifest_text, "source_label": source_label})
    with st.spinner("Analyzing..."):
        raw = core.generate_response_text(model, tokenizer, core.SYSTEM_PROMPT, manifest_text, 1280)
    parsed, errors = core.parse_model_output(raw)
    selected = set(range(len(parsed["findings"]))) if parsed else set()
    st.session_state.messages.append(
        {"role": "assistant", "kind": "analysis", "raw": raw, "parsed": parsed, "errors": errors, "selected": selected}
    )
    st.rerun()


# ------------------------------------------------------------------ sidebar --
with st.sidebar:
    if os.environ.get("DEMO_AUTH", "on") != "off":
        st.header("👤 User")
        st.markdown(f"**{st.user.get('preferred_username') or st.user.get('name') or '?'}** — {', '.join(roles) or 'no role'}")
        if not is_operator:
            st.caption("Viewer: you can list and audit; changing the cluster needs the operator role.")
        st.button("Log out", on_click=st.logout, use_container_width=True)
        st.divider()

    st.header("🛡️ Model")
    adapters = core.list_adapters()
    adapter_path = st.selectbox("LoRA adapter (run)", adapters) if adapters else st.text_input("LoRA adapter path (no runs/*/lora_adapter found)")
    max_seq_length = 4096

    model = tokenizer = None
    if adapter_path:
        # Dropdown paths are repo-root-relative; resolve against REPO_ROOT so
        # this works whatever the process's cwd is.
        adapter_abs_path = str(core.REPO_ROOT / adapter_path)
        if st.button("🔄 Reload model", use_container_width=True):
            get_model.clear()
        with st.spinner(f"Loading base model + adapter ({adapter_path}) ..."):
            model, tokenizer = get_model(adapter_abs_path, max_seq_length)
        st.success(f"Ready: {core.base_model_for_adapter(adapter_abs_path) or '?'}")
    else:
        st.warning("No adapter selected.")

    st.divider()
    st.header("☸️ Cluster")
    ctx = core.kubectl_context()
    reachable = core.cluster_reachable() if ctx else False
    st.markdown(("🟢" if reachable else "🔴") + f" context: `{ctx or 'unavailable'}`")
    if not reachable:
        st.caption("`minikube start` (and optionally `demo/seed_minikube.sh`) to give the agent a cluster to work on.")

    st.divider()
    st.header("🗂️ Audit log")
    audit_log = current_audit_log()
    if audit_log is None:
        st.markdown("⚪ not configured — changes aren't recorded")
    elif audit_log.ping():
        st.markdown("🟢 PostgreSQL connected")
        st.toggle("Show change history", key="show_history")
    else:
        st.markdown("🔴 PostgreSQL unreachable — no change can be applied until it's back")

    if len(st.session_state.messages) > 1:
        st.divider()
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "kind": "greeting", "content": GREETING}]
            st.session_state.pop("agent", None)
            st.rerun()

# --------------------------------------------------------------------- title --
st.title("Kubernetes Cluster Agent")

STATUS_ICON = {"applied": "✅", "declined": "✖️", "rejected": "⛔", "failed": "❌"}


# ------------------------------------------------------------ change history --
def render_history(audit_log) -> None:
    """The recorded changes, newest first, and one of them in full."""
    rows = audit_log.history(100)
    with st.expander(f"📜 Change history ({len(rows)} most recent)", expanded=True):
        if not rows:
            st.info("No recorded changes yet.")
            return
        st.dataframe(
            [{"#": r["id"], "when": r["decided_at"].strftime("%Y-%m-%d %H:%M:%S"), "status": f"{STATUS_ICON.get(r['status'], '')} {r['status']}",
              "user": r["username"], "resource": f"{r['kind']} {r['namespace']}/{r['name']}", "findings": r["findings"],
              "operations": r["operations"]} for r in rows],
            hide_index=True, width="stretch",
        )
        change_id = st.selectbox("Details of change", [r["id"] for r in rows], format_func=lambda i: f"#{i}")
        c = audit_log.change(change_id)
        st.markdown(
            f"**#{c['id']} {STATUS_ICON.get(c['status'], '')} {c['status']}** — `{c['kind']} {c['namespace']}/{c['name']}`, "
            f"by **{c['username']}** ({', '.join(c['user_roles'])}) at {c['decided_at']:%Y-%m-%d %H:%M:%S}  \n"
            f"Model: `{c['model_adapter'] or '?'}`  \nRequest: {c['user_request'] or '—'}  \nResult: {c['result_message']}"
        )
        for f in c["findings"]:
            st.caption(f"{SEVERITY_ICON.get(f['severity'], '⬜')} **{f['rule_id']}** — {f['message']} `{f['path']}`")
        for r in c["created_resources"]:
            st.caption(f"Created {r['kind']} `{r['namespace']}/{r['name']}` (keys: {', '.join(r['data_keys']) or '—'})")
        st.code(c["diff"] or "(no textual changes)", language="diff")


if st.session_state.get("show_history") and audit_log is not None:
    render_history(audit_log)

# ------------------------------------------------------------------ history --
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["kind"] in ("greeting", "chat"):
            st.markdown(msg["content"])
            continue

        if msg["kind"] == "manifest":
            st.caption(msg["source_label"])
            with st.expander(f"manifest ({len(msg['content'].splitlines())} lines)"):
                st.code(msg["content"], language="yaml")
            continue

        if msg["kind"] == "agent":
            for step in msg["steps"]:
                with st.expander("🔧 " + _describe_call(step["tool"], step["arguments"])):
                    st.code(step["result"] or "(no result)", language="text")
            if msg["final"]:
                st.markdown(msg["final"])
            continue

        # kind == "analysis": a pasted manifest, audited directly
        if not msg["parsed"]:
            st.error("Model output did not pass the schema check: " + "; ".join(msg["errors"]))
            with st.expander("raw model output"):
                st.code(msg["raw"])
            continue

        parsed = msg["parsed"]
        findings, patch = parsed["findings"], parsed["patch"]
        manifest_text = st.session_state.messages[i - 1]["content"]  # the preceding user turn

        if not findings:
            st.success("✓ No issues found")
        for j, f in enumerate(findings):
            label = (
                f"{SEVERITY_ICON.get(f['severity'], '⬜')} **{f['rule_id']}** ({f['severity']}) — {f['message']}  \n"
                f"`doc {f['doc']} · {f['path']}`"
            )
            if st.checkbox(label, value=j in msg["selected"], key=f"finding_{i}_{j}"):
                msg["selected"].add(j)
            else:
                msg["selected"].discard(j)

        chosen = {id(op) for op in core.patch_ops_for_findings([findings[j] for j in msg["selected"]], patch)}
        chosen |= {id(op) for op in core.unmatched_patch_ops(findings, patch)}
        selected_ops = [op for op in patch if id(op) in chosen]

        if patch and selected_ops:
            patched_yaml, patch_error = core.apply_patch_to_manifest(manifest_text, selected_ops)
            if patch_error:
                st.error(patch_error)
            else:
                diff = "\n".join(
                    difflib.unified_diff(manifest_text.splitlines(), patched_yaml.splitlines(), fromfile="before", tofile="after", lineterm="")
                )
                with st.expander(f"Diff ({len(selected_ops)} of {len(patch)} change(s) selected)", expanded=True):
                    st.code(diff or "(no textual changes)", language="diff")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ Apply fix + re-analyze", use_container_width=True, key=f"apply_{i}", disabled=model is None):
                        send_manifest(model, tokenizer, patched_yaml, "✅ fix applied, re-sent for analysis")
                with c2:
                    st.download_button("⬇️ Download patched YAML", patched_yaml, file_name="patched.yaml", use_container_width=True, key=f"dl_{i}")

        for j, res_yaml in enumerate(parsed.get("new_resources") or []):
            with st.expander(f"New resource needed ({j + 1})"):
                st.code(res_yaml, language="yaml")

# ------------------------------------------------- pending change (if any) --
# Shown below the conversation, not inside the message that proposed it, so
# it stays in view while the user asks the agent about it -- questions don't
# decline it; only Confirm or Decline does.
def render_pending_change() -> None:
    agent = st.session_state.get("agent")
    change = agent.toolbox.pending if agent is not None else None
    if change is not None:
        st.warning(
            f"Proposed change to `{change.namespace}/{change.kind}/{change.name}` — passed the server-side dry run. "
            "Nothing has been changed yet."
        )
        for f in change.findings:
            st.caption(f"{SEVERITY_ICON.get(f['severity'], '⬜')} **{f['rule_id']}** — {f['message']}")

        # Image tags: the model's choice is a guess -- let the user set one.
        image_overrides, bad_image = {}, False
        for op in change.image_ops:
            key = f"agent_image_{id(change)}_{op['index']}"
            st.session_state.setdefault(key, op["value"])
            image_overrides[op["index"]] = st.text_input(
                f"Image for container `{op['container']}`", key=key,
                help="The model's suggested tag may not exist in your registry: set the version you actually use.",
            )
            running = change.running_images.get(op["container"])
            if running:
                st.button(
                    f"Use the image running now: {running[:60]}{'…' if len(running) > 60 else ''}",
                    key=f"agent_image_running_{id(change)}_{op['index']}",
                    on_click=lambda k=key, v=running: st.session_state.__setitem__(k, v),
                    help="Pins to the exact digest the pods run today -- guaranteed to exist.",
                )
            if not agent_mod.is_pinned(image_overrides[op["index"]]):
                bad_image = True
                st.error("Pin the image to a tag (not `latest`) or a digest.")

        preview_yaml, preview_error = agent.toolbox.preview(change, image_overrides)
        with st.expander("Diff", expanded=True):
            if preview_error:
                st.error(preview_error)
            else:
                st.code(agent_mod.unified_diff(change.original_yaml, preview_yaml) or "(no textual changes)", language="diff")
        edited = []
        for j, res_yaml in enumerate(change.new_resources):
            edited.append(
                st.text_area(
                    f"New resource {j + 1} (created first)", value=res_yaml, height=140, key=f"agent_newres_{id(change)}_{j}",
                    help="Replace placeholder values (e.g. the real credential) before confirming.",
                )
            )
        has_placeholder = any("REPLACE_WITH_SECRET_VALUE" in r or "<REPLACE" in r for r in edited)
        if has_placeholder:
            st.info("Edit the placeholder value(s) above before confirming.")
        audit_down = agent.toolbox.audit_log is not None and not agent.toolbox.audit_log.ping()
        if audit_down:
            st.error("The audit log (PostgreSQL) is unreachable: changes can't be applied until they can be recorded.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Confirm and apply", key=f"agent_ok_{id(change)}", disabled=not is_operator or has_placeholder or bad_image or bool(preview_error) or audit_down, use_container_width=True):
                run_agent_turn(agent.resume(approved=True, new_resources=edited, image_overrides=image_overrides))
        with c2:
            if st.button("✖️ Decline", key=f"agent_no_{id(change)}", use_container_width=True):
                run_agent_turn(agent.resume(approved=False))


with st.container():
    render_pending_change()

# Auto-scroll to the newest message: st.markdown can't run <script> tags, so
# the anchor is placed with markdown and scrolled into view from a
# components.html iframe that reaches the real page via window.parent.
st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)
components.html(
    """
    <script>
        const el = window.parent.document.getElementById("chat-bottom");
        if (el) { el.scrollIntoView({behavior: "smooth", block: "end"}); }
    </script>
    """,
    height=0,
)

# --------------------------------------------------------------- chat input --
prompt = st.chat_input("Ask the agent, or paste a manifest to audit...", disabled=model is None)
if prompt:
    if core.classify_intent(prompt) == "manifest":
        send_manifest(model, tokenizer, prompt, "✏️ pasted")
    else:
        send_agent_message(get_agent(model, tokenizer, adapter_path, max_seq_length), prompt)
