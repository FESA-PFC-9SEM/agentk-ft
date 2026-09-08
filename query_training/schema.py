"""
Single source of truth for the CLI-agent task's contract: the system prompt,
the set of valid kubectl subcommands, and the {system, user, assistant}
message builder.

This task is completely separate from the manifest-security task in
dataset/ + finetune/. The model here does one thing: read a natural-language
request about a Kubernetes cluster and emit a single kubectl command.
"""

from __future__ import annotations

import json

# Target format "command": the assistant turn is the bare command string.
SYSTEM_PROMPT = (
    "You are a Kubernetes command-line assistant. Given a request in natural "
    "language, reply with a single kubectl command that fulfils it. Output "
    "only the command, on one line, with no prose, no explanation and no "
    "markdown code fences. When the request does not name a resource, use a "
    "short placeholder name."
)

# Target format "plan-command": the assistant turn is a JSON object
# {"plan": "...", "command": "..."}. The short plan gives a small model a few
# tokens to work the request before committing to a command -- measurably
# helps on compositional requests -- and stays machine-parseable (extract
# .command at execution time).
PLAN_COMMAND_SYSTEM_PROMPT = (
    "You are a Kubernetes command-line assistant. Given a request in natural "
    "language, respond with a single JSON object and nothing else: "
    '{"plan": "<a brief plan: which subcommand, flags and arguments you will '
    'use and why>", "command": "<the single kubectl command, one line>"}. '
    "No prose outside the JSON, no markdown code fences. When the request "
    "does not name a resource, use a short placeholder name."
)

TARGET_FORMATS = ("command", "plan-command")

# Every first token that can legally follow `kubectl`, from `kubectl --help`
# (v1.29). Used by the cleaner to reject rows whose "command" isn't actually
# a kubectl invocation, and by the evaluator to score "is this even kubectl".
KUBECTL_VERBS = frozenset(
    {
        "annotate", "api-resources", "api-versions", "apply", "attach",
        "auth", "autoscale", "certificate", "cluster-info", "completion",
        "config", "convert", "cordon", "cp", "create", "debug", "delete",
        "describe", "diff", "drain", "edit", "events", "exec", "explain",
        "expose", "get", "kustomize", "label", "logs", "options", "patch",
        "plugin", "port-forward", "proxy", "replace", "rollout", "run",
        "scale", "set", "taint", "top", "uncordon", "version", "wait",
    }
)

# Verbs that only read cluster state. `--read-only` in clean.py filters the
# dataset down to these, for training an agent that is not allowed to mutate
# the cluster (execution-side guardrails should still enforce this
# independently -- never rely on the model alone).
#
# Coarse, by design: it's a verb-level filter. `exec` is excluded (it runs
# arbitrary commands in a container); `config`/`auth` are excluded because
# `config set-credentials` / `auth reconcile` mutate, even though `config
# view` / `auth can-i` don't; `diff` is included (server-side dry-run only).
READ_ONLY_VERBS = frozenset(
    {
        "get", "describe", "logs", "top", "events", "explain",
        "api-resources", "api-versions", "cluster-info", "version", "diff",
    }
)


def build_messages(
    instruction: str, command: str, *, plan: str | None = None
) -> list[dict[str, str]]:
    """The chat-format record every split .jsonl line wraps in {"messages": ...}.

    plan=None  -> target format "command"      (assistant turn = bare command)
    plan="..." -> target format "plan-command" (assistant turn = JSON object)
    """
    if plan is None:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": command},
        ]
    assistant = json.dumps({"plan": plan, "command": command}, ensure_ascii=False)
    return [
        {"role": "system", "content": PLAN_COMMAND_SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": assistant},
    ]


def extract_command(assistant_text: str) -> str:
    """Inverse of build_messages' assistant turn: given either a bare command
    or a {"plan","command"} JSON object, return the command string. Used by
    evaluate.py / infer.py so they score the same regardless of target
    format."""
    text = assistant_text.strip()
    if text.startswith("{"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(obj, dict) and isinstance(obj.get("command"), str):
            return obj["command"].strip()
    return text
