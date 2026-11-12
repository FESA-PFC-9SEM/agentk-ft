"""
Audit trail of the agent's cluster changes, in PostgreSQL.

Every change the agent proposes and the user decides on -- applied,
declined, rejected before touching the cluster (e.g. an unpinned image, a
failed dry run) or failed while applying -- is recorded with who decided,
when, which model audited the manifest, the manifest before and after, the
diff, the findings, every patch operation and the resources it created.
History is append-only: the app's database role may only INSERT and SELECT,
and triggers refuse UPDATE/DELETE on recorded changes (deploy/audit-schema.sql).

Nothing secret is stored: values at the paths a credential was found
(KSEC-001, by the project's detectors or the model) are masked to their
first 4 characters + "***" in both manifests and in the patch values, and a
created Secret is stored as its key names only.

The database runs in minikube (`audit` namespace, deploy/postgres.yaml) and
is set up by deploy/setup_secure_demo.py, which writes its URL to
.streamlit/secrets.toml ([audit] database_url). AUDIT_DATABASE_URL
overrides it.

History from the command line:
    finetune/.venv/bin/python -m demo.audit_log              # latest changes
    finetune/.venv/bin/python -m demo.audit_log --show 12    # one change in full
"""

from __future__ import annotations

import argparse
import copy
import difflib
import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

STREAMLIT_SECRETS = REPO_ROOT / ".streamlit" / "secrets.toml"
STATUSES = ("applied", "declined", "rejected", "failed")


@dataclass(frozen=True)
class Actor:
    """Who decided on a change: the logged-in Keycloak user."""

    subject: str  # OIDC "sub" claim
    username: str
    email: str | None = None
    roles: tuple[str, ...] = ()


def database_url() -> str | None:
    """AUDIT_DATABASE_URL, else [audit] database_url from
    .streamlit/secrets.toml, else None (auditing not configured)."""
    if os.environ.get("AUDIT_DATABASE_URL"):
        return os.environ["AUDIT_DATABASE_URL"]
    try:
        return tomllib.loads(STREAMLIT_SECRETS.read_text()).get("audit", {}).get("database_url")
    except (OSError, tomllib.TOMLDecodeError):
        return None


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def _tokens(pointer: str) -> list[str]:
    return [t.replace("~1", "/").replace("~0", "~") for t in pointer.split("/")[1:]] if pointer else []


def _mask_at(doc, pointer: str) -> None:
    """Masks the string at `pointer` (or the `value` of the env entry it
    names), in place. A path that doesn't resolve is ignored."""
    from dataset.schema import mask_evidence

    tokens = _tokens(pointer)
    if not tokens:
        return
    parent = doc
    for token in tokens[:-1]:
        try:
            parent = parent[int(token)] if isinstance(parent, list) else parent[token]
        except (KeyError, IndexError, ValueError, TypeError):
            return
    key = tokens[-1]
    try:
        key = int(key) if isinstance(parent, list) else key
        value = parent[key]
    except (KeyError, IndexError, ValueError, TypeError):
        return
    if isinstance(value, str):
        parent[key] = mask_evidence(value)
    elif isinstance(value, dict) and isinstance(value.get("value"), str):
        value["value"] = mask_evidence(value["value"])


def sensitive_paths(docs: list, findings: list[dict] = ()) -> set[tuple[int, str]]:
    """(doc index, JSON Pointer) of every plaintext credential: what the
    project's detectors find (KSEC-001) plus what the model reported."""
    from dataset.detect import detect_file

    try:
        detected = [f.to_dict() for f in detect_file(docs)]
    except Exception:  # noqa: BLE001 -- odd documents: fall back to the model's findings
        detected = []
    return {
        (int(f.get("doc", 0) or 0), str(f.get("path", "")))
        for f in [*detected, *findings]
        if f.get("rule_id") == "KSEC-001" and f.get("path")
    }


def redact_manifest(yaml_text: str, findings: list[dict] = ()) -> str:
    """The manifest with every credential masked, re-serialized (always, so
    two redacted manifests diff cleanly)."""
    docs = list(yaml.safe_load_all(yaml_text))
    for index, pointer in sensitive_paths(docs, findings):
        if 0 <= index < len(docs):
            _mask_at(docs[index], pointer)
    return yaml.safe_dump_all(docs, sort_keys=False)


def redact_value(value, op_path: str, op_doc: int, paths: set[tuple[int, str]]):
    """A patch value, masked if the operation touches a credential's path."""
    from finetune.metrics import paths_related

    if value is None or not any(doc == op_doc and paths_related(path, op_path) for doc, path in paths):
        return value
    holder = {"v": copy.deepcopy(value)}
    _mask_at(holder, "/v")
    return holder["v"]


# ---------------------------------------------------------------------------
# What gets recorded
# ---------------------------------------------------------------------------


def created_resource_summary(resource_yaml: str, default_namespace: str) -> dict:
    """Kind, namespace, name and key names of a created resource -- never
    its values."""
    try:
        doc = yaml.safe_load(resource_yaml)
    except yaml.YAMLError:
        doc = None
    doc = doc if isinstance(doc, dict) else {}
    meta = doc.get("metadata") or {}
    keys = sorted({*(doc.get("data") or {}), *(doc.get("stringData") or {})})
    return {
        "kind": str(doc.get("kind", "?")),
        "namespace": str(meta.get("namespace") or default_namespace),
        "name": str(meta.get("name", "?")),
        "data_keys": keys,
    }


def build_change_record(change, status: str, result: str, chosen_images: dict | None = None, created: list[str] | None = None) -> dict:
    """Everything stored about one decided change (a demo.agent.PendingChange),
    redacted. `chosen_images` maps patch index -> the image the user set
    instead of the model's; `created` are the resources actually created."""
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    chosen_images = {int(k): v for k, v in (chosen_images or {}).items()}
    before_docs = list(yaml.safe_load_all(change.original_yaml))
    paths = sensitive_paths(before_docs, change.findings)
    before = redact_manifest(change.original_yaml, change.findings)
    after = redact_manifest(change.patched_yaml, change.findings)
    kind = (before_docs[0] or {}).get("kind") if before_docs and isinstance(before_docs[0], dict) else None
    operations = []
    for position, op in enumerate(change.patch):
        doc = int(op.get("doc", 0) or 0)
        suggested = op.get("value")
        applied = chosen_images.get(position, suggested)
        operations.append({
            "position": position,
            "op": str(op.get("op", "")),
            "path": str(op.get("path", "")),
            "doc": doc,
            "value": redact_value(applied, str(op.get("path", "")), doc, paths),
            "suggested_value": redact_value(suggested, str(op.get("path", "")), doc, paths) if position in chosen_images else None,
        })
    return {
        "workload": {"kind": kind or change.kind, "namespace": change.namespace, "name": change.name},
        "status": status,
        "proposed_at": change.proposed_at,
        "user_request": change.request or None,
        "result_message": result,
        "dry_run_output": change.dry_run_output or None,
        "manifest_before": before,
        "manifest_after": after,
        "diff": "\n".join(difflib.unified_diff(before.splitlines(), after.splitlines(), fromfile="before", tofile="after", lineterm="")),
        "findings": [
            {
                "position": i,
                "rule_id": str(f.get("rule_id", "")),
                "severity": str(f.get("severity", "")),
                "doc": int(f.get("doc", 0) or 0),
                "path": str(f.get("path", "")),
                "message": str(f.get("message", "")),
                "evidence": f.get("evidence"),
            }
            for i, f in enumerate(change.findings)
        ],
        "operations": operations,
        "created_resources": [created_resource_summary(r, change.namespace) for r in created or []],
    }


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


class AuditLog:
    """Writes and reads the audit trail. One short connection per call: the
    app records a change every few minutes at most."""

    def __init__(self, url: str, connect_timeout: int = 5):
        self.url = url
        self.connect_timeout = connect_timeout

    def _connect(self):
        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(self.url, connect_timeout=self.connect_timeout, row_factory=dict_row)

    def ping(self) -> bool:
        try:
            with self._connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:  # noqa: BLE001 -- any failure means "can't record"
            return False

    def record(self, actor: Actor, adapter: str | None, record: dict) -> int:
        """Stores one change (build_change_record's output) in a single
        transaction; returns its id."""
        from psycopg.types.json import Jsonb

        with self._connect() as conn:
            user_id = conn.execute(
                "INSERT INTO app_user (subject, username, email) VALUES (%s, %s, %s) "
                "ON CONFLICT (subject) DO UPDATE SET username = EXCLUDED.username, email = EXCLUDED.email, last_seen_at = now() "
                "RETURNING id",
                (actor.subject, actor.username, actor.email),
            ).fetchone()["id"]
            w = record["workload"]
            workload_id = conn.execute(
                "WITH inserted AS (INSERT INTO workload (kind, namespace, name) VALUES (%(kind)s, %(namespace)s, %(name)s) "
                "ON CONFLICT DO NOTHING RETURNING id) "
                "SELECT id FROM inserted UNION ALL "
                "SELECT id FROM workload WHERE kind = %(kind)s AND namespace = %(namespace)s AND name = %(name)s LIMIT 1",
                w,
            ).fetchone()["id"]
            change_id = conn.execute(
                "INSERT INTO manifest_change (workload_id, user_id, user_roles, status, proposed_at, model_adapter, user_request, "
                "result_message, dry_run_output, manifest_before, manifest_after, diff) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (workload_id, user_id, list(actor.roles), record["status"], record["proposed_at"], adapter, record["user_request"],
                 record["result_message"], record["dry_run_output"], record["manifest_before"], record["manifest_after"], record["diff"]),
            ).fetchone()["id"]
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO finding (change_id, position, rule_id, severity, doc, path, message, evidence) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    [(change_id, f["position"], f["rule_id"], f["severity"], f["doc"], f["path"], f["message"], f["evidence"])
                     for f in record["findings"]],
                )
                cur.executemany(
                    "INSERT INTO patch_operation (change_id, position, op, path, doc, value, suggested_value) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    [(change_id, o["position"], o["op"], o["path"], o["doc"],
                      None if o["value"] is None else Jsonb(o["value"]),
                      None if o["suggested_value"] is None else Jsonb(o["suggested_value"]))
                     for o in record["operations"]],
                )
                cur.executemany(
                    "INSERT INTO created_resource (change_id, kind, namespace, name, data_keys) VALUES (%s, %s, %s, %s, %s)",
                    [(change_id, r["kind"], r["namespace"], r["name"], r["data_keys"]) for r in record["created_resources"]],
                )
        return change_id

    def history(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            return conn.execute("SELECT * FROM change_history ORDER BY decided_at DESC, id DESC LIMIT %s", (limit,)).fetchall()

    def change(self, change_id: int) -> dict | None:
        """One change in full: the change row with its workload and user,
        plus its findings, operations and created resources."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT c.*, w.kind, w.namespace, w.name, u.subject, u.username, u.email "
                "FROM manifest_change c JOIN workload w ON w.id = c.workload_id JOIN app_user u ON u.id = c.user_id "
                "WHERE c.id = %s",
                (change_id,),
            ).fetchone()
            if row is None:
                return None
            for table, key in (("finding", "findings"), ("patch_operation", "operations"), ("created_resource", "created_resources")):
                order = "position" if table != "created_resource" else "id"
                row[key] = conn.execute(f"SELECT * FROM {table} WHERE change_id = %s ORDER BY {order}", (change_id,)).fetchall()
            return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=20, help="how many recent changes to list")
    parser.add_argument("--show", type=int, help="print one change in full")
    args = parser.parse_args(argv)
    url = database_url()
    if not url:
        raise SystemExit("auditing isn't configured: run deploy/setup_secure_demo.py (or set AUDIT_DATABASE_URL)")
    log = AuditLog(url)
    if args.show is None:
        rows = log.history(args.limit)
        if not rows:
            print("No recorded changes yet.")
        for r in rows:
            print(f"#{r['id']:<4} {r['decided_at']:%Y-%m-%d %H:%M:%S}  {r['status']:<8}  {r['username']:<10}  "
                  f"{r['kind']} {r['namespace']}/{r['name']}  ({r['findings']} finding(s), {r['operations']} operation(s))")
        return
    c = log.change(args.show)
    if c is None:
        raise SystemExit(f"no change #{args.show}")
    print(f"Change #{c['id']}: {c['status']} -- {c['kind']} {c['namespace']}/{c['name']}")
    print(f"  by {c['username']} ({', '.join(c['user_roles'])}), proposed {c['proposed_at']}, decided {c['decided_at']}")
    print(f"  model: {c['model_adapter']}\n  request: {c['user_request']}\n  result: {c['result_message']}")
    for f in c["findings"]:
        print(f"  finding {f['rule_id']} ({f['severity']}) at {f['path']}: {f['message']}")
    for o in c["operations"]:
        extra = f" (model suggested {o['suggested_value']})" if o["suggested_value"] is not None else ""
        print(f"  {o['op']} {o['path']} = {o['value']}{extra}")
    for r in c["created_resources"]:
        print(f"  created {r['kind']} {r['namespace']}/{r['name']} keys: {', '.join(r['data_keys']) or '-'}")
    print(c["diff"])


if __name__ == "__main__":
    main()
