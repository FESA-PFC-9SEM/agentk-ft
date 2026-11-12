import json
import os

import pytest
import yaml

from demo.agent import Agent, PendingChange
from demo.audit_log import Actor, AuditLog, build_change_record, redact_manifest
from demo.core import apply_patch_to_manifest
from demo.test_agent import FakeCluster, call, make_toolbox, scripted

SECRET = "hunter2-Zq8vT3xK9pLm"
MANIFEST = yaml.safe_dump(
    {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "api", "namespace": "demo"},
        "spec": {"template": {"spec": {"containers": [{
            "name": "api", "image": "registry/api",
            "args": ["serve", "--db-password", SECRET],
            "env": [{"name": "LOG_LEVEL", "value": "info"}, {"name": "DB_PASSWORD", "value": SECRET}],
        }]}}},
    },
    sort_keys=False,
)
CONTAINER = "/spec/template/spec/containers/0"
FINDINGS = [
    {"rule_id": "KSEC-001", "severity": "critical", "doc": 0, "path": f"{CONTAINER}/env/1/value",
     "message": "Plaintext credential in DB_PASSWORD", "evidence": "hunt***"},
    {"rule_id": "KSEC-005", "severity": "medium", "doc": 0, "path": f"{CONTAINER}/image",
     "message": "Container image has no pinned tag", "evidence": "regi***"},
]
PATCH = [
    {"doc": 0, "op": "replace", "path": f"{CONTAINER}/env/1",
     "value": {"name": "DB_PASSWORD", "valueFrom": {"secretKeyRef": {"name": "api-db", "key": "password"}}}},
    {"doc": 0, "op": "replace", "path": f"{CONTAINER}/image", "value": "registry/api:1.0.0"},
]
NEW_SECRET = yaml.safe_dump({"apiVersion": "v1", "kind": "Secret", "metadata": {"name": "api-db"}, "stringData": {"password": SECRET}})


def pending_change(**overrides) -> PendingChange:
    patched, error = apply_patch_to_manifest(MANIFEST, PATCH)
    assert error is None
    fields = dict(kind="deployment", name="api", namespace="demo", findings=FINDINGS, original_yaml=MANIFEST, patched_yaml=patched,
                  new_resources=[NEW_SECRET], dry_run_output="deployment.apps/api configured (server dry run)", patch=PATCH,
                  request="fix the api deployment")
    return PendingChange(**{**fields, **overrides})


def test_redact_manifest_masks_every_credential_and_nothing_else():
    redacted = redact_manifest(MANIFEST)
    assert SECRET not in redacted
    doc = yaml.safe_load(redacted)
    container = doc["spec"]["template"]["spec"]["containers"][0]
    assert container["env"][1]["value"] == "hunt***"
    assert container["args"] == ["serve", "--db-password", "hunt***"]  # split-flag form, found by the detectors
    assert container["env"][0]["value"] == "info"


def test_change_record_never_contains_a_secret():
    record = build_change_record(pending_change(), "applied", "Applied", chosen_images={1: "registry/api:2.3.4"}, created=[NEW_SECRET])
    assert SECRET not in json.dumps(record, default=str)
    assert record["workload"] == {"kind": "Deployment", "namespace": "demo", "name": "api"}
    assert record["created_resources"] == [{"kind": "Secret", "namespace": "demo", "name": "api-db", "data_keys": ["password"]}]
    image_op = record["operations"][1]
    assert (image_op["value"], image_op["suggested_value"]) == ("registry/api:2.3.4", "registry/api:1.0.0")
    assert record["operations"][0]["suggested_value"] is None
    assert "-          value: hunt***\n+          valueFrom:" in record["diff"]
    assert [f["evidence"] for f in record["findings"]] == ["hunt***", "regi***"]


def test_change_record_masks_a_plaintext_value_written_by_the_patch():
    patch = [{"doc": 0, "op": "replace", "path": f"{CONTAINER}/env/1/value", "value": "another-Secret-9x8y7z"}]
    patched, _ = apply_patch_to_manifest(MANIFEST, patch)
    record = build_change_record(pending_change(patch=patch, patched_yaml=patched), "declined", "Declined")
    assert record["operations"][0]["value"] == "anot***"
    assert "another-Secret-9x8y7z" not in record["manifest_after"]


def test_unknown_status_is_refused():
    with pytest.raises(ValueError):
        build_change_record(pending_change(), "maybe", "?")


# --------------------------------------------------- recording from the agent --


class FakeAuditLog:
    def __init__(self, reachable=True, fail=False):
        self.reachable, self.fail, self.records = reachable, fail, []

    def ping(self):
        return self.reachable

    def record(self, actor, adapter, record):
        if self.fail:
            raise RuntimeError("connection reset")
        self.records.append((actor, adapter, record))
        return len(self.records)


def audited_agent(cluster, log):
    toolbox = make_toolbox(cluster)
    toolbox.audit_log, toolbox.actor, toolbox.adapter = log, Actor("sub-1", "operator", roles=("viewer", "operator")), "runs/x/lora_adapter"
    planner = scripted(call("audit_manifest", kind="Deployment", name="web", namespace="demo"),
                       call("apply_fix", kind="Deployment", name="web", namespace="demo"), "Done.")
    agent = Agent(planner, toolbox)
    list(agent.ask("fix the web deployment in demo"))
    assert agent.waiting
    return agent


def test_an_applied_change_is_recorded_with_who_and_what():
    cluster, log = FakeCluster(), FakeAuditLog()
    agent = audited_agent(cluster, log)
    events = list(agent.resume(approved=True))
    assert cluster.applied()
    actor, adapter, record = log.records[0]
    assert (actor.username, adapter, record["status"]) == ("operator", "runs/x/lora_adapter", "applied")
    assert record["user_request"] == "fix the web deployment in demo"
    assert record["operations"][0]["value"] == "nginx:1.25"
    assert "(audit log: change #1)" in events[0].text


def test_a_declined_change_is_recorded_and_not_applied():
    cluster, log = FakeCluster(), FakeAuditLog()
    agent = audited_agent(cluster, log)
    list(agent.resume(approved=False))
    assert not cluster.applied()
    assert [r[2]["status"] for r in log.records] == ["declined"]


def test_nothing_is_applied_while_the_audit_log_is_unreachable():
    cluster, log = FakeCluster(), FakeAuditLog(reachable=False)
    agent = audited_agent(cluster, log)
    events = list(agent.resume(approved=True))
    assert not cluster.applied() and not log.records
    assert agent.waiting  # still pending: the user can retry once the database is back
    assert "audit log" in events[0].text and events[0].text.startswith("Not applied")


def test_a_rejected_image_is_recorded_as_rejected():
    cluster, log = FakeCluster(), FakeAuditLog()
    agent = audited_agent(cluster, log)
    list(agent.resume(approved=True, image_overrides={0: "nginx:latest"}))
    assert not cluster.applied()
    assert log.records[0][2]["status"] == "rejected"


def test_a_recording_failure_after_applying_is_reported():
    cluster, log = FakeCluster(), FakeAuditLog(fail=True)
    agent = audited_agent(cluster, log)
    events = list(agent.resume(approved=True))
    assert cluster.applied()
    assert "NOT recorded in the audit log: connection reset" in events[0].text


# ------------------------------------------------- against a real PostgreSQL --


@pytest.mark.skipif(not os.environ.get("AUDIT_TEST_DATABASE_URL"), reason="set AUDIT_TEST_DATABASE_URL to a scratch database with the audit schema")
def test_record_and_read_back_from_postgres():
    log = AuditLog(os.environ["AUDIT_TEST_DATABASE_URL"])
    assert log.ping()
    record = build_change_record(pending_change(), "applied", "Applied", chosen_images={1: "registry/api:2.3.4"}, created=[NEW_SECRET])
    change_id = log.record(Actor("sub-test", "tester", "t@example.com", ("viewer", "operator")), "runs/x/lora_adapter", record)
    stored = log.change(change_id)
    assert (stored["status"], stored["username"], stored["kind"], stored["user_roles"]) == ("applied", "tester", "Deployment", ["viewer", "operator"])
    assert [f["rule_id"] for f in stored["findings"]] == ["KSEC-001", "KSEC-005"]
    assert stored["operations"][1]["suggested_value"] == "registry/api:1.0.0"
    assert stored["created_resources"][0]["data_keys"] == ["password"]
    assert SECRET not in json.dumps(stored, default=str)
    assert log.history(5)[0]["id"] == change_id
    # Append-only, also through the app's own role.
    import psycopg

    with pytest.raises(psycopg.errors.InsufficientPrivilege), psycopg.connect(log.url) as conn:
        conn.execute("DELETE FROM manifest_change WHERE id = %s", (change_id,))
