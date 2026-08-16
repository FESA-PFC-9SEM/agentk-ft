"""
Single source of truth for the model's contract: system prompt, rule
taxonomy, response dataclasses, and schema validator. Everything else in the
pipeline (mutate.py, build.py, generation/*) imports from this module instead
of redefining any of these structures.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SEVERITIES = ("low", "medium", "high", "critical")
PATCH_OPS = ("replace", "remove", "add")

RULES = {
    "KSEC-001": "Plaintext credential (password, token, API key, "
                "connection string, private key)",
    "KSEC-002": "Insecure securityContext (privileged, runAsUser 0, "
                "allowPrivilegeEscalation, added capabilities)",
    "KSEC-005": "Unpinned container image (latest tag or missing tag)",
    "KSEC-006": "Selector/label mismatch (spec.selector.matchLabels doesn't "
                "match the pod template's own labels, breaking routing/discovery)",
    "KSEC-007": "Probe port mismatch (liveness/readiness/startup probe targets "
                "a port not declared in the container's ports)",
    "KSEC-008": "Resource requests exceed limits (spec.containers[].resources."
                "requests is greater than resources.limits for the same resource)",
}
# KSEC-003 (host access), KSEC-004 (permissive RBAC) and KSEC-009 (dangling
# volume reference) are fully implemented -- detect_ksecNNN/mutate_ksecNNN
# and their tests are still in the codebase -- but deliberately excluded
# from RULES/MUTATORS, so the dataset pipeline never generates or labels
# examples for them and the model is never told to detect something it was
# never shown. Re-enabling is a code change (add back here and to
# dataset/mutate.py's MUTATORS), not a flag.

RULE_IDS = frozenset(RULES)

_RULES_BLOCK = "\n".join(f"- {rule_id}: {description[0].lower()}{description[1:]}." for rule_id, description in RULES.items())

SYSTEM_PROMPT = """\
You are a Kubernetes manifest security and configuration auditor. You receive \
a manifest file (possibly multi-document, separated by '---') and must \
respond with a single JSON object, with no prose and no markdown code fences.

Rules you must detect:
""" + _RULES_BLOCK + """

Response format (pure JSON, exactly these keys):
{
  "findings": [
    {"rule_id": "KSEC-XXX", "severity": "low|medium|high|critical",
     "doc": 0, "path": "/spec/...", "message": "...", "evidence": "Tr0u***"}
  ],
  "patch": [{"doc": 0, "op": "replace|remove|add", "path": "/spec/...", "value": {}}],
  "new_resources": ["<complete YAML for resources that must be created>"],
  "notes": ["..."]
}

Format rules:
- "doc" is the zero-based index of the YAML document within the file.
- "path" is a JSON Pointer (RFC 6901) into that document.
- "patch" follows RFC 6902. Never rewrite the whole manifest: each operation \
must be the smallest change that fixes the corresponding finding.
- "evidence" is always masked: the first 4 characters of the value followed \
by "***". A secret value must never appear in full in the response.
- "new_resources" is only used when the fix requires creating a resource \
that doesn't exist yet (for example, externalizing a credential into a Secret).
- A clean manifest has all four arrays empty: findings, patch, new_resources \
and notes.
- Respond with the JSON object only. No text before or after it, no code \
fences (```)."""


_EVIDENCE_RE = re.compile(r"^.{1,4}\*\*\*$", re.DOTALL)


@dataclass
class Finding:
    rule_id: str
    severity: str
    doc: int
    path: str
    message: str
    evidence: str = ""

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "doc": self.doc,
            "path": self.path,
            "message": self.message,
            "evidence": self.evidence,
        }


@dataclass
class PatchOp:
    doc: int
    op: str
    path: str
    value: object = None

    def to_dict(self) -> dict:
        d = {"doc": self.doc, "op": self.op, "path": self.path}
        if self.op != "remove":
            d["value"] = self.value
        return d


@dataclass
class Response:
    findings: list[Finding] = field(default_factory=list)
    patch: list[PatchOp] = field(default_factory=list)
    new_resources: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "patch": [p.to_dict() for p in self.patch],
            "new_resources": list(self.new_resources),
            "notes": list(self.notes),
        }

    def is_clean(self) -> bool:
        return not (self.findings or self.patch or self.new_resources)


def mask_evidence(value: str) -> str:
    """Masks a sensitive value: keeps only the first 4 characters."""
    prefix = str(value)[:4]
    return f"{prefix}***"


def escape_json_pointer_token(token: str) -> str:
    """Escapes a single JSON Pointer (RFC 6901) reference token: '~' becomes
    '~0' and '/' becomes '~1', in that order. Needed whenever a free-form key
    (not a fixed field name or a list index) is embedded in a path -- e.g.
    Kubernetes label keys, which routinely contain '/' (app.kubernetes.io/name)."""
    return token.replace("~", "~0").replace("/", "~1")


def validate_response(obj: dict) -> list[str]:
    """Validates a response dict against the schema. Returns a list of
    errors (empty means valid). Never raises: the caller decides what to do
    with the errors (discard the example, log it, etc)."""
    errors: list[str] = []

    if not isinstance(obj, dict):
        return ["response is not a JSON object"]

    for key in ("findings", "patch", "new_resources", "notes"):
        if key not in obj:
            errors.append(f"missing required key: {key}")
        elif not isinstance(obj[key], list):
            errors.append(f"key '{key}' should be a list")

    if errors:
        return errors

    for i, finding in enumerate(obj["findings"]):
        prefix = f"findings[{i}]"
        if not isinstance(finding, dict):
            errors.append(f"{prefix}: should be an object")
            continue
        for key in ("rule_id", "severity", "doc", "path", "message", "evidence"):
            if key not in finding:
                errors.append(f"{prefix}: missing required key: {key}")
        if finding.get("rule_id") not in RULE_IDS:
            errors.append(f"{prefix}: invalid rule_id: {finding.get('rule_id')!r}")
        if finding.get("severity") not in SEVERITIES:
            errors.append(f"{prefix}: invalid severity: {finding.get('severity')!r}")
        if not isinstance(finding.get("doc"), int):
            errors.append(f"{prefix}: doc should be an integer")
        path = finding.get("path")
        if not isinstance(path, str) or not path.startswith("/"):
            errors.append(f"{prefix}: invalid path: {path!r}")
        evidence = finding.get("evidence")
        if not isinstance(evidence, str) or not _EVIDENCE_RE.match(evidence):
            errors.append(f"{prefix}: evidence is not masked correctly: {evidence!r}")

    for i, op in enumerate(obj["patch"]):
        prefix = f"patch[{i}]"
        if not isinstance(op, dict):
            errors.append(f"{prefix}: should be an object")
            continue
        if op.get("op") not in PATCH_OPS:
            errors.append(f"{prefix}: invalid op: {op.get('op')!r}")
        if not isinstance(op.get("doc"), int):
            errors.append(f"{prefix}: doc should be an integer")
        path = op.get("path")
        if not isinstance(path, str) or not path.startswith("/"):
            errors.append(f"{prefix}: invalid path: {path!r}")
        if op.get("op") != "remove" and "value" not in op:
            errors.append(f"{prefix}: operation '{op.get('op')}' requires 'value'")

    for i, res in enumerate(obj["new_resources"]):
        if not isinstance(res, str):
            errors.append(f"new_resources[{i}]: should be a YAML string")

    for i, note in enumerate(obj["notes"]):
        if not isinstance(note, str):
            errors.append(f"notes[{i}]: should be a string")

    return errors
