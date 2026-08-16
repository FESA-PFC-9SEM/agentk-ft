"""
Structural deduplication. Reduces each document to a "skeleton": strips
identity (name, namespace, labels, annotations, selectors) and collapses all
leaf strings/numbers to placeholders, keeping only the shape. Two manifests
that only differ in resource names collide on the same hash -- which is
expected, since the real corpus has many copies of the same template with
different names/namespaces.
"""

from __future__ import annotations

import hashlib
import json

_DROP_ANYWHERE = frozenset(
    {
        "labels",
        "annotations",
        "selector",
        "matchLabels",
        "matchExpressions",
        "resourceVersion",
        "uid",
        "creationTimestamp",
        "selfLink",
        "generation",
        "managedFields",
        "ownerReferences",
        "status",
    }
)
_DROP_IN_METADATA = frozenset({"name", "namespace"})
_KEEP_LITERAL = frozenset({"kind", "apiVersion"})


def skeleton(node, in_metadata: bool = False):
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            key_str = str(key)
            if key_str in _DROP_ANYWHERE:
                continue
            if in_metadata and key_str in _DROP_IN_METADATA:
                continue
            if key_str in _KEEP_LITERAL:
                out[key_str] = value
                continue
            out[key_str] = skeleton(value, in_metadata=(key_str == "metadata"))
        return out
    if isinstance(node, list):
        return [skeleton(item, in_metadata) for item in node]
    if isinstance(node, bool) or node is None:
        return node
    if isinstance(node, (int, float)):
        return "<N>"
    return "<S>"


def skeleton_hash(doc) -> str:
    payload = json.dumps(skeleton(doc), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dedup(docs: list) -> tuple[list[int], float]:
    """Returns (kept indices, survival rate) after structural dedup. The
    first document of each skeleton is the one that survives."""
    seen: set[str] = set()
    kept: list[int] = []
    for i, doc in enumerate(docs):
        h = skeleton_hash(doc)
        if h in seen:
            continue
        seen.add(h)
        kept.append(i)
    survival_rate = len(kept) / len(docs) if docs else 0.0
    return kept, survival_rate
