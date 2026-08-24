"""
Composes dataset/mutate.py's per-rule mutators into single examples carrying
2+ SIMULTANEOUS defects, instead of the one-defect-per-example strategy in
dataset/build.py. Motivated by a real-world generalization gap: a model
trained only on single-defect examples missed a second plaintext credential
when a manifest actually had two (see README.md's design decisions).

Reuses mutate.py's MUTATORS as building blocks rather than duplicating rule
logic -- each rule still owns exactly one way to inject and detect its own
defect. What's new here is only the composition: applying several mutators
to the same document in sequence, chaining their inverse patches, and
re-deriving the combined finding set from the final document state.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

import jsonpatch

from dataset.detect import detect_all
from dataset.mutate import MUTATORS
from dataset.schema import Finding, PatchOp, RULE_IDS

# How many times the SAME rule can fire in one example -- e.g. two separate
# plaintext credentials (KSEC-001 twice), matching a real manifest that had
# exactly that shape (two passwords in one Pod) and which a model trained
# only on "each rule at most once" examples missed the second finding on.
# Each mutator's own precondition (an existing var name excluded for
# KSEC-001's env variant, an AssertionError for the hard-precondition rules
# once their target area is already dirty) naturally caps how many times a
# repeat can actually succeed on a given document.
MAX_REPEATS_PER_RULE = 2


@dataclass
class MultiMutationResult:
    mutated_doc: dict
    canonical: dict
    findings: list[Finding]
    patch: list[PatchOp]
    new_resources: list[str] = field(default_factory=list)
    applied_rule_ids: list[str] = field(default_factory=list)


def mutate_multi_defect(
    canonical_doc: dict,
    rng: random.Random,
    doc_index: int = 0,
    min_defects: int = 2,
    max_defects: int = 4,
    ksec001_candidate_names=None,
) -> MultiMutationResult | None:
    """Injects between min_defects and max_defects simultaneous defects
    (findings, not necessarily distinct rule types -- the same rule can fire
    more than once, up to MAX_REPEATS_PER_RULE, e.g. two separate plaintext
    credentials) into canonical_doc. Returns None if the doc isn't a usable
    base (already dirty) or if fewer than min_defects mutations turned out to
    be applicable (e.g. the doc has only one container)."""
    existing = [f for f in detect_all(canonical_doc, doc_index) if f.rule_id in RULE_IDS]
    if existing:
        return None

    max_possible = len(MUTATORS) * MAX_REPEATS_PER_RULE
    target_count = rng.randint(min_defects, min(max_defects, max_possible))
    rule_pool = list(MUTATORS) * MAX_REPEATS_PER_RULE
    rng.shuffle(rule_pool)

    current_doc = canonical_doc
    running_canonical = copy.deepcopy(canonical_doc)
    applied_rule_ids: list[str] = []
    patch: list[PatchOp] = []
    new_resources: list[str] = []

    for rule_id in rule_pool:
        if len(applied_rule_ids) >= target_count:
            break
        mutator = MUTATORS[rule_id]
        try:
            if rule_id == "KSEC-001":
                result = mutator(current_doc, rng, doc_index, candidate_names=ksec001_candidate_names)
            else:
                result = mutator(current_doc, rng, doc_index)
        except AssertionError:
            # A repeat attempt whose precondition no longer holds (e.g. this
            # rule already fired on the only container it could target) --
            # skip it like any other inapplicable attempt, rather than
            # aborting the whole composition over one exhausted rule.
            continue
        if result is None:
            continue
        # Undoing N chained mutations means undoing the most recent one
        # first, so each new stage's patch goes in front of what came
        # before it.
        patch = result.patch + patch
        new_resources += result.new_resources

        # Most mutators' own "canonical" is just their input doc unchanged --
        # undoing the injected defect reproduces it exactly. KSEC-001's
        # env-variant is the one exception: its "canonical" also ADDS a
        # secretKeyRef env entry (the fixed-forward form), a field that
        # never existed in current_doc at all. Diffing the mutator's own
        # input against its own canonical captures exactly that kind of
        # addition (empty diff for every other rule) without special-casing
        # KSEC-001 here -- and replaying the same diff against the running
        # canonical carries it into the final target.
        canonical_delta = jsonpatch.make_patch(current_doc, result.canonical).patch
        if canonical_delta:
            running_canonical = jsonpatch.apply_patch(running_canonical, canonical_delta)

        current_doc = result.mutated_doc
        applied_rule_ids.append(rule_id)

    if len(applied_rule_ids) < 2:
        return None

    findings = [f for f in detect_all(current_doc, doc_index) if f.rule_id in RULE_IDS]
    assert len(findings) == len(applied_rule_ids), (
        f"expected {len(applied_rule_ids)} findings after injecting {applied_rule_ids}, got {len(findings)}"
    )

    return MultiMutationResult(
        mutated_doc=current_doc,
        canonical=running_canonical,
        findings=findings,
        patch=patch,
        new_resources=new_resources,
        applied_rule_ids=applied_rule_ids,
    )
