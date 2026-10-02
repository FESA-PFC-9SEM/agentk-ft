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

import collections
import copy
import random
from dataclasses import dataclass, field

import jsonpatch

from dataset.detect import detect_all, detect_file
from dataset.mutate import MUTATORS, FileMutationResult, mutate_ksec006_service
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

# Relative weight of each rule when choosing which defects to inject (rules
# not listed weigh 1). With 11 rules sharing the budget uniformly, plaintext
# credentials and unpinned images -- the two defects real manifests carry
# most (every hand-written scenario has them) -- fell to ~50% of positives,
# and a model trained on that missed credentials it used to catch. Weighting
# them up restores them to roughly two thirds of positives without
# dropping any rule.
RULE_WEIGHTS = {"KSEC-001": 3.0, "KSEC-005": 3.0}

# Defect counts favour 2-4 (the usual real-world case) with a thinner tail up
# to max_defects, so the model also learns to keep listing findings on a
# badly broken file (scenarios/8-newrelic.yaml has 9) instead of stopping at
# the 4 it had only ever seen.
_TAIL_START, _TAIL_WEIGHT = 5, 0.4


def _sample_defect_count(rng: random.Random, min_defects: int, max_defects: int) -> int:
    counts = list(range(min_defects, max_defects + 1))
    return rng.choices(counts, weights=[_TAIL_WEIGHT if k >= _TAIL_START else 1.0 for k in counts])[0]


def _weighted_order(steps: list[str], rng: random.Random, rule_weights: dict) -> list[str]:
    """A random permutation of `steps` where heavier rules tend to come
    first (Efraimidis-Spirakis weighted sampling without replacement) --
    the composers apply steps in order until the target count is reached,
    so earlier means more likely to be injected."""
    def weight(step):
        return rule_weights.get(_step_rule_id(step), 1.0)
    return sorted(steps, key=lambda step: rng.random() ** (1.0 / weight(step)), reverse=True)


@dataclass
class MultiMutationResult:
    mutated_doc: dict
    canonical: dict
    findings: list[Finding]
    patch: list[PatchOp]
    new_resources: list[str] = field(default_factory=list)
    applied_rule_ids: list[str] = field(default_factory=list)


def _round_trips(mutated_docs: list[dict], patch: list[PatchOp], canonical_docs: list[dict], single: bool = False) -> bool:
    """Whether the chained patch, applied per document, reproduces the
    running canonical exactly. Checked after every composition step: a step
    whose own fix changes a list's length in the canonical (KSEC-011
    collapsing several password entries into one) can leave an EARLIER
    step's index-based op pointing past the end -- found on a full build
    once KSEC-012 started fixing env values by index. Any such interaction
    rejects just that step instead of failing the whole build."""
    try:
        reconstructed = list(mutated_docs)
        for i in range(len(mutated_docs)):
            ops = [{k: v for k, v in op.to_dict().items() if k != "doc"} for op in patch if single or op.doc == i]
            if ops:
                reconstructed[i] = jsonpatch.apply_patch(mutated_docs[i], ops)
        return reconstructed == canonical_docs
    except Exception:
        return False


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
    target_count = _sample_defect_count(rng, min_defects, min(max_defects, max_possible))
    rule_pool = _weighted_order(list(MUTATORS) * MAX_REPEATS_PER_RULE, rng, RULE_WEIGHTS)

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
        new_patch = result.patch + patch

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
        try:
            new_canonical = jsonpatch.apply_patch(running_canonical, canonical_delta) if canonical_delta else running_canonical
        except Exception:
            continue
        if not _round_trips([result.mutated_doc], new_patch, [new_canonical], single=True):
            continue

        patch = new_patch
        new_resources += result.new_resources
        running_canonical = new_canonical
        current_doc = result.mutated_doc
        applied_rule_ids.append(rule_id)

    if len(applied_rule_ids) < max(1, min_defects):
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


# ---------------------------------------------------------------------------
# Multi-document files
# ---------------------------------------------------------------------------

# The cross-document KSEC-006 mutation (a Service's selector no longer matching
# its workload), as a step alongside the per-document MUTATORS.
SERVICE_SELECTOR_STEP = "KSEC-006/service"


@dataclass
class MultiFileMutationResult:
    mutated_docs: list[dict]
    canonical_docs: list[dict]
    findings: list[Finding]
    patch: list[PatchOp]
    new_resources: list[str] = field(default_factory=list)
    applied_rule_ids: list[str] = field(default_factory=list)


def _step_rule_id(step: str) -> str:
    return "KSEC-006" if step == SERVICE_SELECTOR_STEP else step


def _file_rule_counts(docs: list[dict]) -> collections.Counter:
    return collections.Counter(f.rule_id for f in detect_file(docs) if f.rule_id in RULE_IDS)


def _apply_file_step(docs: list[dict], rng: random.Random, step: str, ksec001_candidate_names=None):
    """One mutation somewhere in a file: the Service-selector mutation, or a
    per-document mutator on the first member (in random order) it applies
    to, with that member's index as doc_index. Returns a FileMutationResult
    only if it changed nothing but its own rule's finding count at FILE
    level -- MUTATORS' own guard can't see cross-document findings."""
    if step == SERVICE_SELECTOR_STEP:
        result = mutate_ksec006_service(docs, rng)
    else:
        result = None
        order = list(range(len(docs)))
        rng.shuffle(order)
        mutator = MUTATORS[step]
        for i in order:
            try:
                if step == "KSEC-001":
                    doc_result = mutator(docs[i], rng, i, candidate_names=ksec001_candidate_names)
                else:
                    doc_result = mutator(docs[i], rng, i)
            except AssertionError:
                continue
            if doc_result is None:
                continue
            mutated, canonical = list(docs), list(docs)
            mutated[i], canonical[i] = doc_result.mutated_doc, doc_result.canonical
            result = FileMutationResult(mutated, canonical, doc_result.findings, doc_result.patch, doc_result.new_resources)
            break
    if result is None:
        return None
    added = _file_rule_counts(result.mutated_docs) - _file_rule_counts(result.canonical_docs)
    expected = collections.Counter({_step_rule_id(step): len(result.findings)})
    if added != expected or (_file_rule_counts(result.canonical_docs) - _file_rule_counts(result.mutated_docs)):
        return None
    return result


def mutate_single_defect_file(
    docs: list[dict], rng: random.Random, rule_id: str, ksec001_candidate_names=None
) -> FileMutationResult | None:
    """One defect of `rule_id` injected somewhere in a multi-document file.
    KSEC-006 is either its single-document form (a workload's own selector)
    or the cross-document one (a Service's), chosen at random."""
    if _file_rule_counts(docs):
        return None
    steps = [rule_id]
    if rule_id == "KSEC-006":
        steps.append(SERVICE_SELECTOR_STEP)
        rng.shuffle(steps)
    for step in steps:
        result = _apply_file_step(docs, rng, step, ksec001_candidate_names)
        if result is not None:
            return result
    return None


def mutate_multi_defect_file(
    docs: list[dict],
    rng: random.Random,
    min_defects: int = 2,
    max_defects: int = 4,
    ksec001_candidate_names=None,
) -> MultiFileMutationResult | None:
    """mutate_multi_defect for a whole multi-document file: each step is a
    per-document mutator on some member, or the cross-document Service
    mutation. Same chaining of inverse patches and fixed-forward canonical
    deltas, kept per document."""
    if _file_rule_counts(docs):
        return None

    steps = (list(MUTATORS) + [SERVICE_SELECTOR_STEP]) * MAX_REPEATS_PER_RULE
    target_count = _sample_defect_count(rng, min_defects, min(max_defects, len(steps)))
    steps = _weighted_order(steps, rng, RULE_WEIGHTS)

    current = list(docs)
    running_canonical = copy.deepcopy(docs)
    applied_rule_ids: list[str] = []
    patch: list[PatchOp] = []
    new_resources: list[str] = []
    expected_findings = 0

    for step in steps:
        if len(applied_rule_ids) >= target_count:
            break
        result = _apply_file_step(current, rng, step, ksec001_candidate_names)
        if result is None:
            continue
        new_patch = result.patch + patch
        new_canonical = list(running_canonical)
        try:
            for i, (before, target) in enumerate(zip(current, result.canonical_docs)):
                delta = jsonpatch.make_patch(before, target).patch
                if delta:
                    new_canonical[i] = jsonpatch.apply_patch(new_canonical[i], delta)
        except Exception:
            continue
        if not _round_trips(result.mutated_docs, new_patch, new_canonical):
            continue
        patch = new_patch
        new_resources += result.new_resources
        running_canonical = new_canonical
        current = result.mutated_docs
        applied_rule_ids.append(_step_rule_id(step))
        expected_findings += len(result.findings)

    if len(applied_rule_ids) < max(1, min_defects):
        return None

    findings = [f for f in detect_file(current) if f.rule_id in RULE_IDS]
    assert len(findings) == expected_findings, (
        f"expected {expected_findings} findings after injecting {applied_rule_ids}, got {len(findings)}"
    )
    return MultiFileMutationResult(
        mutated_docs=current,
        canonical_docs=running_canonical,
        findings=findings,
        patch=patch,
        new_resources=new_resources,
        applied_rule_ids=applied_rule_ids,
    )
