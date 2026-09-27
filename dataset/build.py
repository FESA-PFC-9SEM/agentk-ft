"""
Pipeline orchestration: real corpus (parquet) -> dataset.jsonl.

Flow: load shards -> filter to valid manifests (skip Helm templates with
'{{' and YAML that doesn't parse) -> deduplicate structurally -> drop
documents with a real secret -> normalize into canonical form -> inject one
defect per rule (plus a slice of clean negatives) -> validate the round-trip
of every patch (100%, not sampled) -> write train/val/test.jsonl, split by
source repository (never randomly, to avoid leaking near-identical forks
across splits).

Usage:
    python -m dataset.build --limit 1000 --total 300
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import posixpath
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import jsonpatch
import pandas as pd
import yaml

from dataset.dedup import dedup, skeleton_hash
from dataset.detect import detect_file, detect_semantic, detect_structural
from dataset.k8s import POD_TEMPLATE_KINDS, get_pod_labels, get_service_selector, label_selector_matches
from dataset.multi_mutate import mutate_multi_defect, mutate_multi_defect_file, mutate_single_defect_file
from dataset.mutate import FAKE_SECRET_VAR_NAMES, MUTATORS
from dataset.normalize import normalize_document
from dataset.scanning import find_secrets
from dataset.schema import SYSTEM_PROMPT, Response, validate_response

RULE_IDS = tuple(MUTATORS)

_IDENTIFIER_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{2,39}$")


def _is_harvestable_key_name(key: str, reason: str) -> bool:
    """True if a SecretHit's key (never its value) looks like a real
    variable/field name worth reusing as an injectable KSEC-001 identifier.
    Excludes high-entropy-only hits, where the key isn't required to look
    sensitive at all and could be a list index or an unrelated field name --
    only hits scanning.py matched on the key itself (sensitive-key regex,
    PEM, connection string) have a key worth harvesting."""
    if reason.startswith("high entropy"):
        return False
    return bool(_IDENTIFIER_NAME_RE.match(key))


@dataclass
class Record:
    doc: dict
    repo: str
    path: str


@dataclass
class Bundle:
    """A multi-document file assembled from sibling files of one repository
    directory (see find_sibling_bundles)."""

    docs: list[dict]
    repo: str


# ---------------------------------------------------------------------------
# Loading and filtering
# ---------------------------------------------------------------------------


def load_records(corpus_dir: Path, limit: int | None) -> list[Record]:
    """Reads the parquet shards and returns one Record per valid YAML
    document (with 'kind' and 'apiVersion'), skipping Helm templates ('{{')
    and YAML that doesn't parse. `limit`, if given, caps the number of
    parquet ROWS read (not the final number of documents)."""
    records: list[Record] = []
    rows_read = 0
    shard_paths = sorted(corpus_dir.glob("*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(f"no parquet shard found in {corpus_dir}")

    for shard_path in shard_paths:
        df = pd.read_parquet(shard_path, columns=["content", "max_stars_repo_name", "max_stars_repo_path"])
        for _, row in df.iterrows():
            if limit is not None and rows_read >= limit:
                return records
            rows_read += 1
            text = row["content"]
            if not isinstance(text, str) or "{{" in text:
                continue
            try:
                loaded = list(yaml.safe_load_all(text))
            except yaml.YAMLError:
                continue
            for doc in loaded:
                if isinstance(doc, dict) and "kind" in doc and "apiVersion" in doc:
                    records.append(
                        Record(
                            doc=doc,
                            repo=str(row["max_stars_repo_name"]),
                            path=str(row["max_stars_repo_path"]),
                        )
                    )
    return records


def load_synthetic_records(synthetic_dir: Path) -> list[Record]:
    """Reads every *.curated.jsonl from generation/output/ (already filtered
    by curate.py) and returns one Record per document. Each synthetic
    document gets its own unique fake "repository" for split purposes -- it
    already went through structural dedup in curate.py, so there's no risk
    of leaking near-duplicates across train/val/test the way real forks
    would."""
    records: list[Record] = []
    if not synthetic_dir.exists():
        return records
    for jsonl_path in sorted(synthetic_dir.glob("*.curated.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                mode = item.get("mode", "synthetic")
                index = item.get("index", 0)
                try:
                    docs = list(yaml.safe_load_all(item.get("manifest_yaml", "")))
                except yaml.YAMLError:
                    continue
                for doc_idx, doc in enumerate(docs):
                    if isinstance(doc, dict) and "kind" in doc and "apiVersion" in doc:
                        records.append(
                            Record(
                                doc=doc,
                                repo=f"synthetic:{mode}:{index}:{doc_idx}",
                                path=jsonl_path.name,
                            )
                        )
    return records


# ---------------------------------------------------------------------------
# Round-trip and serialization
# ---------------------------------------------------------------------------


def assert_round_trip(mutated_doc: dict, patch, canonical: dict, context: str) -> None:
    ops = [{k: v for k, v in p.to_dict().items() if k != "doc"} for p in patch]
    reconstructed = jsonpatch.apply_patch(mutated_doc, ops)
    if reconstructed != canonical:
        raise RuntimeError(f"round-trip failure in {context}: patch does not reproduce the canonical form")


def assert_round_trip_file(mutated_docs: list[dict], patch, canonical_docs: list[dict], context: str) -> None:
    """assert_round_trip for a multi-document file: each op applies to the
    document its `doc` index names."""
    reconstructed = list(mutated_docs)
    for i in range(len(mutated_docs)):
        ops = [{k: v for k, v in p.to_dict().items() if k != "doc"} for p in patch if p.doc == i]
        if ops:
            reconstructed[i] = jsonpatch.apply_patch(mutated_docs[i], ops)
    if reconstructed != canonical_docs or any(p.doc >= len(mutated_docs) for p in patch):
        raise RuntimeError(f"round-trip failure in {context}: patch does not reproduce the canonical form")


def doc_to_yaml(doc: dict | list) -> str:
    """One document as YAML, or a list of them as a multi-document file."""
    if isinstance(doc, list):
        return "---\n".join(doc_to_yaml(d) for d in doc)
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)


def make_example(doc_for_input: dict | list, response: Response, repo: str, rule_id: str) -> dict:
    errors = validate_response(response.to_dict())
    if errors:
        raise RuntimeError(f"response fails schema validation ({rule_id}, repo={repo}): {errors}")
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": doc_to_yaml(doc_for_input)},
            {"role": "assistant", "content": json.dumps(response.to_dict(), ensure_ascii=False)},
        ]
    }


# ---------------------------------------------------------------------------
# Multi-document files ("sibling bundles")
# ---------------------------------------------------------------------------

# Probability of adding one unrelated sibling (ConfigMap, HPA, Ingress, ...)
# to a bundle, so the documents carrying defects aren't always at index 0/1.
_EXTRA_SIBLING_PROBABILITY = 0.3


def find_sibling_bundles(records: list[Record], rng: random.Random) -> list[Bundle]:
    """Multi-document files assembled from the corpus, which stores one
    document per row (real multi-document files are split upstream). A
    workload and the Service selecting it usually live in sibling files of
    one repository directory (deployment.yaml + service.yaml), so each
    Service there that selects exactly ONE workload becomes a
    [workload, Service] file, in random order, sometimes with one extra
    sibling that neither creates pods nor is a Service (so it can't make the
    pairing ambiguous). Must run on records BEFORE structural dedup: dedup
    strips labels and selectors, so it collapses almost every Service into a
    handful of skeletons."""
    groups: dict[tuple[str, str], list[Record]] = collections.defaultdict(list)
    for r in records:
        if not r.repo.startswith("synthetic:"):
            groups[(r.repo, posixpath.dirname(r.path))].append(r)

    bundles = []
    for (repo, _), members in groups.items():
        workloads = [(r.doc, labels) for r in members if (labels := get_pod_labels(r.doc)[0]) is not None]
        if not workloads:
            continue
        extras = [r.doc for r in members if r.doc.get("kind") not in POD_TEMPLATE_KINDS and r.doc.get("kind") != "Service"]
        for r in members:
            selector = get_service_selector(r.doc)
            if selector is None:
                continue
            selected = [doc for doc, labels in workloads if label_selector_matches(selector, labels)]
            if len(selected) != 1:
                continue
            docs = [selected[0], r.doc]
            rng.shuffle(docs)
            if extras and rng.random() < _EXTRA_SIBLING_PROBABILITY:
                docs.insert(rng.randrange(len(docs) + 1), rng.choice(extras))
            bundles.append(Bundle(docs=docs, repo=repo))
    return bundles


def canonicalize_bundle(bundle: Bundle) -> Bundle | None:
    """The same per-document gauntlet build() runs (no plaintext secret,
    normalizable, no pre-existing semantic finding) for every member, plus
    no finding at FILE level -- e.g. a cross-document selector mismatch.
    None if any member or the file as a whole fails."""
    canonical_docs = []
    for doc in bundle.docs:
        if find_secrets(doc):
            return None
        canonical = normalize_document(doc)
        if canonical is None:
            return None
        if detect_structural(canonical):
            raise RuntimeError(f"bug in normalize.py: doc from {bundle.repo} is still dirty after normalization")
        canonical_docs.append(canonical)
    if any(f.rule_id in RULE_IDS for f in detect_file(canonical_docs)):
        return None
    return Bundle(docs=canonical_docs, repo=bundle.repo)


def usable_bundles(candidates: list[Bundle]) -> list[Bundle]:
    """Canonical bundles, deduplicated on their members' structural
    skeletons (the same notion of "duplicate" as for single documents)."""
    seen = set()
    out = []
    for candidate in candidates:
        bundle = canonicalize_bundle(candidate)
        if bundle is None:
            continue
        key = tuple(skeleton_hash(d) for d in bundle.docs)
        if key in seen:
            continue
        seen.add(key)
        out.append(bundle)
    return out


def bundle_examples(
    bundles: list[Bundle],
    strategy: str,
    positive_target: int,
    negative_target: int,
    rng: random.Random,
    ksec001_candidate_names,
    min_defects: int,
    max_defects: int,
) -> tuple[list[tuple[str, dict]], int, int]:
    """(examples, positives emitted, negatives emitted) from multi-document
    bundles. Positives follow the strategy (one defect, balanced across
    rules; or 2+ simultaneous defects anywhere in the file); negatives are
    the canonical bundles themselves."""
    examples: list[tuple[str, dict]] = []
    positives = 0
    if strategy == "single-defect":
        per_rule_cap = 2 * max(1, positive_target // len(RULE_IDS))
        pools: dict[str, list] = {rid: [] for rid in RULE_IDS}
        for rid in RULE_IDS:
            order = list(bundles)
            rng.shuffle(order)
            for b in order:
                if len(pools[rid]) >= per_rule_cap:
                    break
                try:
                    result = mutate_single_defect_file(b.docs, rng, rid, ksec001_candidate_names)
                except AssertionError:
                    continue
                if result is not None:
                    pools[rid].append((b, result))
        quotas = _resolve_quotas(positive_target, pools)
        for rid in RULE_IDS:
            for b, result in pools[rid][: quotas[rid]]:
                assert_round_trip_file(result.mutated_docs, result.patch, result.canonical_docs, f"{rid}/file/{b.repo}")
                response = Response(findings=result.findings, patch=result.patch, new_resources=result.new_resources)
                examples.append((b.repo, make_example(result.mutated_docs, response, b.repo, rid)))
                positives += 1
    else:
        order = list(bundles)
        rng.shuffle(order)
        for b in order:
            if positives >= positive_target:
                break
            try:
                result = mutate_multi_defect_file(
                    b.docs, rng, min_defects, max_defects, ksec001_candidate_names=ksec001_candidate_names
                )
            except AssertionError:
                continue
            if result is None:
                continue
            assert_round_trip_file(result.mutated_docs, result.patch, result.canonical_docs, f"multi/file/{b.repo}")
            response = Response(findings=result.findings, patch=result.patch, new_resources=result.new_resources)
            label = "multi-file:" + "+".join(sorted(result.applied_rule_ids))
            examples.append((b.repo, make_example(result.mutated_docs, response, b.repo, label)))
            positives += 1

    negatives = 0
    order = list(bundles)
    rng.shuffle(order)
    for b in order[:negative_target]:
        examples.append((b.repo, make_example(b.docs, Response(), b.repo, "negative")))
        negatives += 1
    return examples, positives, negatives


# ---------------------------------------------------------------------------
# Split by repository (deterministic, never random)
# ---------------------------------------------------------------------------


def split_bucket(repo: str, train_ratio: float, val_ratio: float) -> str:
    digest = hashlib.sha256(repo.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def build(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Reading corpus shards...", file=sys.stderr)
    records = load_records(corpus_dir, args.limit)
    corpus_docs_read = len(records)

    synthetic_docs_read = 0
    if args.synthetic_dir:
        synthetic_records = load_synthetic_records(Path(args.synthetic_dir))
        synthetic_docs_read = len(synthetic_records)
        records.extend(synthetic_records)
        print(f"Synthetic documents merged in: {synthetic_docs_read}", file=sys.stderr)

    docs_read = len(records)
    print(f"Valid documents read (corpus + synthetic): {docs_read}", file=sys.stderr)

    all_records = records  # pre-dedup: sibling bundles need the Services dedup collapses
    kept_idx, survival_rate = dedup([r.doc for r in records])
    records = [records[i] for i in kept_idx]
    unique_after_dedup = len(records)

    dirty_count = 0
    clean_records = []
    harvested_key_names: set[str] = set()
    for r in records:
        hits = find_secrets(r.doc)
        if hits:
            dirty_count += 1
            for hit in hits:
                if _is_harvestable_key_name(hit.key, hit.reason):
                    harvested_key_names.add(hit.key)
        else:
            clean_records.append(r)

    kind_distribution = collections.Counter(r.doc.get("kind", "?") for r in records)

    canonical_records: list[Record] = []
    dropped_unfixable = 0
    dropped_preexisting_semantic = 0
    for r in clean_records:
        canonical = normalize_document(r.doc)
        if canonical is None:
            dropped_unfixable += 1
            continue
        if detect_structural(canonical):
            raise RuntimeError(f"bug in normalize.py: doc from {r.repo} is still dirty after normalization")
        # normalize.py only guarantees rules 001-005; a real document can
        # already violate a semantic rule (006+). Kept, it would either become
        # a "clean" negative with a real unlabeled defect, or a single-defect
        # example whose label misses its second defect -- both teach the model
        # to ignore that defect.
        if any(f.rule_id in RULE_IDS for f in detect_semantic(canonical)):
            dropped_preexisting_semantic += 1
            continue
        canonical_records.append(Record(doc=canonical, repo=r.repo, path=r.path))

    # Union of the curated base pool with names actually seen in real corpus
    # documents (key names only -- never the leaked values, which were
    # already discarded above along with the whole document). This gives
    # KSEC-001 far more realistic lexical/naming-convention diversity than
    # the fixed base pool alone.
    ksec001_candidate_names = sorted(set(FAKE_SECRET_VAR_NAMES) | harvested_key_names)

    diagnostic = {
        "docs_read": docs_read,
        "corpus_docs_read": corpus_docs_read,
        "synthetic_docs_read": synthetic_docs_read,
        "unique_after_dedup": unique_after_dedup,
        "dedup_survival_rate": round(survival_rate, 4),
        "dropped_as_dirty_secret": dirty_count,
        "dropped_as_unfixable_rbac": dropped_unfixable,
        "dropped_preexisting_semantic_finding": dropped_preexisting_semantic,
        "usable_canonical_docs": len(canonical_records),
        "kind_distribution": dict(kind_distribution.most_common()),
        "harvested_credential_key_names": len(harvested_key_names),
        "ksec001_candidate_name_pool_size": len(ksec001_candidate_names),
    }
    print("\n=== Corpus diagnostic ===", file=sys.stderr)
    print(json.dumps(diagnostic, indent=2, ensure_ascii=False), file=sys.stderr)

    if not canonical_records:
        raise RuntimeError("no usable canonical document: pipeline cannot generate examples")

    strategy = getattr(args, "strategy", "single-defect")
    min_defects = getattr(args, "min_defects", 2)
    max_defects = getattr(args, "max_defects", 4)
    multi_doc_ratio = getattr(args, "multi_doc_ratio", 0.0)

    negative_target = round(args.total * args.negative_ratio)
    positive_target = args.total - negative_target

    # Multi-document examples first, with their own rng so that
    # --multi-doc-ratio 0 reproduces the single-document dataset exactly.
    examples = []
    docs_per_example: collections.Counter = collections.Counter()
    bundle_positives = bundle_negatives = 0
    if multi_doc_ratio > 0:
        bundle_rng = random.Random(args.seed + 1)
        candidates = find_sibling_bundles(all_records, bundle_rng)
        bundles = usable_bundles(candidates)
        diagnostic["bundle_candidates"] = len(candidates)
        diagnostic["usable_bundles"] = len(bundles)
        print(f"Multi-document bundles: {len(bundles)} usable of {len(candidates)} candidates", file=sys.stderr)
        if bundles:
            bundle_out, bundle_positives, bundle_negatives = bundle_examples(
                bundles,
                strategy,
                round(positive_target * multi_doc_ratio),
                min(round(negative_target * multi_doc_ratio), len(bundles)),
                bundle_rng,
                ksec001_candidate_names,
                min_defects,
                max_defects,
            )
            examples.extend(bundle_out)
            for _, example in bundle_out:
                docs_per_example[example["messages"][1]["content"].count("\n---\n") + 1] += 1
    doc_positive_target = positive_target - bundle_positives
    doc_negative_target = negative_target - bundle_negatives

    mutation_precondition_failures = 0

    if strategy == "single-defect":
        # Pool of applicable mutations per rule: tries to mutate every
        # canonical doc once; keeps the result so we don't mutate twice with
        # divergent rng states between the counting phase and the sampling
        # phase.
        pools: dict[str, list[tuple[Record, object]]] = {rid: [] for rid in RULE_IDS}
        order = list(canonical_records)
        rng.shuffle(order)
        for rid in RULE_IDS:
            mutator = MUTATORS[rid]
            for r in order:
                try:
                    if rid == "KSEC-001":
                        result = mutator(r.doc, rng, candidate_names=ksec001_candidate_names)
                    else:
                        result = mutator(r.doc, rng)
                except AssertionError:
                    # A mutator's own internal invariant didn't hold for this
                    # particular document (e.g. a malformed field from noisy
                    # generated input defeated the "this mutation always
                    # produces a finding" assumption). This is a per-document,
                    # per-rule skip, not a pipeline failure: the same document
                    # is still tried against every other rule normally.
                    mutation_precondition_failures += 1
                    continue
                if result is not None:
                    pools[rid].append((r, result))

        quotas = _resolve_quotas(doc_positive_target, pools)
        quotas["negative"] = doc_negative_target

        for rid in RULE_IDS:
            pool = pools[rid]
            rng.shuffle(pool)
            for r, result in pool[: quotas[rid]]:
                assert_round_trip(result.mutated_doc, result.patch, result.canonical, context=f"{rid}/{r.repo}")
                response = Response(
                    findings=result.findings,
                    patch=result.patch,
                    new_resources=result.new_resources,
                    notes=[],
                )
                example = make_example(result.mutated_doc, response, r.repo, rid)
                examples.append((r.repo, example))
    else:
        # Multi-defect strategy: each example carries 2+ SIMULTANEOUS
        # findings, composed by dataset/multi_mutate.py from the same
        # per-rule mutators used above -- see README.md's "Dataset
        # generation strategies" for why this exists (a model trained only
        # on single-defect examples missed a second finding when a
        # real-world manifest actually had two).
        quotas = {"positive": doc_positive_target, "negative": doc_negative_target}

        order = list(canonical_records)
        rng.shuffle(order)
        defect_count_distribution: collections.Counter = collections.Counter()
        doc_positives = 0
        for r in order:
            if doc_positives >= doc_positive_target:
                break
            try:
                result = mutate_multi_defect(
                    r.doc,
                    rng,
                    min_defects=min_defects,
                    max_defects=max_defects,
                    ksec001_candidate_names=ksec001_candidate_names,
                )
            except AssertionError:
                mutation_precondition_failures += 1
                continue
            if result is None:
                continue
            assert_round_trip(result.mutated_doc, result.patch, result.canonical, context=f"multi/{r.repo}")
            response = Response(
                findings=result.findings,
                patch=result.patch,
                new_resources=result.new_resources,
                notes=[],
            )
            rule_label = "multi:" + "+".join(sorted(result.applied_rule_ids))
            example = make_example(result.mutated_doc, response, r.repo, rule_label)
            examples.append((r.repo, example))
            doc_positives += 1
            defect_count_distribution[len(result.applied_rule_ids)] += 1

    negative_quota = quotas["negative"]
    negative_pool = list(canonical_records)
    rng.shuffle(negative_pool)
    for r in negative_pool[:negative_quota]:
        response = Response()
        example = make_example(r.doc, response, r.repo, "negative")
        examples.append((r.repo, example))

    rng.shuffle(examples)

    splits = {"train": [], "val": [], "test": []}
    for repo, example in examples:
        splits[split_bucket(repo, args.train_ratio, args.val_ratio)].append(example)

    for split_name, split_examples in splits.items():
        out_path = output_dir / f"{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for example in split_examples:
                fh.write(json.dumps(example, ensure_ascii=False) + "\n")

    diagnostic["strategy"] = strategy
    diagnostic["quotas"] = quotas
    diagnostic["mutation_precondition_failures"] = mutation_precondition_failures
    diagnostic["examples_emitted"] = {k: len(v) for k, v in splits.items()}
    diagnostic["examples_emitted"]["total"] = sum(len(v) for v in splits.values())
    docs_per_example[1] = diagnostic["examples_emitted"]["total"] - sum(docs_per_example.values())
    diagnostic["multi_doc_examples"] = {"positive": bundle_positives, "negative": bundle_negatives}
    diagnostic["docs_per_example"] = dict(sorted(docs_per_example.items()))
    if strategy == "multi-defect":
        diagnostic["defect_count_distribution"] = dict(sorted(defect_count_distribution.items()))

    diag_path = output_dir / "diagnostic.json"
    diag_path.write_text(json.dumps(diagnostic, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Examples emitted ===", file=sys.stderr)
    print(json.dumps(diagnostic["examples_emitted"], indent=2), file=sys.stderr)
    if mutation_precondition_failures:
        print(
            f"\n[WARNING] {mutation_precondition_failures} mutation attempt(s) hit an internal "
            "invariant and were skipped (see mutation_precondition_failures in diagnostic.json). "
            "A handful is expected from noisy generated input; a large count is worth investigating.",
            file=sys.stderr,
        )
    print(f"\nSplits written to {output_dir}/", file=sys.stderr)

    return diagnostic


def _resolve_quotas(positive_target: int, pools: dict[str, list]) -> dict[str, int]:
    """Decides how many positive examples per rule, splitting positive_target
    evenly but never exceeding how much each rule actually managed to mutate
    (available pool)."""
    per_rule_target = positive_target // len(RULE_IDS)

    quotas = {}
    for rid in RULE_IDS:
        available = len(pools[rid])
        quotas[rid] = min(per_rule_target, available)

    # redistributes what's left over (rules with a small pool) to the
    # others, as far as available material allows.
    leftover = positive_target - sum(quotas.values())
    if leftover > 0:
        hungry = sorted(RULE_IDS, key=lambda rid: len(pools[rid]) - quotas[rid], reverse=True)
        for rid in hungry:
            room = len(pools[rid]) - quotas[rid]
            take = min(room, leftover)
            quotas[rid] += take
            leftover -= take
            if leftover <= 0:
                break

    return quotas


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", default="corpus")
    parser.add_argument("--synthetic-dir", default=None, help="folder with *.curated.jsonl from generation/curate.py")
    parser.add_argument(
        "--strategy",
        choices=["single-defect", "multi-defect"],
        default="single-defect",
        help="single-defect (default): one injected finding per example. "
        "multi-defect: 2+ simultaneous findings per example, see README.md's "
        "'Dataset generation strategies' section",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="defaults to dataset/output for --strategy single-defect, "
        "dataset/output-multi-defect for --strategy multi-defect",
    )
    parser.add_argument("--min-defects", type=int, default=2, help="multi-defect strategy only")
    parser.add_argument("--max-defects", type=int, default=4, help="multi-defect strategy only")
    parser.add_argument("--limit", type=int, default=None, help="cap on parquet rows read (smoke test)")
    parser.add_argument("--total", type=int, default=2000, help="total number of examples to generate")
    parser.add_argument("--negative-ratio", type=float, default=0.35)
    parser.add_argument(
        "--multi-doc-ratio",
        type=float,
        default=0.3,
        help="share of positives and negatives built as multi-document files from sibling files "
        "(a workload + the Service selecting it); 0 = single-document examples only",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = "dataset/output" if args.strategy == "single-defect" else "dataset/output-multi-defect"
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    build(args)


if __name__ == "__main__":
    main()
