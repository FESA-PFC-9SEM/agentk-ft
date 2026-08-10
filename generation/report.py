"""
Diversity metrics for the synthetic manifests, to catch generation collapse
(the classic "a thousand variations of nginx in the default namespace")
before spending hours of GPU time generating the whole batch at scale.

Usage:
    python -m generation.report --input generation/output/base.curated.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from dataset.dedup import dedup, skeleton_hash
from dataset.k8s import get_pod_spec, iter_containers

_TOKEN_RE = re.compile(r"[A-Za-z0-9_.\-/:]+")


def _load_records(input_path: Path) -> list[dict]:
    records = []
    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _first_doc(record: dict) -> dict | None:
    try:
        docs = [d for d in yaml.safe_load_all(record["manifest_yaml"]) if isinstance(d, dict)]
    except yaml.YAMLError:
        return None
    return docs[0] if docs else None


def _all_docs(record: dict) -> list[dict]:
    try:
        return [d for d in yaml.safe_load_all(record["manifest_yaml"]) if isinstance(d, dict)]
    except yaml.YAMLError:
        return []


def _top_ngrams(texts: list[str], n: int = 3, top_k: int = 15) -> list[tuple[str, int]]:
    counter = collections.Counter()
    for text in texts:
        tokens = _TOKEN_RE.findall(text)
        for i in range(len(tokens) - n + 1):
            counter[" ".join(tokens[i : i + n])] += 1
    return counter.most_common(top_k)


def compute_metrics(records: list[dict]) -> dict:
    docs_by_record = [(_first_doc(r), r) for r in records]
    docs_by_record = [(d, r) for d, r in docs_by_record if d is not None]

    kind_dist = collections.Counter(d.get("kind", "?") for d, _ in docs_by_record)
    namespace_dist = collections.Counter(
        d.get("metadata", {}).get("namespace", "(default)") for d, _ in docs_by_record
    )

    image_dist = collections.Counter()
    container_counts = collections.Counter()
    for d, _ in docs_by_record:
        pod_spec, prefix = get_pod_spec(d)
        if pod_spec is None:
            continue
        containers = list(iter_containers(pod_spec, prefix))
        container_counts[len(containers)] += 1
        for _cpath, container in containers:
            image = container.get("image")
            if isinstance(image, str):
                repo = image.split("@")[0].rsplit(":", 1)[0]
                image_dist[repo] += 1

    all_docs = [d for r in records for d in _all_docs(r)]
    kept_idx, structural_uniqueness_ratio = dedup(all_docs)

    seed_archetypes = collections.defaultdict(set)
    for record in records:
        seed = record.get("seed", {})
        archetype = (seed.get("domain"), seed.get("stack"), seed.get("kind"))
        doc = _first_doc(record)
        if doc is not None:
            seed_archetypes[archetype].add(skeleton_hash(doc))
    unique_manifests_per_seed = {
        " / ".join(str(p) for p in k): len(v) for k, v in sorted(seed_archetypes.items(), key=lambda kv: -len(kv[1]))
    }

    texts = [r.get("manifest_yaml", "") for r in records]
    top_ngrams = _top_ngrams(texts)

    return {
        "total_records": len(records),
        "kind_distribution": dict(kind_dist.most_common()),
        "namespace_distribution": dict(namespace_dist.most_common(20)),
        "image_distribution": dict(image_dist.most_common(20)),
        "container_count_distribution": dict(sorted(container_counts.items())),
        "structural_uniqueness_ratio": round(structural_uniqueness_ratio, 4),
        "unique_manifests_per_seed_archetype": unique_manifests_per_seed,
        "top_ngrams": top_ngrams,
    }


def write_csv(metrics: dict, output_dir: Path) -> None:
    csv_path = output_dir / "report.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write("metric,key,value\n")
        for metric_name in ("kind_distribution", "namespace_distribution", "image_distribution", "container_count_distribution"):
            for key, value in metrics[metric_name].items():
                fh.write(f'{metric_name},"{key}",{value}\n')
        fh.write(f"structural_uniqueness_ratio,-,{metrics['structural_uniqueness_ratio']}\n")
        for phrase, count in metrics["top_ngrams"]:
            fh.write(f'top_ngram,"{phrase}",{count}\n')


def write_png(metrics: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def _bar(ax, dist: dict, title: str, top_k: int = 12):
        items = list(dist.items())[:top_k]
        if not items:
            ax.set_title(f"{title} (no data)")
            return
        labels, values = zip(*items)
        ax.barh([str(l) for l in labels][::-1], values[::-1])
        ax.set_title(title)

    _bar(axes[0][0], metrics["kind_distribution"], "Kind distribution")
    _bar(axes[0][1], metrics["namespace_distribution"], "Namespace distribution")
    _bar(axes[1][0], metrics["image_distribution"], "Image distribution")
    _bar(axes[1][1], metrics["container_count_distribution"], "Containers per manifest")

    fig.tight_layout()
    fig.savefig(output_dir / "distributions.png", dpi=120)
    plt.close(fig)


def report(args: argparse.Namespace) -> dict:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "report"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _load_records(input_path)
    metrics = compute_metrics(records)

    write_csv(metrics, output_dir)
    write_png(metrics, output_dir)

    print(json.dumps({k: v for k, v in metrics.items() if k != "unique_manifests_per_seed_archetype"}, indent=2, ensure_ascii=False))
    print(f"\nCSV and PNG written to {output_dir}/")
    return metrics


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    report(args)


if __name__ == "__main__":
    main()
