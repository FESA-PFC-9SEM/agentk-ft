"""
Corpus-derived vocabulary of names (command binaries, path directories,
common path prefixes) and the "misspelled name" check KSEC-012 is built on.

The idea is deliberately generic, not a hand-written list of mistakes: a
token is suspicious when it's RARE in the real corpus yet one small edit
away from a token that's VERY COMMON there -- `python5` next to `python3`,
`/hom/` next to `/home/`. A path whose tail is a very common path but whose
first directory is unknown (`/variavel/run/secrets/kubernetes.io/...` for
`/var/run/secrets/kubernetes.io/...`) is flagged the same way. Rarity and
commonness come from counts over the deduplicated corpus
(dataset/name_vocab.json, rebuilt with `python -m dataset.names`), so the
check adapts to whatever the corpus treats as normal.
"""

from __future__ import annotations

import argparse
import collections
import functools
import json
import re
from pathlib import Path

from dataset.k8s import get_pod_spec, iter_containers

VOCAB_PATH = Path(__file__).with_name("name_vocab.json")

# A token seen at least this many times in the corpus is "known" -- never
# reported as a misspelling of something else.
KNOWN_MIN = 3
# A token must be seen at least this many times to be the suggested
# correction ("very common").
COMMON_MIN = 30
# Shortest token checked: very short names (sh, ssh, cp, npx) are one edit
# away from too many others to say anything. Binary names need one more
# character than path directories for the same reason (gcp/cp, kpm/npm).
MIN_LEN = 3
MIN_BINARY_LEN = 4
# Path prefixes this deep or deeper are matched for the wrong-first-directory
# case (the tail after the first directory must be at least 3 segments).
MIN_PREFIX_DEPTH = 4
MAX_PREFIX_DEPTH = 6

PATH_RE = re.compile(r"(?:^|(?<=[\s=:\"'(,]))(/[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*)")


def one_edit_apart(a: str, b: str) -> bool:
    """True if a and b differ by exactly one insertion, deletion,
    substitution, or swap of two adjacent characters."""
    if a == b:
        return False
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = [i for i in range(la) if a[i] != b[i]]
        if len(diff) == 1:
            return True
        return len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]
    if la > lb:
        a, b = b, a
    for i in range(len(b)):
        if b[:i] + b[i + 1 :] == a:
            return True
    return False


def command_binary(container: dict) -> str | None:
    """Basename of the executable a container's `command` starts (None if
    there's no literal command, or it's a shell expression)."""
    command = container.get("command")
    if not isinstance(command, list) or not command or not isinstance(command[0], str):
        return None
    first = command[0].strip()
    if not first or any(c in first for c in " $`{}()"):
        return None
    return first.rsplit("/", 1)[-1]


def container_strings(cpath: str, container: dict):
    """(json pointer, string) for the free-text fields a misspelled path or
    binary can hide in: command, args and literal env values."""
    for key in ("command", "args"):
        values = container.get(key)
        if isinstance(values, list):
            for i, value in enumerate(values):
                if isinstance(value, str):
                    yield f"{cpath}/{key}/{i}", value
    env = container.get("env")
    if isinstance(env, list):
        for i, entry in enumerate(env):
            if isinstance(entry, dict) and isinstance(entry.get("value"), str):
                yield f"{cpath}/env/{i}/value", entry["value"]


def paths_in(text: str) -> list[str]:
    return PATH_RE.findall(text)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


def build_vocab(docs) -> dict:
    binaries: collections.Counter = collections.Counter()
    segments: collections.Counter = collections.Counter()
    prefixes: collections.Counter = collections.Counter()
    for doc in docs:
        pod_spec, prefix = get_pod_spec(doc)
        if pod_spec is None:
            continue
        for cpath, container in iter_containers(pod_spec, prefix):
            binary = command_binary(container)
            if binary:
                binaries[binary] += 1
            texts = [value for _, value in container_strings(cpath, container)]
            for mount in container.get("volumeMounts") or []:
                if isinstance(mount, dict) and isinstance(mount.get("mountPath"), str):
                    texts.append(mount["mountPath"])
            if isinstance(container.get("workingDir"), str):
                texts.append(container["workingDir"])
            for text in texts:
                for path in paths_in(text):
                    parts = path.strip("/").split("/")
                    segments[parts[0]] += 1
                    for depth in range(MIN_PREFIX_DEPTH, min(len(parts), MAX_PREFIX_DEPTH) + 1):
                        prefixes["/".join(parts[:depth])] += 1
    keep = lambda counter: {k: v for k, v in sorted(counter.items()) if v >= KNOWN_MIN}
    return {"binaries": keep(binaries), "path_segments": keep(segments), "path_prefixes": keep(prefixes)}


@functools.lru_cache(maxsize=1)
def load_vocab() -> dict:
    vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    common_tails: dict[str, str] = {}
    for prefix, count in sorted(vocab["path_prefixes"].items(), key=lambda kv: -kv[1]):
        if count < COMMON_MIN:
            continue
        first, _, tail = prefix.partition("/")
        common_tails.setdefault(tail, first)
    vocab["common_tails"] = common_tails
    return vocab


_SEPARATORS = "_-."


def is_style_variant(a: str, b: str) -> bool:
    """Differences that are naming style, not typos -- measured on the real
    corpus, where these were the bulk of the matches: a numbered copy
    (/target1, /target2 next to /target), another separator (/work_dir vs
    /work-dir, init_container.sh), or a plural (/containers vs /container)."""
    short, long_ = sorted((a, b), key=len)
    if long_.startswith(short) and (long_[len(short):].isdigit() or long_[len(short):] == "s"):
        return True
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        return len(diff) == 1 and a[diff[0]] in _SEPARATORS and b[diff[0]] in _SEPARATORS
    return False


def _nearest_common(token: str, counts: dict, min_len: int = MIN_LEN) -> str | None:
    if len(token) < min_len or counts.get(token, 0) >= KNOWN_MIN:
        return None
    best = None
    for candidate, count in counts.items():
        if count >= COMMON_MIN and one_edit_apart(token, candidate) and not is_style_variant(token, candidate):
            if best is None or count > counts[best]:
                best = candidate
    return best


def misspelled_binary(binary: str) -> str | None:
    """The very common binary `binary` is evidently a misspelling of, or None."""
    return _nearest_common(binary, load_vocab()["binaries"], MIN_BINARY_LEN)


def misspelled_path(path: str) -> str | None:
    """The corrected path if `path`'s first directory is a misspelling of a
    very common one (/hom/x -> /home/x), or if its tail is a very common
    path under an unknown first directory (/variavel/run/secrets/... ->
    /var/run/secrets/...); None otherwise."""
    vocab = load_vocab()
    parts = path.strip("/").split("/")
    first, rest = parts[0], parts[1:]
    if vocab["path_segments"].get(first, 0) >= KNOWN_MIN:
        return None
    fixed = _nearest_common(first, vocab["path_segments"])
    if fixed is None:
        for depth in range(min(len(parts), MAX_PREFIX_DEPTH), MIN_PREFIX_DEPTH - 1, -1):
            right = vocab["common_tails"].get("/".join(parts[1:depth]))
            if right and right != first:
                fixed = right
                break
    if fixed is None:
        return None
    return "/" + "/".join([fixed, *rest])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Rebuild dataset/name_vocab.json from the corpus.")
    parser.add_argument("--corpus-dir", default="corpus")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    from dataset.build import load_records
    from dataset.dedup import dedup

    records = load_records(Path(args.corpus_dir), args.limit)
    kept, _ = dedup([r.doc for r in records])
    vocab = build_vocab(records[i].doc for i in kept)
    VOCAB_PATH.write_text(json.dumps(vocab, indent=0, sort_keys=True), encoding="utf-8")
    print({k: len(v) for k, v in vocab.items()}, "->", VOCAB_PATH)


if __name__ == "__main__":
    main()
