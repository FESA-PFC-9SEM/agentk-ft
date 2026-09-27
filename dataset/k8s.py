"""
Kubernetes manifest navigation helpers, shared by detect.py, normalize.py and
mutate.py. The main job here is resolving, for any kind with a pod template,
where the PodSpec actually lives -- especially the CronJob case, which nests
the PodSpec four levels below spec (spec.jobTemplate.spec.template.spec).
"""

from __future__ import annotations

from dataclasses import dataclass

POD_TEMPLATE_KINDS = frozenset(
    {"Pod", "Deployment", "StatefulSet", "DaemonSet", "ReplicaSet", "ReplicationController", "Job", "CronJob"}
)
RBAC_ROLE_KINDS = frozenset({"Role", "ClusterRole"})
RBAC_BINDING_KINDS = frozenset({"RoleBinding", "ClusterRoleBinding"})
# Kinds where spec.selector.matchLabels must select the pod template's own
# labels (spec.template.metadata.labels), in the same document. Job/CronJob
# are excluded: their selector is normally auto-populated/immutable rather
# than hand-written, so a mismatch there isn't the same kind of human error.
# ReplicationController (the legacy ReplicaSet) is included: its selector is
# a flat label map rather than matchLabels, see get_selector_match_labels.
WORKLOAD_SELECTOR_KINDS = frozenset({"Deployment", "StatefulSet", "DaemonSet", "ReplicaSet", "ReplicationController"})

SENSITIVE_HOST_PATHS = (
    "/",
    "/etc",
    "/var/run/docker.sock",
    "/proc",
    "/root",
    "/var/lib/kubelet",
    "/boot",
    "/sys",
    "/home",
)

DEFAULT_PINNED_TAG = "1.0.0"


def is_sensitive_host_path(path) -> bool:
    if not isinstance(path, str) or not path:
        return False
    p = path.rstrip("/") or "/"
    for sensitive in SENSITIVE_HOST_PATHS:
        s = sensitive.rstrip("/") or "/"
        if p == s or p.startswith(s + "/"):
            return True
    return False


def is_unpinned_image(image: str) -> bool:
    if "@" in image:  # digest-pinned
        return False
    tail = image[image.rfind("/") + 1 :]
    if ":" not in tail:
        return True
    return tail.rsplit(":", 1)[1] == "latest"


def split_image(image: str) -> tuple[str, str | None]:
    """Splits `image` into (repository, tag). tag is None if absent."""
    tail = image[image.rfind("/") + 1 :]
    if ":" not in tail:
        return image, None
    repo_len = len(image) - len(tail)
    name, tag = tail.rsplit(":", 1)
    return image[:repo_len] + name, tag


def image_basename(image) -> str | None:
    """Repository name only, without registry/namespace/tag/digest, lowercased:
    "docker.io/bitnami/postgresql:15@sha256:ab" -> "postgresql"."""
    if not isinstance(image, str) or not image:
        return None
    repo, _ = split_image(image.split("@", 1)[0])
    return repo.rsplit("/", 1)[-1].lower() or None


_DOCKER_HUB_REGISTRIES = frozenset({"docker.io", "index.docker.io", "registry-1.docker.io"})


def image_registry_and_path(image) -> tuple[str, tuple[str, ...]] | None:
    """(registry, lowercased repository path) with tag/digest stripped and
    Docker Hub's implicit defaults made explicit: "postgres" and
    "docker.io/library/postgres" both give ("docker.io", ("postgres",)),
    "bitnami/redis" gives ("docker.io", ("bitnami", "redis"))."""
    if not isinstance(image, str) or not image:
        return None
    repo, _ = split_image(image.split("@", 1)[0])
    parts = repo.lower().split("/")
    registry = "docker.io"
    if len(parts) > 1 and ("." in parts[0] or ":" in parts[0] or parts[0] == "localhost"):
        registry = "docker.io" if parts[0] in _DOCKER_HUB_REGISTRIES else parts[0]
        parts = parts[1:]
    if registry == "docker.io" and len(parts) == 2 and parts[0] == "library":
        parts = parts[1:]
    if not parts or not all(parts):
        return None
    return registry, tuple(parts)


def official_image_name(image) -> str | None:
    """Like image_basename, but only for Docker Official Images ("postgres",
    "library/postgres", "docker.io/library/postgres"); None for anything under
    a vendor namespace or another registry. Vendor rebuilds of the same
    software (bitnami/postgresql, ...) often use different env var names, so
    rules about an image's own entrypoint contract must not match them."""
    located = image_registry_and_path(image)
    if located is None or located[0] != "docker.io" or len(located[1]) != 1:
        return None
    return located[1][0]


# Servers that speak their own wire protocol, not HTTP: an httpGet probe
# against one can never succeed. Keyed by image_basename, so vendor rebuilds
# (bitnami/postgresql) match too -- the protocol doesn't change with the
# packager. Exporters/sidecars that do serve HTTP have their own basenames
# (redis-exporter, postgres-exporter) and deliberately don't match. The value
# is the server's default port, used when a fix has to add a probe from scratch.
NON_HTTP_IMAGE_PORTS = {
    "postgres": 5432,
    "postgresql": 5432,
    "mysql": 3306,
    "mariadb": 3306,
    "redis": 6379,
    "mongo": 27017,
    "mongodb": 27017,
    "memcached": 11211,
}

@dataclass(frozen=True)
class EnvRequirement:
    """One env var an image's entrypoint refuses to start without. `primary`
    is what a fix adds; any var in `accepted` satisfies it (aliases, *_FILE,
    explicit allow-empty/random opt-outs). A `literal` requirement is a
    setting rather than a secret (ACCEPT_EULA=Y), so a fix adds it as a
    plain value instead of a secretKeyRef. `only_on_init`: the entrypoint
    only checks it when initializing an empty data directory."""

    primary: str
    accepted: frozenset
    literal: str | None = None
    only_on_init: bool = True


@dataclass(frozen=True)
class ImageEnvContract:
    """What KSEC-011 knows about one image's startup contract.
    - `server_binaries`: first args that still start the server (anything
      else non-flag-like is treated as a client job and skipped);
    - `data_dir` (+ `data_dir_env`, an env override such as PGDATA): where
      the entrypoint looks for an existing database; None if it checks its
      requirements on every start regardless of data;
    - `role_env`: (var, values meaning "primary") -- a container whose var is
      set to anything else is a replica/secondary, which reads a different
      variable (bitnami's *_MASTER_* / INITIAL_PRIMARY_*), so it's skipped."""

    label: str
    requirements: tuple
    server_binaries: frozenset = frozenset()
    data_dir: str | None = None
    data_dir_env: str | None = None
    role_env: tuple = ()


def _password(primary: str, *accepted: str) -> EnvRequirement:
    return EnvRequirement(primary, frozenset({primary, *accepted}))


def _bitnami_password(primary: str, *accepted: str) -> EnvRequirement:
    # Bitnami validates on every start, not only on first initialization.
    return EnvRequirement(
        primary, frozenset({primary, f"{primary}_FILE", "ALLOW_EMPTY_PASSWORD", *accepted}), only_on_init=False
    )


_MYSQL_ROOT = _password(
    "MYSQL_ROOT_PASSWORD", "MYSQL_ROOT_PASSWORD_FILE", "MYSQL_ALLOW_EMPTY_PASSWORD", "MYSQL_RANDOM_ROOT_PASSWORD"
)
_MYSQL_CONTRACT = dict(requirements=(_MYSQL_ROOT,), server_binaries=frozenset({"mysqld"}), data_dir="/var/lib/mysql")
_BITNAMI_PRIMARY = frozenset({"master", "primary"})

# Keyed by required_env_image_key. Three image families, matched strictly
# because each family names its variables differently:
# - Docker Official Images (postgres/mysql/mariadb/percona) and Percona's own
#   percona/percona-server (same entrypoint as the official mysql image);
# - Bitnami on Docker Hub (bitnami/*), which uses its own names but also
#   accepts the official POSTGRES_* aliases -- the Bitnami chart itself sets
#   POSTGRES_PASSWORD -- and ALLOW_EMPTY_PASSWORD=yes as the opt-out;
# - Microsoft SQL Server on mcr.microsoft.com, which needs the EULA accepted
#   AND an SA password -- two independent requirements.
IMAGE_ENV_CONTRACTS = {
    "postgres": ImageEnvContract(
        "postgres",
        (_password("POSTGRES_PASSWORD", "POSTGRES_PASSWORD_FILE", "POSTGRES_HOST_AUTH_METHOD"),),
        server_binaries=frozenset({"postgres"}),
        data_dir="/var/lib/postgresql/data",
        data_dir_env="PGDATA",
    ),
    "mysql": ImageEnvContract("mysql", **_MYSQL_CONTRACT),
    "percona": ImageEnvContract("percona", **_MYSQL_CONTRACT),
    "percona/percona-server": ImageEnvContract("percona/percona-server", **_MYSQL_CONTRACT),
    "mariadb": ImageEnvContract(
        "mariadb",
        (
            _password(
                "MARIADB_ROOT_PASSWORD",
                "MARIADB_ROOT_PASSWORD_FILE",
                "MARIADB_ALLOW_EMPTY_ROOT_PASSWORD",
                "MARIADB_RANDOM_ROOT_PASSWORD",
                *_MYSQL_ROOT.accepted,
            ),
        ),
        server_binaries=frozenset({"mysqld", "mariadbd"}),
        data_dir="/var/lib/mysql",
    ),
    "bitnami/postgresql": ImageEnvContract(
        "bitnami/postgresql",
        (_bitnami_password("POSTGRES_PASSWORD", "POSTGRESQL_PASSWORD", "POSTGRESQL_PASSWORD_FILE"),),
        role_env=(("POSTGRESQL_REPLICATION_MODE", _BITNAMI_PRIMARY), ("POSTGRES_REPLICATION_MODE", _BITNAMI_PRIMARY)),
    ),
    "bitnami/mysql": ImageEnvContract(
        "bitnami/mysql",
        (_bitnami_password("MYSQL_ROOT_PASSWORD"),),
        role_env=(("MYSQL_REPLICATION_MODE", _BITNAMI_PRIMARY),),
    ),
    "bitnami/mariadb": ImageEnvContract(
        "bitnami/mariadb",
        (_bitnami_password("MARIADB_ROOT_PASSWORD", "MYSQL_ROOT_PASSWORD", "MYSQL_ROOT_PASSWORD_FILE"),),
        role_env=(("MARIADB_REPLICATION_MODE", _BITNAMI_PRIMARY),),
    ),
    "bitnami/redis": ImageEnvContract("bitnami/redis", (_bitnami_password("REDIS_PASSWORD"),)),
    "bitnami/mongodb": ImageEnvContract(
        "bitnami/mongodb",
        (_bitnami_password("MONGODB_ROOT_PASSWORD"),),
        role_env=(("MONGODB_REPLICA_SET_MODE", frozenset({"primary"})),),
    ),
    "mssql": ImageEnvContract(
        "mssql",
        (
            EnvRequirement("ACCEPT_EULA", frozenset({"ACCEPT_EULA"}), literal="Y", only_on_init=False),
            _password("MSSQL_SA_PASSWORD", "SA_PASSWORD", "MSSQL_SA_PASSWORD_FILE"),
        ),
        server_binaries=frozenset({"/opt/mssql/bin/sqlservr"}),
        data_dir="/var/opt/mssql",
    ),
}

_MSSQL_PATHS = frozenset({("mssql", "server"), ("mssql", "rhel", "server"), ("azure-sql-edge",)})


def required_env_image_key(image) -> str | None:
    """The IMAGE_ENV_CONTRACTS key for `image`, or None if it isn't one of
    the known images. Private mirrors and look-alike names
    (registry.corp/postgres, localhost:32000/percona, mcr's oss/bitnami
    mirror) deliberately don't match: the image's identity -- and so its
    entrypoint contract -- can't be confirmed from the name."""
    located = image_registry_and_path(image)
    if located is None:
        return None
    registry, parts = located
    if registry == "mcr.microsoft.com":
        return "mssql" if parts in _MSSQL_PATHS else None
    if registry != "docker.io":
        return None
    key = "/".join(parts)
    return key if key in IMAGE_ENV_CONTRACTS and key != "mssql" else None


def is_init_container_path(cpath: str) -> bool:
    return "/initContainers/" in cpath


def _paths_overlap(a: str, b: str) -> bool:
    a, b = a.rstrip("/") or "/", b.rstrip("/") or "/"
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def _env_value(container: dict, name: str):
    """The literal value of env var `name`; "" if set via valueFrom (present
    but unknowable); None if absent."""
    for e in container.get("env") or []:
        if isinstance(e, dict) and e.get("name") == name:
            return e["value"] if isinstance(e.get("value"), str) else ""
    return None


def _data_dir_may_be_prepopulated(contract: ImageEnvContract, container: dict, pod_spec: dict) -> bool:
    """True if an init container writes to the volume backing this server's
    data directory -- the replica-cloning pattern (xtrabackup `clone-mysql`,
    kubegres `setup-replica-data-directory`) that copies an existing
    database in before the server starts, so the entrypoint never asks for
    a password. Found on the real corpus: without this, those replicas were
    flagged as "will exit at startup", which they don't."""
    data_dir = contract.data_dir
    if data_dir is None:
        return False
    if contract.data_dir_env:
        data_dir = _env_value(container, contract.data_dir_env) or data_dir
    data_volumes = {
        m.get("name")
        for m in container.get("volumeMounts") or []
        if isinstance(m, dict) and isinstance(m.get("mountPath"), str) and _paths_overlap(m["mountPath"], data_dir)
    }
    for init in pod_spec.get("initContainers") or []:
        if not isinstance(init, dict):
            continue
        for m in init.get("volumeMounts") or []:
            if isinstance(m, dict) and m.get("name") in data_volumes and not m.get("readOnly"):
                return True
    return False


def required_env_for_container(cpath: str, container: dict, pod_spec: dict) -> tuple[str, tuple] | None:
    """(image label, the EnvRequirements that apply) if this container runs
    one of IMAGE_ENV_CONTRACTS' servers and at least one requirement can be
    checked from the manifest; None otherwise. Not checkable:
    - init containers (typically `pg_isready`-style waiters, not servers);
    - a `command` override or client-style args (psql/pg_dump jobs);
    - envFrom (the var may come from a ConfigMap/Secret we can't see);
    - imagePullPolicy: Never -- a locally built image that merely shares the
      official name (seen on the real corpus: custom "mysql" images with
      their own entrypoint), not the official image;
    - a replica role (see ImageEnvContract.role_env), or one set via
      valueFrom, whose value can't be read;
    - requirements the entrypoint only checks on first initialization, when
      an init container writes to the data directory's volume (see
      _data_dir_may_be_prepopulated)."""
    if is_init_container_path(cpath):
        return None
    key = required_env_image_key(container.get("image"))
    if key is None:
        return None
    contract = IMAGE_ENV_CONTRACTS[key]
    if container.get("command") or container.get("envFrom") or container.get("imagePullPolicy") == "Never":
        return None
    args = container.get("args")
    if isinstance(args, list) and args:
        first = str(args[0])
        if not first.startswith("-") and first not in contract.server_binaries:
            return None
    for var, primary_values in contract.role_env:
        value = _env_value(container, var)
        if value is not None and value.strip().lower() not in primary_values:
            return None
    requirements = contract.requirements
    if _data_dir_may_be_prepopulated(contract, container, pod_spec):
        requirements = tuple(r for r in requirements if not r.only_on_init)
    if not requirements:
        return None
    return contract.label, requirements


def get_pod_spec(doc) -> tuple[dict | None, str]:
    """Returns (pod_spec_dict, json_pointer_prefix) for the document's kind,
    or (None, "") if the kind has no PodSpec or the structure is missing."""
    if not isinstance(doc, dict):
        return None, ""
    kind = doc.get("kind")
    spec = doc.get("spec")

    if kind == "Pod":
        if isinstance(spec, dict):
            return spec, "/spec"
        return None, ""

    if kind in ("Deployment", "StatefulSet", "DaemonSet", "ReplicaSet", "ReplicationController", "Job"):
        if isinstance(spec, dict) and isinstance(spec.get("template"), dict):
            tspec = spec["template"].get("spec")
            if isinstance(tspec, dict):
                return tspec, "/spec/template/spec"
        return None, ""

    if kind == "CronJob":
        if isinstance(spec, dict) and isinstance(spec.get("jobTemplate"), dict):
            jspec = spec["jobTemplate"].get("spec")
            if isinstance(jspec, dict) and isinstance(jspec.get("template"), dict):
                tspec = jspec["template"].get("spec")
                if isinstance(tspec, dict):
                    return tspec, "/spec/jobTemplate/spec/template/spec"
        return None, ""

    return None, ""


def declared_volume_names(doc: dict, pod_spec: dict) -> set:
    """Every volume name a container in this pod may legitimately mount: the
    PodSpec's own `volumes`, plus -- for a StatefulSet -- the names of its
    `volumeClaimTemplates`, which the controller turns into per-replica PVC
    volumes that never appear under `volumes`. (Missing the latter made
    KSEC-009 flag 817 valid StatefulSets on the real corpus.)"""
    volumes = pod_spec.get("volumes")
    names = {v.get("name") for v in volumes if isinstance(v, dict)} if isinstance(volumes, list) else set()
    if doc.get("kind") == "StatefulSet":
        templates = (doc.get("spec") or {}).get("volumeClaimTemplates")
        if isinstance(templates, list):
            names |= {(t.get("metadata") or {}).get("name") for t in templates if isinstance(t, dict)}
    names.discard(None)
    return names


def iter_containers(pod_spec: dict, prefix: str):
    """Iterates containers and initContainers of a PodSpec, yielding
    (container_json_pointer_prefix, container_dict)."""
    for list_key in ("containers", "initContainers"):
        containers = pod_spec.get(list_key)
        if not isinstance(containers, list):
            continue
        for i, container in enumerate(containers):
            if isinstance(container, dict):
                yield f"{prefix}/{list_key}/{i}", container


def get_selector_match_labels(doc) -> tuple[dict | None, str]:
    """Returns (matchLabels_dict, json_pointer_prefix) for
    spec.selector.matchLabels, or (None, "") if absent/not applicable. A
    ReplicationController's selector is itself the flat label map
    (spec.selector), so it's returned as-is."""
    if not isinstance(doc, dict) or doc.get("kind") not in WORKLOAD_SELECTOR_KINDS:
        return None, ""
    spec = doc.get("spec")
    if not isinstance(spec, dict):
        return None, ""
    selector = spec.get("selector")
    if not isinstance(selector, dict):
        return None, ""
    if doc.get("kind") == "ReplicationController":
        flat = {k: v for k, v in selector.items() if not isinstance(v, (dict, list))}
        return (selector, "/spec/selector") if selector and len(flat) == len(selector) else (None, "")
    match_labels = selector.get("matchLabels")
    if not isinstance(match_labels, dict):
        return None, ""
    return match_labels, "/spec/selector/matchLabels"


def get_template_labels(doc) -> tuple[dict | None, str]:
    """Returns (labels_dict, json_pointer_prefix) for the pod template's own
    metadata.labels (spec.template.metadata.labels), or (None, "") if
    absent/not applicable."""
    if not isinstance(doc, dict) or doc.get("kind") not in WORKLOAD_SELECTOR_KINDS:
        return None, ""
    spec = doc.get("spec")
    if not isinstance(spec, dict) or not isinstance(spec.get("template"), dict):
        return None, ""
    metadata = spec["template"].get("metadata")
    if not isinstance(metadata, dict):
        return None, ""
    labels = metadata.get("labels")
    if not isinstance(labels, dict):
        return None, ""
    return labels, "/spec/template/metadata/labels"


def get_pod_labels(doc) -> tuple[dict | None, str]:
    """(labels, json_pointer_prefix) of the pods a document creates -- what a
    Service's selector has to match: a Pod's own metadata.labels, a CronJob's
    jobTemplate pod template, or any other workload's spec.template. (None,
    "") for kinds that don't create pods or have no labels."""
    if not isinstance(doc, dict) or doc.get("kind") not in POD_TEMPLATE_KINDS:
        return None, ""
    kind = doc["kind"]
    if kind == "Pod":
        node, prefix = doc, ""
    else:
        spec = doc.get("spec")
        prefix = "/spec"
        if kind == "CronJob":
            spec = ((spec or {}).get("jobTemplate") or {}).get("spec") if isinstance(spec, dict) else None
            prefix = "/spec/jobTemplate/spec"
        if not isinstance(spec, dict) or not isinstance(spec.get("template"), dict):
            return None, ""
        node, prefix = spec["template"], f"{prefix}/template"
    metadata = node.get("metadata")
    labels = metadata.get("labels") if isinstance(metadata, dict) else None
    if not isinstance(labels, dict) or not labels:
        return None, ""
    return labels, f"{prefix}/metadata/labels"


def get_service_selector(doc) -> dict | None:
    """A Service's spec.selector (a flat label map), or None if absent/empty
    -- a selector-less Service (manually managed Endpoints) selects nothing
    by design."""
    if not isinstance(doc, dict) or doc.get("kind") != "Service":
        return None
    spec = doc.get("spec")
    selector = spec.get("selector") if isinstance(spec, dict) else None
    if not isinstance(selector, dict) or not selector:
        return None
    if any(isinstance(v, (dict, list)) for v in selector.values()):
        return None
    return selector


def label_selector_matches(selector: dict, labels: dict) -> bool:
    return all(k in labels and str(labels[k]) == str(v) for k, v in selector.items())


def pod_label_sets(docs: list) -> list[tuple[int, dict]]:
    """(doc index, pod labels) for every document in a file that creates pods."""
    out = []
    for i, doc in enumerate(docs):
        labels, _ = get_pod_labels(doc)
        if labels is not None:
            out.append((i, labels))
    return out


def get_container_ports(container: dict) -> tuple[set[int], set[str]]:
    """Returns (declared_port_numbers, declared_port_names) for a container's
    spec.containers[].ports list."""
    numbers: set[int] = set()
    names: set[str] = set()
    ports = container.get("ports")
    if not isinstance(ports, list):
        return numbers, names
    for port in ports:
        if not isinstance(port, dict):
            continue
        if isinstance(port.get("containerPort"), int):
            numbers.add(port["containerPort"])
        if isinstance(port.get("name"), str):
            names.add(port["name"])
    return numbers, names


PROBE_FIELDS = ("livenessProbe", "readinessProbe", "startupProbe")

_QUANTITY_BINARY_SUFFIXES = {"Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40, "Pi": 2**50, "Ei": 2**60}
_QUANTITY_DECIMAL_SUFFIXES = {"n": 1e-9, "u": 1e-6, "m": 1e-3, "k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "P": 1e15, "E": 1e18}


def parse_quantity(value) -> float | None:
    """Parses a Kubernetes resource quantity (e.g. "500m", "1Gi", "2") into
    a float in base units (cores for cpu, bytes for memory). Returns None if
    the value can't be parsed."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    s = value.strip()
    for suffix, mult in _QUANTITY_BINARY_SUFFIXES.items():
        if s.endswith(suffix):
            try:
                return float(s[: -len(suffix)]) * mult
            except ValueError:
                return None
    for suffix, mult in _QUANTITY_DECIMAL_SUFFIXES.items():
        if s.endswith(suffix):
            try:
                return float(s[: -len(suffix)]) * mult
            except ValueError:
                return None
    try:
        return float(s)
    except ValueError:
        return None
