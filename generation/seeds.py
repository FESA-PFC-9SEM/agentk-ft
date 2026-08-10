"""
Combinatorial seed sampler for synthetic generation. The classic failure
mode of this kind of pipeline is asking "generate a manifest" a thousand
times and getting a thousand variations of nginx in the default namespace.
Every call to generate.py draws a Seed from here, which becomes an explicit
constraint in the prompt and is recorded alongside the generated manifest so
diversity can be audited afterwards (generation/report.py). All randomness
goes through a configurable random.Random(seed) -- no global random.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

DOMAINS = (
    "fintech",
    "healthcare",
    "logistics",
    "e-commerce",
    "education",
    "gaming",
    "media-streaming",
    "telecom",
    "government",
    "energy",
)

NAMING_CONVENTIONS = ("kebab-case", "camelCase", "team-prefix", "environment-suffix")

SERVICE_NOUNS = (
    "gateway",
    "ledger",
    "invoice",
    "claims",
    "shipment",
    "catalog",
    "checkout",
    "enrollment",
    "telemetry",
    "billing",
    "auth",
    "notification",
    "inventory",
    "scheduler",
    "reporting",
    "recommendation",
    "search",
    "pricing",
    "fraud-detection",
    "audit-log",
)

TEAM_CODES = ("plat", "core", "growth", "sre", "data", "sec", "mobile", "web")
ENVIRONMENTS = ("dev", "staging", "prod", "qa")

STACK_ENV_VARS = {
    "JVM": ("JAVA_OPTS", "SPRING_PROFILES_ACTIVE", "DB_POOL_SIZE", "JVM_HEAP_MAX", "MANAGEMENT_SERVER_PORT"),
    "Node": ("NODE_ENV", "PORT", "NPM_CONFIG_LOGLEVEL", "SESSION_TIMEOUT", "LOG_FORMAT"),
    "Go": ("GOMAXPROCS", "LOG_LEVEL", "HTTP_TIMEOUT", "GO_ENV", "METRICS_PORT"),
    "Python": ("PYTHONUNBUFFERED", "DJANGO_SETTINGS_MODULE", "GUNICORN_WORKERS", "FLASK_ENV", "WORKERS_PER_CORE"),
    ".NET": (
        "ASPNETCORE_ENVIRONMENT",
        "DOTNET_RUNNING_IN_CONTAINER",
        "Logging__LogLevel__Default",
        "ASPNETCORE_URLS",
    ),
}

WORKLOAD_KIND_WEIGHTS = {
    "Deployment": 40,
    "Pod": 15,
    "StatefulSet": 12,
    "DaemonSet": 8,
    "Job": 15,
    "CronJob": 10,
}
RBAC_KIND_WEIGHTS = {"Role": 35, "ClusterRole": 25, "RoleBinding": 25, "ClusterRoleBinding": 15}
# "base" mode: mostly workloads, with RBAC deliberately oversampled relative
# to its real proportion in the corpus (where RBAC is scarce).
BASE_KIND_WEIGHTS = {
    **{k: v * 0.75 for k, v in WORKLOAD_KIND_WEIGHTS.items()},
    **{k: v * 0.25 for k, v in RBAC_KIND_WEIGHTS.items()},
}

NAMESPACE_TEMPLATES = ("default", "{domain}-{env}", "{team}-{env}", "{service}-{env}")

QUOTE_STYLES = ("single", "double", "unquoted")
BLOCK_STYLES = ("literal", "folded", "none")
INDENT_SIZES = (2, 4)
LIST_STYLES = ("block", "inline")
# Language of any comments in the generated YAML -- a genuine diversity axis
# for the training data itself, independent of the pipeline's own language.
COMMENT_LANGUAGES = ("pt-BR", "en", None)


def _slugify(words: list[str]) -> str:
    return "-".join(w.lower().replace("_", "-") for w in words)


def _format_service_name(words: list[str], convention: str, team: str, env: str) -> str:
    if convention == "kebab-case":
        return _slugify(words)
    if convention == "camelCase":
        parts = [w.lower() for w in words]
        return parts[0] + "".join(p.capitalize() for p in parts[1:])
    if convention == "team-prefix":
        return f"{team}-{_slugify(words)}"
    if convention == "environment-suffix":
        return f"{_slugify(words)}-{env}"
    raise ValueError(f"unknown naming convention: {convention}")


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights.keys())
    values = [weights[k] for k in keys]
    return rng.choices(keys, weights=values, k=1)[0]


@dataclass
class YamlStyle:
    quote_style: str
    block_style: str
    indent: int
    list_style: str


@dataclass
class Seed:
    seed_id: int
    mode: str
    domain: str
    service_name: str
    naming_convention: str
    stack: str
    typical_env_vars: tuple[str, ...]
    kind: str
    namespace: str
    namespace_convention: str
    has_labels: bool
    has_annotations: bool
    yaml_style: YamlStyle
    has_comments: bool
    comment_language: str | None
    multi_doc: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_prompt_constraints(self) -> str:
        lines = [
            f"- Business domain: {self.domain}",
            f"- Service/resource name: {self.service_name} (naming convention: {self.naming_convention})",
            f"- Application stack: {self.stack} (typical environment variables for this stack: "
            f"{', '.join(self.typical_env_vars)})",
            f"- Kubernetes resource kind to generate: {self.kind}",
            f"- Namespace: {self.namespace} (convention: {self.namespace_convention})",
            f"- Labels: {'include realistic labels (app, team, env, etc)' if self.has_labels else 'do not include labels'}",
            f"- Annotations: {'include at least one plausible annotation' if self.has_annotations else 'do not include annotations'}",
            f"- Quoting style for YAML strings: {self.yaml_style.quote_style}",
            f"- Block style for multi-line values: {self.yaml_style.block_style}",
            f"- Indentation: {self.yaml_style.indent} spaces",
            f"- List style: {self.yaml_style.list_style}",
        ]
        if self.has_comments:
            lang = "Portuguese" if self.comment_language == "pt-BR" else "English"
            lines.append(f"- Include explanatory comments in {lang}")
        else:
            lines.append("- Do not include comments")
        lines.append(
            "- The file must contain multiple YAML documents separated by '---'"
            if self.multi_doc
            else "- The file must contain a single YAML document"
        )
        return "\n".join(lines)


def sample_seed(rng: random.Random, mode: str = "base") -> Seed:
    """Draws a Seed. `mode` decides the kind pool: 'rbac' forces an RBAC kind
    (Role/ClusterRole/RoleBinding/ClusterRoleBinding); 'hard-negative' and
    'base' use workload kinds ('base' also includes a slice of RBAC,
    oversampled relative to its real proportion in the corpus)."""
    seed_id = rng.randrange(2**31)

    if mode == "rbac":
        kind = _weighted_choice(rng, RBAC_KIND_WEIGHTS)
    elif mode == "hard-negative":
        kind = _weighted_choice(rng, WORKLOAD_KIND_WEIGHTS)
    else:
        kind = _weighted_choice(rng, BASE_KIND_WEIGHTS)

    domain = rng.choice(DOMAINS)
    naming_convention = rng.choice(NAMING_CONVENTIONS)
    stack = rng.choice(list(STACK_ENV_VARS))
    n_words = rng.choice((1, 2))
    words = rng.sample(SERVICE_NOUNS, n_words)
    team = rng.choice(TEAM_CODES)
    env = rng.choice(ENVIRONMENTS)
    service_name = _format_service_name(words, naming_convention, team, env)

    namespace_template = rng.choice(NAMESPACE_TEMPLATES)
    namespace = namespace_template.format(domain=domain, team=team, env=env, service=_slugify(words))

    yaml_style = YamlStyle(
        quote_style=rng.choice(QUOTE_STYLES),
        block_style=rng.choice(BLOCK_STYLES),
        indent=rng.choice(INDENT_SIZES),
        list_style=rng.choice(LIST_STYLES),
    )
    has_comments = rng.random() < 0.4
    comment_language = rng.choice(("pt-BR", "en")) if has_comments else None

    return Seed(
        seed_id=seed_id,
        mode=mode,
        domain=domain,
        service_name=service_name,
        naming_convention=naming_convention,
        stack=stack,
        typical_env_vars=STACK_ENV_VARS[stack],
        kind=kind,
        namespace=namespace,
        namespace_convention=namespace_template,
        has_labels=rng.random() < 0.8,
        has_annotations=rng.random() < 0.35,
        yaml_style=yaml_style,
        has_comments=has_comments,
        comment_language=comment_language,
        multi_doc=rng.random() < 0.15,
    )
