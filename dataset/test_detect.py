import pytest

from dataset.detect import (
    detect_ksec001,
    detect_ksec002,
    detect_ksec003,
    detect_ksec004,
    detect_ksec005,
    detect_ksec006,
    detect_ksec007,
    detect_ksec008,
    detect_ksec009,
    detect_ksec010,
    detect_ksec011,
    detect_file,
)
from dataset.k8s import image_basename, official_image_name, required_env_image_key


def _pod(container_extra=None, pod_spec_extra=None):
    container = {"name": "app", "image": "myapp:1.2.3"}
    if container_extra:
        container.update(container_extra)
    spec = {"containers": [container]}
    if pod_spec_extra:
        spec.update(pod_spec_extra)
    return {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "p"}, "spec": spec}


def test_ksec001_clean_pod():
    doc = _pod({"env": [{"name": "SESSION_TIMEOUT", "value": "3600"}]})
    assert detect_ksec001(doc) == []


def test_ksec001_dirty_pod():
    doc = _pod({"env": [{"name": "DB_PASSWORD", "value": "S3cr3tR34lLeak99"}]})
    findings = detect_ksec001(doc)
    assert len(findings) == 1
    assert findings[0].rule_id == "KSEC-001"
    assert findings[0].evidence == "S3cr***"


def test_ksec002_privileged():
    doc = _pod({"securityContext": {"privileged": True}})
    findings = detect_ksec002(doc)
    assert any(f.path.endswith("/privileged") for f in findings)


def test_ksec002_run_as_root():
    doc = _pod({"securityContext": {"runAsUser": 0}})
    findings = detect_ksec002(doc)
    assert any(f.path.endswith("/runAsUser") for f in findings)


def test_ksec002_clean_container_is_not_flagged():
    doc = _pod({"securityContext": {"runAsNonRoot": True}})
    assert detect_ksec002(doc) == []


def test_ksec002_absent_security_context_is_not_flagged():
    doc = _pod()
    assert detect_ksec002(doc) == []


def test_ksec003_host_network():
    doc = _pod(pod_spec_extra={"hostNetwork": True})
    findings = detect_ksec003(doc)
    assert any(f.path.endswith("/hostNetwork") for f in findings)


def test_ksec003_sensitive_hostpath():
    doc = _pod(
        pod_spec_extra={
            "volumes": [{"name": "docker", "hostPath": {"path": "/var/run/docker.sock"}}]
        }
    )
    findings = detect_ksec003(doc)
    assert len(findings) == 1
    assert findings[0].severity == "critical"


def test_ksec003_benign_hostpath_not_flagged():
    doc = _pod(pod_spec_extra={"volumes": [{"name": "data", "hostPath": {"path": "/data/app"}}]})
    assert detect_ksec003(doc) == []


def test_ksec004_wildcard_role():
    doc = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": {"name": "r"},
        "rules": [{"apiGroups": ["*"], "resources": ["*"], "verbs": ["*"]}],
    }
    findings = detect_ksec004(doc)
    assert len(findings) == 3  # apiGroups, resources, verbs


def test_ksec004_cluster_admin_binding():
    doc = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "ClusterRoleBinding",
        "metadata": {"name": "b"},
        "roleRef": {"kind": "ClusterRole", "name": "cluster-admin", "apiGroup": "rbac.authorization.k8s.io"},
        "subjects": [{"kind": "ServiceAccount", "name": "sa", "namespace": "default"}],
    }
    findings = detect_ksec004(doc)
    assert len(findings) == 1
    assert findings[0].severity == "critical"


def test_ksec004_scoped_role_not_flagged():
    doc = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": {"name": "r"},
        "rules": [{"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list"]}],
    }
    assert detect_ksec004(doc) == []


def test_ksec005_latest_tag():
    doc = _pod({"image": "myapp:latest"})
    findings = detect_ksec005(doc)
    assert len(findings) == 1


def test_ksec005_missing_tag():
    doc = _pod({"image": "myapp"})
    findings = detect_ksec005(doc)
    assert len(findings) == 1


def test_ksec005_pinned_tag_not_flagged():
    doc = _pod({"image": "myapp:1.2.3"})
    assert detect_ksec005(doc) == []


def test_ksec005_digest_pinned_not_flagged():
    doc = _pod({"image": "myapp@sha256:" + "a" * 64})
    assert detect_ksec005(doc) == []


def test_ksec005_registry_with_port_and_no_tag():
    doc = _pod({"image": "registry.internal:5000/myapp"})
    findings = detect_ksec005(doc)
    assert len(findings) == 1


def test_ksec005_registry_with_port_and_tag_is_clean():
    doc = _pod({"image": "registry.internal:5000/myapp:1.0.0"})
    assert detect_ksec005(doc) == []


def test_cronjob_nested_pod_spec():
    doc = {
        "apiVersion": "batch/v1",
        "kind": "CronJob",
        "metadata": {"name": "c"},
        "spec": {
            "jobTemplate": {
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [
                                {"name": "app", "image": "myapp:latest", "securityContext": {"privileged": True}}
                            ]
                        }
                    }
                }
            }
        },
    }
    img_findings = detect_ksec005(doc)
    sc_findings = detect_ksec002(doc)
    assert len(img_findings) == 1
    assert img_findings[0].path == "/spec/jobTemplate/spec/template/spec/containers/0/image"
    assert len(sc_findings) == 1


def _deployment(selector_labels, template_labels, container_extra=None):
    container = {"name": "app", "image": "myapp:1.2.3"}
    if container_extra:
        container.update(container_extra)
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "d"},
        "spec": {
            "selector": {"matchLabels": selector_labels},
            "template": {
                "metadata": {"labels": template_labels},
                "spec": {"containers": [container]},
            },
        },
    }


def test_ksec006_matching_selector_not_flagged():
    doc = _deployment({"app": "x"}, {"app": "x", "tier": "web"})
    assert detect_ksec006(doc) == []


def test_ksec006_mismatched_selector_flagged():
    doc = _deployment({"app": "x"}, {"app": "y"})
    findings = detect_ksec006(doc)
    assert len(findings) == 1
    assert findings[0].path == "/spec/selector/matchLabels/app"


def test_ksec006_not_applicable_to_pod():
    assert detect_ksec006(_pod()) == []


def test_ksec006_label_key_with_slash_is_escaped_in_path():
    # label keys routinely contain '/' (app.kubernetes.io/name) -- the path
    # must escape it per RFC 6901 or the JSON Pointer is unparseable.
    doc = _deployment({"app.kubernetes.io/name": "x"}, {"app.kubernetes.io/name": "y"})
    findings = detect_ksec006(doc)
    assert len(findings) == 1
    assert findings[0].path == "/spec/selector/matchLabels/app.kubernetes.io~1name"


def test_ksec007_port_mismatch_flagged():
    doc = _pod({"ports": [{"containerPort": 8080}], "livenessProbe": {"httpGet": {"path": "/health", "port": 9090}}})
    findings = detect_ksec007(doc)
    assert len(findings) == 1
    assert findings[0].path == "/spec/containers/0/livenessProbe/httpGet/port"


def test_ksec007_matching_port_not_flagged():
    doc = _pod({"ports": [{"containerPort": 8080}], "livenessProbe": {"httpGet": {"path": "/health", "port": 8080}}})
    assert detect_ksec007(doc) == []


def test_ksec007_named_port_mismatch_flagged():
    doc = _pod(
        {
            "ports": [{"containerPort": 8080, "name": "http"}],
            "readinessProbe": {"tcpSocket": {"port": "grpc"}},
        }
    )
    findings = detect_ksec007(doc)
    assert len(findings) == 1


def test_ksec007_no_declared_ports_not_flagged():
    # sem ports declaradas, nao ha base para comparar -- evita falso positivo
    doc = _pod({"livenessProbe": {"httpGet": {"path": "/health", "port": 9090}}})
    assert detect_ksec007(doc) == []


def test_ksec008_requests_exceed_limits_flagged():
    doc = _pod({"resources": {"requests": {"cpu": "1000m"}, "limits": {"cpu": "500m"}}})
    findings = detect_ksec008(doc)
    assert len(findings) == 1
    assert findings[0].path == "/spec/containers/0/resources/requests/cpu"


def test_ksec008_requests_within_limits_not_flagged():
    doc = _pod({"resources": {"requests": {"cpu": "250m", "memory": "128Mi"}, "limits": {"cpu": "500m", "memory": "256Mi"}}})
    assert detect_ksec008(doc) == []


def test_ksec008_memory_binary_suffix_comparison():
    doc = _pod({"resources": {"requests": {"memory": "2Gi"}, "limits": {"memory": "1024Mi"}}})
    findings = detect_ksec008(doc)
    assert len(findings) == 1


def test_ksec008_missing_limits_not_flagged():
    doc = _pod({"resources": {"requests": {"cpu": "500m"}}})
    assert detect_ksec008(doc) == []


def test_ksec009_dangling_volume_mount_flagged():
    doc = _pod(
        {"volumeMounts": [{"name": "cofnig-volume", "mountPath": "/etc/app"}]},
        pod_spec_extra={"volumes": [{"name": "config-volume", "configMap": {"name": "app-config"}}]},
    )
    findings = detect_ksec009(doc)
    assert len(findings) == 1
    assert findings[0].path == "/spec/containers/0/volumeMounts/0/name"


def test_ksec009_matching_volume_mount_not_flagged():
    doc = _pod(
        {"volumeMounts": [{"name": "config-volume", "mountPath": "/etc/app"}]},
        pod_spec_extra={"volumes": [{"name": "config-volume", "configMap": {"name": "app-config"}}]},
    )
    assert detect_ksec009(doc) == []


# ---------------------------------------------------------------------------
# Image identity helpers (k8s.py) used by KSEC-010/011
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "image,expected",
    [
        ("postgres", "postgres"),
        ("postgres:15-alpine", "postgres"),
        ("docker.io/bitnami/postgresql:15", "postgresql"),
        ("redis@sha256:" + "a" * 64, "redis"),
        ("registry.example.com:5000/team/Redis:7", "redis"),
        ("", None),
        (None, None),
    ],
)
def test_image_basename(image, expected):
    assert image_basename(image) == expected


@pytest.mark.parametrize(
    "image,expected",
    [
        ("postgres:15", "postgres"),
        ("library/postgres:15", "postgres"),
        ("docker.io/library/postgres:15", "postgres"),
        ("docker.io/postgres", "postgres"),
        ("mysql@sha256:" + "b" * 64, "mysql"),
        ("bitnami/mysql:8", None),
        ("ghcr.io/org/postgres:15", None),
        ("localhost:5000/postgres", None),
        ("quay.io/postgres", None),
    ],
)
def test_official_image_name(image, expected):
    assert official_image_name(image) == expected


# ---------------------------------------------------------------------------
# KSEC-010 -- httpGet probe on a non-HTTP server
# ---------------------------------------------------------------------------


def _db_pod(image, container_extra=None, init=False):
    container = {"name": "db", "image": image}
    if container_extra:
        container.update(container_extra)
    key = "initContainers" if init else "containers"
    spec = {key: [container]}
    if init:
        spec["containers"] = [{"name": "app", "image": "myapp:1.2.3"}]
    return {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "p"}, "spec": spec}


def test_ksec010_httpget_liveness_on_postgres_is_high():
    doc = _db_pod("postgres:15", {"livenessProbe": {"httpGet": {"path": "/", "port": 5432}}})
    findings = detect_ksec010(doc)
    assert len(findings) == 1
    assert findings[0].path == "/spec/containers/0/livenessProbe/httpGet"
    assert findings[0].severity == "high"
    assert findings[0].evidence == "post***"


def test_ksec010_httpget_readiness_is_medium():
    doc = _db_pod("redis:7", {"readinessProbe": {"httpGet": {"path": "/", "port": 6379}}})
    assert [f.severity for f in detect_ksec010(doc)] == ["medium"]


def test_ksec010_flags_any_port_since_the_container_only_runs_the_server():
    doc = _db_pod("mongo:6", {"livenessProbe": {"httpGet": {"path": "/", "port": 8080}}})
    assert len(detect_ksec010(doc)) == 1


def test_ksec010_vendor_rebuild_matches_by_basename():
    doc = _db_pod("bitnami/postgresql:15", {"livenessProbe": {"httpGet": {"path": "/", "port": 5432}}})
    assert len(detect_ksec010(doc)) == 1


@pytest.mark.parametrize(
    "probe",
    [
        {"tcpSocket": {"port": 5432}},
        {"exec": {"command": ["pg_isready", "-U", "postgres"]}},
    ],
)
def test_ksec010_non_http_checks_are_clean(probe):
    assert detect_ksec010(_db_pod("postgres:15", {"livenessProbe": probe})) == []


def test_ksec010_http_images_and_exporters_are_not_flagged():
    probe = {"livenessProbe": {"httpGet": {"path": "/metrics", "port": 9121}}}
    assert detect_ksec010(_db_pod("myapp:1.2.3", probe)) == []
    assert detect_ksec010(_db_pod("oliver006/redis_exporter:v1.55.0", probe)) == []


# ---------------------------------------------------------------------------
# KSEC-011 -- missing required env var for the image
# ---------------------------------------------------------------------------


def test_ksec011_postgres_without_password_is_flagged():
    findings = detect_ksec011(_db_pod("postgres:15", {"env": [{"name": "POSTGRES_DB", "value": "app"}]}))
    assert len(findings) == 1
    assert findings[0].path == "/spec/containers/0/env"
    assert findings[0].severity == "high"
    assert "POSTGRES_PASSWORD" in findings[0].message


def test_ksec011_no_env_at_all_is_flagged():
    assert len(detect_ksec011(_db_pod("mysql:8"))) == 1


@pytest.mark.parametrize(
    "image,var",
    [
        ("postgres:15", "POSTGRES_PASSWORD"),
        ("postgres:15", "POSTGRES_PASSWORD_FILE"),
        ("postgres:15", "POSTGRES_HOST_AUTH_METHOD"),
        ("mysql:8", "MYSQL_ROOT_PASSWORD"),
        ("mysql:8", "MYSQL_RANDOM_ROOT_PASSWORD"),
        ("mariadb:11", "MARIADB_ROOT_PASSWORD"),
        ("mariadb:11", "MYSQL_ROOT_PASSWORD"),
    ],
)
def test_ksec011_any_accepted_var_satisfies(image, var):
    env = [{"name": var, "valueFrom": {"secretKeyRef": {"name": "s", "key": "k"}}}]
    assert detect_ksec011(_db_pod(image, {"env": env})) == []


@pytest.mark.parametrize(
    "extra",
    [
        {"envFrom": [{"secretRef": {"name": "db-env"}}]},
        {"command": ["pg_dump", "-h", "db"]},
        {"args": ["psql", "-h", "db"]},
    ],
)
def test_ksec011_uncheckable_containers_are_skipped(extra):
    assert detect_ksec011(_db_pod("postgres:15", extra)) == []


def test_ksec011_server_flags_in_args_are_still_checked():
    assert len(detect_ksec011(_db_pod("postgres:15", {"args": ["-c", "max_connections=200"]}))) == 1
    assert len(detect_ksec011(_db_pod("mysql:8", {"args": ["mysqld", "--skip-name-resolve"]}))) == 1


def test_ksec011_init_containers_and_unknown_images_are_skipped():
    assert detect_ksec011(_db_pod("postgres:15", init=True)) == []
    # Official redis/mongo have no mandatory env var.
    assert detect_ksec011(_db_pod("redis:7")) == []
    assert detect_ksec011(_db_pod("mongo:7")) == []
    # Mirrors/look-alikes: identity (and so the contract) can't be confirmed.
    assert detect_ksec011(_db_pod("localhost:32000/percona:8")) == []
    assert detect_ksec011(_db_pod("mcr.microsoft.com/oss/bitnami/redis:6.0.8")) == []
    assert detect_ksec011(_db_pod("registry.corp/mssql:2019")) == []


@pytest.mark.parametrize(
    "image,expected",
    [
        ("postgres:15", "postgres"),
        ("docker.io/library/mysql:8", "mysql"),
        ("percona:8.0", "percona"),
        ("percona/percona-server:8.0", "percona/percona-server"),
        ("bitnami/postgresql:15", "bitnami/postgresql"),
        ("docker.io/bitnami/redis:7.2", "bitnami/redis"),
        ("bitnami/mongodb@sha256:" + "c" * 64, "bitnami/mongodb"),
        ("mcr.microsoft.com/mssql/server:2022-latest", "mssql"),
        ("mcr.microsoft.com/mssql/rhel/server:2019-latest", "mssql"),
        ("mcr.microsoft.com/azure-sql-edge:latest", "mssql"),
        ("mcr.microsoft.com/mssql-tools", None),
        ("mssql", None),
        ("bitnami/nginx:1.25", None),
        ("quay.io/bitnami/redis:6", None),
        ("perconalab/percona-xtradb-cluster-operator:1.4", None),
    ],
)
def test_required_env_image_key(image, expected):
    assert required_env_image_key(image) == expected


@pytest.mark.parametrize(
    "image,var",
    [
        ("bitnami/postgresql:15", "POSTGRES_PASSWORD"),
        ("bitnami/postgresql:15", "POSTGRESQL_PASSWORD"),
        ("bitnami/postgresql:15", "ALLOW_EMPTY_PASSWORD"),
        ("bitnami/mysql:8", "MYSQL_ROOT_PASSWORD_FILE"),
        ("bitnami/mariadb:11", "MARIADB_ROOT_PASSWORD"),
        ("bitnami/redis:7", "REDIS_PASSWORD"),
        ("bitnami/redis:7", "ALLOW_EMPTY_PASSWORD"),
        ("bitnami/mongodb:7", "MONGODB_ROOT_PASSWORD"),
        ("percona:8.0", "MYSQL_ROOT_PASSWORD"),
    ],
)
def test_ksec011_vendor_images_accept_their_own_vars(image, var):
    assert len(detect_ksec011(_db_pod(image))) == 1
    env = [{"name": var, "valueFrom": {"secretKeyRef": {"name": "s", "key": "k"}}}]
    assert detect_ksec011(_db_pod(image, {"env": env})) == []


@pytest.mark.parametrize(
    "image,var,value",
    [
        ("bitnami/postgresql:15", "POSTGRESQL_REPLICATION_MODE", "slave"),
        ("bitnami/mysql:8", "MYSQL_REPLICATION_MODE", "slave"),
        ("bitnami/mongodb:7", "MONGODB_REPLICA_SET_MODE", "secondary"),
        ("bitnami/mongodb:7", "MONGODB_REPLICA_SET_MODE", "arbiter"),
    ],
)
def test_ksec011_bitnami_replicas_are_skipped(image, var, value):
    # Replicas read the primary's password from a different variable.
    assert detect_ksec011(_db_pod(image, {"env": [{"name": var, "value": value}]})) == []
    assert len(detect_ksec011(_db_pod(image, {"env": [{"name": var, "value": "master" if "MONGO" not in var else "primary"}]}))) == 1


def test_ksec011_mssql_has_two_independent_requirements():
    image = "mcr.microsoft.com/mssql/server:2022-latest"
    messages = sorted(f.message for f in detect_ksec011(_db_pod(image)))
    assert len(messages) == 2
    assert "ACCEPT_EULA" in messages[0] and "MSSQL_SA_PASSWORD" in messages[1]
    eula_only = {"env": [{"name": "ACCEPT_EULA", "value": "Y"}]}
    assert ["MSSQL_SA_PASSWORD" in f.message for f in detect_ksec011(_db_pod(image, eula_only))] == [True]
    both = {"env": [{"name": "ACCEPT_EULA", "value": "Y"}, {"name": "SA_PASSWORD", "valueFrom": {"secretKeyRef": {"name": "s", "key": "k"}}}]}
    assert detect_ksec011(_db_pod(image, both)) == []


def test_ksec011_bitnami_checks_even_with_a_prepopulated_data_dir():
    # Bitnami validates its password on every start, so the replica-cloning
    # exemption doesn't apply.
    doc = _replica_pod("/mnt/data")
    db = doc["spec"]["containers"][0]
    db["image"] = "bitnami/postgresql:15"
    db["volumeMounts"] = [{"name": "data", "mountPath": "/bitnami/postgresql"}]
    assert len(detect_ksec011(doc)) == 1


def test_ksec011_mssql_prepopulated_data_dir_still_needs_the_eula():
    doc = _replica_pod("/mnt/data")
    db = doc["spec"]["containers"][0]
    db["image"] = "mcr.microsoft.com/mssql/server:2022-latest"
    db["volumeMounts"] = [{"name": "data", "mountPath": "/var/opt/mssql"}]
    findings = detect_ksec011(doc)
    assert len(findings) == 1 and "ACCEPT_EULA" in findings[0].message


def test_ksec011_locally_built_image_with_official_name_is_skipped():
    # Real corpus case: custom "mysql" images built into the cluster's
    # docker daemon, with their own entrypoint.
    assert detect_ksec011(_db_pod("mysql:v1", {"imagePullPolicy": "Never"})) == []


def _replica_pod(init_mount_path, init_volume="data", pgdata=None):
    db = {"name": "db", "image": "postgres:15", "volumeMounts": [{"name": "data", "mountPath": "/var/lib/postgresql/data"}]}
    if pgdata:
        db["volumeMounts"] = [{"name": "data", "mountPath": "/pgdata"}]
        db["env"] = [{"name": "PGDATA", "value": "/pgdata/db"}]
    init = {"name": "clone", "image": "busybox:1.36", "volumeMounts": [{"name": init_volume, "mountPath": init_mount_path}]}
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": "p"},
        "spec": {
            "initContainers": [init],
            "containers": [db],
            "volumes": [{"name": "data", "emptyDir": {}}, {"name": "conf", "emptyDir": {}}],
        },
    }


def test_ksec011_init_container_writing_the_data_volume_is_skipped():
    # Replica-cloning pattern: an existing database is copied in before the
    # server starts, so the entrypoint never asks for a password.
    assert detect_ksec011(_replica_pod("/mnt/data")) == []
    assert detect_ksec011(_replica_pod("/mnt/data", pgdata=True)) == []


def test_ksec011_init_container_writing_only_a_config_volume_is_still_flagged():
    doc = _replica_pod("/mnt/conf", init_volume="conf")
    doc["spec"]["containers"][0]["volumeMounts"].append({"name": "conf", "mountPath": "/etc/postgresql"})
    assert len(detect_ksec011(doc)) == 1


def test_ksec011_read_only_init_mount_of_the_data_volume_is_still_flagged():
    doc = _replica_pod("/mnt/data")
    doc["spec"]["initContainers"][0]["volumeMounts"][0]["readOnly"] = True
    assert len(detect_ksec011(doc)) == 1


def _statefulset(mount_name, claim_template_name="data"):
    return {
        "apiVersion": "apps/v1",
        "kind": "StatefulSet",
        "metadata": {"name": "db"},
        "spec": {
            "template": {
                "spec": {"containers": [{"name": "db", "image": "myapp:1.2.3", "volumeMounts": [{"name": mount_name, "mountPath": "/data"}]}]}
            },
            "volumeClaimTemplates": [{"metadata": {"name": claim_template_name}, "spec": {"accessModes": ["ReadWriteOnce"]}}],
        },
    }


def test_ksec009_statefulset_claim_template_mount_is_not_flagged():
    # volumeClaimTemplates become per-replica volumes that never appear under
    # `volumes` -- 817 valid StatefulSets in the real corpus mount them.
    assert detect_ksec009(_statefulset("data")) == []


def test_ksec009_statefulset_typo_of_a_claim_template_is_flagged():
    findings = detect_ksec009(_statefulset("dtaa"))
    assert len(findings) == 1
    assert findings[0].path == "/spec/template/spec/containers/0/volumeMounts/0/name"


def test_ksec009_claim_templates_only_count_for_statefulsets():
    doc = _statefulset("data")
    doc["kind"] = "Deployment"
    assert len(detect_ksec009(doc)) == 1


# ---------------------------------------------------------------------------
# ReplicationController (legacy workload kind)
# ---------------------------------------------------------------------------


def _replication_controller(selector, labels, image="nginx"):
    return {
        "apiVersion": "v1",
        "kind": "ReplicationController",
        "metadata": {"name": "rc"},
        "spec": {
            "selector": selector,
            "template": {"metadata": {"labels": labels}, "spec": {"containers": [{"name": "c", "image": image}]}},
        },
    }


def test_replication_controller_containers_are_checked():
    # scenarios/5-nginx.yaml and 7-elasticsearch.yaml: unpinned images inside
    # an RC used to be invisible to every container rule.
    findings = detect_ksec005(_replication_controller({"app": "web"}, {"app": "web"}, image="nginx"))
    assert [f.path for f in findings] == ["/spec/template/spec/containers/0/image"]


def test_replication_controller_flat_selector_mismatch():
    assert detect_ksec006(_replication_controller({"app": "web"}, {"app": "web"}, image="nginx:1.25")) == []
    findings = detect_ksec006(_replication_controller({"app": "wbe"}, {"app": "web"}, image="nginx:1.25"))
    assert [f.path for f in findings] == ["/spec/selector/app"]


# ---------------------------------------------------------------------------
# KSEC-006 across documents: Service selector vs workloads in the same file
# ---------------------------------------------------------------------------


def _service(selector):
    return {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "s"}, "spec": {"selector": selector, "ports": [{"port": 80}]}}


def _file_ksec006(docs):
    return [f for f in detect_file(docs) if f.rule_id == "KSEC-006"]


def test_service_matching_its_workload_is_clean():
    workload = _deployment({"app": "web"}, {"app": "web", "tier": "fe"}, {"image": "nginx:1.25"})
    assert _file_ksec006([workload, _service({"app": "web"})]) == []


def test_service_selector_typo_is_flagged_on_the_service():
    # scenarios/10-mongodb.yaml shape: the Service comes first.
    workload = _deployment({"app": "mongodb-app"}, {"app": "mongodb-app"}, {"image": "mongo:7"})
    findings = _file_ksec006([_service({"app": "nonexistent-mongodb"}), workload])
    assert len(findings) == 1
    assert findings[0].doc == 0
    assert findings[0].path == "/spec/selector/app"
    assert findings[0].evidence == "none***"


def test_service_with_one_wrong_key_among_several_flags_only_that_key():
    workload = _deployment({"app": "web"}, {"app": "web", "tier": "fe"}, {"image": "nginx:1.25"})
    findings = _file_ksec006([workload, _service({"app": "web", "tier": "be"})])
    assert [f.path for f in findings] == ["/spec/selector/tier"]


def test_service_whose_keys_no_workload_has_is_not_flagged():
    # Most likely selects a workload defined in another file.
    workload = _deployment({"app": "web"}, {"app": "web"}, {"image": "nginx:1.25"})
    assert _file_ksec006([workload, _service({"component": "db"})]) == []


def test_service_alone_or_selectorless_is_not_flagged():
    assert _file_ksec006([_service({"app": "web"})]) == []
    workload = _deployment({"app": "web"}, {"app": "web"}, {"image": "nginx:1.25"})
    selectorless = {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "s"}, "spec": {"ports": [{"port": 80}]}}
    assert _file_ksec006([workload, selectorless]) == []


def test_service_matching_any_one_of_several_workloads_is_clean():
    a = _deployment({"app": "a"}, {"app": "a"}, {"image": "nginx:1.25"})
    b = _replication_controller({"app": "b"}, {"app": "b"}, image="nginx:1.25")
    pod = _pod({"image": "nginx:1.25"})
    pod["metadata"]["labels"] = {"app": "c"}
    for target in ("a", "b", "c"):
        assert _file_ksec006([a, b, pod, _service({"app": target})]) == []


def test_detect_file_tags_each_documents_findings_with_its_index():
    workload = _deployment({"app": "web"}, {"app": "web"}, {"image": "nginx"})
    findings = detect_file([_service({"app": "web"}), workload])
    assert [(f.rule_id, f.doc) for f in findings] == [("KSEC-005", 1)]
