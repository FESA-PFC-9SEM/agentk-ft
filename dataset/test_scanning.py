"""Tests for dataset/scanning.py. The first test is the most important one in
the project: it guarantees the walker recognizes the Kubernetes {name: X,
value: Y} pattern and uses X as the semantic key, not the literal key "value"."""

from dataset.scanning import (
    find_cli_embedded_secrets,
    find_secrets,
    is_placeholder,
    shannon_entropy,
    walk_leaves,
)


def test_env_name_value_pair():
    doc = {
        "spec": {
            "containers": [
                {
                    "name": "app",
                    "env": [
                        {"name": "DB_PASSWORD", "value": "S3cr3tR34lLeak99"},
                    ],
                }
            ]
        }
    }
    hits = find_secrets(doc)
    assert len(hits) == 1
    hit = hits[0]
    assert hit.key == "DB_PASSWORD"
    assert hit.value == "S3cr3tR34lLeak99"
    assert hit.path.endswith("/value")


def test_naive_leaf_walk_would_miss_it():
    # Documents the bug this module exists to avoid: testing the literal
    # dict key ("value") against the sensitive-name regex would never
    # match, since "value" isn't a sensitive name.
    from dataset.scanning import SENSITIVE_KEY_RE

    assert not SENSITIVE_KEY_RE.search("value")
    assert not SENSITIVE_KEY_RE.search("name")


def test_env_value_from_secret_ref_is_not_flagged():
    doc = {
        "env": [
            {
                "name": "DB_PASSWORD",
                "valueFrom": {
                    "secretKeyRef": {"name": "db-creds", "key": "password"}
                },
            }
        ]
    }
    assert find_secrets(doc) == []


def test_placeholder_values_are_ignored():
    doc = {
        "env": [
            {"name": "DB_PASSWORD", "value": "changeme"},
            {"name": "API_TOKEN", "value": "${API_TOKEN}"},
            {"name": "SECRET_KEY", "value": "<CHANGE_ME>"},
            {"name": "AUTH_TOKEN", "value": "xxx"},
        ]
    }
    assert find_secrets(doc) == []


def test_high_entropy_generic_value_flagged():
    doc = {"data": {"config-value": "aK9!zQ2m#Lp8xR4vT7wY1nB6cJ0d"}}
    hits = find_secrets(doc)
    assert len(hits) == 1
    assert "entropy" in hits[0].reason


def test_connection_string_with_credentials_flagged():
    doc = {"data": {"DATABASE_URL": "postgres://admin:S3cr3tPass@db.internal:5432/app"}}
    hits = find_secrets(doc)
    assert len(hits) == 1
    assert "connection string" in hits[0].reason


def test_pem_private_key_flagged():
    doc = {
        "data": {
            "tls.key": "-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBAK...\n-----END RSA PRIVATE KEY-----"
        }
    }
    hits = find_secrets(doc)
    assert len(hits) == 1
    assert "PEM" in hits[0].reason


def test_hard_negative_harmless_var_named_like_secret():
    # SESSION_TIMEOUT doesn't match the sensitive-key regex, and the value
    # "3600" has neither the length nor the entropy to look like a secret.
    doc = {"env": [{"name": "SESSION_TIMEOUT", "value": "3600"}]}
    assert find_secrets(doc) == []


def test_shannon_entropy_basic():
    assert shannon_entropy("") == 0.0
    assert shannon_entropy("aaaa") == 0.0
    assert shannon_entropy("ab") == 1.0


def test_is_placeholder():
    assert is_placeholder("changeme")
    assert is_placeholder("${DB_PASSWORD}")
    assert is_placeholder("<REPLACE_ME>")
    assert is_placeholder("")
    assert not is_placeholder("S3cr3tR34lLeak99")


def test_walk_leaves_configmap_data_key_is_semantic():
    doc = {"data": {"password": "S3cr3tR34lLeak99"}}
    leaves = list(walk_leaves(doc))
    assert ("/data/password", "password", "S3cr3tR34lLeak99") in leaves


def test_credentials_file_path_is_not_a_secret():
    doc = {"env": [{"name": "GOOGLE_APPLICATION_CREDENTIALS", "value": "/secrets/gcloud_key/key.json"}]}
    assert find_secrets(doc) == []


def test_secret_reference_name_suffix_is_not_flagged():
    doc = {"env": [{"name": "TLS_SECRET_NAME", "value": "kotsadm-tls"}]}
    assert find_secrets(doc) == []


def test_kebab_case_resource_identifier_not_flagged():
    doc = {"data": {"region": "eu-central-1", "app": "rpi3firmware-manager"}}
    assert find_secrets(doc) == []


def test_cli_flag_and_regex_pattern_values_not_flagged():
    doc = {
        "args": ["--timeout=60s", "--max-volumes-per-node=10"],
        "pattern": r"^[0-9]+[\.]?[0-9]*([KMGTPE]i|[kMGTPE])?$",
    }
    assert find_secrets(doc) == []


def test_weak_default_password_still_flagged():
    # A weak/default password is still a real plaintext credential.
    doc = {"env": [{"name": "MYSQL_ROOT_PASSWORD", "value": "admin"}]}
    hits = find_secrets(doc)
    assert len(hits) == 1
    assert hits[0].key == "MYSQL_ROOT_PASSWORD"


def test_base64_encoded_secret_under_sensitive_key_flagged():
    doc = {"data": {"mysql-root-password": "YUFFRlNFWmtYMg=="}}
    hits = find_secrets(doc)
    assert len(hits) == 1


def test_token_header_name_var_is_hard_negative_not_flagged():
    # The name of an HTTP header, not an actual token -- the classic
    # hard-negative pattern called out in the project brief.
    doc = {"env": [{"name": "API_TOKEN_HEADER", "value": "X-Api-Token"}]}
    assert find_secrets(doc) == []


# ---------------------------------------------------------------------------
# Credentials embedded in command/args (CLI flags, basic-auth URLs, bearer
# tokens) -- these are invisible to the generic leaf walk by design, since it
# excludes any value containing a space/slash/colon to avoid flagging
# ordinary shell commands and URLs.
# ---------------------------------------------------------------------------


def test_password_flag_in_args_is_flagged():
    doc = {
        "spec": {
            "containers": [
                {"name": "app", "image": "myapp:1.0.0", "args": ["--password=S3cr3tR34lLeak99"]}
            ]
        }
    }
    hits = find_secrets(doc)
    assert len(hits) == 1
    assert hits[0].value == "S3cr3tR34lLeak99"
    assert "CLI flag" in hits[0].reason


def test_basic_auth_url_in_command_is_flagged():
    doc = {
        "spec": {
            "containers": [
                {
                    "name": "app",
                    "image": "myapp:1.0.0",
                    "command": ["curl", "-sS", "https://admin:S3cr3tR34lLeak99@internal.example.com/report"],
                }
            ]
        }
    }
    hits = find_cli_embedded_secrets(doc)
    assert len(hits) == 1
    assert "basic-auth" in hits[0].reason


def test_bearer_token_in_args_is_flagged():
    doc = {
        "spec": {
            "containers": [
                {
                    "name": "app",
                    "image": "myapp:1.0.0",
                    "args": ["-H", "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.fake.token"],
                }
            ]
        }
    }
    hits = find_cli_embedded_secrets(doc)
    assert len(hits) == 1
    assert "bearer token" in hits[0].reason


def test_ordinary_cli_flags_in_args_not_flagged():
    doc = {"spec": {"containers": [{"name": "app", "args": ["--timeout=60s", "--verbose"]}]}}
    assert find_cli_embedded_secrets(doc) == []


def test_placeholder_password_flag_not_flagged():
    doc = {"spec": {"containers": [{"name": "app", "args": ["--password=${DB_PASSWORD}"]}]}}
    assert find_cli_embedded_secrets(doc) == []


def test_cli_embedded_secret_masked_correctly():
    from dataset.schema import mask_evidence

    doc = {"spec": {"containers": [{"name": "app", "args": ["--api-key=Tr0ub4dor3xyz991"]}]}}
    hits = find_secrets(doc)
    assert len(hits) == 1
    assert mask_evidence(hits[0].value) == "Tr0u***"


def _args_doc(args):
    return {"spec": {"containers": [{"name": "app", "image": "myapp:1.0.0", "args": args}]}}


def test_split_flag_value_is_flagged_once_at_the_value():
    # Real scenario shape (scenarios/1-orion.yaml): flag and value are
    # separate list elements. The random-looking value would also trip the
    # entropy heuristic on its own -- still exactly one hit.
    hits = find_secrets(_args_doc(["-dbhost", "mongo:27017", "-dbuser", "admin", "-dbpwd", "Xk9q2LmP7vRt"]))
    assert len(hits) == 1
    assert hits[0].path == "/spec/containers/0/args/5"
    assert hits[0].value == "Xk9q2LmP7vRt"
    assert "separate CLI flag" in hits[0].reason


def test_split_flag_weak_numeric_password_is_flagged():
    hits = find_cli_embedded_secrets(_args_doc(["--db-password", "123456789"]))
    assert [h.value for h in hits] == ["123456789"]


def test_split_flag_hard_negatives():
    assert find_secrets(_args_doc(["--password-file", "/run/secrets/db"])) == []
    assert find_secrets(_args_doc(["--password", "$(DB_PASSWORD)"])) == []
    assert find_secrets(_args_doc(["--password", "--verbose"])) == []
    assert find_secrets(_args_doc(["--secret", "YOUR SECRET HERE"])) == []
    # --secret/--token flags often take a resource NAME (real corpus cases).
    assert find_secrets(_args_doc(["--secret", "webhook-certs"])) == []
    assert find_secrets(_args_doc(["--initial-cluster-token", "skydns-etcd"])) == []


def test_split_flag_hyphenated_value_after_a_password_flag_is_still_flagged():
    hits = find_cli_embedded_secrets(_args_doc(["--password", "my-secret-pass"]))
    assert [h.value for h in hits] == ["my-secret-pass"]


def test_license_key_is_a_sensitive_key():
    doc = {"env": [{"name": "NEW_RELIC_LICENSE_KEY", "value": "abc123def456ghi789jkl"}]}
    assert [h.key for h in find_secrets(doc)] == ["NEW_RELIC_LICENSE_KEY"]


def test_upper_case_algorithm_names_are_not_secrets():
    # scenarios/1-orion.yaml: `-dbAuthMech SCRAM-SHA-256` was flagged as a
    # high-entropy secret.
    assert find_secrets(_args_doc(["-dbAuthMech", "SCRAM-SHA-256"])) == []
    assert find_secrets({"cipher": "TLS-AES-256-GCM-SHA384"}) == []


def test_mixed_case_hyphenated_random_value_is_still_entropy_checked():
    assert find_secrets({"blob": "Ab3kD9-x7Qp2Zr8W"}) != []


def test_assignment_is_judged_as_the_value_under_its_name():
    # scenarios/7-elasticsearch.yaml: a sysctl init container.
    doc = {"command": ["sysctl", "-w", "vm.max_map_count=262144"]}
    assert find_secrets(doc) == []
    hits = find_secrets({"command": ["env", "DB_PASSWORD=hunter2hunter2"]})
    assert [h.value for h in hits] == ["DB_PASSWORD=hunter2hunter2"]


def test_base64_padding_is_not_mistaken_for_an_assignment():
    assert find_secrets({"blob": "c2VjcmV0cGFzc3dvcmQxMjM0NTY3OA=="}) != []
