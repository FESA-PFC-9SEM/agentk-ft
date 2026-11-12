-- Audit trail of the cluster agent's changes (database `audit`).
--
-- Applied by deploy/setup_secure_demo.py as the superuser, after it has
-- created the `audit_writer` login role. Idempotent: safe to re-run.
-- docs/audit-database.md describes the tables and shows the entity
-- diagram (generated from this file by deploy/audit_er_diagram.py).
--
-- One manifest_change row per change the agent proposed and the user
-- decided on (confirmed or declined), with its findings, patch operations
-- and the resources it created. Stored manifests are redacted: plaintext
-- credentials are masked (first 4 characters + "***") and a created
-- Secret's values are never stored, only its key names.

CREATE TABLE IF NOT EXISTS app_user (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    subject       text        NOT NULL UNIQUE,  -- OIDC "sub" claim (Keycloak user id)
    username      text        NOT NULL,
    email         text,
    first_seen_at timestamptz NOT NULL DEFAULT now(),
    last_seen_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS workload (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    kind          text        NOT NULL,
    namespace     text        NOT NULL,
    name          text        NOT NULL,
    first_seen_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (kind, namespace, name)
);

CREATE TABLE IF NOT EXISTS manifest_change (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    workload_id     bigint      NOT NULL REFERENCES workload (id),
    user_id         bigint      NOT NULL REFERENCES app_user (id),
    user_roles      text[]      NOT NULL,       -- the user's roles at decision time
    status          text        NOT NULL CHECK (status IN ('applied', 'declined', 'rejected', 'failed')),
    proposed_at     timestamptz NOT NULL,
    decided_at      timestamptz NOT NULL DEFAULT now(),
    model_adapter   text,                       -- the fine-tuned adapter that audited the manifest
    user_request    text,                       -- the chat message that led to the change
    result_message  text        NOT NULL,
    dry_run_output  text,
    manifest_before text        NOT NULL,       -- redacted
    manifest_after  text        NOT NULL,       -- redacted; what was applied (or would have been)
    diff            text        NOT NULL        -- unified diff of the two redacted manifests
);
CREATE INDEX IF NOT EXISTS manifest_change_workload_idx ON manifest_change (workload_id, decided_at DESC);
CREATE INDEX IF NOT EXISTS manifest_change_decided_idx ON manifest_change (decided_at DESC);

CREATE TABLE IF NOT EXISTS finding (
    id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    change_id bigint NOT NULL REFERENCES manifest_change (id),
    position  int    NOT NULL,
    rule_id   text   NOT NULL,                  -- KSEC-001 .. KSEC-012
    severity  text   NOT NULL,
    doc       int    NOT NULL DEFAULT 0,
    path      text   NOT NULL,                  -- JSON Pointer into the manifest
    message   text   NOT NULL,
    evidence  text,                             -- masked
    UNIQUE (change_id, position)
);

CREATE TABLE IF NOT EXISTS patch_operation (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    change_id       bigint NOT NULL REFERENCES manifest_change (id),
    position        int    NOT NULL,
    op              text   NOT NULL,            -- RFC 6902: add, remove, replace, ...
    path            text   NOT NULL,
    doc             int    NOT NULL DEFAULT 0,
    value           jsonb,                      -- the value applied (redacted)
    suggested_value jsonb,                      -- the model's value, when the user chose another (image tags)
    UNIQUE (change_id, position)
);

CREATE TABLE IF NOT EXISTS created_resource (
    id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    change_id bigint NOT NULL REFERENCES manifest_change (id),
    kind      text   NOT NULL,
    namespace text   NOT NULL,
    name      text   NOT NULL,
    data_keys text[] NOT NULL DEFAULT '{}'      -- key names only, never values
);

-- One row per change, for listings.
CREATE OR REPLACE VIEW change_history AS
SELECT c.id,
       c.decided_at,
       c.status,
       u.username,
       w.kind,
       w.namespace,
       w.name,
       (SELECT count(*) FROM finding f WHERE f.change_id = c.id)         AS findings,
       (SELECT count(*) FROM patch_operation p WHERE p.change_id = c.id) AS operations,
       c.model_adapter,
       c.result_message
FROM manifest_change c
JOIN workload w ON w.id = c.workload_id
JOIN app_user u ON u.id = c.user_id;

-- History is append-only: a recorded change, and everything attached to
-- it, can't be edited or deleted -- not even by the tables' owner.
CREATE OR REPLACE FUNCTION audit_forbid_modification() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'the audit trail is append-only: % on % is not allowed', TG_OP, TG_TABLE_NAME;
END
$$;

DO $$
DECLARE
    t text;
BEGIN
    FOREACH t IN ARRAY ARRAY['manifest_change', 'finding', 'patch_operation', 'created_resource'] LOOP
        EXECUTE format('DROP TRIGGER IF EXISTS append_only ON %I', t);
        EXECUTE format('CREATE TRIGGER append_only BEFORE UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION audit_forbid_modification()', t);
    END LOOP;
END
$$;

-- The app's role: read and append, nothing else. It may refresh a user's
-- name, e-mail and last-seen time, never anything else of a row.
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM PUBLIC;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT CONNECT ON DATABASE audit TO audit_writer;
GRANT USAGE ON SCHEMA public TO audit_writer;
GRANT SELECT, INSERT ON app_user, workload, manifest_change, finding, patch_operation, created_resource TO audit_writer;
GRANT UPDATE (username, email, last_seen_at) ON app_user TO audit_writer;
GRANT SELECT ON change_history TO audit_writer;
