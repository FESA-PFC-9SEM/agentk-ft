import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_er_diagram import SCHEMA, main, parse_schema  # noqa: E402


def test_schema_is_parsed_into_tables_and_relationships():
    tables = {t.name: t for t in parse_schema(SCHEMA.read_text())}
    assert set(tables) == {"app_user", "workload", "manifest_change", "finding", "patch_operation", "created_resource"}
    fks = {(t.name, c.name, c.references) for t in tables.values() for c in t.columns if c.references}
    assert fks == {
        ("manifest_change", "workload_id", "workload"), ("manifest_change", "user_id", "app_user"),
        ("finding", "change_id", "manifest_change"), ("patch_operation", "change_id", "manifest_change"),
        ("created_resource", "change_id", "manifest_change"),
    }
    assert [c.type for c in tables["manifest_change"].columns if c.name == "user_roles"] == ["text[]"]


def test_docs_diagram_matches_the_schema():
    main(["--check"])  # raises SystemExit if docs/audit-database.md or the .dot file is stale
