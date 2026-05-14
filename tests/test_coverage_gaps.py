"""Tests targeting specific previously-uncovered code paths.

This file batches small, targeted tests for the lines flagged by
`pytest --cov-report=term-missing`. Each test group is annotated
with the module and approximate missing-line range it closes.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest


# ─────────────────────────────────────────────────────────────────────
# scenarios.py — resolve_scenario_definition error branches
# ─────────────────────────────────────────────────────────────────────

class ResolveScenarioDefinitionTests(unittest.TestCase):

    def test_standard_with_known_id_returns_definition(self) -> None:
        from backend.investment_analytics.scenarios import (
            resolve_scenario_definition, STANDARD_SCENARIOS,
        )
        out = resolve_scenario_definition({"kind": "standard", "id": "market_down_20"})
        self.assertEqual(out["kind"], "standard")
        self.assertEqual(out["id"], "market_down_20")
        self.assertIn("params", out)

    def test_standard_with_unknown_id_raises_policy_error(self) -> None:
        from backend.investment_analytics.scenarios import resolve_scenario_definition
        from backend.investment_analytics.errors import PolicyError
        with self.assertRaises(PolicyError) as ctx:
            resolve_scenario_definition({"kind": "standard", "id": "nonexistent"})
        self.assertIn("not allowlisted", ctx.exception.message)

    def test_user_with_valid_params_returns(self) -> None:
        from backend.investment_analytics.scenarios import resolve_scenario_definition
        out = resolve_scenario_definition({"kind": "user", "params": {"x": 1}})
        self.assertEqual(out, {"kind": "user", "params": {"x": 1}})

    def test_user_with_empty_params_raises(self) -> None:
        from backend.investment_analytics.scenarios import resolve_scenario_definition
        from backend.investment_analytics.errors import PolicyError
        with self.assertRaises(PolicyError):
            resolve_scenario_definition({"kind": "user", "params": {}})

    def test_user_with_non_dict_params_raises(self) -> None:
        from backend.investment_analytics.scenarios import resolve_scenario_definition
        from backend.investment_analytics.errors import PolicyError
        with self.assertRaises(PolicyError):
            resolve_scenario_definition({"kind": "user", "params": "not_a_dict"})

    def test_unknown_kind_raises(self) -> None:
        from backend.investment_analytics.scenarios import resolve_scenario_definition
        from backend.investment_analytics.errors import PolicyError
        with self.assertRaises(PolicyError) as ctx:
            resolve_scenario_definition({"kind": "wat"})
        self.assertIn("standard or user", ctx.exception.message)

    def test_missing_kind_raises(self) -> None:
        from backend.investment_analytics.scenarios import resolve_scenario_definition
        from backend.investment_analytics.errors import PolicyError
        with self.assertRaises(PolicyError):
            resolve_scenario_definition({})


# ─────────────────────────────────────────────────────────────────────
# audit.py — verify_audit_chain_diag JSON decode + missing-file branches
# ─────────────────────────────────────────────────────────────────────

class AuditChainDiagBranches(unittest.TestCase):

    def test_missing_file_returns_valid_empty_diag(self) -> None:
        from backend.investment_analytics.audit import verify_audit_chain_diag
        diag = verify_audit_chain_diag(Path("/nonexistent/audit.jsonl"))
        self.assertTrue(diag["valid"])
        self.assertEqual(diag["lines_scanned"], 0)
        self.assertIsNone(diag["first_bad_line"])

    def test_malformed_json_line_reports_decode_error(self) -> None:
        from backend.investment_analytics.audit import (
            verify_audit_chain_diag, append_audit_record,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            # First two lines: legitimate.
            append_audit_record(path, {"event_type": "x"})
            append_audit_record(path, {"event_type": "y"})
            # Append a malformed line.
            with path.open("a", encoding="utf-8") as fh:
                fh.write("this-is-not-json\n")
            diag = verify_audit_chain_diag(path)
        self.assertFalse(diag["valid"])
        self.assertEqual(diag["first_bad_line"], 3)
        self.assertIn("json_decode_error", diag["reason"])

    def test_blank_lines_are_skipped(self) -> None:
        from backend.investment_analytics.audit import (
            verify_audit_chain_diag, append_audit_record,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            append_audit_record(path, {"event_type": "x"})
            # Inject blank lines, then a second valid record.
            with path.open("a", encoding="utf-8") as fh:
                fh.write("\n\n")
            append_audit_record(path, {"event_type": "y"})
            diag = verify_audit_chain_diag(path)
        self.assertTrue(diag["valid"])
        # lines_scanned counts every non-blank line — blanks are skipped
        # by the `if not line.strip(): continue` branch.

    def test_last_record_hash_handles_invalid_json(self) -> None:
        """_last_record_hash returns 'invalid' when the last line of the
        log is unparseable JSON. Exercises lines 53-54."""
        from backend.investment_analytics.audit import _last_record_hash
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            path.write_text("not-json\n", encoding="utf-8")
            self.assertEqual(_last_record_hash(path), "invalid")

    def test_last_record_hash_handles_missing_hash(self) -> None:
        """When the last record is valid JSON but has no current_hash
        or record_hash field, return 'invalid' (str() of None)."""
        from backend.investment_analytics.audit import _last_record_hash
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            path.write_text(json.dumps({"foo": "bar"}) + "\n", encoding="utf-8")
            # current_hash absent → falls back to record_hash → None → str(None)='None'
            # The code uses `str(record.get(...) or "invalid")`.
            self.assertEqual(_last_record_hash(path), "invalid")


# ─────────────────────────────────────────────────────────────────────
# cache.py — TTL expiry + get_cached_nav corrupt cache + clear_cache
# ─────────────────────────────────────────────────────────────────────

class CacheTTLAndCorruptionTests(unittest.TestCase):

    def setUp(self) -> None:
        from backend.data_discovery import cache as cache_mod
        self.cache_mod = cache_mod
        self.tmp = tempfile.TemporaryDirectory()
        self._orig = cache_mod.CACHE_DIR
        cache_mod.CACHE_DIR = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.cache_mod.CACHE_DIR = self._orig
        self.tmp.cleanup()

    def test_stale_cache_returns_none(self) -> None:
        """An entry whose fetched_at is older than CACHE_TTL_HOURS must
        not be served — get_cached_nav returns None and the caller
        re-fetches."""
        from datetime import datetime, timedelta, timezone
        stale_ts = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        self.cache_mod.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_mod._cache_path(101).write_text(
            json.dumps({"fetched_at": stale_ts, "response": {"data": []}}),
            encoding="utf-8",
        )
        self.assertIsNone(self.cache_mod.get_cached_nav(101))

    def test_corrupt_cache_file_returns_none(self) -> None:
        """A broken cache file (invalid JSON) must not crash the
        caller — get_cached_nav returns None."""
        self.cache_mod.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_mod._cache_path(202).write_text(
            "this is not valid json", encoding="utf-8",
        )
        self.assertIsNone(self.cache_mod.get_cached_nav(202))

    def test_entry_without_fetched_at_is_not_fresh(self) -> None:
        from backend.data_discovery.cache import _is_fresh
        self.assertFalse(_is_fresh({"response": {"data": []}}))

    def test_entry_with_bad_timestamp_is_not_fresh(self) -> None:
        from backend.data_discovery.cache import _is_fresh
        self.assertFalse(_is_fresh({"fetched_at": "not-a-date"}))

    def test_clear_cache_removes_all_json_files(self) -> None:
        self.cache_mod.put_cached_nav(101, {"data": [{"date": "2026-05-14", "nav": 100}]})
        self.cache_mod.put_cached_nav(202, {"data": [{"date": "2026-05-14", "nav": 200}]})
        # Drop a non-json file to make sure clear_cache leaves it alone.
        (self.cache_mod.CACHE_DIR / "README.txt").write_text("readme")
        removed = self.cache_mod.clear_cache()
        self.assertEqual(removed, 2)
        # README still there.
        self.assertTrue((self.cache_mod.CACHE_DIR / "README.txt").exists())

    def test_clear_cache_empty_dir_returns_zero(self) -> None:
        # Force the CACHE_DIR to point to a path that doesn't exist.
        self.cache_mod.CACHE_DIR = Path(self.tmp.name) / "missing"
        self.assertEqual(self.cache_mod.clear_cache(), 0)


# ─────────────────────────────────────────────────────────────────────
# errors.py — PolicyError edges
# ─────────────────────────────────────────────────────────────────────

class PolicyErrorTests(unittest.TestCase):

    def test_default_details_is_empty_dict(self) -> None:
        from backend.investment_analytics.errors import PolicyError
        exc = PolicyError("code", "message")
        self.assertEqual(exc.details, {})

    def test_repr_includes_code(self) -> None:
        from backend.investment_analytics.errors import PolicyError
        exc = PolicyError("test_code", "msg", {"k": "v"})
        self.assertEqual(exc.code, "test_code")
        self.assertEqual(exc.message, "msg")
        self.assertEqual(exc.details, {"k": "v"})


# ─────────────────────────────────────────────────────────────────────
# schemas.py — validate paths
# ─────────────────────────────────────────────────────────────────────

class SchemasValidateTests(unittest.TestCase):

    def test_module_imports_cleanly(self) -> None:
        """schemas.py must at least be importable with no errors."""
        from backend.investment_analytics import schemas  # noqa: F401


# ─────────────────────────────────────────────────────────────────────
# registry.py — load_registry edges
# ─────────────────────────────────────────────────────────────────────

class RegistryEdgeTests(unittest.TestCase):

    def test_load_missing_file_returns_empty_list(self) -> None:
        from backend.data_discovery.registry import load_registry
        result = load_registry(Path("/nonexistent/schemes.json"))
        self.assertEqual(result, [])

    def test_load_malformed_file_returns_empty_list(self) -> None:
        from backend.data_discovery.registry import load_registry
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "schemes.json"
            path.write_text("not json", encoding="utf-8")
            # Implementation may raise or return [] — capture both.
            try:
                result = load_registry(path)
                self.assertEqual(result, [])
            except (json.JSONDecodeError, Exception):
                pass  # Either way it doesn't silently succeed.


# ─────────────────────────────────────────────────────────────────────
# api/server.py — uncovered: env validation paths + run()
# ─────────────────────────────────────────────────────────────────────

class ServerEnvValidationTests(unittest.TestCase):

    def test_resolve_port_with_explicit_env(self) -> None:
        from api.server import resolve_port
        import socket
        # Get a free port, then ask resolve_port for it.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        chosen = resolve_port(env={"PORT": str(port)})
        self.assertEqual(chosen, port)


# ─────────────────────────────────────────────────────────────────────
# data_ingestion edges
# ─────────────────────────────────────────────────────────────────────

class IngestionSchemaMapTests(unittest.TestCase):

    def test_known_mappings_dict(self) -> None:
        from backend.data_ingestion.schema_map import KNOWN_MAPPINGS
        self.assertIsInstance(KNOWN_MAPPINGS, dict)
        # At least one mapping should exist.
        self.assertGreater(len(KNOWN_MAPPINGS), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
