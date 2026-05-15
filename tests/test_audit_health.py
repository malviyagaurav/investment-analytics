"""Tests for the extended /health endpoint (Step 8).

Coverage map:
  preserved fields           × 1   (existing consumers must continue to work)
  audit sub-object shape     × 1   (typed payload present + verification_duration_ms)
  top-level status mapping   × 5   (valid / partial / invalid / empty / unverifiable
                                    + data-dir-missing override)
  permission-error fallback  × 1   (OSError from verify_audit_chain_multi
                                    surfaces as typed unverifiable + 200, NEVER 500)
  read-only polling          × 1   (polling /health N× does not change line count)
  http status code           × 1   (always 200 — typed status in body)
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from starlette.testclient import TestClient

from api import main as api_main


class HealthBaselineTests(unittest.TestCase):
    """Existing consumers' contract: preserved fields, correct types,
    status code 200, top-level status='ok' on a clean chain."""

    def setUp(self) -> None:
        self.client = TestClient(api_main.app)

    def test_preserved_fields_present(self) -> None:
        # /health must continue to expose every field its existing
        # consumers rely on. NEW fields land alongside; no field
        # gets renamed or dropped.
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        for key in ("status", "audit_chain_valid",
                    "data_directory_accessible", "registered_sources"):
            self.assertIn(key, data, f"missing preserved field {key!r}")
        self.assertIn(data["status"], ("ok", "degraded", "unhealthy"))
        self.assertIsInstance(data["audit_chain_valid"], bool)
        self.assertIsInstance(data["data_directory_accessible"], bool)
        self.assertIsInstance(data["registered_sources"], int)

    def test_audit_sub_object_present_with_typed_shape(self) -> None:
        resp = self.client.get("/health")
        data = resp.json()
        self.assertIn("audit", data)
        audit = data["audit"]
        for key in ("overall_status", "epochs_total", "epochs_valid",
                    "epochs_failed", "orphan_chains_total", "total_lines",
                    "per_epoch", "per_orphan", "verification_duration_ms"):
            self.assertIn(key, audit, f"missing audit.{key}")
        self.assertIn(audit["overall_status"],
                      {"valid", "partial_failure", "invalid",
                       "empty", "unverifiable"})
        self.assertIsInstance(audit["verification_duration_ms"], int)
        self.assertGreaterEqual(audit["verification_duration_ms"], 0)
        # total_lines is the aggregate of per_epoch lines_scanned.
        expected_total = sum(
            (e.get("lines_scanned") or 0) for e in audit["per_epoch"]
        )
        self.assertEqual(audit["total_lines"], expected_total)

    def test_http_status_is_always_200(self) -> None:
        # Even under unhealthy conditions (data dir missing) we keep
        # 200 — audit-chain issues are diagnostic, not LB-eviction signals.
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        with patch.object(api_main, "ROOT", Path("/nonexistent/path")):
            resp = self.client.get("/health")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["status"], "unhealthy")


class HealthStatusMappingTests(unittest.TestCase):
    """Top-level status (ok / degraded / unhealthy) derived from the
    typed audit overall_status. Preserves the 'no fake green' rule:
    invalid + unverifiable always escalate to unhealthy; partial
    surfaces as degraded; empty stays ok (structurally clean)."""

    def setUp(self) -> None:
        self.client = TestClient(api_main.app)

    def _patch_verify(self, **fields):
        """Build a synthetic verify_audit_chain_multi result and
        patch it in for one request."""
        defaults = {
            "overall_status": "valid",
            "epochs_total": 1, "epochs_valid": 1, "epochs_failed": 0,
            "orphan_chains_total": 0,
            "per_epoch": [{"epoch": 1, "file": "audit.jsonl",
                           "status": "valid", "lines_scanned": 100,
                           "first_bad_line": None, "reason": None}],
            "per_orphan": [],
        }
        defaults.update(fields)
        return patch.object(api_main, "verify_audit_chain_multi",
                            return_value=defaults)

    def test_valid_maps_to_ok(self) -> None:
        with self._patch_verify(overall_status="valid"):
            data = self.client.get("/health").json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["audit_chain_valid"])
        self.assertEqual(data["audit"]["overall_status"], "valid")

    def test_partial_failure_maps_to_degraded(self) -> None:
        with self._patch_verify(overall_status="partial_failure",
                                epochs_valid=1, epochs_failed=1):
            data = self.client.get("/health").json()
        self.assertEqual(data["status"], "degraded")
        self.assertFalse(data["audit_chain_valid"])
        self.assertEqual(data["audit"]["overall_status"], "partial_failure")

    def test_invalid_maps_to_unhealthy(self) -> None:
        with self._patch_verify(overall_status="invalid",
                                epochs_valid=0, epochs_failed=1):
            data = self.client.get("/health").json()
        self.assertEqual(data["status"], "unhealthy")
        self.assertFalse(data["audit_chain_valid"])

    def test_empty_maps_to_ok_not_degraded(self) -> None:
        # Decision Q2: empty chain is structurally clean. HTTP says ok.
        # (CLI exit code is still 2 — different signal for different
        # audience, by design.)
        with self._patch_verify(overall_status="empty",
                                epochs_total=0, epochs_valid=0,
                                per_epoch=[]):
            data = self.client.get("/health").json()
        self.assertEqual(data["status"], "ok")
        self.assertFalse(data["audit_chain_valid"])
        self.assertEqual(data["audit"]["overall_status"], "empty")

    def test_unverifiable_maps_to_unhealthy(self) -> None:
        with self._patch_verify(overall_status="unverifiable",
                                epochs_total=0, per_epoch=[],
                                reason="epochs.json could not be parsed"):
            data = self.client.get("/health").json()
        self.assertEqual(data["status"], "unhealthy")
        self.assertIn("reason", data["audit"])


class HealthPermissionErrorTests(unittest.TestCase):
    """If the chain primitive raises (OSError / PermissionError), the
    /health endpoint must degrade into a typed diagnostic — never let
    FastAPI emit a 500 stack trace from a polling endpoint."""

    def setUp(self) -> None:
        self.client = TestClient(api_main.app)

    def test_permission_error_returns_typed_unverifiable_not_500(self) -> None:
        with patch.object(api_main, "verify_audit_chain_multi",
                          side_effect=PermissionError("denied")):
            resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200,
                         "permission error must NOT bubble as 500")
        data = resp.json()
        self.assertEqual(data["status"], "unhealthy")
        self.assertEqual(data["audit"]["overall_status"], "unverifiable")
        self.assertIn("permission_error", data["audit"]["reason"])

    def test_generic_oserror_returns_typed_unverifiable(self) -> None:
        with patch.object(api_main, "verify_audit_chain_multi",
                          side_effect=OSError("disk vanished")):
            resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "unhealthy")
        self.assertEqual(data["audit"]["overall_status"], "unverifiable")
        self.assertIn("permission_error", data["audit"]["reason"])


class HealthPollingSafetyTests(unittest.TestCase):
    """The cornerstone of Step 8: polling /health is read-only.
    Repeated polls must NEVER produce an audit append, must NEVER
    mutate any on-disk state."""

    def test_repeated_polls_do_not_change_audit_chain(self) -> None:
        # We poll the live /health (which reads the live audit.jsonl).
        # Capture the byte content before and after N polls.
        client = TestClient(api_main.app)
        path = api_main.AUDIT_PATH
        before = path.read_bytes()
        for _ in range(5):
            resp = client.get("/health")
            self.assertEqual(resp.status_code, 200)
        after = path.read_bytes()
        self.assertEqual(before, after,
                         "polling /health must be byte-for-byte read-only "
                         "against the live audit chain")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
