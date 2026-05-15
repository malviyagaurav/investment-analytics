"""Tests for `python -m backend.audit verify`.

Coverage map:
  exit codes      × 5  (valid / partial_failure / empty / invalid / unverifiable)
  human output    × 2  (valid + partial_failure surfaces line + reason)
  --json output   × 1  (matches verify_audit_chain_multi shape)
  --audit-dir     × 1  (honors override)
  read-only       × 1  (audit.jsonl line count unchanged after verify)
  subcommand req  × 1  (running with no subcommand fails — argparse default)
"""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from backend import audit as audit_cli
from backend.investment_analytics.audit import (
    _canonical_json,
    append_audit_record,
)
from backend.investment_analytics.audit_migrate import EPOCHS_SCHEMA_FINGERPRINT


def _seed_chain(path: Path, n: int = 3) -> None:
    for i in range(n):
        append_audit_record(path, {"event_type": "test", "i": i,
                                   "schema_version": "v1"})


def _run(argv: list) -> tuple[int, str]:
    """Run the CLI's _main, capturing stdout."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = audit_cli._main(argv)
    return rc, buf.getvalue()


class VerifyExitCodeTests(unittest.TestCase):
    """Each typed overall_status maps to a specific exit code so ops
    scripts can chain `python -m backend.audit verify && deploy`."""

    def test_valid_chain_returns_exit_0(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            audit_dir = Path(td)
            _seed_chain(audit_dir / "audit.jsonl", n=3)
            rc, out = _run(["verify", "--audit-dir", str(audit_dir)])
            self.assertEqual(rc, 0, out)
            self.assertIn("valid", out)

    def test_empty_dir_returns_exit_2(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rc, out = _run(["verify", "--audit-dir", td])
            self.assertEqual(rc, 2)
            self.assertIn("empty", out)

    def test_invalid_chain_returns_exit_3(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            audit_dir = Path(td)
            path = audit_dir / "audit.jsonl"
            _seed_chain(path, n=3)
            # Corrupt the second record so verification fails.
            lines = path.read_text(encoding="utf-8").splitlines()
            tampered = json.loads(lines[1])
            tampered["payload_hash"] = "f" * 64
            lines[1] = _canonical_json(tampered)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            rc, out = _run(["verify", "--audit-dir", str(audit_dir)])
            self.assertEqual(rc, 3, out)
            self.assertIn("invalid", out)

    def test_partial_failure_returns_exit_2(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            audit_dir = Path(td)
            path_a = audit_dir / "epoch_1.jsonl"
            path_b = audit_dir / "audit.jsonl"
            _seed_chain(path_a, n=2)
            _seed_chain(path_b, n=2)
            # Corrupt the second file.
            lines = path_b.read_text(encoding="utf-8").splitlines()
            tampered = json.loads(lines[1])
            tampered["payload_hash"] = "f" * 64
            lines[1] = _canonical_json(tampered)
            path_b.write_text("\n".join(lines) + "\n", encoding="utf-8")

            index = {
                "schema_version": "v1",
                "schema_fingerprint": EPOCHS_SCHEMA_FINGERPRINT,
                "epochs": [
                    {"epoch": 1, "file": "epoch_1.jsonl", "status": "closed",
                     "chain_root_type": "genesis"},
                    {"epoch": 2, "file": "audit.jsonl", "status": "open",
                     "chain_root_type": "rotated_handoff"},
                ],
                "orphan_chains": [],
                "registered_at": "2026-05-15T00:00:00+00:00",
            }
            (audit_dir / "epochs.json").write_text(json.dumps(index))

            rc, out = _run(["verify", "--audit-dir", str(audit_dir)])
            self.assertEqual(rc, 2, out)
            self.assertIn("partial_failure", out)

    def test_unverifiable_returns_exit_3(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            audit_dir = Path(td)
            (audit_dir / "epochs.json").write_text("{ not valid json")
            rc, out = _run(["verify", "--audit-dir", str(audit_dir)])
            self.assertEqual(rc, 3, out)
            self.assertIn("unverifiable", out)


class VerifyOutputShapeTests(unittest.TestCase):
    """Human output is for at-a-glance scanning; --json output is
    the typed verify_audit_chain_multi return passed through."""

    def test_human_output_lists_per_epoch_status(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            audit_dir = Path(td)
            _seed_chain(audit_dir / "audit.jsonl", n=3)
            _, out = _run(["verify", "--audit-dir", str(audit_dir)])
            self.assertIn("audit chain: valid", out)
            self.assertIn("epoch 1", out)
            self.assertIn("audit.jsonl", out)
            self.assertIn("total lines", out)

    def test_human_output_surfaces_first_bad_line_and_reason(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            audit_dir = Path(td)
            path = audit_dir / "audit.jsonl"
            _seed_chain(path, n=3)
            lines = path.read_text(encoding="utf-8").splitlines()
            tampered = json.loads(lines[1])
            tampered["payload_hash"] = "f" * 64
            lines[1] = _canonical_json(tampered)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            _, out = _run(["verify", "--audit-dir", str(audit_dir)])
            self.assertIn("first bad line", out)
            self.assertIn("reason:", out)

    def test_json_output_matches_verify_audit_chain_multi_shape(self) -> None:
        from backend.investment_analytics.audit import verify_audit_chain_multi
        with tempfile.TemporaryDirectory() as td:
            audit_dir = Path(td)
            _seed_chain(audit_dir / "audit.jsonl", n=3)
            _, out = _run(["verify", "--audit-dir", str(audit_dir), "--json"])
            parsed = json.loads(out)
            expected = verify_audit_chain_multi(audit_dir)
            # Compare the keys we care about — both calls run independently
            # so transient ordering of dict keys is irrelevant after parse.
            for key in ("overall_status", "epochs_total", "epochs_valid",
                        "epochs_failed", "orphan_chains_total",
                        "per_epoch", "per_orphan"):
                self.assertEqual(parsed[key], expected[key],
                                 f"key {key!r} mismatched between CLI JSON "
                                 f"and verify_audit_chain_multi")


class VerifyOperationalDisciplineTests(unittest.TestCase):

    def test_audit_dir_override_is_honored(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            audit_dir = Path(td) / "nested" / "deep"
            audit_dir.mkdir(parents=True)
            _seed_chain(audit_dir / "audit.jsonl", n=2)
            rc, out = _run(["verify", "--audit-dir", str(audit_dir)])
            self.assertEqual(rc, 0, out)
            self.assertIn("valid", out)

    def test_verify_does_not_mutate_the_chain(self) -> None:
        # Read-only invariant: line count must be byte-identical
        # before and after a verify run. No hidden writes.
        with tempfile.TemporaryDirectory() as td:
            audit_dir = Path(td)
            path = audit_dir / "audit.jsonl"
            _seed_chain(path, n=4)
            before_bytes = path.read_bytes()
            for _ in range(3):
                rc, _ = _run(["verify", "--audit-dir", str(audit_dir)])
                self.assertEqual(rc, 0)
            after_bytes = path.read_bytes()
            self.assertEqual(before_bytes, after_bytes,
                             "verify CLI must be byte-for-byte read-only")

    def test_missing_subcommand_exits_nonzero(self) -> None:
        # argparse with required=True rejects argv=[] with SystemExit(2).
        with self.assertRaises(SystemExit) as cm:
            _run([])
        self.assertEqual(cm.exception.code, 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
