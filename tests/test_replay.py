"""Tests for backend.evidence.replay.

Coverage map:

  exact_match               × 1
  semantically_equivalent   × 1 (patched computed_at)
  expected_divergence       × 2 (methodology bump, registry hash change)
  unreproducible            × 4 (no evidence_ref, file missing,
                                 unsupported kind, run_id not found)
  invalid_replay            × 4 (chain broken, hash mismatch,
                                 handler raises, differs-without-driver)
  discipline                × 5 (parent_run_id link, --dry-run quietness,
                                 replay-of-replay → unreproducible,
                                 watchlist replay has no side-effects,
                                 chain stays valid across mixed sequence)
  CLI                       × 2 (happy path, dry-run prints, no row)
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.evidence import replay as rp
from backend.evidence.store import emit_evidence
from backend.investment_analytics import methodology
from backend.investment_analytics.audit import verify_audit_chain
from backend.investment_analytics.ranking import (
    CategoryRanking,
    FundMetrics,
    RankedFund,
)


def _f(code: int, **kw) -> FundMetrics:
    base = dict(
        scheme_code=code,
        fund_name=f"Fund {code} Direct Plan - Growth",
        fund_house=f"AMC {code // 100}",
        excess_return_pct=2.0, max_drawdown_pct=-15.0, consistency_pct=55.0,
        volatility_pct=12.0, downside_capture_ratio=1.0,
        fund_cagr_pct=10.0, benchmark_cagr_pct=8.0,
        aligned_points=1500, history_years=8.0, drawdown_trough_date=None,
    )
    base.update(kw)
    return FundMetrics(**base)


def _ranking(category: str, codes: list, computed_at="2026-05-14T00:00:00+00:00") -> CategoryRanking:
    funds = [_f(c, excess_return_pct=4.0 - i) for i, c in enumerate(codes)]
    ranked = [
        RankedFund(rank=i + 1, fund=f, dominance_count=len(funds) - 1 - i,
                   total_peers=len(funds), confidence_level="High",
                   strengths=[], weaknesses=[])
        for i, f in enumerate(funds)
    ]
    return CategoryRanking(
        category=category, benchmark_name="Nifty 100", benchmark_code=999,
        benchmark_fallback=False, ranked=ranked, excluded=[],
        computed_at=computed_at, total_funds_in_category=len(funds),
    )


class ReplayTestBase(unittest.TestCase):
    """Shared fixture: isolated audit + evidence + registry under tmp."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp.name)
        self.audit_path = self.data_dir / "audit" / "audit.jsonl"
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        # A non-empty registry file so capture_provenance_inputs gets a
        # real sha256 (rather than None) and our handlers don't see a
        # "missing registry" signal unexpectedly.
        self.registry_path = self.data_dir / "registry" / "schemes.json"
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text("[]", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _read_events(self) -> list:
        if not self.audit_path.exists():
            return []
        with self.audit_path.open("r", encoding="utf-8") as h:
            return [json.loads(line)["event"] for line in h if line.strip()]

    def _emit_ranking_snapshot(
        self,
        category: str = "Equity Scheme - Large Cap Fund",
        codes=(101, 102, 103),
        computed_at="2026-05-14T00:00:00+00:00",
    ) -> dict:
        """Emit a real ranking_snapshot audit row + evidence file using
        the production emit_evidence path. Returns the audit record."""
        from backend.investment_analytics.ranking import ranking_to_dict
        result = _ranking(category, list(codes), computed_at=computed_at)
        payload = ranking_to_dict(result)
        return emit_evidence(
            audit_log_path=self.audit_path,
            evidence_kind="ranking_snapshot",
            audit_event={
                "event_type": "rank_category",
                "subject_token": "test",
                "category": category,
                "ranked_count": len(result.ranked),
                "excluded_count": 0,
                "benchmark_code": result.benchmark_code,
                "benchmark_fallback": False,
                "schema_version": "v1",
            },
            payload=payload,
        )


# ── exact_match ───────────────────────────────────────────────────────


class ExactMatchTests(ReplayTestBase):

    def test_exact_match_when_handler_produces_identical_payload(self) -> None:
        record = self._emit_ranking_snapshot()
        run_id = record["event"]["run_id"]

        # Patch rank_category at the package boundary so the handler's
        # lazy `from ...ranking import rank_category` resolves to our
        # deterministic stub. Returns the SAME ranking (same codes,
        # same computed_at) as the original emit.
        def fake_rank(category, _path):
            return _ranking(category, [101, 102, 103])

        with patch("backend.investment_analytics.ranking.rank_category",
                   side_effect=fake_rank):
            result = rp.replay_run(
                self.audit_path, run_id,
                verify_chain=False,
                emit_audit=False,
                registry_path=self.registry_path,
            )
        self.assertEqual(result["state"], "exact_match",
                         f"state={result['state']} reason={result.get('reason')}")
        self.assertEqual(result["differences"], [])
        self.assertEqual(result["divergence_drivers"], [])
        self.assertEqual(result["audit_event_ref"]["run_id"], run_id)


# ── semantically_equivalent ───────────────────────────────────────────


class SemanticallyEquivalentTests(ReplayTestBase):

    def test_volatile_computed_at_difference_classified_as_equivalent(self) -> None:
        record = self._emit_ranking_snapshot(
            computed_at="2026-05-14T10:00:00+00:00",
        )
        run_id = record["event"]["run_id"]

        def fake_rank(category, _path):
            return _ranking(category, [101, 102, 103],
                            computed_at="2026-05-15T11:11:11+00:00")  # later

        with patch("backend.investment_analytics.ranking.rank_category",
                   side_effect=fake_rank):
            result = rp.replay_run(
                self.audit_path, run_id,
                verify_chain=False, emit_audit=False,
                registry_path=self.registry_path,
            )
        self.assertEqual(result["state"], "semantically_equivalent",
                         f"reason={result.get('reason')}")
        self.assertIn("computed_at", result["differences"])
        self.assertEqual(result["divergence_drivers"], [])


# ── expected_divergence ───────────────────────────────────────────────


class ExpectedDivergenceTests(ReplayTestBase):

    def test_methodology_bump_produces_expected_divergence(self) -> None:
        record = self._emit_ranking_snapshot()
        run_id = record["event"]["run_id"]

        # Bump methodology AFTER the emit. Handler returns a payload
        # with different content too (the methodology change would
        # plausibly affect ranking output; we model that here).
        original_versions = dict(methodology.METHODOLOGY_VERSIONS)
        methodology.METHODOLOGY_VERSIONS["equity_metric"] = "v2"

        def fake_rank(category, _path):
            return _ranking(category, [999, 998, 997])  # different codes

        try:
            with patch("backend.investment_analytics.ranking.rank_category",
                       side_effect=fake_rank):
                result = rp.replay_run(
                    self.audit_path, run_id,
                    verify_chain=False, emit_audit=False,
                    registry_path=self.registry_path,
                )
        finally:
            methodology.METHODOLOGY_VERSIONS.clear()
            methodology.METHODOLOGY_VERSIONS.update(original_versions)

        self.assertEqual(result["state"], "expected_divergence",
                         f"reason={result.get('reason')}")
        kinds = [d["kind"] for d in result["divergence_drivers"]]
        self.assertIn("methodology_changed", kinds)

    def test_registry_hash_change_produces_expected_divergence(self) -> None:
        # The production emit path does NOT pass registry_path to
        # capture_provenance_inputs today, so recorded.registry_hash is
        # None for live emitters — asymmetric None is intentionally
        # NOT flagged as a driver. To exercise the driver itself we
        # construct the recorded envelope manually with an explicit
        # registry_hash captured at "emit time", then mutate the
        # registry so the replay capture diverges. This also keeps
        # the test forward-compatible: when emitters start passing
        # registry_path through, the same logic catches real drift.
        from backend.investment_analytics.audit import append_audit_record
        from backend.investment_analytics.methodology import current_methodology
        from backend.investment_analytics.provenance import capture_provenance_inputs
        from backend.evidence.store import write_evidence
        from backend.investment_analytics.evidence_envelope import build_event_envelope
        from backend.investment_analytics.ranking import ranking_to_dict

        # Capture inputs WITH the registry_path so registry_hash is
        # populated; freeze that capture into recorded.inputs.
        recorded_inputs = capture_provenance_inputs(registry_path=self.registry_path)
        self.assertIsNotNone(recorded_inputs["registry_hash"])

        result_obj = _ranking(
            "Equity Scheme - Large Cap Fund", [101, 102, 103],
        )
        payload = ranking_to_dict(result_obj)
        audit_event = {
            "event_type": "rank_category",
            "subject_token": "test",
            "category": "Equity Scheme - Large Cap Fund",
            "ranked_count": 3, "excluded_count": 0,
            "benchmark_code": 999, "benchmark_fallback": False,
            "schema_version": "v1",
        }
        # Build envelope + write evidence + append audit with the
        # explicit recorded_inputs we just captured.
        envelope = build_event_envelope(
            {**audit_event, "payload": payload},
            evidence_kind="ranking_snapshot",
            inputs=recorded_inputs,
        )
        run_id = envelope["run_id"]
        evidence_ref = write_evidence(
            self.audit_path.parent, "ranking_snapshot", envelope,
        )
        append_audit_record(
            self.audit_path,
            {**audit_event, "evidence_ref": evidence_ref},
            evidence_kind="ranking_snapshot",
            run_id=run_id,
            inputs=recorded_inputs,
        )

        # Mutate the registry between emit and replay — the replay's
        # capture_provenance_inputs(registry_path=...) will produce a
        # different sha than the one frozen into recorded.inputs.
        self.registry_path.write_text(
            '[{"scheme_code": 1, "scheme_name": "Test"}]', encoding="utf-8",
        )

        def fake_rank(category, _path):
            return _ranking(category, [999, 998, 997])  # different content too

        with patch("backend.investment_analytics.ranking.rank_category",
                   side_effect=fake_rank):
            result = rp.replay_run(
                self.audit_path, run_id,
                verify_chain=False, emit_audit=False,
                registry_path=self.registry_path,
            )
        self.assertEqual(result["state"], "expected_divergence",
                         f"reason={result.get('reason')}")
        kinds = [d["kind"] for d in result["divergence_drivers"]]
        self.assertIn("registry_hash_changed", kinds)


# ── unreproducible (× 4) ──────────────────────────────────────────────


class UnreproducibleTests(ReplayTestBase):

    def test_run_id_not_found_classified_as_unreproducible(self) -> None:
        # Emit nothing — chain has no rows. Lookup fails.
        result = rp.replay_run(
            self.audit_path, run_id="00000000-0000-4000-8000-000000000000",
            verify_chain=False, emit_audit=False,
            registry_path=self.registry_path,
        )
        self.assertEqual(result["state"], "unreproducible")
        self.assertEqual(result["missing_inputs"], ["audit_record"])

    def test_no_evidence_ref_on_row_classified_as_unreproducible(self) -> None:
        # Emit a no_change-style audit-only row (no evidence_ref).
        from backend.investment_analytics.audit import append_audit_record
        record = append_audit_record(
            self.audit_path,
            {"event_type": "no_change", "snapshot_date": "2026-05-14",
             "reason": "weekend", "schema_version": "v1"},
            evidence_kind="watchlist_run",
        )
        run_id = record["event"]["run_id"]
        result = rp.replay_run(
            self.audit_path, run_id,
            verify_chain=False, emit_audit=False,
            registry_path=self.registry_path,
        )
        self.assertEqual(result["state"], "unreproducible")
        self.assertIn("evidence_ref", result["missing_inputs"])

    def test_evidence_file_missing_classified_as_unreproducible(self) -> None:
        record = self._emit_ranking_snapshot()
        run_id = record["event"]["run_id"]
        # Delete the evidence file out from under replay.
        ev_path = self.data_dir / record["event"]["evidence_ref"]["path"]
        ev_path.unlink()
        result = rp.replay_run(
            self.audit_path, run_id,
            verify_chain=False, emit_audit=False,
            registry_path=self.registry_path,
        )
        self.assertEqual(result["state"], "unreproducible")
        self.assertEqual(result["missing_inputs"], ["evidence_file"])

    def test_unsupported_evidence_kind_classified_as_unreproducible(self) -> None:
        # Synthesize a row with an evidence_kind that has no handler.
        # We use experiment_run (in EVIDENCE_KINDS but no emitter yet,
        # and no replay handler — exactly the targeted case).
        record = emit_evidence(
            audit_log_path=self.audit_path,
            evidence_kind="experiment_run",
            audit_event={"event_type": "experiment_run", "schema_version": "v1"},
            payload={"opaque": True},
        )
        run_id = record["event"]["run_id"]
        result = rp.replay_run(
            self.audit_path, run_id,
            verify_chain=False, emit_audit=False,
            registry_path=self.registry_path,
        )
        self.assertEqual(result["state"], "unreproducible")
        self.assertIn("experiment_run", result["reason"])

    def test_registry_missing_classified_as_unreproducible(self) -> None:
        record = self._emit_ranking_snapshot()
        run_id = record["event"]["run_id"]
        self.registry_path.unlink()
        result = rp.replay_run(
            self.audit_path, run_id,
            verify_chain=False, emit_audit=False,
            registry_path=self.registry_path,
        )
        self.assertEqual(result["state"], "unreproducible")
        self.assertEqual(result["missing_inputs"], ["registry"])


# ── invalid_replay (× 4) ──────────────────────────────────────────────


class InvalidReplayTests(ReplayTestBase):

    def test_chain_verification_failure_classified_as_invalid_replay(self) -> None:
        record = self._emit_ranking_snapshot()
        run_id = record["event"]["run_id"]

        # Corrupt the audit chain by appending a bogus line that breaks
        # the hash linkage. verify_audit_chain will reject it.
        with self.audit_path.open("a", encoding="utf-8") as h:
            h.write(json.dumps({
                "timestamp": "2026-05-14T00:00:00+00:00",
                "schema_version": "v1",
                "analyzer_version": "mf_v2",
                "chain_epoch": 1,
                "prev_hash": "ZZZ_BROKEN_LINKAGE",
                "payload_hash": "x",
                "current_hash": "y",
                "event": {"event_type": "junk"},
            }) + "\n")

        result = rp.replay_run(
            self.audit_path, run_id,
            verify_chain=True, emit_audit=False,
            registry_path=self.registry_path,
        )
        self.assertEqual(result["state"], "invalid_replay")
        self.assertIn("chain", result["reason"])

    def test_evidence_hash_mismatch_classified_as_invalid_replay(self) -> None:
        record = self._emit_ranking_snapshot()
        run_id = record["event"]["run_id"]
        # Flip a byte in the evidence file.
        ev_path = self.data_dir / record["event"]["evidence_ref"]["path"]
        ev_path.write_bytes(ev_path.read_bytes() + b" ")
        result = rp.replay_run(
            self.audit_path, run_id,
            verify_chain=False, emit_audit=False,
            registry_path=self.registry_path,
        )
        self.assertEqual(result["state"], "invalid_replay")
        self.assertIn("hash mismatch", result["reason"].lower())

    def test_handler_raising_classified_as_invalid_replay(self) -> None:
        record = self._emit_ranking_snapshot()
        run_id = record["event"]["run_id"]

        def raising(*_a, **_kw):
            raise RuntimeError("synthetic handler failure")

        with patch("backend.investment_analytics.ranking.rank_category",
                   side_effect=raising):
            result = rp.replay_run(
                self.audit_path, run_id,
                verify_chain=False, emit_audit=False,
                registry_path=self.registry_path,
            )
        self.assertEqual(result["state"], "invalid_replay")
        self.assertIn("synthetic handler failure", result["reason"])

    def test_differs_without_driver_classified_as_invalid_replay(self) -> None:
        # Content diverges (different codes) AND no input driver
        # changed (methodology fixed, registry fixed, code_sha fixed).
        # Per the approved decision default, this is invalid_replay.
        record = self._emit_ranking_snapshot()
        run_id = record["event"]["run_id"]

        def fake_rank(category, _path):
            return _ranking(category, [999, 998, 997])  # different content

        with patch("backend.investment_analytics.ranking.rank_category",
                   side_effect=fake_rank):
            result = rp.replay_run(
                self.audit_path, run_id,
                verify_chain=False, emit_audit=False,
                registry_path=self.registry_path,
            )
        self.assertEqual(result["state"], "invalid_replay",
                         f"reason={result.get('reason')}")
        self.assertEqual(result["divergence_drivers"], [])


# ── Discipline (linkage, dry-run, replay-of-replay, side-effects, chain) ──


class ReplayDisciplineTests(ReplayTestBase):

    def test_emitted_replay_result_links_via_parent_run_id(self) -> None:
        record = self._emit_ranking_snapshot()
        original_run_id = record["event"]["run_id"]

        def fake_rank(category, _path):
            return _ranking(category, [101, 102, 103])

        with patch("backend.investment_analytics.ranking.rank_category",
                   side_effect=fake_rank):
            rp.replay_run(
                self.audit_path, original_run_id,
                verify_chain=False, emit_audit=True,
                registry_path=self.registry_path,
            )

        events = self._read_events()
        replay_rows = [e for e in events if e.get("event_type") == "replay_run"]
        self.assertEqual(len(replay_rows), 1)
        row = replay_rows[0]
        self.assertEqual(row["evidence_kind"], "replay_result")
        self.assertEqual(row["parent_run_id"], original_run_id)
        self.assertEqual(row["audit_event_ref"]["run_id"], original_run_id)

    def test_dry_run_writes_no_audit_row(self) -> None:
        record = self._emit_ranking_snapshot()
        run_id = record["event"]["run_id"]
        before = len(self._read_events())

        def fake_rank(category, _path):
            return _ranking(category, [101, 102, 103])

        with patch("backend.investment_analytics.ranking.rank_category",
                   side_effect=fake_rank):
            rp.replay_run(
                self.audit_path, run_id,
                verify_chain=False, emit_audit=False,
                registry_path=self.registry_path,
            )
        after = len(self._read_events())
        self.assertEqual(before, after,
                         "dry-run must not append any audit row")

    def test_replaying_a_replay_result_row_is_unreproducible(self) -> None:
        # First, emit + replay to land a replay_result row on the chain.
        original = self._emit_ranking_snapshot()
        original_run_id = original["event"]["run_id"]

        def fake_rank(category, _path):
            return _ranking(category, [101, 102, 103])

        with patch("backend.investment_analytics.ranking.rank_category",
                   side_effect=fake_rank):
            rp.replay_run(
                self.audit_path, original_run_id,
                verify_chain=False, emit_audit=True,
                registry_path=self.registry_path,
            )

        # Find the replay_result row's run_id; replay THAT one.
        events = self._read_events()
        replay_row = next(e for e in events if e.get("event_type") == "replay_run")
        replay_run_id = replay_row["run_id"]

        result = rp.replay_run(
            self.audit_path, replay_run_id,
            verify_chain=False, emit_audit=False,
            registry_path=self.registry_path,
        )
        self.assertEqual(result["state"], "unreproducible")
        self.assertIn("replay_result", result["reason"])

    def test_watchlist_replay_has_no_side_effects(self) -> None:
        # Emit a watchlist_run audit row directly via emit_evidence with
        # a tiny config so the rerun is cheap. Mock the rank_* leaves so
        # the handler never hits real ranking code.
        cfg = {
            "equity_categories": ["Equity Scheme - Large Cap Fund"],
            "debt_categories": [],
            "include_gold": False,
            "top_n_to_persist": 5,
        }
        original = emit_evidence(
            audit_log_path=self.audit_path,
            evidence_kind="watchlist_run",
            audit_event={
                "event_type": "watchlist_run",
                "snapshot_date": "2026-05-14",
                "equity_count": 1, "debt_count": 0, "gold_included": False,
                "ok_count": 1, "error_count": 0, "errored_categories": [],
                "registry_problem": None, "schema_version": "v1",
            },
            payload={
                "snapshot_date": "2026-05-14",
                "config": cfg,
                "summary": {"Equity Scheme - Large Cap Fund": "ok"},
                "registry_path": str(self.registry_path),
                "snapshots_dir": "/tmp/ignored",
            },
        )
        original_run_id = original["event"]["run_id"]

        snapshots_dir = self.data_dir / "snapshots"
        sidecar_path = self.data_dir / "watchlist" / "last_run.json"
        self.assertFalse(snapshots_dir.exists())
        self.assertFalse(sidecar_path.exists())

        # Handler will call rank_category + _validate_registry; mock both.
        from backend.jobs import watchlist as wl
        with patch("backend.investment_analytics.ranking.rank_category",
                   side_effect=lambda c, p: _ranking(c, [101, 102, 103])), \
             patch.object(wl, "_validate_registry", return_value=None):
            rp.replay_run(
                self.audit_path, original_run_id,
                verify_chain=False, emit_audit=False,
                registry_path=self.registry_path,
            )

        # Side-effect guarantees:
        self.assertFalse(snapshots_dir.exists(),
                         "watchlist replay must not write snapshot files")
        self.assertFalse(sidecar_path.exists(),
                         "watchlist replay must not write the sidecar")
        # And no second watchlist_run audit row should appear.
        events = self._read_events()
        watchlist_rows = [e for e in events if e.get("event_type") == "watchlist_run"]
        self.assertEqual(len(watchlist_rows), 1,
                         "watchlist replay must not append a new watchlist_run row")

    def test_chain_remains_valid_across_mixed_sequence(self) -> None:
        # watchlist_run → replay_result (success path)
        original = self._emit_ranking_snapshot()
        run_id = original["event"]["run_id"]

        def fake_rank(category, _path):
            return _ranking(category, [101, 102, 103])

        with patch("backend.investment_analytics.ranking.rank_category",
                   side_effect=fake_rank):
            rp.replay_run(
                self.audit_path, run_id,
                verify_chain=False, emit_audit=True,
                registry_path=self.registry_path,
            )

        # And a second replay row.
        with patch("backend.investment_analytics.ranking.rank_category",
                   side_effect=fake_rank):
            rp.replay_run(
                self.audit_path, run_id,
                verify_chain=False, emit_audit=True,
                registry_path=self.registry_path,
            )

        self.assertTrue(verify_audit_chain(self.audit_path),
                        "chain must remain valid after replay appends")
        # Sequence sanity:
        types = [e["event_type"] for e in self._read_events()]
        self.assertEqual(types, ["rank_category", "replay_run", "replay_run"])


# ── CLI ──────────────────────────────────────────────────────────────


class CliTests(ReplayTestBase):

    def test_cli_happy_path_prints_classification_and_emits_row(self) -> None:
        record = self._emit_ranking_snapshot()
        run_id = record["event"]["run_id"]
        before = len(self._read_events())

        def fake_rank(category, _path):
            return _ranking(category, [101, 102, 103])

        with patch("backend.investment_analytics.ranking.rank_category",
                   side_effect=fake_rank):
            rc = rp._main([
                run_id,
                "--audit-path", str(self.audit_path),
                "--registry-path", str(self.registry_path),
                "--no-verify-chain",
            ])
        self.assertEqual(rc, 0)
        after = len(self._read_events())
        self.assertEqual(after, before + 1,
                         "CLI default path must append one replay_result row")

    def test_cli_dry_run_does_not_append_row(self) -> None:
        record = self._emit_ranking_snapshot()
        run_id = record["event"]["run_id"]
        before = len(self._read_events())

        def fake_rank(category, _path):
            return _ranking(category, [101, 102, 103])

        with patch("backend.investment_analytics.ranking.rank_category",
                   side_effect=fake_rank):
            rc = rp._main([
                run_id,
                "--audit-path", str(self.audit_path),
                "--registry-path", str(self.registry_path),
                "--no-verify-chain",
                "--dry-run",
            ])
        self.assertEqual(rc, 0)
        self.assertEqual(len(self._read_events()), before,
                         "--dry-run must not append a row")

    def test_cli_unreproducible_exits_nonzero(self) -> None:
        rc = rp._main([
            "00000000-0000-4000-8000-000000000000",
            "--audit-path", str(self.audit_path),
            "--registry-path", str(self.registry_path),
            "--no-verify-chain",
            "--dry-run",
        ])
        self.assertEqual(rc, 2, "unreproducible should map to exit code 2")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
