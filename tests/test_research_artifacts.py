"""Tests for backend.research_artifacts (Step 13).

Coverage map:

  Governance / closed enums:
    - EVIDENCE_KINDS contains threshold_recommendation
    - ADOPTION_STATUSES is closed (5 values)
    - Step 13 only permits emitting adoption_status="proposed"
    - forward statuses (under_review/adopted/rejected/superseded)
      are refused by emit_threshold_recommendation

  Emit happy path:
    - projecting from a calibration_report with non-null
      recommendation succeeds
    - emitted row carries derived_from_calibration_report_run_id
    - parent_run_id chains to either the supersedes target or the
      cited calibration_report
    - scope is projected from calibration_report.calibration_scope

  Refusal-projection guard (LOAD-BEARING):
    - emit REFUSES when cited calibration_report has
      recommendation=null (typed refusal)
    - emit REFUSES when cited run_id is not a calibration_report
    - emit REFUSES when cited calibration targets an unregistered
      target
    - replay handler REFUSES same conditions at replay time
      (mirrors emit-side guard; maps to invalid_replay)

  Supersedes chain:
    - re-emit same target without supersedes → refused
    - supersedes_run_id citing nonexistent row → refused
    - supersedes_run_id citing already-superseded row (branching)
      → refused
    - valid linear chain → both rows present, include_superseded
      filter respects the chain

  Replay:
    - handler registered in REPLAY_HANDLERS
    - replay of valid row reproduces semantically
    - replay where cited calibration has become refusal →
      invalid_replay (handler raises)

  Production isolation:
    - HIGH_CORRELATION_THRESHOLD byte-unmutated by emit
    - METHODOLOGY_VERSIONS byte-unmutated
    - METHODOLOGY_SCHEMA_VERSION still "v3" (no bump)

  Chain integrity:
    - verify_audit_chain passes across calibration → recommend →
      supersede sequence
"""
from __future__ import annotations

import json
import math
import random
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import patch

from backend.calibration import (
    CALIBRATION_TARGETS,
    run_calibration,
)
from backend.calibration import runner as cal_runner
from backend.calibration import targets as cal_targets
from backend.evidence import replay as rp
from backend.evidence.store import emit_evidence
from backend.investment_analytics import methodology as meth_mod
from backend.investment_analytics.audit import verify_audit_chain
from backend.investment_analytics.evidence_envelope import EVIDENCE_KINDS
from backend.research_artifacts import (
    ADOPTION_STATUSES,
    STEP_13_PERMITTED_EMIT_STATUSES,
    emit_threshold_recommendation,
    find_threshold_recommendations,
)
from backend.research_artifacts import runner as ra_runner


# ── Governance / closed enums ────────────────────────────────────────


class GovernanceTests(unittest.TestCase):

    def test_threshold_recommendation_in_evidence_kinds(self) -> None:
        self.assertIn("threshold_recommendation", EVIDENCE_KINDS)

    def test_adoption_statuses_closed_enum_five_values(self) -> None:
        self.assertEqual(ADOPTION_STATUSES, {
            "proposed", "under_review", "adopted",
            "rejected", "superseded",
        })

    def test_step_13_permits_only_proposed_emit_status(self) -> None:
        self.assertEqual(STEP_13_PERMITTED_EMIT_STATUSES, {"proposed"})

    def test_step_13_added_no_methodology_component(self) -> None:
        # Step 13 itself does NOT add a methodology component or bump
        # METHODOLOGY_SCHEMA_VERSION — threshold_recommendation is a
        # typed projection, not new inference. (Step 14 later added
        # reliability_weighting and bumped schema to v4; that is its
        # extension, not Step 13's.) This test pins the Step-13
        # invariant by checking the relevant component names are
        # absent from the methodology registry.
        for unwanted in (
            "threshold_recommender",
            "recommendation_engine",
            "research_artifact",
        ):
            self.assertNotIn(
                unwanted, meth_mod.METHODOLOGY_VERSIONS,
                f"Step 13 must not have added {unwanted!r} to "
                f"METHODOLOGY_VERSIONS",
            )

    def test_no_new_methodology_component_added(self) -> None:
        # Step 13 must NOT add a methodology component. Adding one
        # would bump METHODOLOGY_SCHEMA_VERSION and require this
        # test to update. Test pins the explicit non-addition.
        unwanted_keys = {
            "threshold_recommender",
            "recommendation_engine",
            "research_artifact",
        }
        present = unwanted_keys & set(meth_mod.METHODOLOGY_VERSIONS.keys())
        self.assertEqual(present, set(),
                         f"unexpected methodology components: {present}")


# ── Harness ──────────────────────────────────────────────────────────


def _write_synthetic_fund_cache(
    cache_dir: Path, scheme_code: int, scheme_category: str,
    start: date, n_days: int, target_ann_vol_pct: float, seed: int = 0,
) -> None:
    rng = random.Random(seed)
    daily_vol = (target_ann_vol_pct / 100.0) / math.sqrt(252)
    nav = 100.0
    points: List[Tuple[date, float]] = []
    cursor = start
    while len(points) < n_days:
        if cursor.weekday() < 5:
            nav = nav * (1.0 + rng.gauss(0.0, daily_vol))
            points.append((cursor, nav))
        cursor = cursor + timedelta(days=1)
    points.reverse()
    data = [{"date": d.strftime("%d-%m-%Y"), "nav": f"{v:.4f}"}
            for d, v in points]
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{scheme_code}.json").write_text(json.dumps({
        "fetched_at": "2026-05-16T00:00:00+00:00",
        "scheme_code": scheme_code,
        "response": {
            "meta": {"scheme_code": scheme_code,
                     "scheme_name": f"Test Fund {scheme_code}",
                     "scheme_category": scheme_category},
            "data": data,
        },
    }), encoding="utf-8")


class _RecommendHarness(unittest.TestCase):
    """Isolated audit + cache + registry so tests don't touch live
    state. Provides helpers to seed a deep-history equity universe,
    emit synthetic regime_summary rows, and produce calibration
    reports (real recommendations OR typed refusals)."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp.name)
        self.audit_path = self.data_dir / "audit" / "audit.jsonl"
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.data_dir / "registry" / "schemes.json"
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text("[]", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _seed_equity_universe(self, n_funds: int = 12) -> None:
        for i in range(n_funds):
            _write_synthetic_fund_cache(
                self.cache_dir,
                scheme_code=900_000 + i,
                scheme_category="Equity Scheme - Large Cap Fund",
                start=date(2018, 1, 1),
                n_days=3000,
                target_ann_vol_pct=15.0,
                seed=i + 1,
            )

    def _emit_regime_summary(self, *,
                             window_start: str, window_end: str,
                             confidence: float = 1.0,
                             coverage_quality: str = "high") -> str:
        import hashlib
        applied_bands = {
            "low_threshold_pct": 12.0,
            "high_threshold_pct": 20.0,
            "crisis_threshold_pct": 35.0,
        }
        window_components = {
            "window_start_date":  window_start,
            "window_end_date":    window_end,
            "signal_scheme_code": 999_001,
            "applied_bands":      applied_bands,
        }
        window_hash = hashlib.sha256(json.dumps(
            window_components, sort_keys=True,
            separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")).hexdigest()
        regime_signature = {
            "signal_kind": "nifty50_realized_vol",
            "taxonomy_version": "v1",
            "classifier_version": "v1",
            "window_hash": window_hash,
        }
        payload = {
            "schema_version": "v1",
            "taxonomy_version": "v1",
            "regime_classifier_version": "v1",
            "classification_semantics": "descriptive_not_causal",
            "regime_class": "normal_vol",
            "window_start_date": window_start,
            "window_end_date": window_end,
            "window_coverage_days": 252,
            "classification_confidence": confidence,
            "classification_stability": 1.0,
            "boundary_separation": confidence,
            "classification_basis": {
                "signal_kind": "nifty50_realized_vol",
                "signal_scheme_code": 999_001,
                "annualized_vol_pct": 15.0,
                "applied_bands": applied_bands,
                "min_coverage_days": 60,
                "boundary_confidence_margin_pct": 10.0,
            },
            "signal_quality": {
                "history_depth_days": 2500,
                "missing_data_pct": 3.0,
                "coverage_quality": coverage_quality,
            },
            "regime_signature": regime_signature,
            "supersedes_run_id": None,
        }
        audit_event = {
            "event_type": "regime_summary",
            "regime_class": "normal_vol",
            "window_start_date": window_start,
            "window_end_date": window_end,
            "window_coverage_days": 252,
            "classification_confidence": confidence,
            "taxonomy_version": "v1",
            "regime_classifier_version": "v1",
            "classification_semantics": "descriptive_not_causal",
            "coverage_quality": coverage_quality,
            "supersedes_run_id": None,
            "schema_version": "v1",
        }
        record = emit_evidence(
            audit_log_path=self.audit_path,
            evidence_kind="regime_summary",
            audit_event=audit_event,
            payload=payload,
        )
        return record["event"]["run_id"]

    def _emit_calibration_recommendation(self) -> dict:
        """Run the real calibration engine and emit a recommendation
        row (substrate seeded above). Returns the audit record."""
        self._seed_equity_universe(n_funds=12)
        self._emit_regime_summary(
            window_start="2024-01-02", window_end="2024-09-30",
            confidence=1.0, coverage_quality="high",
        )
        return run_calibration(
            target="portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            audit_path=self.audit_path,
            cache_dir=self.cache_dir,
        )

    def _emit_calibration_refusal(self) -> dict:
        """Empty chain → calibration emits a typed refusal."""
        return run_calibration(
            target="portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            audit_path=self.audit_path,
            cache_dir=self.cache_dir,
        )


# ── Emit happy path ──────────────────────────────────────────────────


class EmitHappyPathTests(_RecommendHarness):

    def test_emit_succeeds_from_calibration_recommendation(self) -> None:
        cal = self._emit_calibration_recommendation()
        self.assertIsNotNone(cal["event"]["recommendation"])
        cal_run_id = cal["event"]["run_id"]
        rec = emit_threshold_recommendation(
            cal_run_id,
            audit_path=self.audit_path,
        )
        self.assertEqual(rec["event"]["evidence_kind"],
                         "threshold_recommendation")
        self.assertEqual(rec["event"]["event_type"],
                         "threshold_recommendation")
        self.assertEqual(rec["event"]["adoption_status"], "proposed")
        self.assertEqual(
            rec["event"]["derived_from_calibration_report_run_id"],
            cal_run_id,
        )

    def test_parent_run_id_chains_to_calibration_when_no_supersedes(self) -> None:
        cal = self._emit_calibration_recommendation()
        cal_run_id = cal["event"]["run_id"]
        rec = emit_threshold_recommendation(
            cal_run_id,
            audit_path=self.audit_path,
        )
        # Initial recommendation: parent is the cited calibration_report.
        self.assertEqual(rec["event"]["parent_run_id"], cal_run_id)

    def test_evidence_payload_carries_full_provenance(self) -> None:
        cal = self._emit_calibration_recommendation()
        rec = emit_threshold_recommendation(
            cal["event"]["run_id"],
            audit_path=self.audit_path,
        )
        ev_path = self.data_dir / rec["event"]["evidence_ref"]["path"]
        payload = json.loads(ev_path.read_text(encoding="utf-8"))["payload"]
        self.assertEqual(payload["target_canonical_id"],
                         "portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD")
        self.assertEqual(payload["adoption_status"], "proposed")
        self.assertEqual(payload["methodology_kind"], "data_driven_variant")
        self.assertIn("recommendation_scope", payload)
        self.assertIn("valid_within_regimes", payload["recommendation_scope"])
        self.assertIsNotNone(payload["recommended_value"])


# ── Refusal-projection guard (LOAD-BEARING) ──────────────────────────


class RefusalProjectionGuardTests(_RecommendHarness):

    def test_emit_refuses_from_calibration_refusal(self) -> None:
        # Empty chain → calibration emits insufficient_substrate refusal.
        cal = self._emit_calibration_refusal()
        self.assertEqual(cal["event"]["refusal_reason"],
                         "insufficient_substrate")
        self.assertIsNone(cal["event"]["recommendation"])
        # Attempting to wrap a refusal MUST fail.
        with self.assertRaises(ValueError) as ctx:
            emit_threshold_recommendation(
                cal["event"]["run_id"],
                audit_path=self.audit_path,
            )
        self.assertIn("calibration refusal", str(ctx.exception))

    def test_emit_refuses_when_cited_run_is_not_calibration_report(self) -> None:
        # Emit a regime_summary, then try to wrap it as if it were
        # a calibration. Must refuse with a typed error.
        rs_run_id = self._emit_regime_summary(
            window_start="2024-01-02", window_end="2024-09-30",
        )
        with self.assertRaises(ValueError) as ctx:
            emit_threshold_recommendation(
                rs_run_id, audit_path=self.audit_path,
            )
        self.assertIn("expected 'calibration_report'", str(ctx.exception))

    def test_emit_refuses_unknown_run_id(self) -> None:
        with self.assertRaises(KeyError):
            emit_threshold_recommendation(
                "00000000-0000-4000-8000-000000000000",
                audit_path=self.audit_path,
            )

    def test_emit_refuses_forward_adoption_status(self) -> None:
        # Step 13 emits ONLY adoption_status="proposed". Forward
        # statuses are Step 15's domain and must be refused here.
        cal = self._emit_calibration_recommendation()
        for forward_status in ("under_review", "adopted",
                               "rejected", "superseded"):
            with self.assertRaises(ValueError) as ctx:
                emit_threshold_recommendation(
                    cal["event"]["run_id"],
                    audit_path=self.audit_path,
                    adoption_status=forward_status,
                )
            self.assertIn("Step 13", str(ctx.exception))

    def test_emit_refuses_unknown_adoption_status(self) -> None:
        cal = self._emit_calibration_recommendation()
        with self.assertRaises(ValueError):
            emit_threshold_recommendation(
                cal["event"]["run_id"],
                audit_path=self.audit_path,
                adoption_status="bogus",
            )


# ── Supersedes chain ─────────────────────────────────────────────────


class SupersedesChainTests(_RecommendHarness):

    def test_re_emit_same_target_without_supersedes_refused(self) -> None:
        cal = self._emit_calibration_recommendation()
        cal_run_id = cal["event"]["run_id"]
        emit_threshold_recommendation(cal_run_id, audit_path=self.audit_path)
        with self.assertRaises(ValueError) as ctx:
            emit_threshold_recommendation(
                cal_run_id, audit_path=self.audit_path,
            )
        self.assertIn("supersedes_run_id", str(ctx.exception))

    def test_supersedes_unknown_run_id_refused(self) -> None:
        cal = self._emit_calibration_recommendation()
        emit_threshold_recommendation(cal["event"]["run_id"],
                                      audit_path=self.audit_path)
        with self.assertRaises(ValueError):
            emit_threshold_recommendation(
                cal["event"]["run_id"],
                audit_path=self.audit_path,
                supersedes_run_id="bogus-id",
            )

    def test_supersedes_with_no_prior_row_refused(self) -> None:
        cal = self._emit_calibration_recommendation()
        with self.assertRaises(ValueError):
            emit_threshold_recommendation(
                cal["event"]["run_id"],
                audit_path=self.audit_path,
                supersedes_run_id="any-id",
            )

    def test_branching_refused(self) -> None:
        # A → B (B supersedes A). Trying to emit C that ALSO
        # supersedes A creates a branch — refused.
        cal = self._emit_calibration_recommendation()
        first = emit_threshold_recommendation(
            cal["event"]["run_id"], audit_path=self.audit_path,
        )
        emit_threshold_recommendation(
            cal["event"]["run_id"],
            audit_path=self.audit_path,
            supersedes_run_id=first["event"]["run_id"],
        )  # legitimate next step in the chain
        with self.assertRaises(ValueError) as ctx:
            emit_threshold_recommendation(
                cal["event"]["run_id"],
                audit_path=self.audit_path,
                supersedes_run_id=first["event"]["run_id"],
            )
        self.assertIn("already superseded", str(ctx.exception))

    def test_linear_chain_preserved_under_include_superseded_filter(self) -> None:
        cal = self._emit_calibration_recommendation()
        first = emit_threshold_recommendation(
            cal["event"]["run_id"], audit_path=self.audit_path,
        )
        second = emit_threshold_recommendation(
            cal["event"]["run_id"],
            audit_path=self.audit_path,
            supersedes_run_id=first["event"]["run_id"],
        )
        third = emit_threshold_recommendation(
            cal["event"]["run_id"],
            audit_path=self.audit_path,
            supersedes_run_id=second["event"]["run_id"],
        )
        # Full history
        rows = find_threshold_recommendations(
            audit_path=self.audit_path, include_superseded=True,
        )
        self.assertEqual(len(rows), 3)
        # Only the current claim
        current = find_threshold_recommendations(
            audit_path=self.audit_path, include_superseded=False,
        )
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0]["run_id"], third["event"]["run_id"])


# ── Replay ───────────────────────────────────────────────────────────


class ReplayTests(_RecommendHarness):

    def test_handler_registered(self) -> None:
        self.assertIn("threshold_recommendation", rp.REPLAY_HANDLERS)

    def test_replay_reproduces_valid_recommendation(self) -> None:
        # Verification 1a: a valid threshold_recommendation replay
        # MUST land on exact_match or semantically_equivalent — not
        # expected_divergence, not invalid_replay. The cited
        # calibration has a non-null recommendation, so the
        # refusal-projection guard does not fire; methodology is
        # unchanged within the isolated test chain, so no driver
        # fires; the projection is deterministic; both sides match.
        cal = self._emit_calibration_recommendation()
        rec = emit_threshold_recommendation(
            cal["event"]["run_id"], audit_path=self.audit_path,
        )
        original_audit = ra_runner.DEFAULT_AUDIT_PATH
        original_cal_audit = cal_runner.DEFAULT_AUDIT_PATH
        original_cache = cal_targets.DEFAULT_CACHE_DIR
        ra_runner.DEFAULT_AUDIT_PATH = self.audit_path
        cal_runner.DEFAULT_AUDIT_PATH = self.audit_path
        cal_targets.DEFAULT_CACHE_DIR = self.cache_dir
        try:
            result = rp.replay_run(
                self.audit_path, rec["event"]["run_id"],
                verify_chain=False, emit_audit=False,
                registry_path=self.registry_path,
            )
        finally:
            ra_runner.DEFAULT_AUDIT_PATH = original_audit
            cal_runner.DEFAULT_AUDIT_PATH = original_cal_audit
            cal_targets.DEFAULT_CACHE_DIR = original_cache
        self.assertIn(result["state"],
                      ("exact_match", "semantically_equivalent"),
                      f"state={result['state']} "
                      f"reason={result.get('reason')} — "
                      f"valid recommendation replay MUST NOT land on "
                      f"expected_divergence or invalid_replay")
        self.assertNotEqual(result["state"], "expected_divergence")
        self.assertNotEqual(result["state"], "invalid_replay")

    def test_replay_refuses_when_cited_calibration_is_refusal(self) -> None:
        # Craft a scenario where a recommendation row exists but the
        # cited calibration is (or has become) a refusal. We bypass
        # the emit-side guard by patching it; this exercises the
        # REPLAY-side refusal-projection guard specifically.

        # Step 1: emit a real calibration recommendation + recommendation
        cal = self._emit_calibration_recommendation()
        rec = emit_threshold_recommendation(
            cal["event"]["run_id"], audit_path=self.audit_path,
        )
        rec_run_id = rec["event"]["run_id"]

        # Step 2: append a SYNTHETIC calibration refusal with a
        # specific run_id, then rewrite the recommendation row's
        # derived_from to cite that refusal. Since the audit chain
        # is hash-linked, we can't actually mutate the recommendation
        # row. Instead we test the handler function directly with a
        # mocked recorded payload.

        # Direct handler invocation with a synthetic recorded_envelope
        # citing a (synthesized) calibration refusal run_id.
        from backend.evidence.store import emit_evidence as _emit
        refusal_record = run_calibration(
            target="portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            audit_path=self.audit_path,
            cache_dir=Path(self.tmp.name) / "empty_cache",
        )
        # Verify the second calibration was indeed a refusal.
        self.assertIsNone(refusal_record["event"]["recommendation"])

        # Construct a fake recorded_envelope that cites the refusal.
        fake_recorded_envelope = {
            "payload": {
                "schema_version": "v1",
                "target_canonical_id": (
                    "portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD"),
                "recommended_value": 0.85,
                "adoption_status": "proposed",
                "derived_from_calibration_report_run_id":
                    refusal_record["event"]["run_id"],
                "supersedes_run_id": None,
            }
        }
        original_audit = ra_runner.DEFAULT_AUDIT_PATH
        ra_runner.DEFAULT_AUDIT_PATH = self.audit_path
        try:
            with self.assertRaises(RuntimeError) as ctx:
                rp._replay_threshold_recommendation(
                    audit_event={},
                    recorded_envelope=fake_recorded_envelope,
                    registry_path=self.registry_path,
                )
            self.assertIn("typed refusal", str(ctx.exception))
        finally:
            ra_runner.DEFAULT_AUDIT_PATH = original_audit


# ── Production isolation ─────────────────────────────────────────────


class ProductionIsolationTests(_RecommendHarness):

    def test_high_correlation_threshold_unchanged(self) -> None:
        from backend.investment_analytics.portfolio_health import correlation
        before = correlation.HIGH_CORRELATION_THRESHOLD
        cal = self._emit_calibration_recommendation()
        emit_threshold_recommendation(
            cal["event"]["run_id"], audit_path=self.audit_path,
        )
        self.assertEqual(correlation.HIGH_CORRELATION_THRESHOLD, before,
                         "production threshold must NEVER be mutated by "
                         "threshold_recommendation emit")

    def test_methodology_versions_unchanged(self) -> None:
        before = dict(meth_mod.METHODOLOGY_VERSIONS)
        cal = self._emit_calibration_recommendation()
        emit_threshold_recommendation(
            cal["event"]["run_id"], audit_path=self.audit_path,
        )
        self.assertEqual(before, dict(meth_mod.METHODOLOGY_VERSIONS))


# ── Chain integrity ──────────────────────────────────────────────────


class ChainIntegrityTests(_RecommendHarness):

    def test_chain_valid_across_recommend_supersede_sequence(self) -> None:
        cal = self._emit_calibration_recommendation()
        cal_run_id = cal["event"]["run_id"]
        first = emit_threshold_recommendation(
            cal_run_id, audit_path=self.audit_path,
        )
        second = emit_threshold_recommendation(
            cal_run_id, audit_path=self.audit_path,
            supersedes_run_id=first["event"]["run_id"],
        )
        emit_threshold_recommendation(
            cal_run_id, audit_path=self.audit_path,
            supersedes_run_id=second["event"]["run_id"],
        )
        self.assertTrue(verify_audit_chain(self.audit_path))


# ── Find / query ─────────────────────────────────────────────────────


class FindThresholdRecommendationsTests(_RecommendHarness):

    def test_empty_chain_returns_empty(self) -> None:
        self.assertEqual(
            find_threshold_recommendations(audit_path=self.audit_path),
            [],
        )

    def test_filter_by_adoption_status(self) -> None:
        cal = self._emit_calibration_recommendation()
        emit_threshold_recommendation(
            cal["event"]["run_id"], audit_path=self.audit_path,
        )
        rows = find_threshold_recommendations(
            audit_path=self.audit_path, adoption_status="proposed",
        )
        self.assertEqual(len(rows), 1)
        empty = find_threshold_recommendations(
            audit_path=self.audit_path, adoption_status="adopted",
        )
        self.assertEqual(empty, [])


# ── Pre-commit explicit verifications (Step 13 sign-off) ────────────


class ExplicitVerificationTests(_RecommendHarness):
    """The five properties the reviewer asked to verify explicitly
    before committing Step 13. Each test maps to one numbered
    requirement and is named to make the mapping obvious in the
    test runner output."""

    # ── 1) Replay classification — full-flow state assertion ────────

    def test_v1b_replay_of_refusal_cited_recommendation_state_is_invalid_replay(
        self,
    ) -> None:
        """Verification 1b: when the cited calibration_report has
        become (or always was) a refusal, replaying the
        threshold_recommendation that depends on it must classify
        as invalid_replay — NOT expected_divergence. The handler
        raises; Step 7's contract maps raises to invalid_replay."""
        # Build a synthetic threshold_recommendation row that cites
        # a refusal. We bypass the emit-side guard (which would
        # never let this be created in practice) so the test
        # exercises the REPLAY-side guard end-to-end through
        # replay_run, not just the handler in isolation.

        refusal_record = run_calibration(
            target="portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            audit_path=self.audit_path,
            cache_dir=Path(self.tmp.name) / "empty_cache",
        )
        self.assertIsNone(refusal_record["event"]["recommendation"])
        refusal_run_id = refusal_record["event"]["run_id"]

        # Direct emit_evidence — bypasses runner validation — to plant
        # a recommendation row that cites a refusal. Mirrors the kind
        # of chain corruption / future-state-drift the replay-side
        # guard is designed to catch.
        fake_payload = {
            "schema_version": "v1",
            "target_canonical_id":
                "portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            "recommended_value": 0.85,
            "recommendation_scope": {
                "valid_within_regimes": [],
                "assumed_stationarity": "test",
                "known_limitations": [],
            },
            "adoption_status": "proposed",
            "derived_from_calibration_report_run_id": refusal_run_id,
            "supersedes_run_id": None,
            "methodology_kind": "data_driven_variant",
            "non_semantic_metadata": {},
        }
        fake_event = {
            "event_type": "threshold_recommendation",
            "target_canonical_id": fake_payload["target_canonical_id"],
            "recommended_value": 0.85,
            "adoption_status": "proposed",
            "derived_from_calibration_report_run_id": refusal_run_id,
            "supersedes_run_id": None,
            "schema_version": "v1",
        }
        rec = emit_evidence(
            audit_log_path=self.audit_path,
            evidence_kind="threshold_recommendation",
            audit_event=fake_event,
            payload=fake_payload,
            parent_run_id=refusal_run_id,
        )
        rec_run_id = rec["event"]["run_id"]

        original_audit = ra_runner.DEFAULT_AUDIT_PATH
        ra_runner.DEFAULT_AUDIT_PATH = self.audit_path
        try:
            result = rp.replay_run(
                self.audit_path, rec_run_id,
                verify_chain=False, emit_audit=False,
                registry_path=self.registry_path,
            )
        finally:
            ra_runner.DEFAULT_AUDIT_PATH = original_audit

        self.assertEqual(
            result["state"], "invalid_replay",
            f"got state={result['state']}, reason={result.get('reason')}",
        )
        # Reason must reference the refusal cause, not generic drift.
        self.assertIn("typed refusal", result["reason"].lower(),
                      f"reason={result['reason']}")

    # ── 2) Supersedes lineage integrity ─────────────────────────────

    def test_v2a_supersedes_cannot_cross_targets(self) -> None:
        """Verification 2a: supersedes_run_id citing a row for a
        different target is refused. Today the registry has one
        target; we synthesize a foreign-target row directly via
        emit_evidence to exercise the runner's per-target filter."""
        cal = self._emit_calibration_recommendation()
        target_x = cal["event"]["target"]

        # Plant a foreign-target threshold_recommendation row.
        foreign_event = {
            "event_type": "threshold_recommendation",
            "target_canonical_id": "portfolio_health.foreign.OTHER_TARGET",
            "recommended_value": 0.50,
            "adoption_status": "proposed",
            "derived_from_calibration_report_run_id": "synthetic",
            "supersedes_run_id": None,
            "schema_version": "v1",
        }
        foreign_payload = {
            "schema_version": "v1",
            "target_canonical_id": "portfolio_health.foreign.OTHER_TARGET",
            "recommended_value": 0.50,
            "recommendation_scope": {"valid_within_regimes": [],
                                     "assumed_stationarity": "x",
                                     "known_limitations": []},
            "adoption_status": "proposed",
            "derived_from_calibration_report_run_id": "synthetic",
            "supersedes_run_id": None,
            "methodology_kind": "data_driven_variant",
            "non_semantic_metadata": {},
        }
        foreign = emit_evidence(
            audit_log_path=self.audit_path,
            evidence_kind="threshold_recommendation",
            audit_event=foreign_event,
            payload=foreign_payload,
        )

        # Try to supersede a row for target_x using the foreign-target row.
        with self.assertRaises(ValueError) as ctx:
            emit_threshold_recommendation(
                cal["event"]["run_id"],
                audit_path=self.audit_path,
                supersedes_run_id=foreign["event"]["run_id"],
            )
        # Refusal message references the absence in target_x's history.
        self.assertIn(target_x, str(ctx.exception))

    def test_v2b_include_superseded_false_yields_one_active_per_target(
        self,
    ) -> None:
        """Verification 2b: after a linear supersedes chain of N
        recommendations for ONE target, include_superseded=False
        returns exactly one row — the tail of the chain."""
        cal = self._emit_calibration_recommendation()
        cal_id = cal["event"]["run_id"]

        prev_id = None
        chain: List[str] = []
        for _ in range(4):
            row = emit_threshold_recommendation(
                cal_id, audit_path=self.audit_path,
                supersedes_run_id=prev_id,
            )
            chain.append(row["event"]["run_id"])
            prev_id = row["event"]["run_id"]

        full = find_threshold_recommendations(
            audit_path=self.audit_path, include_superseded=True,
        )
        active = find_threshold_recommendations(
            audit_path=self.audit_path, include_superseded=False,
        )
        self.assertEqual(len(full), 4)
        self.assertEqual(len(active), 1,
                         "exactly one active recommendation per target")
        self.assertEqual(active[0]["run_id"], chain[-1],
                         "active row must be the tail of the supersedes chain")

    # ── 3) No hidden production mutation ────────────────────────────

    def test_v3_no_hidden_mutation_of_production_constants(self) -> None:
        """Verification 3: byte equality before vs after a full
        recommend-emit cycle on:
          - HIGH_CORRELATION_THRESHOLD (production threshold)
          - METHODOLOGY_VERSIONS (methodology registry)
          - CALIBRATION_TARGETS (calibration target registry)
        """
        from backend.investment_analytics.portfolio_health import correlation
        from backend.calibration.config import CALIBRATION_TARGETS as CT

        before_threshold = correlation.HIGH_CORRELATION_THRESHOLD
        before_methodology = dict(meth_mod.METHODOLOGY_VERSIONS)
        before_methodology_schema = meth_mod.METHODOLOGY_SCHEMA_VERSION
        before_calibration_targets = frozenset(CT)

        cal = self._emit_calibration_recommendation()
        emit_threshold_recommendation(
            cal["event"]["run_id"], audit_path=self.audit_path,
        )

        self.assertEqual(correlation.HIGH_CORRELATION_THRESHOLD,
                         before_threshold)
        self.assertEqual(dict(meth_mod.METHODOLOGY_VERSIONS),
                         before_methodology)
        self.assertEqual(meth_mod.METHODOLOGY_SCHEMA_VERSION,
                         before_methodology_schema)
        self.assertEqual(frozenset(CT), before_calibration_targets)

    # ── 4) Chain semantics ──────────────────────────────────────────

    def test_v4a_parent_run_id_first_recommendation_anchors_to_calibration(
        self,
    ) -> None:
        cal = self._emit_calibration_recommendation()
        rec = emit_threshold_recommendation(
            cal["event"]["run_id"], audit_path=self.audit_path,
        )
        self.assertEqual(rec["event"]["parent_run_id"],
                         cal["event"]["run_id"])

    def test_v4b_parent_run_id_subsequent_anchors_to_prior_recommendation(
        self,
    ) -> None:
        cal = self._emit_calibration_recommendation()
        first = emit_threshold_recommendation(
            cal["event"]["run_id"], audit_path=self.audit_path,
        )
        second = emit_threshold_recommendation(
            cal["event"]["run_id"], audit_path=self.audit_path,
            supersedes_run_id=first["event"]["run_id"],
        )
        # Subsequent recommendation's parent_run_id MUST anchor to
        # the prior recommendation (not back to the calibration).
        # This preserves the supersedes-chain lineage as the
        # authoritative genealogy.
        self.assertEqual(second["event"]["parent_run_id"],
                         first["event"]["run_id"])
        self.assertNotEqual(second["event"]["parent_run_id"],
                            cal["event"]["run_id"])

    def test_v4c_no_orphan_recommendation_rows(self) -> None:
        """Verification 4c: every threshold_recommendation row in
        the chain has a non-null parent_run_id — first recommendations
        anchor to their cited calibration_report, subsequent
        recommendations anchor to the prior recommendation. There
        are no orphan rows."""
        cal = self._emit_calibration_recommendation()
        cal_id = cal["event"]["run_id"]
        prev_id = None
        for _ in range(3):
            row = emit_threshold_recommendation(
                cal_id, audit_path=self.audit_path,
                supersedes_run_id=prev_id,
            )
            prev_id = row["event"]["run_id"]
        rows = find_threshold_recommendations(
            audit_path=self.audit_path, include_superseded=True,
        )
        self.assertEqual(len(rows), 3)
        for r in rows:
            self.assertIsNotNone(
                r.get("parent_run_id"),
                f"orphan recommendation row: run_id={r.get('run_id')}",
            )

    # ── 5) Refusal anti-laundering symmetry ─────────────────────────

    def test_v5_cli_returns_exit_code_2_for_refusal_projection(self) -> None:
        """Verification 5c: the CLI surface also honors the
        refusal-projection guard. Calling `recommend` with a cited
        calibration that is a refusal must exit with code 2 (typed
        refusal exit), not 0 (success) and not 4 (bad arg)."""
        from backend.research_artifacts.__main__ import _main as ra_main

        refusal = run_calibration(
            target="portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            audit_path=self.audit_path,
            cache_dir=Path(self.tmp.name) / "empty_cache",
        )
        self.assertIsNone(refusal["event"]["recommendation"])

        # Run via CLI. Capture stderr for cleanliness; we only check
        # the exit code here.
        import io
        from contextlib import redirect_stderr, redirect_stdout
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            exit_code = ra_main([
                "recommend",
                refusal["event"]["run_id"],
                "--audit-path", str(self.audit_path),
            ])
        self.assertEqual(
            exit_code, 2,
            f"expected exit code 2 (typed refusal), got {exit_code}. "
            f"stderr: {err_buf.getvalue()}",
        )
        # stderr should mention "refused" and the refusal context.
        self.assertIn("refused", err_buf.getvalue().lower())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
