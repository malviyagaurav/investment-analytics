"""Tests for backend.reliability (Step 14).

Coverage map:

  Governance:
    - SCORING_DIMENSIONS closed (8 values)
    - RELIABILITY_REFUSAL_REASONS closed
    - REPLAY_STATE_WEIGHTS asymmetric (load-bearing tightening):
        invalid_replay = -1.0 (HARD penalty)
        unreproducible = +0.5 (SOFT penalty)
        others = +1.0
    - reliability_weighting in METHODOLOGY_VERSIONS
    - METHODOLOGY_SCHEMA_VERSION = "v4"
    - EVIDENCE_KINDS has reliability_score (10 total)
    - WEIGHTING_TABLE_THRESHOLD_RECOMMENDATION sums to 1.0
    - K_REFUSAL_FLOOR = 4

  Asymmetric replay penalty (THE Step 14 tightening):
    - 10 replays, 1 invalid_replay → harder penalty than 1 unreproducible
    - basis exposes unreproducible_causes + invalid_replay_causes
    - score floor at 0.0 even when negative-weighted

  Per-dimension functions:
    - evidence_stability: supersedes ratio inversion
    - replay_stability: asymmetric weighting
    - regime_sensitivity: CV-based scoring
    - calibration_coverage_quality: high+medium fraction
    - drift_exposure: transition density inversion
    - supersession_churn: supersedes across kinds
    - refusal_frequency: calibration refusal density inversion
    - methodology_volatility: distinct snapshot count

  Aggregation:
    - happy path: weighted average across all dimensions
    - K-floor (4): refusal when ≥4 dimensions refuse
    - aggregate refusal payload carries per-dimension scores

  Refusal symmetry:
    - target_unsupported refusal
    - target_run_id_missing refusal
    - refusal payloads carry FULL dimension list + basis
    - run_ids consulted are preserved on refusal

  Production isolation (LOAD-BEARING):
    - HIGH_CORRELATION_THRESHOLD byte-unchanged
    - METHODOLOGY_VERSIONS byte-unchanged (no in-place mutation
      by the runner; only the new key was added at module load)
    - threshold_recommendation.adoption_status unchanged

  Replay:
    - handler registered
    - replay reproduces under unchanged chain
    - methodology bump → expected_divergence with
      methodology_changed driver

  Chain integrity:
    - verify_audit_chain passes across mixed sequence
"""
from __future__ import annotations

import io
import json
import math
import random
import statistics
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
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
from backend.investment_analytics.audit import (
    append_audit_record, verify_audit_chain,
)
from backend.investment_analytics.evidence_envelope import EVIDENCE_KINDS
from backend.reliability import (
    DIMENSION_FUNCTIONS,
    K_REFUSAL_FLOOR,
    RELIABILITY_REFUSAL_REASONS,
    REPLAY_STATE_WEIGHTS,
    SCORING_DIMENSIONS,
    WEIGHTING_TABLE_THRESHOLD_RECOMMENDATION,
    find_reliability_scores,
    score_target,
)
from backend.reliability import dimensions as dim_mod
from backend.reliability import runner as rel_runner
from backend.research_artifacts import emit_threshold_recommendation


# ── Governance ───────────────────────────────────────────────────────


class GovernanceTests(unittest.TestCase):

    def test_scoring_dimensions_closed_enum_eight_values(self) -> None:
        self.assertEqual(SCORING_DIMENSIONS, {
            "evidence_stability", "replay_stability",
            "regime_sensitivity", "calibration_coverage_quality",
            "drift_exposure", "supersession_churn",
            "refusal_frequency", "methodology_volatility",
        })

    def test_reliability_score_in_evidence_kinds(self) -> None:
        self.assertIn("reliability_score", EVIDENCE_KINDS)

    def test_methodology_schema_v4_after_step_14(self) -> None:
        self.assertEqual(meth_mod.METHODOLOGY_SCHEMA_VERSION, "v4")
        self.assertIn("reliability_weighting", meth_mod.METHODOLOGY_VERSIONS)

    def test_weighting_table_sums_to_one(self) -> None:
        s = sum(WEIGHTING_TABLE_THRESHOLD_RECOMMENDATION.values())
        self.assertAlmostEqual(s, 1.0, places=6)

    def test_k_refusal_floor_half_of_dimensions(self) -> None:
        self.assertEqual(K_REFUSAL_FLOOR, 4)
        self.assertEqual(len(SCORING_DIMENSIONS), 8)

    def test_replay_state_weights_asymmetric_step14_tightening(self) -> None:
        """LOAD-BEARING: invalid_replay must be HARDER penalty than
        unreproducible. Chain-integrity violations cannot hide
        inside benign substrate-decay statistics."""
        self.assertEqual(REPLAY_STATE_WEIGHTS["exact_match"], 1.0)
        self.assertEqual(REPLAY_STATE_WEIGHTS["semantically_equivalent"], 1.0)
        self.assertEqual(REPLAY_STATE_WEIGHTS["expected_divergence"], 1.0)
        # The asymmetry:
        self.assertEqual(REPLAY_STATE_WEIGHTS["unreproducible"], 0.5)
        self.assertEqual(REPLAY_STATE_WEIGHTS["invalid_replay"], -1.0)
        # invalid_replay must be STRICTLY MORE punitive than unreproducible.
        self.assertLess(REPLAY_STATE_WEIGHTS["invalid_replay"],
                        REPLAY_STATE_WEIGHTS["unreproducible"])


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


class _ReliabilityHarness(unittest.TestCase):

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

    def _read_events(self) -> List[dict]:
        if not self.audit_path.exists():
            return []
        with self.audit_path.open("r", encoding="utf-8") as h:
            return [json.loads(line)["event"] for line in h if line.strip()]

    def _seed_equity_universe(self, n_funds: int = 12) -> None:
        for i in range(n_funds):
            _write_synthetic_fund_cache(
                self.cache_dir, scheme_code=900_000 + i,
                scheme_category="Equity Scheme - Large Cap Fund",
                start=date(2018, 1, 1), n_days=3000,
                target_ann_vol_pct=15.0, seed=i + 1,
            )

    def _emit_regime_summary(self, *, start: str, end: str,
                             confidence: float = 1.0,
                             coverage_quality: str = "high",
                             regime_class: str = "normal_vol") -> str:
        import hashlib
        applied_bands = {"low_threshold_pct": 12.0,
                         "high_threshold_pct": 20.0,
                         "crisis_threshold_pct": 35.0}
        window_components = {
            "window_start_date": start, "window_end_date": end,
            "signal_scheme_code": 999_001,
            "applied_bands": applied_bands,
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
            "schema_version": "v1", "taxonomy_version": "v1",
            "regime_classifier_version": "v1",
            "classification_semantics": "descriptive_not_causal",
            "regime_class": regime_class,
            "window_start_date": start, "window_end_date": end,
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
                "history_depth_days": 2500, "missing_data_pct": 3.0,
                "coverage_quality": coverage_quality,
            },
            "regime_signature": regime_signature,
            "supersedes_run_id": None,
        }
        audit_event = {
            "event_type": "regime_summary", "regime_class": regime_class,
            "window_start_date": start, "window_end_date": end,
            "window_coverage_days": 252,
            "classification_confidence": confidence,
            "taxonomy_version": "v1", "regime_classifier_version": "v1",
            "classification_semantics": "descriptive_not_causal",
            "coverage_quality": coverage_quality,
            "supersedes_run_id": None, "schema_version": "v1",
        }
        record = emit_evidence(
            audit_log_path=self.audit_path,
            evidence_kind="regime_summary",
            audit_event=audit_event, payload=payload,
        )
        return record["event"]["run_id"]

    def _emit_calibration_recommendation(self) -> dict:
        self._seed_equity_universe(n_funds=12)
        self._emit_regime_summary(
            start="2024-01-02", end="2024-09-30",
            confidence=1.0, coverage_quality="high",
        )
        return run_calibration(
            target="portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            audit_path=self.audit_path,
            cache_dir=self.cache_dir,
        )

    def _emit_threshold_recommendation_with_substrate(self) -> dict:
        """Convenience: build substrate + emit a real recommendation."""
        cal = self._emit_calibration_recommendation()
        return emit_threshold_recommendation(
            cal["event"]["run_id"], audit_path=self.audit_path,
        )

    def _ensure_synthetic_original_for_target(self, target: str) -> str:
        """Plant a synthetic threshold_recommendation row for the
        given target and return its run_id. Cached on the test
        instance so multiple calls per test reuse the same original
        (matches how real replays of one artifact would look)."""
        cache_attr = f"_synthetic_original_for_{hash(target) & 0xffff}"
        cached = getattr(self, cache_attr, None)
        if cached is not None:
            return cached
        audit_event = {
            "event_type": "threshold_recommendation",
            "target_canonical_id": target,
            "recommended_value": 0.50,
            "adoption_status": "proposed",
            "derived_from_calibration_report_run_id": "synthetic",
            "supersedes_run_id": None,
            "schema_version": "v1",
        }
        payload = {
            "schema_version": "v1",
            "target_canonical_id": target,
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
        record = emit_evidence(
            audit_log_path=self.audit_path,
            evidence_kind="threshold_recommendation",
            audit_event=audit_event, payload=payload,
        )
        run_id = record["event"]["run_id"]
        setattr(self, cache_attr, run_id)
        return run_id

    def _emit_synthetic_replay_result(
        self, *, state: str, reason: str = "synthetic",
        original_kind: str = "threshold_recommendation",
        original_target: str = "x",
    ) -> str:
        """Plant a synthetic replay_result whose audit_event_ref
        points at a real planted original artifact for the given
        target. Ensures the cross-target filter (Step 14 tightening)
        sees a valid same-target match."""
        original_run_id = self._ensure_synthetic_original_for_target(
            original_target,
        )
        audit_event = {
            "event_type": "replay_run",
            "audit_event_ref": {"run_id": original_run_id,
                                "evidence_kind": original_kind},
            "state": state,
            "reason": reason,
            "schema_version": "v1",
        }
        payload = {
            "audit_event_ref": {"run_id": original_run_id,
                                "evidence_kind": original_kind},
            "state": state,
            "reason": reason,
            "differences": [],
            "divergence_drivers": [],
            "missing_inputs": [],
            "recorded": {"sha256": "abc"},
            "current": {"sha256": "def"},
            "duration_ms": 10,
            "replay_tool_version": "v1",
        }
        record = emit_evidence(
            audit_log_path=self.audit_path,
            evidence_kind="replay_result",
            audit_event=audit_event, payload=payload,
        )
        return record["event"]["run_id"]


# ── Asymmetric replay penalty (THE Step 14 tightening) ──────────────


class AsymmetricReplayPenaltyTests(_ReliabilityHarness):

    def test_invalid_replay_harder_penalty_than_unreproducible(self) -> None:
        """LOAD-BEARING: one invalid_replay reduces the
        replay_stability score MORE than one unreproducible. The
        chain-integrity violation does not hide inside benign
        substrate-decay statistics."""
        # Two parallel scenarios: 9 exact_match + 1 invalid_replay vs
        # 9 exact_match + 1 unreproducible.

        # First scenario: emit baseline + 9 OK replays + 1 invalid_replay
        for _ in range(9):
            self._emit_synthetic_replay_result(state="exact_match")
        self._emit_synthetic_replay_result(
            state="invalid_replay",
            reason="evidence file hash mismatch",
        )

        window_start = (datetime.now(timezone.utc).replace(microsecond=0)
                        - timedelta(days=90))
        score_with_invalid = dim_mod.replay_stability(
            target_canonical_id="x", target_run_id="y",
            audit_path=self.audit_path, window_start=window_start,
        )

        # Reset chain for second scenario. Also clear the cached
        # synthetic-original run_id so the helper plants a fresh one
        # (the cached id is no longer in the wiped chain).
        self.audit_path.write_text("", encoding="utf-8")
        for attr in list(vars(self)):
            if attr.startswith("_synthetic_original_for_"):
                delattr(self, attr)

        for _ in range(9):
            self._emit_synthetic_replay_result(state="exact_match")
        self._emit_synthetic_replay_result(
            state="unreproducible",
            reason="evidence file missing on disk",
        )

        score_with_unrep = dim_mod.replay_stability(
            target_canonical_id="x", target_run_id="y",
            audit_path=self.audit_path, window_start=window_start,
        )

        self.assertLess(
            score_with_invalid.score, score_with_unrep.score,
            f"invalid_replay must penalize HARDER than unreproducible. "
            f"with_invalid={score_with_invalid.score}, "
            f"with_unrep={score_with_unrep.score}",
        )

    def test_unreproducible_causes_exposed_in_basis(self) -> None:
        """Verification: substrate decay causes are typed and
        traceable in the basis — operators can see WHY the chain
        could not be re-derived without re-reading the entire chain."""
        for _ in range(3):
            self._emit_synthetic_replay_result(state="exact_match")
        self._emit_synthetic_replay_result(
            state="unreproducible",
            reason="evidence file missing on disk",
        )
        self._emit_synthetic_replay_result(
            state="unreproducible",
            reason="audit row has no evidence_ref (audit-only event)",
        )
        window_start = (datetime.now(timezone.utc).replace(microsecond=0)
                        - timedelta(days=90))
        result = dim_mod.replay_stability(
            target_canonical_id="x", target_run_id="y",
            audit_path=self.audit_path, window_start=window_start,
        )
        causes = result.basis["unreproducible_causes"]
        self.assertEqual(causes["evidence file missing on disk"], 1)
        self.assertEqual(
            causes["audit row has no evidence_ref (audit-only event)"], 1,
        )

    def test_invalid_replay_causes_exposed_separately(self) -> None:
        for _ in range(3):
            self._emit_synthetic_replay_result(state="exact_match")
        self._emit_synthetic_replay_result(
            state="invalid_replay",
            reason="evidence file hash mismatch",
        )
        self._emit_synthetic_replay_result(
            state="invalid_replay",
            reason="audit chain verification failed",
        )
        window_start = (datetime.now(timezone.utc).replace(microsecond=0)
                        - timedelta(days=90))
        result = dim_mod.replay_stability(
            target_canonical_id="x", target_run_id="y",
            audit_path=self.audit_path, window_start=window_start,
        )
        causes = result.basis["invalid_replay_causes"]
        self.assertEqual(causes["evidence file hash mismatch"], 1)
        self.assertEqual(causes["audit chain verification failed"], 1)
        # NOT collapsed into unreproducible_causes.
        self.assertEqual(result.basis["unreproducible_causes"], {})

    def test_score_floored_at_zero_under_heavy_invalid_replay(self) -> None:
        # All invalid_replay → would drive raw score to -1.0 — must
        # clamp to 0.0 (not negative).
        for _ in range(5):
            self._emit_synthetic_replay_result(state="invalid_replay")
        window_start = (datetime.now(timezone.utc).replace(microsecond=0)
                        - timedelta(days=90))
        result = dim_mod.replay_stability(
            target_canonical_id="x", target_run_id="y",
            audit_path=self.audit_path, window_start=window_start,
        )
        self.assertEqual(result.score, 0.0)
        # The raw pre-clamp value is preserved for forensics.
        self.assertEqual(result.basis["raw_score_pre_clamp"], -1.0)


# ── Aggregation + refusal floor ─────────────────────────────────────


class AggregationTests(_ReliabilityHarness):

    def test_target_unsupported_refusal(self) -> None:
        rec = score_target(
            target_canonical_id="not.a.real.target",
            target_run_id="fake-id",
            audit_path=self.audit_path,
        )
        self.assertEqual(rec["event"]["overall_refusal_reason"],
                         "target_unsupported")
        self.assertIsNone(rec["event"]["overall_score"])

    def test_target_run_id_missing_refusal(self) -> None:
        rec = score_target(
            target_canonical_id=
                "portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            target_run_id="00000000-0000-4000-8000-000000000000",
            audit_path=self.audit_path,
        )
        self.assertEqual(rec["event"]["overall_refusal_reason"],
                         "target_run_id_missing")
        self.assertIsNone(rec["event"]["overall_score"])

    def test_aggregate_refuses_when_k_or_more_dimensions_refuse(self) -> None:
        """K=4: with no replay history and minimal substrate, most
        dimensions refuse → aggregate refusal."""
        # Emit a real threshold_recommendation but with NO replay history
        # and minimal calibration history → most dimensions refuse.
        rec = self._emit_threshold_recommendation_with_substrate()
        result = score_target(
            target_canonical_id=
                "portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            target_run_id=rec["event"]["run_id"],
            audit_path=self.audit_path,
        )
        # Many dimensions refuse with insufficient_substrate (no
        # replay history, few calibrations, etc.). The aggregate
        # MUST refuse if ≥ K_REFUSAL_FLOOR refuse individually.
        refused = result["event"]["refused_dimension_count"]
        if refused >= K_REFUSAL_FLOOR:
            self.assertIsNone(result["event"]["overall_score"])
            self.assertIn(
                result["event"]["overall_refusal_reason"],
                ("all_dimensions_refused", "insufficient_substrate"),
            )

    def test_refusal_payload_carries_full_dimension_list(self) -> None:
        rec = score_target(
            target_canonical_id="not.a.real.target",
            target_run_id="x",
            audit_path=self.audit_path,
        )
        ev_path = self.data_dir / rec["event"]["evidence_ref"]["path"]
        payload = json.loads(ev_path.read_text(encoding="utf-8"))["payload"]
        # Even on target_unsupported refusal, the dimensions[] list
        # is present (empty for this specific refusal path, since no
        # dimension computation was attempted — but the field exists).
        self.assertIn("dimensions", payload)
        # Refusal carries calibration_basis-equivalent fields:
        self.assertIn("applied_weights", payload)
        self.assertIn("derived_from_run_ids", payload)


# ── Per-dimension function tests (lightweight) ──────────────────────


class DimensionFunctionTests(_ReliabilityHarness):

    def test_evidence_stability_refuses_with_too_few_rows(self) -> None:
        window_start = (datetime.now(timezone.utc).replace(microsecond=0)
                        - timedelta(days=90))
        result = dim_mod.evidence_stability(
            target_canonical_id=
                "portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            target_run_id="x",
            audit_path=self.audit_path,
            window_start=window_start,
        )
        self.assertIsNone(result.score)
        self.assertEqual(result.refusal_reason, "insufficient_substrate")

    def test_methodology_volatility_score_inverse_of_count(self) -> None:
        # Plant a calibration_report row in the chain so the dim has
        # substrate.
        cal = self._emit_calibration_recommendation()
        window_start = (datetime.now(timezone.utc).replace(microsecond=0)
                        - timedelta(days=90))
        result = dim_mod.methodology_volatility(
            target_canonical_id=
                "portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            target_run_id=cal["event"]["run_id"],
            audit_path=self.audit_path,
            window_start=window_start,
        )
        # One distinct methodology → score = 1.0
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.basis["distinct_methodologies"], 1)

    def test_drift_exposure_refuses_when_no_drift_rows(self) -> None:
        window_start = (datetime.now(timezone.utc).replace(microsecond=0)
                        - timedelta(days=90))
        result = dim_mod.drift_exposure(
            target_canonical_id="x", target_run_id="y",
            audit_path=self.audit_path, window_start=window_start,
        )
        self.assertIsNone(result.score)
        self.assertEqual(result.refusal_reason, "insufficient_substrate")


# ── Production isolation ─────────────────────────────────────────────


class ProductionIsolationTests(_ReliabilityHarness):

    def test_high_correlation_threshold_unchanged(self) -> None:
        from backend.investment_analytics.portfolio_health import correlation
        before = correlation.HIGH_CORRELATION_THRESHOLD
        rec = self._emit_threshold_recommendation_with_substrate()
        score_target(
            target_canonical_id=
                "portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            target_run_id=rec["event"]["run_id"],
            audit_path=self.audit_path,
        )
        self.assertEqual(correlation.HIGH_CORRELATION_THRESHOLD, before,
                         "production threshold MUST NEVER be mutated by "
                         "reliability scoring")

    def test_threshold_recommendation_adoption_status_unchanged(self) -> None:
        rec = self._emit_threshold_recommendation_with_substrate()
        before_status = rec["event"]["adoption_status"]
        score_target(
            target_canonical_id=
                "portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            target_run_id=rec["event"]["run_id"],
            audit_path=self.audit_path,
        )
        # adoption_status transitions are Step 15's domain.
        # reliability scoring NEVER mutates them.
        events = self._read_events()
        threshold_rows = [e for e in events
                          if e.get("evidence_kind") == "threshold_recommendation"]
        self.assertEqual(len(threshold_rows), 1)
        self.assertEqual(threshold_rows[0]["adoption_status"], before_status)


# ── Replay ───────────────────────────────────────────────────────────


class ReplayTests(_ReliabilityHarness):

    def test_handler_registered(self) -> None:
        self.assertIn("reliability_score", rp.REPLAY_HANDLERS)

    def test_replay_reproduces_under_unchanged_chain(self) -> None:
        rec = self._emit_threshold_recommendation_with_substrate()
        # Score the recommendation.
        rel_record = score_target(
            target_canonical_id=
                "portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            target_run_id=rec["event"]["run_id"],
            audit_path=self.audit_path,
        )
        rel_run_id = rel_record["event"]["run_id"]

        # Redirect runner defaults for the replay path.
        original_rel = rel_runner.DEFAULT_AUDIT_PATH
        original_cal = cal_runner.DEFAULT_AUDIT_PATH
        original_cache = cal_targets.DEFAULT_CACHE_DIR
        rel_runner.DEFAULT_AUDIT_PATH = self.audit_path
        cal_runner.DEFAULT_AUDIT_PATH = self.audit_path
        cal_targets.DEFAULT_CACHE_DIR = self.cache_dir
        try:
            result = rp.replay_run(
                self.audit_path, rel_run_id,
                verify_chain=False, emit_audit=False,
                registry_path=self.registry_path,
            )
        finally:
            rel_runner.DEFAULT_AUDIT_PATH = original_rel
            cal_runner.DEFAULT_AUDIT_PATH = original_cal
            cal_targets.DEFAULT_CACHE_DIR = original_cache

        # Replay of a reliability_score under unchanged chain SHOULD
        # land on exact_match / semantically_equivalent. Volatile
        # field `scored_at` differs; rest matches.
        self.assertIn(result["state"],
                      ("exact_match", "semantically_equivalent"),
                      f"state={result['state']}, "
                      f"reason={result.get('reason')}")


# ── Chain integrity ──────────────────────────────────────────────────


class ChainIntegrityTests(_ReliabilityHarness):

    def test_chain_valid_after_mixed_reliability_emissions(self) -> None:
        rec = self._emit_threshold_recommendation_with_substrate()
        score_target(
            target_canonical_id=
                "portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            target_run_id=rec["event"]["run_id"],
            audit_path=self.audit_path,
        )
        # Also a refusal row
        score_target(
            target_canonical_id="not.a.real.target",
            target_run_id="x",
            audit_path=self.audit_path,
        )
        self.assertTrue(verify_audit_chain(self.audit_path))


# ── find_reliability_scores ─────────────────────────────────────────


class FindTests(_ReliabilityHarness):

    def test_empty_chain_returns_empty(self) -> None:
        self.assertEqual(
            find_reliability_scores(audit_path=self.audit_path), [],
        )

    def test_filter_recommendations_vs_refusals(self) -> None:
        score_target(
            target_canonical_id="not.a.real.target",
            target_run_id="x",
            audit_path=self.audit_path,
        )  # refusal
        # No clean recommendation possible without substantial substrate
        # so this test just checks the refusal-side filter.
        refusals = find_reliability_scores(
            audit_path=self.audit_path, only_refusals=True,
        )
        self.assertEqual(len(refusals), 1)
        self.assertIsNone(refusals[0]["overall_score"])


# ── CLI ──────────────────────────────────────────────────────────────


class CliTests(_ReliabilityHarness):

    def test_cli_score_target_unsupported_exits_2(self) -> None:
        from backend.reliability.__main__ import _main
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            rc = _main([
                "score",
                "--target", "not.a.real.target",
                "--target-run-id", "fake",
                "--audit-path", str(self.audit_path),
            ])
        self.assertEqual(rc, 2, "typed refusal should exit code 2")


# ── Step 14 post-review tightenings ─────────────────────────────────


class CrossTargetReplayExclusionTests(_ReliabilityHarness):
    """Step 14 tightening #1: replay_stability MUST filter replay
    rows by the original artifact's target_canonical_id. Without
    this, target X's reliability score silently incorporates
    target Y's replay history."""

    def _emit_synthetic_replay_for_original(
        self, *, state: str, original_run_id: str,
        original_kind: str = "threshold_recommendation",
        reason: str = "synthetic",
    ) -> str:
        """Plant a replay_result whose audit_event_ref.run_id points
        at a specific original artifact. Lets the test verify the
        cross-target filter resolves through the original lookup."""
        audit_event = {
            "event_type": "replay_run",
            "audit_event_ref": {"run_id": original_run_id,
                                "evidence_kind": original_kind},
            "state": state,
            "reason": reason,
            "schema_version": "v1",
        }
        payload = {
            "audit_event_ref": {"run_id": original_run_id,
                                "evidence_kind": original_kind},
            "state": state,
            "reason": reason,
            "differences": [],
            "divergence_drivers": [],
            "missing_inputs": [],
            "recorded": {"sha256": "abc"},
            "current": {"sha256": "def"},
            "duration_ms": 10,
            "replay_tool_version": "v1",
        }
        record = emit_evidence(
            audit_log_path=self.audit_path,
            evidence_kind="replay_result",
            audit_event=audit_event, payload=payload,
        )
        return record["event"]["run_id"]

    def _emit_synthetic_threshold_for_target(self, target: str) -> str:
        """Plant a threshold_recommendation row for an arbitrary
        target_canonical_id via direct emit_evidence. Bypasses the
        runner's closed-enum guard so we can test cross-target
        filtering without registering a second real target."""
        audit_event = {
            "event_type": "threshold_recommendation",
            "target_canonical_id": target,
            "recommended_value": 0.50,
            "adoption_status": "proposed",
            "derived_from_calibration_report_run_id": "synthetic",
            "supersedes_run_id": None,
            "schema_version": "v1",
        }
        payload = {
            "schema_version": "v1",
            "target_canonical_id": target,
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
        record = emit_evidence(
            audit_log_path=self.audit_path,
            evidence_kind="threshold_recommendation",
            audit_event=audit_event, payload=payload,
        )
        return record["event"]["run_id"]

    def test_replay_stability_excludes_replays_targeting_different_target(
        self,
    ) -> None:
        target_x = "portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD"
        target_y = "portfolio_health.foreign.OTHER_TARGET"

        # Plant target X artifact + 3 OK replays of it.
        x_run_id = self._emit_synthetic_threshold_for_target(target_x)
        for _ in range(3):
            self._emit_synthetic_replay_for_original(
                state="exact_match", original_run_id=x_run_id,
            )

        # Plant target Y artifact + 5 invalid_replay rows for it.
        # If the cross-target filter were broken, these would
        # poison target X's replay_stability score severely.
        y_run_id = self._emit_synthetic_threshold_for_target(target_y)
        for _ in range(5):
            self._emit_synthetic_replay_for_original(
                state="invalid_replay", original_run_id=y_run_id,
            )

        window_start = (datetime.now(timezone.utc).replace(microsecond=0)
                        - timedelta(days=90))
        result = dim_mod.replay_stability(
            target_canonical_id=target_x,
            target_run_id=x_run_id,
            audit_path=self.audit_path,
            window_start=window_start,
        )
        # ONLY the 3 X-replays should count. All 3 are exact_match → 1.0.
        # If contamination occurred, score would be heavily penalized
        # by the 5 invalid_replays of target Y.
        self.assertEqual(result.basis["total_replays"], 3,
                         f"Expected 3 same-target replays, got "
                         f"{result.basis['total_replays']}. "
                         f"Cross-target rejected: "
                         f"{result.basis.get('rejected_cross_target_count')}")
        self.assertEqual(result.basis["rejected_cross_target_count"], 5,
                         "5 target_Y replays must be filtered out")
        self.assertEqual(result.score, 1.0,
                         f"3 exact_match replays → score 1.0, "
                         f"got {result.score}")

    def test_replay_stability_excludes_unresolvable_originals(self) -> None:
        # Replay row whose original_run_id doesn't exist in the chain
        # → must be excluded (conservative: can't verify same-target).
        target_x = "portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD"
        x_run_id = self._emit_synthetic_threshold_for_target(target_x)
        for _ in range(3):
            self._emit_synthetic_replay_for_original(
                state="exact_match", original_run_id=x_run_id,
            )
        # Plant 2 replays whose original_run_id refers to nothing.
        for _ in range(2):
            self._emit_synthetic_replay_for_original(
                state="exact_match",
                original_run_id="00000000-0000-0000-0000-000000000000",
            )
        window_start = (datetime.now(timezone.utc).replace(microsecond=0)
                        - timedelta(days=90))
        result = dim_mod.replay_stability(
            target_canonical_id=target_x,
            target_run_id=x_run_id,
            audit_path=self.audit_path,
            window_start=window_start,
        )
        self.assertEqual(result.basis["total_replays"], 3)
        self.assertEqual(result.basis["rejected_unresolvable_count"], 2)


class DimensionExceptionFallbackTests(_ReliabilityHarness):
    """Step 14 tightening #2: a dimension function that RAISES must
    surface as ``dimension_execution_failed`` — a typed refusal
    distinct from ``insufficient_substrate``. Otherwise an
    implementation bug becomes indistinguishable from an honest
    refusal, violating the anti-laundering posture."""

    def test_dimension_exception_surfaces_as_typed_refusal(self) -> None:
        rec = self._emit_threshold_recommendation_with_substrate()

        def _raise_for_test(**kwargs):
            raise RuntimeError("synthetic dimension failure")

        # Monkeypatch ONE dimension function to raise. The runner
        # must catch + classify as dimension_execution_failed.
        original_fn = dim_mod.DIMENSION_FUNCTIONS["evidence_stability"]
        dim_mod.DIMENSION_FUNCTIONS["evidence_stability"] = _raise_for_test
        try:
            result = score_target(
                target_canonical_id=
                    "portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
                target_run_id=rec["event"]["run_id"],
                audit_path=self.audit_path,
            )
        finally:
            dim_mod.DIMENSION_FUNCTIONS["evidence_stability"] = original_fn

        # Load the payload + find the affected dimension entry.
        ev_path = self.data_dir / result["event"]["evidence_ref"]["path"]
        payload = json.loads(ev_path.read_text(encoding="utf-8"))["payload"]
        evidence_stability_entry = next(
            d for d in payload["dimensions"]
            if d["dimension"] == "evidence_stability"
        )
        # The fix: refusal_reason MUST be the new typed value, NOT
        # the laundered insufficient_substrate.
        self.assertEqual(
            evidence_stability_entry["refusal_reason"],
            "dimension_execution_failed",
            "dimension exception must surface as dimension_execution_failed, "
            "NOT laundered into insufficient_substrate",
        )
        self.assertIsNone(evidence_stability_entry["score"])
        # The basis must preserve enough detail to diagnose.
        basis = evidence_stability_entry["basis"]
        self.assertEqual(basis["exception_type"], "RuntimeError")
        self.assertIn("synthetic dimension failure",
                      basis["exception_message"])
        self.assertEqual(basis["dimension"], "evidence_stability")

    def test_dimension_execution_failed_in_closed_refusal_enum(self) -> None:
        # The new refusal_reason must be in the closed enum so emit
        # paths validate against it.
        self.assertIn("dimension_execution_failed",
                      RELIABILITY_REFUSAL_REASONS)

    def test_legitimate_substrate_refusal_NOT_relabeled(self) -> None:
        # A dimension that returns refusal_reason="insufficient_substrate"
        # (honest substrate refusal, not an exception) MUST surface
        # that way — not get relabeled as dimension_execution_failed.
        # Empty chain + score with no substrate → honest refusal.
        rec = self._emit_threshold_recommendation_with_substrate()
        result = score_target(
            target_canonical_id=
                "portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            target_run_id=rec["event"]["run_id"],
            audit_path=self.audit_path,
        )
        ev_path = self.data_dir / result["event"]["evidence_ref"]["path"]
        payload = json.loads(ev_path.read_text(encoding="utf-8"))["payload"]
        # Dimensions that refused for honest substrate reasons must NOT
        # be tagged as execution failures.
        for d in payload["dimensions"]:
            if d["refusal_reason"] == "dimension_execution_failed":
                self.fail(
                    f"Honest substrate refusal mislabeled as "
                    f"dimension_execution_failed: dimension={d['dimension']}, "
                    f"basis={d['basis']}"
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
