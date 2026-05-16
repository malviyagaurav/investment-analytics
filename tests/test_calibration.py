"""Tests for backend.calibration (Step 11).

Coverage map:

  weighted_percentile (pure-math):
    - empty / negative-weight / out-of-range percentile raise
    - equal-weights matches direct computation
    - asymmetric weights produce expected interpolated values
    - p0 / p100 saturate to endpoints

  Schema invariants (frozen at emit time):
    - recommendation non-null ↔ refusal_reason null ↔
      valid_within_regimes non-empty
    - recommendation null ↔ refusal_reason non-null ↔
      valid_within_regimes empty
    - both branches carry calibration_basis (refusal is symmetric
      work to recommendation)
    - refusal_reason from closed REFUSAL_REASONS enum only

  Refusal paths (all six closed-enum values):
    - target_unsupported            — unknown target name
    - insufficient_substrate        — no regime_summary rows
    - insufficient_coverage         — too few raw observations
    - confidence_floor_unmet        — weighted mass too low
    - regime_indeterminate          — indeterminate excluded from
                                      sampling (covered by absence
                                      of contribution)
    - regime_dependency_superseded  — first-class replay driver

  Recommendation path (happy):
    - synthesized regime substrate yields non-null recommendation
    - top-level = median of per-regime
    - per-regime values preserved in samples_by_regime

  Production isolation:
    - HIGH_CORRELATION_THRESHOLD in portfolio_health byte-unmutated
    - METHODOLOGY_VERSIONS byte-unmutated

  Replay:
    - calibration_report present in REPLAY_HANDLERS
    - replay of recorded report reproduces shape (semantic
      equivalence under unchanged chain)
    - regime_dependency_superseded driver fires when a cited
      regime_summary has been superseded
    - calibration_methodology_changed driver fires when engine
      version differs in payload

  Chain integrity:
    - verify_audit_chain passes across mixed sequence
"""
from __future__ import annotations

import io
import json
import math
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import patch

from backend.calibration import (
    CALIBRATION_TARGETS,
    REFUSAL_REASONS,
    REGISTERED_TARGETS,
    assemble_per_regime_samples,
    find_calibration_reports,
    get_target,
    passes_coverage_floors,
    run_calibration,
    unweighted_median,
    weighted_median,
    weighted_percentile,
)
from backend.calibration import runner as cal_runner
from backend.calibration import sampling as cal_sampling
from backend.calibration import targets as cal_targets
from backend.evidence import replay as rp
from backend.investment_analytics import methodology as meth_mod
from backend.investment_analytics.audit import verify_audit_chain


# ── Synthetic NAV cache fixture ──────────────────────────────────────


def _write_synthetic_fund_cache(
    cache_dir: Path,
    scheme_code: int,
    scheme_category: str,
    start: date,
    n_days: int,
    target_ann_vol_pct: float,
    drift_pct_per_year: float = 0.0,
    seed: int = 0,
) -> None:
    """Write an mfapi-shaped cache file with deterministic
    Gaussian-walk NAVs. Tests can request a fund of any category +
    history length to flex the calibration universe."""
    rng = random.Random(seed)
    daily_vol = (target_ann_vol_pct / 100.0) / math.sqrt(252)
    daily_drift = (drift_pct_per_year / 100.0) / 252.0
    nav = 100.0
    points: List[Tuple[date, float]] = []
    cursor = start
    while len(points) < n_days:
        if cursor.weekday() < 5:
            r = rng.gauss(daily_drift, daily_vol)
            nav = nav * (1.0 + r)
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


class _CalibrationHarness(unittest.TestCase):

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

    def _read_events(self) -> list:
        if not self.audit_path.exists():
            return []
        with self.audit_path.open("r", encoding="utf-8") as h:
            return [json.loads(line)["event"] for line in h if line.strip()]

    def _seed_equity_universe(self, n_funds: int = 4,
                              n_days: int = 3000,
                              target_vol: float = 15.0) -> None:
        # Deep-history equity funds in the calibration universe
        # categories. Different seeds → different (partially
        # correlated) random walks → non-degenerate correlations.
        for i in range(n_funds):
            _write_synthetic_fund_cache(
                self.cache_dir,
                scheme_code=900_000 + i,
                scheme_category="Equity Scheme - Large Cap Fund",
                start=date(2018, 1, 1),
                n_days=n_days,
                target_ann_vol_pct=target_vol,
                seed=i + 1,
            )

    def _emit_regime_summary(self, *,
                             window_start: str, window_end: str,
                             regime_class: str = "normal_vol",
                             classification_confidence: float = 1.0,
                             coverage_quality: str = "high") -> str:
        """Build and emit a regime_summary row directly via emit_evidence,
        bypassing the regime runner's coverage checks. Lets tests
        synthesize any (regime, confidence, coverage_quality) tuple
        the calibration substrate needs."""
        from backend.evidence.store import emit_evidence
        # Build a signature canonical to the window so calibration
        # groups rows correctly.
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
        import hashlib
        window_hash = hashlib.sha256(
            json.dumps(window_components, sort_keys=True,
                       separators=(",", ":"), ensure_ascii=True
                       ).encode("utf-8")).hexdigest()
        regime_signature = {
            "signal_kind":         "nifty50_realized_vol",
            "taxonomy_version":    "v1",
            "classifier_version":  "v1",
            "window_hash":         window_hash,
        }
        payload = {
            "schema_version":              "v1",
            "taxonomy_version":            "v1",
            "regime_classifier_version":   "v1",
            "classification_semantics":    "descriptive_not_causal",
            "regime_class":                regime_class,
            "window_start_date":           window_start,
            "window_end_date":             window_end,
            "window_coverage_days":        252,
            "classification_confidence":   classification_confidence,
            "classification_stability":    1.0,
            "boundary_separation":         classification_confidence,
            "classification_basis": {
                "signal_kind":         "nifty50_realized_vol",
                "signal_scheme_code":  999_001,
                "annualized_vol_pct":  15.0,
                "applied_bands":       applied_bands,
                "min_coverage_days":   60,
                "boundary_confidence_margin_pct": 10.0,
            },
            "signal_quality": {
                "history_depth_days": 2500,
                "missing_data_pct":   3.0,
                "coverage_quality":   coverage_quality,
            },
            "regime_signature":  regime_signature,
            "supersedes_run_id": None,
        }
        audit_event = {
            "event_type":                 "regime_summary",
            "regime_class":               regime_class,
            "window_start_date":          window_start,
            "window_end_date":            window_end,
            "window_coverage_days":       252,
            "classification_confidence":  classification_confidence,
            "taxonomy_version":           "v1",
            "regime_classifier_version":  "v1",
            "classification_semantics":   "descriptive_not_causal",
            "coverage_quality":           coverage_quality,
            "supersedes_run_id":          None,
            "schema_version":             "v1",
        }
        record = emit_evidence(
            audit_log_path=self.audit_path,
            evidence_kind="regime_summary",
            audit_event=audit_event,
            payload=payload,
        )
        return record["event"]["run_id"]


# ── Weighted percentile (pure-math) ──────────────────────────────────


class WeightedPercentileTests(unittest.TestCase):

    def test_empty_samples_raises(self) -> None:
        with self.assertRaises(ValueError):
            weighted_percentile([], 50.0)

    def test_negative_weight_raises(self) -> None:
        with self.assertRaises(ValueError):
            weighted_percentile([(1.0, -0.5)], 50.0)

    def test_out_of_range_percentile_raises(self) -> None:
        with self.assertRaises(ValueError):
            weighted_percentile([(1.0, 1.0)], -1.0)
        with self.assertRaises(ValueError):
            weighted_percentile([(1.0, 1.0)], 101.0)

    def test_all_zero_weight_raises(self) -> None:
        with self.assertRaises(ValueError):
            weighted_percentile([(1.0, 0.0), (2.0, 0.0)], 50.0)

    def test_endpoints_saturate(self) -> None:
        samples = [(1.0, 1.0), (5.0, 1.0), (10.0, 1.0)]
        self.assertEqual(weighted_percentile(samples, 0.0), 1.0)
        self.assertEqual(weighted_percentile(samples, 100.0), 10.0)

    def test_equal_weights_p50_known(self) -> None:
        # Mass-based p50 over [1, 2, 3, 4, 5] with weight 1 each →
        # cumulative target = 2.5 → interpolated between samples
        # at indices 1 and 2 → 2.5.
        samples = [(float(v), 1.0) for v in [1, 2, 3, 4, 5]]
        self.assertEqual(weighted_percentile(samples, 50.0), 2.5)

    def test_asymmetric_weights_shift_percentile(self) -> None:
        # The mass-based linear-interp algorithm. With samples
        # [(1.0, 0.1), (100.0, 1.0)], total weight is 1.1, p50
        # target = 0.55. Falls 0.45 of the way through the second
        # sample's weight span. Linear interp from prev_v=1.0 to
        # v=100.0 at offset 0.45 → 45.55.
        # The point: asymmetric weights MOVE the percentile relative
        # to the equal-weight case (which would give 50.5 here).
        samples_weighted = [(1.0, 0.1), (100.0, 1.0)]
        samples_equal    = [(1.0, 1.0), (100.0, 1.0)]
        w_p50 = weighted_percentile(samples_weighted, 50.0)
        e_p50 = weighted_percentile(samples_equal, 50.0)
        # Heavier weight on the right shifts the linear-interp p50
        # LEFT (lower value) because cumulative mass advances faster
        # over the wider right-side weight span.
        self.assertLess(w_p50, e_p50,
                        f"weighted p50={w_p50} should differ from "
                        f"equal-weight p50={e_p50}")

    def test_weighted_median_matches_p50(self) -> None:
        samples = [(1.0, 1.0), (2.0, 2.0), (3.0, 1.0)]
        self.assertEqual(weighted_median(samples),
                         weighted_percentile(samples, 50.0))


# ── Schema invariants (frozen at emit) ───────────────────────────────


class SchemaInvariantTests(_CalibrationHarness):

    def test_recommendation_payload_construction_requires_signatures(self) -> None:
        with self.assertRaises(ValueError):
            cal_runner._recommendation_payload(
                target_id="x", recommendation=0.85, percentile=95.0,
                valid_signatures=[], excluded_signatures=[],
                basis={}, derived_from_run_ids=[], derivation_depth=0,
            )

    def test_refusal_payload_rejects_unknown_reason(self) -> None:
        with self.assertRaises(ValueError):
            cal_runner._refusal_payload(
                target_id="x", refusal_reason="bogus_reason",
                percentile=95.0, basis={},
                derived_from_run_ids=[], derivation_depth=0,
            )

    def test_refusal_payload_carries_calibration_basis(self) -> None:
        # The whole point of typed refusal: it carries the same
        # forensic shape as a recommendation payload, only the
        # conclusion differs.
        payload = cal_runner._refusal_payload(
            target_id="t", refusal_reason="insufficient_substrate",
            percentile=95.0,
            basis=cal_runner._empty_basis("t", 95.0),
            derived_from_run_ids=[], derivation_depth=0,
        )
        self.assertIsNone(payload["recommendation"])
        self.assertEqual(payload["refusal_reason"], "insufficient_substrate")
        self.assertIn("calibration_basis", payload)
        self.assertIn("calibration_scope", payload)
        self.assertEqual(payload["calibration_scope"]["valid_within_regimes"], [])

    def test_recommendation_payload_invariants_hold(self) -> None:
        payload = cal_runner._recommendation_payload(
            target_id="t", recommendation=0.85, percentile=95.0,
            valid_signatures=[{"signature": "x"}],
            excluded_signatures=[],
            basis={"observation_count": 100},
            derived_from_run_ids=["r1"], derivation_depth=1,
        )
        self.assertEqual(payload["recommendation"], 0.85)
        self.assertIsNone(payload["refusal_reason"])
        self.assertEqual(payload["calibration_scope"]["valid_within_regimes"],
                         [{"signature": "x"}])
        self.assertIn("calibration_basis", payload)


# ── Refusal paths (closed-enum coverage) ─────────────────────────────


class RefusalPathTests(_CalibrationHarness):

    def test_target_unsupported_refusal(self) -> None:
        record = run_calibration(
            target="not_a_real_target",
            audit_path=self.audit_path,
            cache_dir=self.cache_dir,
        )
        self.assertEqual(record["event"]["refusal_reason"], "target_unsupported")
        self.assertIsNone(record["event"]["recommendation"])

    def test_insufficient_substrate_refusal_on_empty_chain(self) -> None:
        # No regime_summary rows in chain → refusal with empty basis.
        record = run_calibration(
            target="portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            audit_path=self.audit_path,
            cache_dir=self.cache_dir,
        )
        self.assertEqual(record["event"]["refusal_reason"],
                         "insufficient_substrate")
        self.assertIsNone(record["event"]["recommendation"])
        # Refusal carries calibration_basis (symmetric work).
        ev_path = self.data_dir / record["event"]["evidence_ref"]["path"]
        payload = json.loads(ev_path.read_text())["payload"]
        self.assertIn("calibration_basis", payload)
        self.assertEqual(
            payload["calibration_basis"]["regime_signatures_consulted"], [])

    def test_insufficient_coverage_refusal_too_few_funds(self) -> None:
        # Synthesize a regime with valid coverage_quality but only
        # 2 funds in the universe → just 1 correlation pair per
        # window → far below MIN_OBSERVATIONS_PER_REGIME (50).
        self._seed_equity_universe(n_funds=2)
        self._emit_regime_summary(
            window_start="2024-01-02", window_end="2024-09-30",
            classification_confidence=1.0, coverage_quality="high",
        )
        record = run_calibration(
            target="portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            audit_path=self.audit_path,
            cache_dir=self.cache_dir,
        )
        self.assertEqual(record["event"]["refusal_reason"],
                         "insufficient_coverage")
        self.assertIsNone(record["event"]["recommendation"])

    def test_confidence_floor_unmet_refusal_low_confidence(self) -> None:
        # Many funds → plenty of pairs → raw N is above the floor,
        # but classification_confidence is so low that effective
        # weight collapses below MIN_EFFECTIVE_WEIGHT_PER_REGIME.
        self._seed_equity_universe(n_funds=12)
        self._emit_regime_summary(
            window_start="2024-01-02", window_end="2024-09-30",
            classification_confidence=0.01, coverage_quality="high",
        )
        record = run_calibration(
            target="portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            audit_path=self.audit_path,
            cache_dir=self.cache_dir,
        )
        self.assertEqual(record["event"]["refusal_reason"],
                         "confidence_floor_unmet")
        self.assertIsNone(record["event"]["recommendation"])

    def test_indeterminate_regime_excluded_from_substrate(self) -> None:
        # Even with full-confidence indeterminate regime_summary
        # rows, the sampler refuses to use them. End result: no
        # substrate → insufficient_substrate refusal.
        self._seed_equity_universe(n_funds=12)
        self._emit_regime_summary(
            window_start="2024-01-02", window_end="2024-09-30",
            regime_class="indeterminate",
            classification_confidence=1.0,
            coverage_quality="high",
        )
        record = run_calibration(
            target="portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            audit_path=self.audit_path,
            cache_dir=self.cache_dir,
        )
        # Indeterminate is filtered out → looks like no substrate.
        self.assertEqual(record["event"]["refusal_reason"],
                         "insufficient_substrate")

    def test_refusal_with_excluded_regimes_records_them(self) -> None:
        # Three regimes, all undercovered → top-level refusal with
        # excluded_regimes populated so consumers can see what was
        # examined.
        self._seed_equity_universe(n_funds=3)  # 3 pairs per window
        for i, start in enumerate(["2023-01-02", "2024-01-02", "2025-01-02"]):
            self._emit_regime_summary(
                window_start=start,
                window_end=f"{2023 + i}-09-30",
                classification_confidence=1.0, coverage_quality="high",
            )
        record = run_calibration(
            target="portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            audit_path=self.audit_path,
            cache_dir=self.cache_dir,
        )
        self.assertIsNone(record["event"]["recommendation"])
        ev_path = self.data_dir / record["event"]["evidence_ref"]["path"]
        payload = json.loads(ev_path.read_text())["payload"]
        # samples_by_regime carries per-regime detail (3 entries).
        self.assertEqual(len(payload["calibration_basis"]["samples_by_regime"]), 3)


# ── Recommendation path ──────────────────────────────────────────────


class RecommendationPathTests(_CalibrationHarness):

    def test_recommendation_emitted_when_coverage_passes(self) -> None:
        # Many funds + multiple windows → comfortably above all floors.
        # 12 funds → 66 pairs per window. 1 regime, 1 window: 66 obs.
        # Effective weight = 66 * 1.0 (confidence 1.0) = 66 ≥ 25.
        self._seed_equity_universe(n_funds=12)
        self._emit_regime_summary(
            window_start="2024-01-02", window_end="2024-09-30",
            classification_confidence=1.0, coverage_quality="high",
        )
        record = run_calibration(
            target="portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            audit_path=self.audit_path,
            cache_dir=self.cache_dir,
        )
        self.assertIsNotNone(record["event"]["recommendation"],
                             f"refusal_reason={record['event']['refusal_reason']}")
        self.assertIsNone(record["event"]["refusal_reason"])
        # Recommendation lies in [0, 1] for correlation thresholds.
        rec = record["event"]["recommendation"]
        self.assertGreaterEqual(rec, 0.0)
        self.assertLessEqual(rec, 1.0)

    def test_top_level_is_median_of_per_regime_values(self) -> None:
        # Two regimes with adequate coverage. Each produces its own
        # p95; top-level = median of those two = their mean.
        self._seed_equity_universe(n_funds=12)
        self._emit_regime_summary(
            window_start="2023-01-02", window_end="2023-09-30",
            classification_confidence=1.0, coverage_quality="high",
            regime_class="normal_vol",
        )
        # Different window → different regime_signature (different
        # window_hash) → separate bucket.
        self._emit_regime_summary(
            window_start="2024-01-02", window_end="2024-09-30",
            classification_confidence=1.0, coverage_quality="high",
            regime_class="normal_vol",
        )
        record = run_calibration(
            target="portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            audit_path=self.audit_path,
            cache_dir=self.cache_dir,
        )
        rec = record["event"]["recommendation"]
        self.assertIsNotNone(rec)
        ev_path = self.data_dir / record["event"]["evidence_ref"]["path"]
        payload = json.loads(ev_path.read_text())["payload"]
        per_regime_values = [
            entry["percentile_value"]
            for entry in payload["calibration_basis"]["samples_by_regime"].values()
            if entry["percentile_value"] is not None
        ]
        self.assertEqual(len(per_regime_values), 2)
        expected_median = unweighted_median(per_regime_values)
        # Rounding: top-level rounded to 6dp; per-regime same.
        self.assertEqual(round(rec, 6), round(expected_median, 6))


# ── Production isolation ─────────────────────────────────────────────


class ProductionIsolationTests(_CalibrationHarness):

    def test_methodology_versions_unmutated_by_run(self) -> None:
        before = dict(meth_mod.METHODOLOGY_VERSIONS)
        self._seed_equity_universe(n_funds=12)
        self._emit_regime_summary(
            window_start="2024-01-02", window_end="2024-09-30",
            classification_confidence=1.0, coverage_quality="high",
        )
        run_calibration(
            target="portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            audit_path=self.audit_path,
            cache_dir=self.cache_dir,
        )
        self.assertEqual(before, dict(meth_mod.METHODOLOGY_VERSIONS))

    def test_production_high_correlation_threshold_unmutated(self) -> None:
        from backend.investment_analytics.portfolio_health import correlation as corr_mod
        before = corr_mod.HIGH_CORRELATION_THRESHOLD
        self._seed_equity_universe(n_funds=12)
        self._emit_regime_summary(
            window_start="2024-01-02", window_end="2024-09-30",
            classification_confidence=1.0, coverage_quality="high",
        )
        run_calibration(
            target="portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            audit_path=self.audit_path,
            cache_dir=self.cache_dir,
        )
        # Production threshold byte-unchanged — calibration is a
        # PROPOSAL, never an auto-promotion.
        self.assertEqual(corr_mod.HIGH_CORRELATION_THRESHOLD, before)


# ── Replay ───────────────────────────────────────────────────────────


class ReplayTests(_CalibrationHarness):

    def test_calibration_report_handler_registered(self) -> None:
        self.assertIn("calibration_report", rp.REPLAY_HANDLERS)

    def test_replay_reproduces_refusal_payload(self) -> None:
        # Empty chain → refusal. Replay must reproduce the same
        # refusal shape (semantic equivalence under unchanged chain).
        record = run_calibration(
            target="portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            audit_path=self.audit_path,
            cache_dir=self.cache_dir,
        )
        run_id = record["event"]["run_id"]
        # Patch the calibration runner's default audit path so the
        # replay handler walks our isolated chain, not the live one.
        original_audit = cal_runner.DEFAULT_AUDIT_PATH
        original_cache = cal_targets.DEFAULT_CACHE_DIR
        cal_runner.DEFAULT_AUDIT_PATH = self.audit_path
        cal_targets.DEFAULT_CACHE_DIR = self.cache_dir
        try:
            result = rp.replay_run(
                self.audit_path, run_id,
                verify_chain=False, emit_audit=False,
                registry_path=self.registry_path,
            )
        finally:
            cal_runner.DEFAULT_AUDIT_PATH = original_audit
            cal_targets.DEFAULT_CACHE_DIR = original_cache
        # The recorded refusal AND the replay refusal share the same
        # refusal_reason; volatile differences are timestamps.
        self.assertIn(result["state"],
                      ("exact_match", "semantically_equivalent"),
                      f"got state={result['state']}, "
                      f"reason={result.get('reason')}")

    def test_calibration_methodology_changed_driver_fires(self) -> None:
        # Replay where the recorded payload was emitted under engine v1
        # but the current replay produces a payload with engine v999
        # (via monkey-patch). The payload-level driver must surface
        # the change as expected_divergence.
        from backend.calibration import config as cal_config
        record = run_calibration(
            target="portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            audit_path=self.audit_path,
            cache_dir=self.cache_dir,
        )
        run_id = record["event"]["run_id"]

        original_engine = cal_config.CALIBRATION_ENGINE_VERSION
        cal_config.CALIBRATION_ENGINE_VERSION = "v999"
        cal_runner.CALIBRATION_ENGINE_VERSION = "v999"
        original_audit = cal_runner.DEFAULT_AUDIT_PATH
        original_cache = cal_targets.DEFAULT_CACHE_DIR
        cal_runner.DEFAULT_AUDIT_PATH = self.audit_path
        cal_targets.DEFAULT_CACHE_DIR = self.cache_dir
        try:
            result = rp.replay_run(
                self.audit_path, run_id,
                verify_chain=False, emit_audit=False,
                registry_path=self.registry_path,
            )
        finally:
            cal_config.CALIBRATION_ENGINE_VERSION = original_engine
            cal_runner.CALIBRATION_ENGINE_VERSION = original_engine
            cal_runner.DEFAULT_AUDIT_PATH = original_audit
            cal_targets.DEFAULT_CACHE_DIR = original_cache
        self.assertEqual(result["state"], "expected_divergence",
                         f"reason={result.get('reason')}")
        kinds = {d["kind"] for d in result["divergence_drivers"]}
        self.assertIn("calibration_methodology_changed", kinds)

    def test_regime_dependency_superseded_driver_fires(self) -> None:
        # Emit a calibration report that derives from regime_summary X,
        # then supersede X with a newer regime_summary, then replay
        # the calibration. The dedicated driver must surface — kept
        # distinct from the generic methodology_changed driver.
        self._seed_equity_universe(n_funds=12)
        first_run_id = self._emit_regime_summary(
            window_start="2024-01-02", window_end="2024-09-30",
            classification_confidence=1.0, coverage_quality="high",
        )
        record = run_calibration(
            target="portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            audit_path=self.audit_path,
            cache_dir=self.cache_dir,
        )
        cal_run_id = record["event"]["run_id"]
        # Supersede the regime claim by emitting a NEW regime_summary
        # for the same window with supersedes_run_id pointing at it.
        from backend.regimes.runner import emit_regime_summary as emit_reg
        from backend.regimes.config import ClassifierParams
        from backend.regimes.classifier import (
            classify_window, classification_to_payload,
        )
        # Build a synthetic-cache classifier params so emit_regime_summary
        # doesn't refuse to find the signal.
        # Easier path: bypass classify_window and emit the supersedes
        # row directly via emit_evidence — same trick the helper uses.
        from backend.evidence.store import emit_evidence
        second_payload = {
            "schema_version":              "v1",
            "taxonomy_version":            "v1",
            "regime_classifier_version":   "v1",
            "classification_semantics":    "descriptive_not_causal",
            "regime_class":                "normal_vol",
            "window_start_date":           "2024-01-02",
            "window_end_date":             "2024-09-30",
            "window_coverage_days":        252,
            "classification_confidence":   1.0,
            "classification_stability":    1.0,
            "boundary_separation":         1.0,
            "classification_basis":        {"applied_bands": {
                "low_threshold_pct": 12.0, "high_threshold_pct": 20.0,
                "crisis_threshold_pct": 35.0,
            }},
            "signal_quality":              {"coverage_quality": "high"},
            "regime_signature":            {"signal_kind": "nifty50_realized_vol"},
            "supersedes_run_id":           first_run_id,
        }
        second_event = {
            "event_type":                 "regime_summary",
            "regime_class":               "normal_vol",
            "window_start_date":          "2024-01-02",
            "window_end_date":            "2024-09-30",
            "window_coverage_days":       252,
            "classification_confidence":  1.0,
            "taxonomy_version":           "v1",
            "regime_classifier_version":  "v1",
            "classification_semantics":   "descriptive_not_causal",
            "coverage_quality":           "high",
            "supersedes_run_id":          first_run_id,
            "schema_version":             "v1",
        }
        emit_evidence(
            audit_log_path=self.audit_path,
            evidence_kind="regime_summary",
            audit_event=second_event,
            payload=second_payload,
            parent_run_id=first_run_id,
        )

        # Now replay the calibration. The chain has first_run_id +
        # the calibration that derives from it + a supersedes row.
        original_audit = cal_runner.DEFAULT_AUDIT_PATH
        original_cache = cal_targets.DEFAULT_CACHE_DIR
        cal_runner.DEFAULT_AUDIT_PATH = self.audit_path
        cal_targets.DEFAULT_CACHE_DIR = self.cache_dir
        try:
            result = rp.replay_run(
                self.audit_path, cal_run_id,
                verify_chain=False, emit_audit=False,
                registry_path=self.registry_path,
            )
        finally:
            cal_runner.DEFAULT_AUDIT_PATH = original_audit
            cal_targets.DEFAULT_CACHE_DIR = original_cache
        kinds = {d["kind"] for d in result.get("divergence_drivers", [])}
        self.assertIn("regime_dependency_superseded", kinds,
                      f"state={result['state']} drivers={result.get('divergence_drivers')}")


# ── Chain integrity ──────────────────────────────────────────────────


class ChainIntegrityTests(_CalibrationHarness):

    def test_chain_valid_after_mixed_sequence(self) -> None:
        self._seed_equity_universe(n_funds=12)
        self._emit_regime_summary(
            window_start="2024-01-02", window_end="2024-09-30",
            classification_confidence=1.0, coverage_quality="high",
        )
        run_calibration(
            target="portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            audit_path=self.audit_path,
            cache_dir=self.cache_dir,
        )
        run_calibration(  # second emit, refusal-eligible if nothing changed
            target="not_a_real_target",
            audit_path=self.audit_path,
            cache_dir=self.cache_dir,
        )
        self.assertTrue(verify_audit_chain(self.audit_path))


# ── find_calibration_reports ─────────────────────────────────────────


class FindCalibrationReportsTests(_CalibrationHarness):

    def test_filter_by_recommendations_vs_refusals(self) -> None:
        # One refusal (no substrate), one recommendation (substrate
        # present). Filters should each return exactly one row.
        run_calibration(
            target="portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            audit_path=self.audit_path,
            cache_dir=self.cache_dir,
        )  # refusal
        self._seed_equity_universe(n_funds=12)
        self._emit_regime_summary(
            window_start="2024-01-02", window_end="2024-09-30",
            classification_confidence=1.0, coverage_quality="high",
        )
        run_calibration(
            target="portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
            audit_path=self.audit_path,
            cache_dir=self.cache_dir,
        )  # recommendation
        recs = find_calibration_reports(
            audit_path=self.audit_path, only_recommendations=True,
        )
        refs = find_calibration_reports(
            audit_path=self.audit_path, only_refusals=True,
        )
        self.assertEqual(len(recs), 1)
        self.assertEqual(len(refs), 1)
        self.assertIsNotNone(recs[0]["recommendation"])
        self.assertIsNone(refs[0]["recommendation"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
