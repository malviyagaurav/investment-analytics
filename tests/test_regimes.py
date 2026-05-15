"""Tests for backend.regimes (Step 10).

Coverage map:

  Taxonomy + governance (config):
    - REGIME_CLASSES is closed, finite, frozenset
    - DETERMINATE_REGIME_CLASSES excludes indeterminate
    - CLASSIFICATION_SEMANTICS is descriptive_not_causal
    - taxonomy_version + regime_classifier_version are independent
    - METHODOLOGY_VERSIONS now contains regime_classifier
    - METHODOLOGY_SCHEMA_VERSION bumped to v2

  Classifier semantics (classifier):
    - synthesized NAV with known vol lands in each determinate class
    - insufficient coverage → indeterminate (with reason)
    - vol-uncomputable → indeterminate (with reason)
    - end < start → ValueError
    - confidence erodes near band boundaries (barely crossed)
    - signal_quality structure: history_depth_days, missing_data_pct,
      coverage_quality bucket
    - applied bands recorded inline for replay determinism

  Drift (drift):
    - identical windows → vol_delta == 0, transition False
    - transition recognized only when BOTH determinate
    - indeterminate ↔ determinate → transition False (tightening)
    - indeterminate ↔ indeterminate → transition False
    - transition_confidence = min of both confidences
    - magnitude bands: minor / notable / regime_change

  Emit + immutability (runner):
    - regime_summary row appears on chain, lightweight event +
      heavy payload
    - re-emitting same window without supersedes_run_id → ValueError
    - re-emitting with valid supersedes_run_id → both rows present
    - supersedes_run_id pointing at unknown run_id → ValueError
    - find_regime_summaries respects include_superseded
    - parent_run_id chains supersedes link
    - drift_analysis row appears with linked window run_ids
    - METHODOLOGY_VERSIONS unmutated by emit

  Replay (replay handlers):
    - regime_summary replay reproduces with same params
    - drift_analysis replay reproduces
    - both handlers present in REPLAY_HANDLERS
    - replays do NOT create new regime_summary rows
"""
from __future__ import annotations

import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from backend.evidence import replay as rp
from backend.investment_analytics import methodology as meth_mod
from backend.investment_analytics.audit import verify_audit_chain
from backend.regimes import (
    CLASSIFICATION_SEMANTICS,
    ClassifierParams,
    DEFAULT_CLASSIFIER_PARAMS,
    DETERMINATE_REGIME_CLASSES,
    REGIME_CLASSES,
    REGIME_TAXONOMY_VERSION,
    classify_window,
    compute_drift,
    emit_drift_analysis,
    emit_regime_summary,
    find_regime_summaries,
)
from backend.regimes.classifier import RegimeClassification


# ── Synthetic NAV fixture ────────────────────────────────────────────


def _write_synthetic_signal_cache(
    cache_dir: Path,
    scheme_code: int,
    start: date,
    *,
    n_days: int,
    target_ann_vol_pct: float,
    drift_pct_per_year: float = 0.0,
    seed: int = 42,
) -> None:
    """Create a fake mfapi-shaped cache file with a deterministic
    random walk whose realized annualized vol approximates the target.

    Trading-day approximation: skip weekends. mfapi's date format is
    DD-MM-YYYY, newest-first — match that so the classifier's parser
    consumes it.
    """
    import random
    rng = random.Random(seed)
    daily_vol = (target_ann_vol_pct / 100.0) / math.sqrt(252)
    daily_drift = (drift_pct_per_year / 100.0) / 252.0

    nav = 100.0
    points = []
    cursor = start
    while len(points) < n_days:
        if cursor.weekday() < 5:
            r = rng.gauss(daily_drift, daily_vol)
            nav = nav * (1.0 + r)
            points.append((cursor, nav))
        cursor = cursor + timedelta(days=1)

    # mfapi returns newest-first
    points.reverse()
    data = [{"date": d.strftime("%d-%m-%Y"), "nav": f"{v:.4f}"}
            for d, v in points]

    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{scheme_code}.json").write_text(json.dumps({
        "fetched_at": "2026-05-15T00:00:00+00:00",
        "scheme_code": scheme_code,
        "response": {
            "meta": {"scheme_code": scheme_code, "scheme_name": "test"},
            "data": data,
        },
    }), encoding="utf-8")


class _RegimeHarness(unittest.TestCase):
    """Common setUp: isolated audit + cache under tmpdir."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp.name)
        self.audit_path = self.data_dir / "audit" / "audit.jsonl"
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Stub registry FILE (replay_run calls _hash_file via
        # capture_provenance_inputs when classifying drivers; passing a
        # directory raises IsADirectoryError). Tests that exercise
        # the expected_divergence path need a real file here.
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


# ── Taxonomy + governance ────────────────────────────────────────────


class TaxonomyAndGovernanceTests(unittest.TestCase):

    def test_regime_classes_is_closed_frozenset(self) -> None:
        self.assertIsInstance(REGIME_CLASSES, frozenset)
        self.assertEqual(REGIME_CLASSES, {
            "low_vol", "normal_vol", "high_vol",
            "crisis_vol", "indeterminate",
        })

    def test_determinate_regime_classes_excludes_indeterminate(self) -> None:
        self.assertNotIn("indeterminate", DETERMINATE_REGIME_CLASSES)
        self.assertEqual(
            DETERMINATE_REGIME_CLASSES,
            REGIME_CLASSES - {"indeterminate"},
        )

    def test_classification_semantics_is_descriptive_not_causal(self) -> None:
        # Formal anti-overinterpretation control. The system makes
        # descriptive vol-band claims, NOT causal economic claims.
        self.assertEqual(CLASSIFICATION_SEMANTICS, "descriptive_not_causal")

    def test_taxonomy_version_is_independent_of_classifier_version(self) -> None:
        # The two axes are governed separately. v1 / v1 today, but
        # the dataclass field for the classifier version on a result
        # is distinct from the module-level taxonomy version.
        c = RegimeClassification(
            regime_class="low_vol",
            window_start_date="2024-01-01",
            window_end_date="2024-06-30",
            window_coverage_days=126,
            classification_confidence=1.0,
            classification_stability=1.0,
            boundary_separation=1.0,
            classification_basis={},
            signal_quality={},
        )
        # Default values come from config module constants; mutation
        # of one must not silently mirror to the other.
        self.assertEqual(c.taxonomy_version, REGIME_TAXONOMY_VERSION)
        self.assertEqual(c.regime_classifier_version, "v1")

    def test_methodology_versions_contains_regime_classifier(self) -> None:
        self.assertIn("regime_classifier", meth_mod.METHODOLOGY_VERSIONS)
        self.assertEqual(meth_mod.METHODOLOGY_VERSIONS["regime_classifier"], "v1")

    def test_methodology_schema_version_bumped_to_v2(self) -> None:
        self.assertEqual(meth_mod.METHODOLOGY_SCHEMA_VERSION, "v2")


# ── Classifier semantics ─────────────────────────────────────────────


class ClassifierSemanticsTests(_RegimeHarness):

    def _params(self, **overrides) -> ClassifierParams:
        defaults = dict(
            signal_scheme_code=999_001,
            low_threshold_pct=12.0,
            high_threshold_pct=20.0,
            crisis_threshold_pct=35.0,
            min_coverage_days=60,
        )
        defaults.update(overrides)
        return ClassifierParams(**defaults)

    def test_low_vol_window_classifies_as_low_vol(self) -> None:
        _write_synthetic_signal_cache(
            self.cache_dir, 999_001, date(2024, 1, 1),
            n_days=200, target_ann_vol_pct=8.0,
        )
        c = classify_window("2024-01-02", "2024-09-30",
                            params=self._params(), cache_dir=self.cache_dir)
        self.assertEqual(c.regime_class, "low_vol",
                         f"got {c.regime_class}, vol={c.classification_basis.get('annualized_vol_pct')}")

    def test_high_vol_window_classifies_as_high_vol(self) -> None:
        _write_synthetic_signal_cache(
            self.cache_dir, 999_001, date(2024, 1, 1),
            n_days=200, target_ann_vol_pct=25.0,
        )
        c = classify_window("2024-01-02", "2024-09-30",
                            params=self._params(), cache_dir=self.cache_dir)
        self.assertEqual(c.regime_class, "high_vol",
                         f"got {c.regime_class}, vol={c.classification_basis.get('annualized_vol_pct')}")

    def test_crisis_vol_window_classifies_as_crisis_vol(self) -> None:
        _write_synthetic_signal_cache(
            self.cache_dir, 999_001, date(2024, 1, 1),
            n_days=200, target_ann_vol_pct=45.0,
        )
        c = classify_window("2024-01-02", "2024-09-30",
                            params=self._params(), cache_dir=self.cache_dir)
        self.assertEqual(c.regime_class, "crisis_vol")

    def test_insufficient_coverage_returns_indeterminate(self) -> None:
        # 30 days < min_coverage_days=60.
        _write_synthetic_signal_cache(
            self.cache_dir, 999_001, date(2024, 1, 1),
            n_days=30, target_ann_vol_pct=15.0,
        )
        c = classify_window("2024-01-02", "2024-02-15",
                            params=self._params(), cache_dir=self.cache_dir)
        self.assertEqual(c.regime_class, "indeterminate")
        self.assertEqual(c.classification_confidence, 0.0)
        self.assertEqual(c.classification_basis.get("indeterminate_reason"),
                         "insufficient_coverage")

    def test_missing_cache_file_returns_indeterminate(self) -> None:
        # No file written for the signal code.
        c = classify_window("2024-01-01", "2024-12-31",
                            params=self._params(), cache_dir=self.cache_dir)
        self.assertEqual(c.regime_class, "indeterminate")
        self.assertEqual(c.classification_confidence, 0.0)

    def test_end_before_start_raises(self) -> None:
        with self.assertRaises(ValueError):
            classify_window("2024-12-31", "2024-01-01",
                            params=self._params(), cache_dir=self.cache_dir)

    def test_signal_quality_structure_present(self) -> None:
        _write_synthetic_signal_cache(
            self.cache_dir, 999_001, date(2018, 1, 1),
            n_days=2000, target_ann_vol_pct=15.0,
        )
        c = classify_window("2022-01-01", "2023-12-31",
                            params=self._params(), cache_dir=self.cache_dir)
        sq = c.signal_quality
        for key in ("history_depth_days", "missing_data_pct", "coverage_quality"):
            self.assertIn(key, sq)
        self.assertIn(sq["coverage_quality"], ("high", "medium", "low"))
        self.assertGreaterEqual(sq["history_depth_days"], 0)
        self.assertGreaterEqual(sq["missing_data_pct"], 0.0)

    def test_confidence_erodes_near_band_boundary(self) -> None:
        # Build a window whose realized vol is ~12.5% — just above the
        # low/normal boundary at 12%. Confidence should be < 1.0.
        _write_synthetic_signal_cache(
            self.cache_dir, 999_001, date(2024, 1, 1),
            n_days=400, target_ann_vol_pct=12.5, seed=7,
        )
        c = classify_window("2024-01-02", "2025-12-31",
                            params=self._params(), cache_dir=self.cache_dir)
        # Synthetic vol won't land exactly on 12.5%; assert it is in the
        # normal_vol or low_vol band AND that confidence reflects the
        # proximity to a boundary.
        self.assertIn(c.regime_class, {"low_vol", "normal_vol"})
        self.assertLess(c.classification_confidence, 1.0)

    def test_applied_bands_recorded_inline(self) -> None:
        _write_synthetic_signal_cache(
            self.cache_dir, 999_001, date(2024, 1, 1),
            n_days=200, target_ann_vol_pct=15.0,
        )
        params = self._params(
            low_threshold_pct=10.0, high_threshold_pct=18.0,
            crisis_threshold_pct=30.0,
        )
        c = classify_window("2024-01-02", "2024-09-30",
                            params=params, cache_dir=self.cache_dir)
        bands = c.classification_basis.get("applied_bands", {})
        self.assertEqual(bands.get("low_threshold_pct"), 10.0)
        self.assertEqual(bands.get("high_threshold_pct"), 18.0)
        self.assertEqual(bands.get("crisis_threshold_pct"), 30.0)


# ── Drift semantics ──────────────────────────────────────────────────


class DriftSemanticsTests(_RegimeHarness):

    def _params(self, **overrides) -> ClassifierParams:
        defaults = dict(signal_scheme_code=999_001, min_coverage_days=60)
        defaults.update(overrides)
        return ClassifierParams(**defaults)

    def _build_classification(
        self, regime_class: str, vol_pct: float, confidence: float,
        *, boundary_confidence_margin_pct: float = 10.0,
    ) -> RegimeClassification:
        """Synthesize a classification result directly. Bypasses
        classify_window so we control regime_class / vol / confidence
        exactly for the drift semantics tests."""
        return RegimeClassification(
            regime_class=regime_class,
            window_start_date="2024-01-01",
            window_end_date="2024-06-30",
            window_coverage_days=126,
            classification_confidence=confidence,
            classification_stability=confidence,
            boundary_separation=confidence,
            classification_basis={
                "annualized_vol_pct": vol_pct,
                "applied_bands": {
                    "low_threshold_pct": 12.0,
                    "high_threshold_pct": 20.0,
                    "crisis_threshold_pct": 35.0,
                },
                "boundary_confidence_margin_pct": boundary_confidence_margin_pct,
            },
            signal_quality={"coverage_quality": "high"},
        )

    def test_identical_classifications_have_zero_drift(self) -> None:
        a = self._build_classification("normal_vol", 15.0, 1.0)
        b = self._build_classification("normal_vol", 15.0, 1.0)
        d = compute_drift(a, b)
        self.assertEqual(d.vol_delta_pct, 0.0)
        self.assertFalse(d.regime_transition)
        self.assertEqual(d.transition_confidence, 0.0)

    def test_transition_between_determinate_classes(self) -> None:
        a = self._build_classification("normal_vol", 15.0, 0.9)
        b = self._build_classification("crisis_vol", 40.0, 0.8)
        d = compute_drift(a, b)
        self.assertTrue(d.regime_transition)
        self.assertEqual(d.magnitude_band, "regime_change")
        # transition_confidence is bounded by EITHER side's
        # classification_confidence OR the magnitude factor. Here
        # vol=15 sits 5pp under normal_vol's upper boundary (20) and
        # vol=40 sits 5pp over crisis_vol's lower boundary (35), so
        # the magnitude_factor is 5/margin=10 → 0.5 → dominates the
        # side-confidence floor (0.8).
        self.assertEqual(d.transition_confidence, 0.5)

    def test_indeterminate_to_determinate_is_not_transition(self) -> None:
        # Tightening per Step 10 governance review: indeterminate is
        # a confidence state, not a market state. A move into or out
        # of indeterminate is NOT a regime transition.
        a = self._build_classification("indeterminate", 0.0, 0.0)
        b = self._build_classification("crisis_vol", 40.0, 0.9)
        d = compute_drift(a, b)
        self.assertFalse(d.regime_transition,
                         "indeterminate → determinate must NOT be a transition")
        self.assertEqual(d.transition_confidence, 0.0)

    def test_determinate_to_indeterminate_is_not_transition(self) -> None:
        a = self._build_classification("normal_vol", 15.0, 0.9)
        b = self._build_classification("indeterminate", 0.0, 0.0)
        d = compute_drift(a, b)
        self.assertFalse(d.regime_transition)
        self.assertEqual(d.transition_confidence, 0.0)

    def test_indeterminate_to_indeterminate_is_not_transition(self) -> None:
        a = self._build_classification("indeterminate", 0.0, 0.0)
        b = self._build_classification("indeterminate", 0.0, 0.0)
        d = compute_drift(a, b)
        self.assertFalse(d.regime_transition)

    def test_transition_confidence_bounded_by_weaker_side(self) -> None:
        # ADD 4 from review: barely-crossed transitions must not
        # appear as authoritative. Here BOTH the side-confidence AND
        # the magnitude factor collapse — vols sit 0.1pp from the
        # band boundary on either side, so magnitude_factor=0.01,
        # which dominates the 0.05 side-confidence floor.
        a = self._build_classification("normal_vol", 19.9, 0.05)
        b = self._build_classification("high_vol", 20.1, 0.05)
        d = compute_drift(a, b)
        self.assertTrue(d.regime_transition)
        # The honest signal: barely-crossed, near-zero confidence.
        # Either bound (side or magnitude) being low forces the
        # combined value low. Step 11 calibration will refuse to
        # use this as authoritative regime evidence.
        self.assertLess(d.transition_confidence, 0.05)

    def test_magnitude_bands_match_thresholds(self) -> None:
        a = self._build_classification("normal_vol", 15.0, 1.0)
        b_minor = self._build_classification("normal_vol", 16.0, 1.0)
        b_notable = self._build_classification("high_vol", 20.5, 1.0)
        b_change = self._build_classification("crisis_vol", 40.0, 1.0)
        self.assertEqual(compute_drift(a, b_minor).magnitude_band, "minor")
        self.assertEqual(compute_drift(a, b_notable).magnitude_band, "notable")
        self.assertEqual(compute_drift(a, b_change).magnitude_band,
                         "regime_change")


# ── Emit + immutability (runner) ─────────────────────────────────────


class EmitImmutabilityTests(_RegimeHarness):

    def _params(self) -> ClassifierParams:
        return ClassifierParams(signal_scheme_code=999_001, min_coverage_days=60)

    def _seed_cache(self, target_vol_pct: float = 15.0,
                    n_days: int = 200) -> None:
        _write_synthetic_signal_cache(
            self.cache_dir, 999_001, date(2024, 1, 1),
            n_days=n_days, target_ann_vol_pct=target_vol_pct,
        )

    def test_emit_regime_summary_appends_one_row(self) -> None:
        self._seed_cache()
        before = len(self._read_events())
        record = emit_regime_summary(
            "2024-01-02", "2024-09-30",
            params=self._params(), cache_dir=self.cache_dir,
            audit_path=self.audit_path,
        )
        events = self._read_events()
        self.assertEqual(len(events), before + 1)
        # The row's heavy provenance is in the evidence file, not the
        # audit event.
        self.assertIn("evidence_ref", record["event"])
        self.assertEqual(record["event"]["evidence_kind"], "regime_summary")
        self.assertEqual(record["event"]["event_type"], "regime_summary")
        self.assertIsNone(record["event"]["supersedes_run_id"])

    def test_evidence_payload_carries_governance_fields(self) -> None:
        self._seed_cache()
        record = emit_regime_summary(
            "2024-01-02", "2024-09-30",
            params=self._params(), cache_dir=self.cache_dir,
            audit_path=self.audit_path,
        )
        ev_path = self.data_dir / record["event"]["evidence_ref"]["path"]
        envelope = json.loads(ev_path.read_text(encoding="utf-8"))
        payload = envelope["payload"]
        # Step 10 governance additions:
        self.assertIn("classification_confidence", payload)
        self.assertIn("classification_basis", payload)
        self.assertIn("window_coverage_days", payload)
        self.assertIn("signal_quality", payload)
        self.assertIn("classification_semantics", payload)
        self.assertIn("taxonomy_version", payload)
        self.assertIn("regime_classifier_version", payload)
        # Anti-overinterpretation:
        self.assertEqual(payload["classification_semantics"],
                         "descriptive_not_causal")
        # Signal quality sub-fields:
        sq = payload["signal_quality"]
        for k in ("history_depth_days", "missing_data_pct", "coverage_quality"):
            self.assertIn(k, sq)

    def test_re_emit_same_window_without_supersedes_raises(self) -> None:
        # The immutability invariant — silent shadowing is refused.
        self._seed_cache()
        emit_regime_summary(
            "2024-01-02", "2024-09-30",
            params=self._params(), cache_dir=self.cache_dir,
            audit_path=self.audit_path,
        )
        with self.assertRaises(ValueError) as ctx:
            emit_regime_summary(
                "2024-01-02", "2024-09-30",
                params=self._params(), cache_dir=self.cache_dir,
                audit_path=self.audit_path,
            )
        self.assertIn("supersedes_run_id", str(ctx.exception))

    def test_re_emit_with_valid_supersedes_chains_both_rows(self) -> None:
        self._seed_cache()
        first = emit_regime_summary(
            "2024-01-02", "2024-09-30",
            params=self._params(), cache_dir=self.cache_dir,
            audit_path=self.audit_path,
        )
        first_run_id = first["event"]["run_id"]
        second = emit_regime_summary(
            "2024-01-02", "2024-09-30",
            params=self._params(), cache_dir=self.cache_dir,
            audit_path=self.audit_path,
            supersedes_run_id=first_run_id,
        )
        self.assertEqual(second["event"]["supersedes_run_id"], first_run_id)
        self.assertEqual(second["event"]["parent_run_id"], first_run_id)
        # Both rows exist on chain.
        rows = find_regime_summaries(
            audit_path=self.audit_path, include_superseded=True,
        )
        self.assertEqual(len(rows), 2)
        # Filtering superseded → only second row.
        latest = find_regime_summaries(
            audit_path=self.audit_path, include_superseded=False,
        )
        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0]["run_id"], second["event"]["run_id"])

    def test_supersedes_unknown_run_id_raises(self) -> None:
        self._seed_cache()
        emit_regime_summary(
            "2024-01-02", "2024-09-30",
            params=self._params(), cache_dir=self.cache_dir,
            audit_path=self.audit_path,
        )
        with self.assertRaises(ValueError):
            emit_regime_summary(
                "2024-01-02", "2024-09-30",
                params=self._params(), cache_dir=self.cache_dir,
                audit_path=self.audit_path,
                supersedes_run_id="00000000-0000-4000-8000-000000000000",
            )

    def test_supersedes_with_no_prior_row_raises(self) -> None:
        self._seed_cache()
        with self.assertRaises(ValueError):
            emit_regime_summary(
                "2024-01-02", "2024-09-30",
                params=self._params(), cache_dir=self.cache_dir,
                audit_path=self.audit_path,
                supersedes_run_id="any-id",
            )

    def test_emit_does_not_mutate_methodology_versions(self) -> None:
        before = dict(meth_mod.METHODOLOGY_VERSIONS)
        self._seed_cache()
        emit_regime_summary(
            "2024-01-02", "2024-09-30",
            params=self._params(), cache_dir=self.cache_dir,
            audit_path=self.audit_path,
        )
        after = dict(meth_mod.METHODOLOGY_VERSIONS)
        self.assertEqual(before, after)


# ── drift_analysis emission ──────────────────────────────────────────


class DriftEmissionTests(_RegimeHarness):

    def _params(self) -> ClassifierParams:
        return ClassifierParams(signal_scheme_code=999_001, min_coverage_days=60)

    def test_emit_drift_analysis_appends_one_row(self) -> None:
        _write_synthetic_signal_cache(
            self.cache_dir, 999_001, date(2024, 1, 1),
            n_days=400, target_ann_vol_pct=15.0,
        )
        a = classify_window("2024-01-02", "2024-06-30",
                            params=self._params(), cache_dir=self.cache_dir)
        b = classify_window("2024-07-01", "2024-12-31",
                            params=self._params(), cache_dir=self.cache_dir)
        record = emit_drift_analysis(a, b, audit_path=self.audit_path)
        self.assertEqual(record["event"]["evidence_kind"], "drift_analysis")
        self.assertEqual(record["event"]["event_type"], "drift_analysis")
        self.assertIn("transition_confidence", record["event"])
        self.assertIn("magnitude_band", record["event"])

    def test_emit_drift_analysis_carries_window_run_id_links(self) -> None:
        _write_synthetic_signal_cache(
            self.cache_dir, 999_001, date(2024, 1, 1),
            n_days=400, target_ann_vol_pct=15.0,
        )
        # Emit two regime_summary rows first.
        rs_a = emit_regime_summary(
            "2024-01-02", "2024-06-30",
            params=self._params(), cache_dir=self.cache_dir,
            audit_path=self.audit_path,
        )
        rs_b = emit_regime_summary(
            "2024-07-01", "2024-12-31",
            params=self._params(), cache_dir=self.cache_dir,
            audit_path=self.audit_path,
        )
        a = classify_window("2024-01-02", "2024-06-30",
                            params=self._params(), cache_dir=self.cache_dir)
        b = classify_window("2024-07-01", "2024-12-31",
                            params=self._params(), cache_dir=self.cache_dir)
        record = emit_drift_analysis(
            a, b,
            audit_path=self.audit_path,
            window_a_run_id=rs_a["event"]["run_id"],
            window_b_run_id=rs_b["event"]["run_id"],
        )
        self.assertEqual(record["event"]["window_a_run_id"],
                         rs_a["event"]["run_id"])
        self.assertEqual(record["event"]["window_b_run_id"],
                         rs_b["event"]["run_id"])


# ── Replay handlers ──────────────────────────────────────────────────


class ReplayHandlerTests(_RegimeHarness):

    def _params(self) -> ClassifierParams:
        return ClassifierParams(signal_scheme_code=999_001, min_coverage_days=60)

    def test_regime_summary_handler_registered(self) -> None:
        self.assertIn("regime_summary", rp.REPLAY_HANDLERS)

    def test_drift_analysis_handler_registered(self) -> None:
        self.assertIn("drift_analysis", rp.REPLAY_HANDLERS)

    def test_replay_regime_summary_reproduces_with_same_params(self) -> None:
        # Seed the LIVE classifier cache too — the replay handler
        # reads from the production cache by default. We patch the
        # default cache dir for the duration of the replay.
        _write_synthetic_signal_cache(
            self.cache_dir, 999_001, date(2024, 1, 1),
            n_days=400, target_ann_vol_pct=15.0,
        )
        record = emit_regime_summary(
            "2024-01-02", "2024-09-30",
            params=self._params(), cache_dir=self.cache_dir,
            audit_path=self.audit_path,
        )
        run_id = record["event"]["run_id"]

        # Patch the replay handler's default cache dir.
        from backend.regimes import classifier as cls_mod
        original_default = cls_mod.DEFAULT_CACHE_DIR
        cls_mod.DEFAULT_CACHE_DIR = self.cache_dir
        try:
            result = rp.replay_run(
                self.audit_path, run_id,
                verify_chain=False, emit_audit=False,
                registry_path=self.cache_dir,  # unused for this kind
            )
        finally:
            cls_mod.DEFAULT_CACHE_DIR = original_default

        self.assertIn(result["state"],
                      ("exact_match", "semantically_equivalent"),
                      f"got state={result['state']}, "
                      f"reason={result.get('reason')}")

    def test_replay_does_not_create_new_regime_summary_rows(self) -> None:
        _write_synthetic_signal_cache(
            self.cache_dir, 999_001, date(2024, 1, 1),
            n_days=400, target_ann_vol_pct=15.0,
        )
        record = emit_regime_summary(
            "2024-01-02", "2024-09-30",
            params=self._params(), cache_dir=self.cache_dir,
            audit_path=self.audit_path,
        )
        run_id = record["event"]["run_id"]
        rs_count_before = len(find_regime_summaries(
            audit_path=self.audit_path, include_superseded=True,
        ))

        from backend.regimes import classifier as cls_mod
        original_default = cls_mod.DEFAULT_CACHE_DIR
        cls_mod.DEFAULT_CACHE_DIR = self.cache_dir
        try:
            rp.replay_run(
                self.audit_path, run_id,
                verify_chain=False, emit_audit=False,
                registry_path=self.cache_dir,
            )
        finally:
            cls_mod.DEFAULT_CACHE_DIR = original_default

        rs_count_after = len(find_regime_summaries(
            audit_path=self.audit_path, include_superseded=True,
        ))
        self.assertEqual(rs_count_before, rs_count_after,
                         "replay must NOT create new regime_summary rows")


# ── Chain integrity ──────────────────────────────────────────────────


class ChainIntegrityTests(_RegimeHarness):

    def test_chain_valid_after_mixed_regime_sequence(self) -> None:
        _write_synthetic_signal_cache(
            self.cache_dir, 999_001, date(2024, 1, 1),
            n_days=400, target_ann_vol_pct=15.0,
        )
        params = ClassifierParams(signal_scheme_code=999_001,
                                  min_coverage_days=60)
        rs_a = emit_regime_summary(
            "2024-01-02", "2024-06-30",
            params=params, cache_dir=self.cache_dir,
            audit_path=self.audit_path,
        )
        rs_b = emit_regime_summary(
            "2024-07-01", "2024-12-31",
            params=params, cache_dir=self.cache_dir,
            audit_path=self.audit_path,
        )
        a = classify_window("2024-01-02", "2024-06-30",
                            params=params, cache_dir=self.cache_dir)
        b = classify_window("2024-07-01", "2024-12-31",
                            params=params, cache_dir=self.cache_dir)
        emit_drift_analysis(
            a, b,
            audit_path=self.audit_path,
            window_a_run_id=rs_a["event"]["run_id"],
            window_b_run_id=rs_b["event"]["run_id"],
        )
        # supersede the first regime_summary
        emit_regime_summary(
            "2024-01-02", "2024-06-30",
            params=params, cache_dir=self.cache_dir,
            audit_path=self.audit_path,
            supersedes_run_id=rs_a["event"]["run_id"],
        )
        self.assertTrue(verify_audit_chain(self.audit_path))


# ── Step 10 review tightenings — additional coverage ────────────────


class ConfidenceSplitAndFloorTests(_RegimeHarness):
    """Tightening #2: split classification_stability from
    boundary_separation; combined confidence floors at 0.5 when the
    underlying estimate is stable AND coverage_quality is high."""

    def _params(self) -> ClassifierParams:
        return ClassifierParams(signal_scheme_code=999_001,
                                min_coverage_days=60)

    def test_payload_exposes_stability_and_separation_separately(self) -> None:
        _write_synthetic_signal_cache(
            self.cache_dir, 999_001, date(2024, 1, 1),
            n_days=200, target_ann_vol_pct=15.0,
        )
        c = classify_window("2024-01-02", "2024-09-30",
                            params=self._params(),
                            cache_dir=self.cache_dir)
        from backend.regimes.classifier import classification_to_payload
        payload = classification_to_payload(c)
        self.assertIn("classification_stability", payload)
        self.assertIn("boundary_separation", payload)
        self.assertIn("classification_confidence", payload)
        # The three are independent numbers.
        self.assertGreaterEqual(payload["classification_stability"], 0.0)
        self.assertGreaterEqual(payload["boundary_separation"], 0.0)

    def test_floor_at_half_when_stable_and_high_coverage(self) -> None:
        # Build a window with vol close to a boundary BUT high
        # coverage_quality (deep history, low missing pct) and stable
        # estimate (≥ 2*min_coverage observations). Confidence must
        # floor at 0.5 — boundary geometry is not epistemic weakness.
        from backend.regimes.classifier import _classification_confidence
        params = self._params()
        # Window has plenty of observations (well above 2*min_coverage_days)
        # and "high" coverage quality.
        combined, stability, separation = _classification_confidence(
            vol_pct=12.5,  # 0.5pp from low/normal boundary at 12
            coverage_days=400,
            signal_quality={"coverage_quality": "high"},
            params=params,
        )
        # boundary_separation is raw: 0.5/10 = 0.05.
        self.assertLess(separation, 0.5)
        # stability saturates at 1.0 (400 >> 2*60).
        self.assertEqual(stability, 1.0)
        # combined: floor applied because stable + high coverage_quality.
        self.assertGreaterEqual(combined, 0.5)

    def test_no_floor_when_coverage_quality_medium(self) -> None:
        # Same boundary proximity, but coverage_quality medium → no
        # floor, combined confidence can erode to the separation value.
        from backend.regimes.classifier import _classification_confidence
        params = self._params()
        combined, stability, separation = _classification_confidence(
            vol_pct=12.5,
            coverage_days=400,
            signal_quality={"coverage_quality": "medium"},
            params=params,
        )
        # Combined matches raw separation (no floor); near boundary.
        self.assertEqual(combined, round(separation, 4))

    def test_no_floor_when_stability_below_saturation(self) -> None:
        # If the window itself is small (stability < 1.0), the floor
        # MUST NOT apply — even with high coverage_quality.
        from backend.regimes.classifier import _classification_confidence
        params = self._params()
        # coverage_days = min_coverage_days * 1.5 → stability = 0.75
        combined, stability, separation = _classification_confidence(
            vol_pct=12.5,
            coverage_days=int(params.min_coverage_days * 1.5),
            signal_quality={"coverage_quality": "high"},
            params=params,
        )
        self.assertLess(stability, 1.0)
        # Floor must not apply; combined ≤ stability AND ≤ separation.
        self.assertEqual(combined, min(round(separation, 4),
                                       round(stability, 4)))


class CoverageQualityUnitTests(_RegimeHarness):
    """Tightening #3: history_depth_days must count TRADING-DAY
    OBSERVATIONS, not calendar days. The previous mix-up compared
    calendar days against trading-day cutoffs (252*N), silently
    classifying clean windows as 'medium' coverage."""

    def test_history_depth_days_counts_trading_observations(self) -> None:
        # Seed exactly 500 trading-day observations into the synthetic
        # cache. history_depth_days for a window ending at the last
        # observation must equal 500.
        _write_synthetic_signal_cache(
            self.cache_dir, 999_001, date(2018, 1, 1),
            n_days=500, target_ann_vol_pct=15.0,
        )
        c = classify_window(
            "2018-01-02", "2020-12-31",  # window covers all observations
            params=ClassifierParams(signal_scheme_code=999_001,
                                    min_coverage_days=60),
            cache_dir=self.cache_dir,
        )
        # Trading-day count, not calendar days. With 500 observations
        # the history_depth_days must equal 500.
        self.assertEqual(c.signal_quality["history_depth_days"], 500)


class MagnitudeAwareTransitionConfidenceTests(unittest.TestCase):
    """Tightening #1: transition_confidence combines weaker-side
    classification_confidence with magnitude_factor (distance past the
    crossed boundary). Barely-crossed and decisive transitions can no
    longer collapse to the same value."""

    def _mk(self, klass: str, vol_pct: float, conf: float) -> RegimeClassification:
        return RegimeClassification(
            regime_class=klass,
            window_start_date="2024-01-01",
            window_end_date="2024-06-30",
            window_coverage_days=126,
            classification_confidence=conf,
            classification_stability=conf,
            boundary_separation=conf,
            classification_basis={
                "annualized_vol_pct": vol_pct,
                "applied_bands": {
                    "low_threshold_pct": 12.0,
                    "high_threshold_pct": 20.0,
                    "crisis_threshold_pct": 35.0,
                },
                "boundary_confidence_margin_pct": 10.0,
            },
            signal_quality={"coverage_quality": "high"},
        )

    def test_barely_crossed_has_near_zero_confidence(self) -> None:
        # 19.99 → 20.01 sits 0.01pp from the crossed boundary on
        # both sides. magnitude_factor = 0.01/10 = 0.001 → dominates.
        a = self._mk("normal_vol", 19.99, 0.9)
        b = self._mk("high_vol", 20.01, 0.9)
        d = compute_drift(a, b)
        self.assertTrue(d.regime_transition)
        self.assertLess(d.transition_confidence, 0.01)

    def test_decisive_two_band_crossing_has_high_confidence(self) -> None:
        # 8 → 40 crosses both 12 and 35 boundaries decisively.
        # weaker side here is bounded by the magnitude factor, not by
        # classification_confidence.
        a = self._mk("low_vol", 8.0, 0.9)
        b = self._mk("crisis_vol", 40.0, 0.9)
        d = compute_drift(a, b)
        self.assertTrue(d.regime_transition)
        self.assertGreater(d.transition_confidence, 0.3)

    def test_decisive_and_barely_crossed_do_not_collapse(self) -> None:
        # The whole point of magnitude-awareness: two transitions
        # with identical weaker-side classification_confidence must
        # land at very different transition_confidence values when
        # one is decisive and the other is borderline.
        weak_side_conf = 0.9
        barely = compute_drift(
            self._mk("normal_vol", 19.99, weak_side_conf),
            self._mk("high_vol", 20.01, weak_side_conf),
        )
        decisive = compute_drift(
            self._mk("low_vol", 8.0, weak_side_conf),
            self._mk("crisis_vol", 40.0, weak_side_conf),
        )
        self.assertLess(barely.transition_confidence,
                        decisive.transition_confidence / 10,
                        "barely-crossed and decisive transitions must "
                        "produce materially different confidences")


class RegimeSignatureTests(_RegimeHarness):
    """Tightening #6: regime_signature is a canonical identity tuple
    for the (signal, taxonomy, classifier, window) context. Step 11
    calibration joins per-regime statistics on this signature."""

    def test_signature_present_in_payload(self) -> None:
        _write_synthetic_signal_cache(
            self.cache_dir, 999_001, date(2024, 1, 1),
            n_days=200, target_ann_vol_pct=15.0,
        )
        c = classify_window("2024-01-02", "2024-09-30",
                            params=ClassifierParams(signal_scheme_code=999_001,
                                                    min_coverage_days=60),
                            cache_dir=self.cache_dir)
        from backend.regimes.classifier import classification_to_payload
        payload = classification_to_payload(c)
        self.assertIn("regime_signature", payload)
        sig = payload["regime_signature"]
        for key in ("signal_kind", "taxonomy_version", "classifier_version",
                    "window_hash"):
            self.assertIn(key, sig)
        # sha256 hex string is 64 chars.
        self.assertEqual(len(sig["window_hash"]), 64)

    def test_same_window_same_methodology_same_signature(self) -> None:
        # Two classifications under identical (signal, taxonomy,
        # classifier, window definition) MUST converge on identical
        # signatures.
        _write_synthetic_signal_cache(
            self.cache_dir, 999_001, date(2024, 1, 1),
            n_days=200, target_ann_vol_pct=15.0,
        )
        from backend.regimes.classifier import regime_signature
        params = ClassifierParams(signal_scheme_code=999_001, min_coverage_days=60)
        a = classify_window("2024-01-02", "2024-09-30",
                            params=params, cache_dir=self.cache_dir)
        b = classify_window("2024-01-02", "2024-09-30",
                            params=params, cache_dir=self.cache_dir)
        self.assertEqual(regime_signature(a), regime_signature(b))

    def test_different_bands_yield_different_signature(self) -> None:
        # Changing the applied bands (a classifier methodology change)
        # MUST produce a different signature even on the same window —
        # the "interpretation context" is different.
        _write_synthetic_signal_cache(
            self.cache_dir, 999_001, date(2024, 1, 1),
            n_days=200, target_ann_vol_pct=15.0,
        )
        from backend.regimes.classifier import regime_signature
        default = ClassifierParams(signal_scheme_code=999_001,
                                   min_coverage_days=60)
        shifted = ClassifierParams(signal_scheme_code=999_001,
                                   min_coverage_days=60,
                                   low_threshold_pct=10.0)
        a = classify_window("2024-01-02", "2024-09-30",
                            params=default, cache_dir=self.cache_dir)
        b = classify_window("2024-01-02", "2024-09-30",
                            params=shifted, cache_dir=self.cache_dir)
        self.assertNotEqual(regime_signature(a), regime_signature(b))


class PayloadLevelDriverTests(_RegimeHarness):
    """Tightening #4: replay must distinguish classifier-methodology
    drift from taxonomy drift. Both are payload-level provenance
    surfaces; the replay classifier surfaces each as a typed driver.
    """

    def _params(self) -> ClassifierParams:
        return ClassifierParams(signal_scheme_code=999_001,
                                min_coverage_days=60)

    def _seed_and_emit(self) -> str:
        _write_synthetic_signal_cache(
            self.cache_dir, 999_001, date(2024, 1, 1),
            n_days=400, target_ann_vol_pct=15.0,
        )
        record = emit_regime_summary(
            "2024-01-02", "2024-09-30",
            params=self._params(), cache_dir=self.cache_dir,
            audit_path=self.audit_path,
        )
        return record["event"]["run_id"]

    def test_identify_payload_drivers_taxonomy_change(self) -> None:
        from backend.evidence.replay import _identify_payload_drivers
        rec = {"taxonomy_version": "v1", "regime_classifier_version": "v1"}
        cur = {"taxonomy_version": "v2", "regime_classifier_version": "v1"}
        drivers = _identify_payload_drivers(rec, cur)
        kinds = [d["kind"] for d in drivers]
        self.assertIn("taxonomy_changed", kinds)
        self.assertNotIn("classifier_methodology_changed", kinds)

    def test_identify_payload_drivers_classifier_change(self) -> None:
        from backend.evidence.replay import _identify_payload_drivers
        rec = {"taxonomy_version": "v1", "regime_classifier_version": "v1"}
        cur = {"taxonomy_version": "v1", "regime_classifier_version": "v2"}
        drivers = _identify_payload_drivers(rec, cur)
        kinds = [d["kind"] for d in drivers]
        self.assertIn("classifier_methodology_changed", kinds)
        self.assertNotIn("taxonomy_changed", kinds)

    def test_identify_payload_drivers_skip_asymmetric_none(self) -> None:
        # Matches the established envelope-level convention: asymmetric
        # None means "one side did not expose this surface" rather than
        # "the value changed". Avoids noise when older evidence_kinds
        # don't carry the field at all.
        from backend.evidence.replay import _identify_payload_drivers
        rec = {"taxonomy_version": None}
        cur = {"taxonomy_version": "v1"}
        drivers = _identify_payload_drivers(rec, cur)
        kinds = [d["kind"] for d in drivers]
        self.assertNotIn("taxonomy_changed", kinds)

    def test_taxonomy_drift_surfaces_as_typed_driver_via_replay_run(self) -> None:
        # Integration check: end-to-end replay of a recorded
        # regime_summary, with the replay-side payload mutated to
        # simulate a taxonomy bump. The replay machinery must land
        # on expected_divergence with the taxonomy_changed driver.
        from backend.regimes import classifier as cls_mod
        run_id = self._seed_and_emit()
        original_default = cls_mod.DEFAULT_CACHE_DIR
        cls_mod.DEFAULT_CACHE_DIR = self.cache_dir
        original_to_payload = cls_mod.classification_to_payload

        def patched_to_payload(c):
            out = original_to_payload(c)
            out["taxonomy_version"] = "v999"
            return out

        try:
            with patch.object(cls_mod, "classification_to_payload",
                              new=patched_to_payload):
                result = rp.replay_run(
                    self.audit_path, run_id,
                    verify_chain=False, emit_audit=False,
                    registry_path=self.registry_path,
                )
        finally:
            cls_mod.DEFAULT_CACHE_DIR = original_default
        self.assertEqual(result["state"], "expected_divergence",
                         f"reason={result.get('reason')}")
        kinds = {d["kind"] for d in result["divergence_drivers"]}
        self.assertIn("taxonomy_changed", kinds)


class SupersedesDagProtectionTests(_RegimeHarness):
    """Tightening #5: supersedes must form a linear acyclic chain.
    Branching ("two rows both supersede A") and cycles are refused
    at emit time."""

    def _params(self) -> ClassifierParams:
        return ClassifierParams(signal_scheme_code=999_001, min_coverage_days=60)

    def _seed(self) -> None:
        _write_synthetic_signal_cache(
            self.cache_dir, 999_001, date(2024, 1, 1),
            n_days=400, target_ann_vol_pct=15.0,
        )

    def test_branching_supersedes_is_refused(self) -> None:
        self._seed()
        first = emit_regime_summary(
            "2024-01-02", "2024-09-30",
            params=self._params(), cache_dir=self.cache_dir,
            audit_path=self.audit_path,
        )
        first_id = first["event"]["run_id"]
        # First valid supersedes — fine.
        emit_regime_summary(
            "2024-01-02", "2024-09-30",
            params=self._params(), cache_dir=self.cache_dir,
            audit_path=self.audit_path,
            supersedes_run_id=first_id,
        )
        # Second attempt to supersede the SAME prior row creates a
        # branch (two rows both pointing at first_id). Must refuse.
        with self.assertRaises(ValueError) as ctx:
            emit_regime_summary(
                "2024-01-02", "2024-09-30",
                params=self._params(), cache_dir=self.cache_dir,
                audit_path=self.audit_path,
                supersedes_run_id=first_id,
            )
        self.assertIn("already superseded", str(ctx.exception))

    def test_supersedes_chain_remains_linear_under_repeated_updates(self) -> None:
        # The legitimate methodology-update pattern: each supersedes
        # the immediately-prior row, never an older one. Must be
        # accepted (no cycle, no branch).
        self._seed()
        a = emit_regime_summary(
            "2024-01-02", "2024-09-30",
            params=self._params(), cache_dir=self.cache_dir,
            audit_path=self.audit_path,
        )
        b = emit_regime_summary(
            "2024-01-02", "2024-09-30",
            params=self._params(), cache_dir=self.cache_dir,
            audit_path=self.audit_path,
            supersedes_run_id=a["event"]["run_id"],
        )
        c = emit_regime_summary(
            "2024-01-02", "2024-09-30",
            params=self._params(), cache_dir=self.cache_dir,
            audit_path=self.audit_path,
            supersedes_run_id=b["event"]["run_id"],
        )
        rows = find_regime_summaries(audit_path=self.audit_path,
                                     include_superseded=True)
        self.assertEqual(len(rows), 3)
        # Filter to current claim (not-superseded) — exactly one.
        current = find_regime_summaries(audit_path=self.audit_path,
                                        include_superseded=False)
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0]["run_id"], c["event"]["run_id"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
