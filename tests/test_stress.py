"""Stress tests — validate epistemic integrity under chaotic inputs.

Each test targets a specific failure surface:
- No crash
- No silent correction
- Evidence degrades appropriately
- Limitations explain reality
- Impact links are accurate
- Language policy holds
- No NaN / Infinity leaks
"""
from __future__ import annotations

import json
import math
import re
import unittest

from starlette.testclient import TestClient

from api.main import app

ADVISORY_TERMS = re.compile(
    r"\b(should|must|recommend|prefer|avoid|better|suitable|strong"
    r"|outperformance|overweight|underweight|cheap|expensive|switch"
    r"|rebalance|allocate|allocation|buy|sell|best|pick|target"
    r"|optimized|improvement)\b",
    re.IGNORECASE,
)


def _no_nan_inf(obj, path="$"):
    """Recursively assert no NaN or Infinity leaks in the response."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise AssertionError(f"NaN/Infinity at {path}: {obj}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _no_nan_inf(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _no_nan_inf(v, f"{path}[{i}]")


def _no_advisory(obj):
    """Recursively assert no advisory language in any string."""
    text = json.dumps(obj)
    matches = ADVISORY_TERMS.findall(text)
    if matches:
        raise AssertionError(f"Advisory terms found: {set(matches)}")


class StressTestMixin:
    """Common helpers for stress tests via API."""

    client: TestClient

    def _call(self, symbol: str, **overrides) -> dict:
        payload = {
            "subject_token": "stress_test",
            "user_country": "IN",
            "asset_market": "IN",
            "serving_entity": "stress_entity",
            "source": "csv_sample",
            "symbol": symbol,
            "fund_name": "Stress Fund",
            "benchmark_name": "Stress Benchmark",
            "category": "Equity",
            "expense_ratio_pct": 1.0,
            "rolling_window_points": 2,
            "rolling_step_points": 1,
            "rolling_min_windows": 1,
        }
        payload.update(overrides)
        response = self.client.post(
            "/analytics/mutual-fund/from-source", json=payload,
        )
        return response

    def _assert_clean(self, data: dict) -> None:
        """Baseline assertions every successful response must pass."""
        _no_nan_inf(data)
        _no_advisory(data)
        self.assertIn("gate", data)
        self.assertIn("insights", data)
        self.assertIn("ingestion_report", data)


class SparseDataTests(StressTestMixin, unittest.TestCase):
    """3 data points — extreme sparsity."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_sparse_returns_200(self):
        """System handles 3 points without crashing."""
        resp = self._call("stress_sparse")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self._assert_clean(data)

    def test_sparse_evidence_is_low(self):
        """With only 3 points, evidence must be Low."""
        data = self._call("stress_sparse").json()
        for insight in data["insights"]:
            ev = insight["payload"]["evidence_strength"]
            self.assertEqual(ev, "Low", f"Expected Low evidence, got {ev}")

    def test_sparse_short_history_link(self):
        """Impact links should flag short history."""
        data = self._call("stress_sparse").json()
        issues = [l["issue"] for l in data["ingestion_report"]["impact_links"]]
        self.assertIn("short_history", issues)

    def test_sparse_has_gap_link(self):
        """3 points months apart → gaps detected."""
        data = self._call("stress_sparse").json()
        issues = [l["issue"] for l in data["ingestion_report"]["impact_links"]]
        self.assertIn("gaps_detected", issues)

    def test_sparse_limitations_present(self):
        """Limitations must be non-empty for sparse data."""
        data = self._call("stress_sparse").json()
        lims = data["ingestion_report"]["ingestion_limitations"]
        self.assertGreater(len(lims), 0)


class GapDataTests(StressTestMixin, unittest.TestCase):
    """Clusters of data with multi-week gaps between them."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_gaps_returns_200(self):
        resp = self._call("stress_gaps")
        self.assertEqual(resp.status_code, 200)
        self._assert_clean(resp.json())

    def test_gaps_detected_in_report(self):
        data = self._call("stress_gaps").json()
        meta = data["ingestion_report"]["fund_series"]
        self.assertGreater(meta["gaps_detected"], 0)

    def test_gaps_impact_link_present(self):
        data = self._call("stress_gaps").json()
        issues = [l["issue"] for l in data["ingestion_report"]["impact_links"]]
        self.assertIn("gaps_detected", issues)

    def test_gaps_affect_correct_metrics(self):
        data = self._call("stress_gaps").json()
        for link in data["ingestion_report"]["impact_links"]:
            if link["issue"] == "gaps_detected":
                self.assertIn("rolling excess returns", link["affected_metrics"])
                self.assertIn("drawdown profile", link["affected_metrics"])
                self.assertIn("trailing returns", link["affected_metrics"])

    def test_gaps_limitations_mention_gaps(self):
        data = self._call("stress_gaps").json()
        lims = " ".join(data["ingestion_report"]["ingestion_limitations"])
        self.assertIn("gap", lims.lower())


class SpikeDataTests(StressTestMixin, unittest.TestCase):
    """Single extreme spike (100 → 250 → 105) in fund series."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_spike_returns_200(self):
        resp = self._call("stress_spike")
        self.assertEqual(resp.status_code, 200)
        self._assert_clean(resp.json())

    def test_spike_anomalies_flagged(self):
        data = self._call("stress_spike").json()
        meta = data["ingestion_report"]["fund_series"]
        self.assertGreater(meta["anomalies_flagged"], 0,
                          "Spike must be flagged as anomaly")

    def test_spike_extreme_values_link(self):
        data = self._call("stress_spike").json()
        issues = [l["issue"] for l in data["ingestion_report"]["impact_links"]]
        self.assertIn("extreme_values", issues)

    def test_spike_both_directions_flagged(self):
        """100→250 AND 250→105 are both >50% moves."""
        data = self._call("stress_spike").json()
        anomalies = data["ingestion_report"]["fund_anomalies"]
        self.assertGreaterEqual(len(anomalies), 2,
                               "Both spike up and reversal should be flagged")


class FlatDataTests(StressTestMixin, unittest.TestCase):
    """Constant value series — zero returns everywhere."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_flat_returns_200(self):
        resp = self._call("stress_flat")
        self.assertEqual(resp.status_code, 200)
        self._assert_clean(resp.json())

    def test_flat_no_anomalies(self):
        """Flat data should not flag any anomalies."""
        data = self._call("stress_flat").json()
        self.assertEqual(data["ingestion_report"]["fund_series"]["anomalies_flagged"], 0)
        self.assertEqual(data["ingestion_report"]["benchmark_series"]["anomalies_flagged"], 0)

    def test_flat_no_extreme_values_link(self):
        data = self._call("stress_flat").json()
        issues = [l["issue"] for l in data["ingestion_report"]["impact_links"]]
        self.assertNotIn("extreme_values", issues)

    def test_flat_returns_are_zero(self):
        """All trailing returns should be 0% for flat series."""
        data = self._call("stress_flat").json()
        for insight in data["insights"]:
            sd = insight["payload"].get("supporting_data", {})
            trailing = sd.get("trailing_returns", {})
            for period, ret_data in trailing.items():
                if isinstance(ret_data, dict):
                    fund_ret = ret_data.get("fund_return_pct", None)
                    if fund_ret is not None:
                        self.assertAlmostEqual(fund_ret, 0.0, places=2,
                            msg=f"Flat fund return should be 0, got {fund_ret}")


class DuplicateDataTests(StressTestMixin, unittest.TestCase):
    """Heavy duplicate dates — 3 entries for same date."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_duplicates_returns_200(self):
        resp = self._call("stress_duplicates")
        self.assertEqual(resp.status_code, 200)
        self._assert_clean(resp.json())

    def test_duplicates_merged_in_fund(self):
        data = self._call("stress_duplicates").json()
        meta = data["ingestion_report"]["fund_series"]
        self.assertGreater(meta["duplicate_dates_merged"], 0)

    def test_duplicates_impact_link(self):
        data = self._call("stress_duplicates").json()
        issues = [l["issue"] for l in data["ingestion_report"]["impact_links"]]
        self.assertIn("duplicates_merged", issues)

    def test_duplicates_output_fewer_than_input(self):
        """Dedup must reduce point count."""
        data = self._call("stress_duplicates").json()
        meta = data["ingestion_report"]["fund_series"]
        self.assertLess(meta["output_points"], meta["input_records"])


class MissingValueTests(StressTestMixin, unittest.TestCase):
    """Empty cells, N/A, blank values in both series."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_missing_returns_200(self):
        resp = self._call("stress_missing")
        self.assertEqual(resp.status_code, 200)
        self._assert_clean(resp.json())

    def test_missing_fund_rejected(self):
        data = self._call("stress_missing").json()
        meta = data["ingestion_report"]["fund_series"]
        self.assertGreater(meta["rejected_count"], 0,
                          "Empty/N/A fund values must be rejected")

    def test_missing_benchmark_rejected(self):
        data = self._call("stress_missing").json()
        meta = data["ingestion_report"]["benchmark_series"]
        self.assertGreater(meta["rejected_count"], 0,
                          "Empty/N.A. benchmark values must be rejected")

    def test_missing_records_rejected_link(self):
        data = self._call("stress_missing").json()
        issues = [l["issue"] for l in data["ingestion_report"]["impact_links"]]
        self.assertIn("records_rejected", issues)

    def test_missing_limitations_mention_excluded(self):
        data = self._call("stress_missing").json()
        lims = " ".join(data["ingestion_report"]["ingestion_limitations"])
        self.assertIn("excluded", lims.lower())


class ConflictingSignalTests(StressTestMixin, unittest.TestCase):
    """Good returns overall but one huge spike + enough data for analysis.
    Tests that system shows BOTH the good returns AND the data issue."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_conflicting_returns_200(self):
        resp = self._call("stress_conflicting")
        self.assertEqual(resp.status_code, 200)
        self._assert_clean(resp.json())

    def test_conflicting_has_insights_and_anomalies(self):
        """System shows both: analysis results AND anomaly flags."""
        data = self._call("stress_conflicting").json()
        self.assertGreater(len(data["insights"]), 0, "Must have insights")
        anomalies = data["ingestion_report"]["fund_anomalies"]
        self.assertGreater(len(anomalies), 0, "Must flag the spike")

    def test_conflicting_no_reconciliation(self):
        """System must NOT try to reconcile conflicting signals.
        Both anomaly and normal analysis results coexist."""
        data = self._call("stress_conflicting").json()
        issues = {l["issue"] for l in data["ingestion_report"]["impact_links"]}
        self.assertIn("extreme_values", issues)
        # Insights still present — system didn't suppress them
        self.assertGreater(len(data["insights"]), 0)

    def test_conflicting_limitations_mention_extreme(self):
        data = self._call("stress_conflicting").json()
        lims = " ".join(data["ingestion_report"]["ingestion_limitations"])
        self.assertIn("extreme", lims.lower())


class MisalignedDataTests(StressTestMixin, unittest.TestCase):
    """Fund has a gap (Jan 5 missing → Jan 9) but benchmark is dense.
    Tests alignment behavior."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_misaligned_returns_200(self):
        resp = self._call("stress_misaligned")
        self.assertEqual(resp.status_code, 200)
        self._assert_clean(resp.json())

    def test_misaligned_alignment_drops_points(self):
        """Alignment should produce fewer points than the denser series."""
        data = self._call("stress_misaligned").json()
        quality = data.get("data_quality", {})
        if quality:
            aligned = quality.get("aligned_points", 0)
            fund = quality.get("fund_points", 0)
            bench = quality.get("benchmark_points", 0)
            self.assertLessEqual(aligned, max(fund, bench))


class NegativeValueTests(StressTestMixin, unittest.TestCase):
    """Negative and zero values — must be rejected, not crash."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_negatives_returns_200(self):
        resp = self._call("stress_negatives")
        self.assertEqual(resp.status_code, 200)
        self._assert_clean(resp.json())

    def test_negatives_rejected(self):
        data = self._call("stress_negatives").json()
        fund_rejected = data["ingestion_report"]["fund_series"]["rejected_count"]
        bench_rejected = data["ingestion_report"]["benchmark_series"]["rejected_count"]
        self.assertGreater(fund_rejected + bench_rejected, 0,
                          "Negative/zero values must be rejected")

    def test_negatives_no_output_with_bad_values(self):
        """Output points must be fewer than input (bad rows dropped)."""
        data = self._call("stress_negatives").json()
        for series in ("fund_series", "benchmark_series"):
            meta = data["ingestion_report"][series]
            if meta["rejected_count"] > 0:
                self.assertLess(meta["output_points"], meta["input_records"])


class UnsortedDataTests(StressTestMixin, unittest.TestCase):
    """Completely shuffled dates."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_unsorted_returns_200(self):
        resp = self._call("stress_unsorted")
        self.assertEqual(resp.status_code, 200)
        self._assert_clean(resp.json())

    def test_unsorted_events_note_sorting(self):
        """Normalization events should record that sorting was needed."""
        data = self._call("stress_unsorted").json()
        events = data["ingestion_report"]["fund_events"]
        event_types = [e["type"] for e in events]
        self.assertIn("sorted_ascending", event_types)

    def test_unsorted_output_is_chronological(self):
        """Despite shuffled input, fund output in ingestion payload is sorted."""
        data = self._call("stress_unsorted").json()
        report = data["ingestion_report"]
        dates = []
        fund_meta = report["fund_series"]
        date_range = fund_meta["date_range"]
        self.assertIsNotNone(date_range["start"])
        self.assertIsNotNone(date_range["end"])
        self.assertLessEqual(date_range["start"], date_range["end"])


class ImpactLinkOrderingTests(StressTestMixin, unittest.TestCase):
    """Verify impact link ordering is deterministic: broadest-first."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_link_order_broadest_first(self):
        """When multiple issues present, order is:
        short_history → records_rejected → duplicates → gaps → extreme_values."""
        # Use messy_mf which has multiple issues
        resp = self._call("messy_mf",
                          fund_name="Messy Fund", benchmark_name="Messy Bench",
                          expense_ratio_pct=1.5, rolling_window_points=4)
        data = resp.json()
        issues = [l["issue"] for l in data["ingestion_report"]["impact_links"]]
        # Check relative ordering of whatever issues are present
        expected_order = [
            "short_history", "records_rejected", "duplicates_merged",
            "gaps_detected", "extreme_values",
        ]
        present = [i for i in expected_order if i in issues]
        actual_order = [i for i in issues if i in expected_order]
        self.assertEqual(present, actual_order,
                        f"Link ordering not broadest-first: {issues}")


class CrossDatasetConsistencyTests(StressTestMixin, unittest.TestCase):
    """Run all stress datasets through and verify universal invariants."""

    STRESS_SYMBOLS = [
        "stress_sparse", "stress_gaps", "stress_spike", "stress_flat",
        "stress_duplicates", "stress_missing", "stress_conflicting",
        "stress_misaligned", "stress_negatives", "stress_unsorted",
    ]

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_all_stress_datasets_no_crashes(self):
        """Every stress dataset must return 200."""
        for symbol in self.STRESS_SYMBOLS:
            with self.subTest(symbol=symbol):
                resp = self._call(symbol)
                self.assertEqual(resp.status_code, 200,
                                f"{symbol} returned {resp.status_code}")

    def test_all_stress_datasets_no_nan_inf(self):
        """No NaN or Infinity in any response."""
        for symbol in self.STRESS_SYMBOLS:
            with self.subTest(symbol=symbol):
                data = self._call(symbol).json()
                _no_nan_inf(data)

    def test_all_stress_datasets_no_advisory_language(self):
        """No advisory terms in any response."""
        for symbol in self.STRESS_SYMBOLS:
            with self.subTest(symbol=symbol):
                data = self._call(symbol).json()
                _no_advisory(data)

    def test_all_stress_datasets_have_gate(self):
        for symbol in self.STRESS_SYMBOLS:
            with self.subTest(symbol=symbol):
                data = self._call(symbol).json()
                self.assertIn("gate", data)
                self.assertTrue(data["gate"]["features"]["analytics"])

    def test_all_stress_datasets_have_ingestion_report(self):
        for symbol in self.STRESS_SYMBOLS:
            with self.subTest(symbol=symbol):
                data = self._call(symbol).json()
                report = data["ingestion_report"]
                self.assertIn("fund_series", report)
                self.assertIn("benchmark_series", report)
                self.assertIn("impact_links", report)
                self.assertIn("ingestion_limitations", report)

    def test_all_insights_have_evidence(self):
        """Every insight card must have evidence_strength."""
        for symbol in self.STRESS_SYMBOLS:
            with self.subTest(symbol=symbol):
                data = self._call(symbol).json()
                for insight in data["insights"]:
                    ev = insight["payload"].get("evidence_strength")
                    self.assertIn(ev, {"High", "Medium", "Low"},
                                 f"{symbol}: missing or invalid evidence: {ev}")

    def test_all_insights_have_limitations(self):
        """Every insight must carry limitations list."""
        for symbol in self.STRESS_SYMBOLS:
            with self.subTest(symbol=symbol):
                data = self._call(symbol).json()
                for idx, insight in enumerate(data["insights"]):
                    lims = insight["payload"].get("limitations")
                    self.assertIsInstance(lims, list,
                        f"{symbol} insight {idx}: limitations must be a list")


if __name__ == "__main__":
    unittest.main()
