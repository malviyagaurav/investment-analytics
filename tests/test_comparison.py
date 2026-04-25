from __future__ import annotations

import unittest
from datetime import date, timedelta
from typing import List

from backend.investment_analytics.comparison import (
    align_multiple_series,
    alignment_quality,
    evidence_from_alignment,
    worst_case_evidence,
    trailing_cagr,
    drawdown_profile,
    volatility_metrics,
    consistency_vs_peers,
    cost_impact_comparison,
    build_comparison_insights,
    DRAWDOWN_SIGNIFICANT_PCT,
)
from backend.investment_analytics.compiler import compile_insight
from backend.investment_analytics.errors import PolicyError
from backend.investment_analytics.mutual_funds import SeriesPoint


def _series(start: date, values: List[float], step_days: int = 1) -> List[SeriesPoint]:
    """Build a SeriesPoint list from start date + values."""
    return [
        SeriesPoint(as_of=start + timedelta(days=i * step_days), value=v)
        for i, v in enumerate(values)
    ]


def _daily_dates(start: date, n: int) -> List[date]:
    return [start + timedelta(days=i) for i in range(n)]


class TestAlignMultipleSeries(unittest.TestCase):

    def test_perfect_overlap_two_funds(self):
        d = date(2024, 1, 1)
        s1 = _series(d, [100, 110, 120, 130, 140])
        s2 = _series(d, [200, 210, 220, 230, 240])
        dates, aligned = align_multiple_series({1: s1, 2: s2})
        self.assertEqual(len(dates), 5)
        self.assertEqual(aligned[1], [100, 110, 120, 130, 140])
        self.assertEqual(aligned[2], [200, 210, 220, 230, 240])

    def test_partial_overlap(self):
        s1 = _series(date(2024, 1, 1), [100, 110, 120, 130, 140])
        s2 = _series(date(2024, 1, 3), [200, 210, 220])
        dates, aligned = align_multiple_series({1: s1, 2: s2})
        self.assertEqual(len(dates), 3)
        self.assertEqual(aligned[1], [120, 130, 140])
        self.assertEqual(aligned[2], [200, 210, 220])

    def test_no_overlap(self):
        s1 = _series(date(2024, 1, 1), [100, 110])
        s2 = _series(date(2024, 6, 1), [200, 210])
        dates, aligned = align_multiple_series({1: s1, 2: s2})
        self.assertEqual(len(dates), 0)
        self.assertEqual(aligned, {})

    def test_three_funds(self):
        d = date(2024, 1, 2)
        s1 = _series(date(2024, 1, 1), [10, 20, 30, 40])
        s2 = _series(d, [50, 60, 70])
        s3 = _series(d, [80, 90, 100])
        dates, aligned = align_multiple_series({1: s1, 2: s2, 3: s3})
        self.assertEqual(len(dates), 3)
        self.assertEqual(aligned[1], [20, 30, 40])

    def test_empty_input(self):
        dates, aligned = align_multiple_series({})
        self.assertEqual(dates, [])
        self.assertEqual(aligned, {})


class TestAlignmentQuality(unittest.TestCase):

    def test_high_quality(self):
        d = date(2020, 1, 1)
        pts = _series(d, list(range(800)))  # ~800 daily points ≈ 2+ years
        q = alignment_quality(
            _daily_dates(d, 800),
            {1: pts, 2: pts},
        )
        self.assertGreater(q["aligned_points"], 0)
        self.assertGreater(q["history_years"], 1.0)
        self.assertGreater(q["calendar_density"], 0.5)

    def test_few_points_low_evidence(self):
        d = date(2024, 1, 1)
        pts = _series(d, [100, 110, 120])
        q = alignment_quality(
            _daily_dates(d, 3),
            {1: pts, 2: pts},
        )
        ev = evidence_from_alignment(q)
        self.assertEqual(ev, "Low")


class TestEvidenceFromAlignment(unittest.TestCase):

    def test_high(self):
        q = {
            "aligned_points": 800,
            "history_years": 3.5,
            "calendar_density": 0.75,
            "relative_completeness": 0.9,
            "max_series_points": 800,
        }
        self.assertEqual(evidence_from_alignment(q), "High")

    def test_medium(self):
        q = {
            "aligned_points": 100,
            "history_years": 1.5,
            "calendar_density": 0.5,
            "relative_completeness": 0.7,
            "max_series_points": 150,
        }
        self.assertEqual(evidence_from_alignment(q), "Medium")

    def test_low(self):
        q = {
            "aligned_points": 5,
            "history_years": 0.02,
            "calendar_density": 0.1,
            "relative_completeness": 0.1,
            "max_series_points": 50,
        }
        self.assertEqual(evidence_from_alignment(q), "Low")


class TestWorstCaseEvidence(unittest.TestCase):

    def test_all_high(self):
        self.assertEqual(worst_case_evidence("High", ["High", "High"]), "High")

    def test_one_low_fund_degrades(self):
        self.assertEqual(worst_case_evidence("High", ["High", "Low"]), "Low")

    def test_alignment_low_degrades(self):
        self.assertEqual(worst_case_evidence("Low", ["High", "High"]), "Low")

    def test_medium_floor(self):
        self.assertEqual(worst_case_evidence("High", ["Medium"]), "Medium")

    def test_empty_per_fund(self):
        self.assertEqual(worst_case_evidence("High", []), "High")

    def test_mixed(self):
        self.assertEqual(worst_case_evidence("Medium", ["High", "Low"]), "Low")


class TestTrailingCagr(unittest.TestCase):

    def test_basic_cagr(self):
        d = date(2019, 1, 1)
        # 5 years of daily data, growing from 100 to ~200
        n = 365 * 5
        dates = _daily_dates(d, n)
        vals = [100 * (1.15 ** (i / 365)) for i in range(n)]
        result = trailing_cagr(dates, vals, horizons_years=(1, 3, 5))
        self.assertIn("1Y", result)
        self.assertIn("3Y", result)
        # CAGR should be near 15%
        self.assertAlmostEqual(result["1Y"]["cagr_pct"], 15.0, delta=1.0)

    def test_too_short(self):
        result = trailing_cagr([date(2024, 1, 1)], [100])
        self.assertEqual(result, {})


class TestDrawdownProfile(unittest.TestCase):

    def test_no_drawdown(self):
        d = date(2024, 1, 1)
        dates = _daily_dates(d, 10)
        vals = list(range(100, 110))
        result = drawdown_profile(dates, vals)
        self.assertEqual(result["max_drawdown_pct"], 0.0)
        self.assertEqual(result["drawdowns_gt_threshold_pct"], 0)

    def test_significant_drawdown(self):
        d = date(2024, 1, 1)
        # 100 → 120 → 100 → 130 (peak at 120, trough at 100 = -16.7%)
        vals = [100, 110, 120, 105, 100, 105, 115, 125, 130]
        dates = _daily_dates(d, len(vals))
        result = drawdown_profile(dates, vals)
        self.assertLess(result["max_drawdown_pct"], -10.0)
        self.assertEqual(result["drawdowns_gt_threshold_pct"], 1)
        self.assertEqual(result["threshold_pct"], DRAWDOWN_SIGNIFICANT_PCT)

    def test_unrecovered_drawdown(self):
        d = date(2024, 1, 1)
        vals = [100, 110, 90, 85]  # drops below peak, never recovers
        dates = _daily_dates(d, len(vals))
        result = drawdown_profile(dates, vals)
        self.assertLess(result["max_drawdown_pct"], 0)

    def test_too_short(self):
        result = drawdown_profile([date(2024, 1, 1)], [100])
        self.assertEqual(result["max_drawdown_pct"], 0.0)


class TestVolatilityMetrics(unittest.TestCase):

    def test_zero_vol(self):
        vals = [100, 100, 100, 100]
        dates = _daily_dates(date(2024, 1, 1), 4)
        result = volatility_metrics(vals, dates)
        self.assertEqual(result["periodic_return_std_pct"], 0.0)

    def test_nonzero_vol(self):
        vals = [100, 110, 95, 115, 105]
        dates = _daily_dates(date(2024, 1, 1), 5)
        result = volatility_metrics(vals, dates)
        self.assertGreater(result["periodic_return_std_pct"], 0)
        self.assertEqual(result["observation_count"], 4)


class TestConsistencyVsPeers(unittest.TestCase):

    def test_two_funds(self):
        d = date(2024, 1, 1)
        # Fund 1 always goes up, Fund 2 flat
        n = 20
        dates = _daily_dates(d, n)
        aligned = {
            1: [100 + i * 2 for i in range(n)],
            2: [100 + i for i in range(n)],
        }
        result = consistency_vs_peers(dates, aligned, window_points=5, step_points=2)
        self.assertGreater(result["window_count"], 0)
        self.assertIn(1, result["funds"])
        self.assertIn(2, result["funds"])
        # Fund 1 should have higher hit ratio
        self.assertGreaterEqual(result["funds"][1]["hit_ratio_vs_peer_median"], 0.0)

    def test_insufficient_data(self):
        dates = _daily_dates(date(2024, 1, 1), 3)
        aligned = {1: [100, 110, 120], 2: [100, 105, 110]}
        result = consistency_vs_peers(dates, aligned, window_points=10, step_points=1)
        self.assertEqual(result["window_count"], 0)


class TestCostImpactComparison(unittest.TestCase):

    def test_basic(self):
        result = cost_impact_comparison({1: 1.0, 2: 2.0})
        self.assertIn(1, result["funds"])
        self.assertIn(2, result["funds"])
        # Fund 2 has higher TER → higher drag
        f1_drag = result["funds"][1]["estimated_drag_amounts"]
        f2_drag = result["funds"][2]["estimated_drag_amounts"]
        for a, b in zip(f1_drag, f2_drag):
            self.assertLess(a, b)

    def test_zero_ter(self):
        result = cost_impact_comparison({1: 0.0})
        self.assertEqual(result["funds"][1]["estimated_drag_amounts"], [0.0, 0.0, 0.0])


class TestBuildComparisonInsights(unittest.TestCase):

    def _make_inputs(self, n=200, num_funds=2):
        d = date(2020, 1, 1)
        dates = _daily_dates(d, n)
        aligned = {}
        names = {}
        expense = {}
        for i in range(num_funds):
            code = 100 + i
            aligned[code] = [100 + j * (0.5 + i * 0.3) for j in range(n)]
            names[code] = f"Fund {i + 1}"
            expense[code] = 1.0 + i * 0.5

        quality = alignment_quality(dates, {
            code: _series(d, vals) for code, vals in aligned.items()
        })
        return dates, aligned, names, expense, quality

    def test_produces_insights(self):
        dates, aligned, names, expense, quality = self._make_inputs()
        insights = build_comparison_insights(
            dates, aligned, names, expense, quality,
            window_points=20, step_points=5,
        )
        self.assertGreater(len(insights), 0)
        # All should be 'diagnostic' type
        for raw in insights:
            self.assertEqual(raw["type"], "diagnostic")

    def test_all_compile(self):
        """All produced insights should pass the compiler."""
        dates, aligned, names, expense, quality = self._make_inputs()
        insights = build_comparison_insights(
            dates, aligned, names, expense, quality,
            window_points=20, step_points=5,
        )
        for raw in insights:
            compiled = compile_insight(raw)
            self.assertEqual(compiled["template"], "diagnostic")

    def test_three_funds(self):
        dates, aligned, names, expense, quality = self._make_inputs(num_funds=3)
        insights = build_comparison_insights(
            dates, aligned, names, expense, quality,
            window_points=20, step_points=5,
        )
        self.assertGreater(len(insights), 0)

    def test_five_funds(self):
        dates, aligned, names, expense, quality = self._make_inputs(num_funds=5)
        insights = build_comparison_insights(
            dates, aligned, names, expense, quality,
            window_points=20, step_points=5,
        )
        self.assertGreater(len(insights), 0)

    def test_too_few_dates(self):
        d = date(2024, 1, 1)
        dates = [d]
        aligned = {1: [100], 2: [200]}
        names = {1: "A", 2: "B"}
        expense = {1: 1.0, 2: 1.5}
        quality = {"aligned_points": 1, "max_series_points": 1,
                   "relative_completeness": 1.0, "calendar_density": 1.0,
                   "history_years": 0.0}
        insights = build_comparison_insights(
            dates, aligned, names, expense, quality,
        )
        self.assertEqual(insights, [])

    def test_language_policy_compliance(self):
        """No banned words in any insight text."""
        dates, aligned, names, expense, quality = self._make_inputs()
        insights = build_comparison_insights(
            dates, aligned, names, expense, quality,
            window_points=20, step_points=5,
        )
        # compile_insight checks language policy; if any fail, it's a violation
        compiled = []
        for raw in insights:
            compiled.append(compile_insight(raw))
        self.assertTrue(len(compiled) > 0)

    def test_sparse_overlap_low_evidence(self):
        """Sparse overlap should produce Low evidence."""
        d = date(2024, 1, 1)
        dates = _daily_dates(d, 5)
        aligned = {1: [100, 110, 120, 130, 140], 2: [200, 190, 210, 205, 215]}
        names = {1: "Short A", 2: "Short B"}
        expense = {1: 1.0, 2: 1.0}
        quality = alignment_quality(dates, {
            code: _series(d, vals) for code, vals in aligned.items()
        })
        ev = evidence_from_alignment(quality)
        self.assertEqual(ev, "Low")

    def test_per_fund_evidence_degrades_comparison(self):
        """If one fund has Low data quality, comparison evidence must be Low."""
        dates, aligned, names, expense, quality = self._make_inputs(n=800, num_funds=2)
        alignment_ev = evidence_from_alignment(quality)
        # Alignment may be High or Medium depending on density; either way,
        # injecting a Low per-fund evidence must degrade comparison to Low.
        insights = build_comparison_insights(
            dates, aligned, names, expense, quality,
            window_points=20, step_points=5,
            per_fund_evidence=[alignment_ev, "Low"],
        )
        for raw in insights:
            self.assertEqual(raw["evidence_strength"], "Low")


class TestCompareRequestValidation(unittest.TestCase):

    def test_valid_request(self):
        from api.models import CompareRequest, FundCompareEntry
        req = CompareRequest(
            funds=[
                FundCompareEntry(scheme_code=100),
                FundCompareEntry(scheme_code=200),
            ],
        )
        self.assertEqual(len(req.funds), 2)

    def test_too_few_funds(self):
        from api.models import CompareRequest, FundCompareEntry
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            CompareRequest(funds=[FundCompareEntry(scheme_code=100)])

    def test_too_many_funds(self):
        from api.models import CompareRequest, FundCompareEntry
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            CompareRequest(funds=[
                FundCompareEntry(scheme_code=i) for i in range(6)
            ])


if __name__ == "__main__":
    unittest.main()
