from __future__ import annotations

import unittest
from datetime import date, timedelta
from typing import Dict, List

from backend.investment_analytics.portfolio import (
    compute_portfolio_series,
    rolling_period_returns,
    return_distribution,
    _daily_returns,
    _pearson_correlation,
    pairwise_correlations,
    _correlation_label,
    concentration_metrics,
    contribution_analysis,
    build_portfolio_insights,
)
from backend.investment_analytics.comparison import (
    align_multiple_series,
    alignment_quality,
    drawdown_profile,
)
from backend.investment_analytics.compiler import compile_insight
from backend.investment_analytics.errors import PolicyError
from backend.investment_analytics.language_policy import lint_text_tree
from backend.investment_analytics.mutual_funds import SeriesPoint


def _series(start: date, values: List[float], step_days: int = 1) -> List[SeriesPoint]:
    return [
        SeriesPoint(as_of=start + timedelta(days=i * step_days), value=v)
        for i, v in enumerate(values)
    ]


def _daily_dates(start: date, n: int) -> List[date]:
    return [start + timedelta(days=i) for i in range(n)]


def _flat_series(start: date, n: int, value: float = 100.0) -> List[SeriesPoint]:
    return _series(start, [value] * n)


def _rising_series(start: date, n: int, rate: float = 0.10) -> List[SeriesPoint]:
    return _series(start, [100.0 * (1.0 + rate) ** (i / 365.0) for i in range(n)])


# ── compute_portfolio_series ──────────────────────────────────

class TestComputePortfolioSeries(unittest.TestCase):

    def test_flat_funds_returns_flat(self):
        """Two flat funds → flat portfolio."""
        n = 100
        vals = {1: [100.0] * n, 2: [200.0] * n}
        weights = {1: 0.5, 2: 0.5}
        ps = compute_portfolio_series(vals, weights)
        self.assertEqual(len(ps), n)
        for v in ps:
            self.assertAlmostEqual(v, 100.0, places=6)

    def test_single_fund_full_weight(self):
        """One fund with weight 1 → same returns as that fund."""
        n = 50
        vals_raw = [100.0 * (1.0 + 0.0001 * i) for i in range(n)]
        vals = {1: vals_raw, 2: [100.0] * n}
        weights = {1: 1.0, 2: 0.0}
        ps = compute_portfolio_series(vals, weights)
        # Portfolio should track fund 1's return
        expected_final = 100.0 * (vals_raw[-1] / vals_raw[0])
        self.assertAlmostEqual(ps[-1], expected_final, places=4)

    def test_equal_weight_rising(self):
        """Two rising funds with equal weight → rising portfolio."""
        n = 365
        vals = {
            1: [100.0 * (1.10 ** (i / 365)) for i in range(n)],
            2: [100.0 * (1.08 ** (i / 365)) for i in range(n)],
        }
        weights = {1: 0.5, 2: 0.5}
        ps = compute_portfolio_series(vals, weights)
        self.assertGreater(ps[-1], 100.0)
        # Should be between the two individual fund returns
        r1 = vals[1][-1] / vals[1][0]
        r2 = vals[2][-1] / vals[2][0]
        portfolio_return = ps[-1] / ps[0]
        self.assertGreater(portfolio_return, min(r1, r2) - 0.01)
        self.assertLess(portfolio_return, max(r1, r2) + 0.01)

    def test_empty_values(self):
        """Empty series → empty output."""
        ps = compute_portfolio_series({1: [], 2: []}, {1: 0.5, 2: 0.5})
        self.assertEqual(ps, [])

    def test_base_100_start(self):
        """Portfolio always starts at 100."""
        vals = {1: [50.0, 55.0, 60.0], 2: [200.0, 210.0, 220.0]}
        weights = {1: 0.5, 2: 0.5}
        ps = compute_portfolio_series(vals, weights)
        self.assertEqual(ps[0], 100.0)


# ── rolling_period_returns + distribution ─────────────────────

class TestRollingPeriodReturns(unittest.TestCase):

    def test_sufficient_data(self):
        """With enough data, rolling returns are produced."""
        n = 600
        dates = _daily_dates(date(2019, 1, 1), n)
        vals = [100.0 * (1.10 ** (i / 365)) for i in range(n)]
        results = rolling_period_returns(dates, vals, window_points=252, step_points=21)
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIn("cagr_pct", r)
            self.assertIn("start_date", r)
            self.assertIn("end_date", r)

    def test_insufficient_data(self):
        """Too few points → no results."""
        dates = _daily_dates(date(2024, 1, 1), 10)
        vals = [100.0 + i for i in range(10)]
        results = rolling_period_returns(dates, vals, window_points=252, step_points=21)
        self.assertEqual(len(results), 0)

    def test_cagr_reasonable(self):
        """10% growth fund → CAGR close to 10%."""
        n = 600
        dates = _daily_dates(date(2019, 1, 1), n)
        vals = [100.0 * (1.10 ** (i / 365)) for i in range(n)]
        results = rolling_period_returns(dates, vals, window_points=365, step_points=21)
        if results:
            for r in results:
                self.assertAlmostEqual(r["cagr_pct"], 10.0, delta=1.0)


class TestReturnDistribution(unittest.TestCase):

    def test_empty(self):
        dist = return_distribution([])
        self.assertEqual(dist["window_count"], 0)
        self.assertIsNone(dist["median_cagr_pct"])

    def test_with_data(self):
        rolling = [{"cagr_pct": 8.0}, {"cagr_pct": 10.0}, {"cagr_pct": 12.0}, {"cagr_pct": 14.0}]
        dist = return_distribution(rolling)
        self.assertEqual(dist["window_count"], 4)
        self.assertEqual(dist["min_cagr_pct"], 8.0)
        self.assertEqual(dist["max_cagr_pct"], 14.0)
        self.assertIsNotNone(dist["median_cagr_pct"])
        self.assertIsNotNone(dist["p25_cagr_pct"])
        self.assertIsNotNone(dist["p75_cagr_pct"])


# ── Daily returns helper ──────────────────────────────────────

class TestDailyReturns(unittest.TestCase):

    def test_constant(self):
        rets = _daily_returns([100.0] * 10)
        self.assertEqual(len(rets), 9)
        for r in rets:
            self.assertAlmostEqual(r, 0.0, places=10)

    def test_doubling(self):
        rets = _daily_returns([100.0, 200.0])
        self.assertEqual(len(rets), 1)
        self.assertAlmostEqual(rets[0], 1.0, places=6)

    def test_zero_prev(self):
        rets = _daily_returns([0.0, 100.0])
        self.assertEqual(rets[0], 0.0)


# ── Pearson correlation ───────────────────────────────────────

class TestPearsonCorrelation(unittest.TestCase):

    def test_identical(self):
        xs = [0.01, -0.02, 0.03, -0.01, 0.02]
        corr = _pearson_correlation(xs, xs)
        self.assertAlmostEqual(corr, 1.0, places=6)

    def test_opposite(self):
        xs = [0.01, -0.02, 0.03, -0.01, 0.02]
        ys = [-0.01, 0.02, -0.03, 0.01, -0.02]
        corr = _pearson_correlation(xs, ys)
        self.assertAlmostEqual(corr, -1.0, places=6)

    def test_too_few_points(self):
        corr = _pearson_correlation([0.01, 0.02], [0.01, 0.02])
        self.assertEqual(corr, 0.0)

    def test_zero_variance(self):
        xs = [0.0, 0.0, 0.0, 0.0]
        ys = [0.01, 0.02, 0.03, 0.04]
        corr = _pearson_correlation(xs, ys)
        self.assertEqual(corr, 0.0)


# ── Pairwise correlations ────────────────────────────────────

class TestPairwiseCorrelations(unittest.TestCase):

    def test_two_identical_funds(self):
        n = 100
        vals = [100.0 * (1.0 + 0.001 * i) for i in range(n)]
        result = pairwise_correlations(
            {1: vals, 2: vals},
            {1: "Fund A", 2: "Fund B"},
        )
        self.assertEqual(result["pair_count"], 1)
        self.assertAlmostEqual(result["pairs"][0]["correlation"], 1.0, places=4)

    def test_three_funds(self):
        n = 100
        vals1 = [100.0 + i for i in range(n)]
        vals2 = [100.0 + 2 * i for i in range(n)]
        vals3 = [100.0 - i for i in range(n)]
        result = pairwise_correlations(
            {1: vals1, 2: vals2, 3: vals3},
            {1: "A", 2: "B", 3: "C"},
        )
        # 3 funds → 3 pairs
        self.assertEqual(result["pair_count"], 3)
        self.assertIsNotNone(result["average_correlation"])

    def test_fund_names_in_pairs(self):
        n = 50
        vals = [100.0 + i for i in range(n)]
        result = pairwise_correlations(
            {10: vals, 20: vals},
            {10: "Fund X", 20: "Fund Y"},
        )
        self.assertEqual(result["pairs"][0]["fund_a"], "Fund X")
        self.assertEqual(result["pairs"][0]["fund_b"], "Fund Y")


# ── Correlation label ─────────────────────────────────────────

class TestCorrelationLabel(unittest.TestCase):

    def test_very_high(self):
        self.assertEqual(_correlation_label(0.95), "very high")
        self.assertEqual(_correlation_label(-0.85), "very high")

    def test_high(self):
        self.assertEqual(_correlation_label(0.65), "high")

    def test_moderate(self):
        self.assertEqual(_correlation_label(0.45), "moderate")

    def test_low(self):
        self.assertEqual(_correlation_label(0.25), "low")

    def test_very_low(self):
        self.assertEqual(_correlation_label(0.05), "very low")


# ── Concentration metrics ─────────────────────────────────────

class TestConcentrationMetrics(unittest.TestCase):

    def test_equal_weights(self):
        conc = concentration_metrics({1: 0.5, 2: 0.5})
        self.assertAlmostEqual(conc["hhi"], 0.5, places=4)
        self.assertAlmostEqual(conc["effective_fund_count"], 2.0, places=2)
        self.assertEqual(conc["actual_fund_count"], 2)

    def test_skewed_weights(self):
        conc = concentration_metrics({1: 0.9, 2: 0.1})
        self.assertGreater(conc["hhi"], 0.5)
        self.assertLess(conc["effective_fund_count"], 2.0)
        self.assertAlmostEqual(conc["largest_weight"], 0.9, places=4)
        self.assertAlmostEqual(conc["smallest_weight"], 0.1, places=4)

    def test_single_fund(self):
        conc = concentration_metrics({1: 1.0})
        self.assertAlmostEqual(conc["hhi"], 1.0, places=4)
        self.assertAlmostEqual(conc["effective_fund_count"], 1.0, places=2)

    def test_equal_four(self):
        conc = concentration_metrics({1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25})
        self.assertAlmostEqual(conc["hhi"], 0.25, places=4)
        self.assertAlmostEqual(conc["effective_fund_count"], 4.0, places=2)


# ── Contribution analysis ─────────────────────────────────────

class TestContributionAnalysis(unittest.TestCase):

    def test_basic(self):
        n = 365
        dates = _daily_dates(date(2019, 1, 1), n)
        vals = {
            1: [100.0 * (1.10 ** (i / 365)) for i in range(n)],
            2: [100.0 * (1.05 ** (i / 365)) for i in range(n)],
        }
        weights = {1: 0.6, 2: 0.4}
        result = contribution_analysis(dates, vals, weights, {1: "A", 2: "B"})
        self.assertIn("portfolio_return_pct", result)
        self.assertEqual(len(result["funds"]), 2)
        # Each fund should have weighted return
        for f in result["funds"]:
            self.assertIn("weighted_return_pct", f)
            self.assertIn("fund_max_drawdown_pct", f)

    def test_too_few_dates(self):
        result = contribution_analysis(
            [date(2024, 1, 1)], {1: [100.0]}, {1: 1.0}, {1: "A"},
        )
        self.assertEqual(result["funds"], [])

    def test_weighted_returns_sum_to_portfolio(self):
        """Sum of weighted returns should equal portfolio return."""
        n = 200
        dates = _daily_dates(date(2020, 1, 1), n)
        vals = {
            1: [100.0 * (1.12 ** (i / 365)) for i in range(n)],
            2: [100.0 * (1.06 ** (i / 365)) for i in range(n)],
        }
        weights = {1: 0.7, 2: 0.3}
        result = contribution_analysis(dates, vals, weights, {1: "X", 2: "Y"})
        total_weighted = sum(f["weighted_return_pct"] for f in result["funds"])
        self.assertAlmostEqual(total_weighted, result["portfolio_return_pct"], places=2)


# ── build_portfolio_insights ──────────────────────────────────

class TestBuildPortfolioInsights(unittest.TestCase):

    def _make_data(self, n=600, funds=2):
        start = date(2019, 1, 1)
        rates = [0.10, 0.08, 0.12, 0.06, 0.09][:funds]
        series_map = {}
        for i in range(funds):
            code = i + 1
            series_map[code] = _rising_series(start, n, rates[i])
        dates, aligned = align_multiple_series(series_map)
        q = alignment_quality(dates, series_map)
        names = {code: f"Fund {code}" for code in series_map}
        equal_w = 1.0 / funds
        weights = {code: equal_w for code in series_map}
        return dates, aligned, names, weights, q

    def test_produces_seven_insights(self):
        dates, aligned, names, weights, q = self._make_data(n=600, funds=2)
        insights = build_portfolio_insights(dates, aligned, names, weights, q)
        self.assertEqual(len(insights), 7)

    def test_all_insights_compile(self):
        dates, aligned, names, weights, q = self._make_data(n=600, funds=2)
        insights = build_portfolio_insights(dates, aligned, names, weights, q)
        for raw in insights:
            compiled = compile_insight(raw)
            self.assertEqual(compiled["template"], "diagnostic")

    def test_all_insights_pass_language_lint(self):
        dates, aligned, names, weights, q = self._make_data(n=600, funds=2)
        insights = build_portfolio_insights(dates, aligned, names, weights, q)
        for idx, raw in enumerate(insights):
            matches = lint_text_tree(raw)
            self.assertEqual(
                len(matches), 0,
                f"Insight {idx} has lint violations: {[(m.path, m.pattern) for m in matches]}",
            )

    def test_evidence_high_for_good_data(self):
        dates, aligned, names, weights, q = self._make_data(n=600, funds=2)
        insights = build_portfolio_insights(dates, aligned, names, weights, q)
        for raw in insights:
            self.assertIn(raw["evidence_strength"], ("High", "Medium"))

    def test_evidence_degrades_with_low_fund_evidence(self):
        dates, aligned, names, weights, q = self._make_data(n=600, funds=2)
        insights = build_portfolio_insights(
            dates, aligned, names, weights, q,
            per_fund_evidence=["Low", "High"],
        )
        for raw in insights:
            self.assertEqual(raw["evidence_strength"], "Low")

    def test_three_funds(self):
        dates, aligned, names, weights, q = self._make_data(n=600, funds=3)
        insights = build_portfolio_insights(dates, aligned, names, weights, q)
        self.assertEqual(len(insights), 7)
        # Should have 3 pairs in correlation insight
        corr_insight = [i for i in insights if "correlation" in i["observation"].lower()]
        self.assertTrue(len(corr_insight) > 0)

    def test_five_funds(self):
        dates, aligned, names, weights, q = self._make_data(n=600, funds=5)
        insights = build_portfolio_insights(dates, aligned, names, weights, q)
        self.assertEqual(len(insights), 7)

    def test_short_data_fewer_insights(self):
        """With very short data, some insights may be missing."""
        start = date(2024, 1, 1)
        series_map = {
            1: _series(start, [100.0, 101.0, 102.0]),
            2: _series(start, [100.0, 100.5, 101.0]),
        }
        dates, aligned = align_multiple_series(series_map)
        q = alignment_quality(dates, series_map)
        insights = build_portfolio_insights(
            dates, aligned, {1: "A", 2: "B"}, {1: 0.5, 2: 0.5}, q,
        )
        # Should still produce some insights (at least drawdown, volatility, etc.)
        self.assertGreater(len(insights), 0)
        for raw in insights:
            compiled = compile_insight(raw)
            self.assertEqual(compiled["template"], "diagnostic")

    def test_empty_dates_no_insights(self):
        insights = build_portfolio_insights(
            [], {}, {}, {}, {
                "aligned_points": 0,
                "relative_completeness": 0.0,
                "history_years": 0.0,
                "calendar_density": 0.0,
            },
        )
        self.assertEqual(len(insights), 0)


# ── Request validation (PortfolioAggregateRequest) ────────────

class TestPortfolioAggregateRequestValidation(unittest.TestCase):

    def test_valid_request(self):
        from api.models import PortfolioAggregateRequest
        req = PortfolioAggregateRequest(
            funds=[
                {"scheme_code": 100, "weight": 0.5},
                {"scheme_code": 200, "weight": 0.5},
            ],
        )
        self.assertEqual(len(req.funds), 2)

    def test_min_two_funds(self):
        from api.models import PortfolioAggregateRequest
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            PortfolioAggregateRequest(
                funds=[{"scheme_code": 100, "weight": 1.0}],
            )

    def test_max_five_funds(self):
        from api.models import PortfolioAggregateRequest
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            PortfolioAggregateRequest(
                funds=[
                    {"scheme_code": i, "weight": 1 / 6}
                    for i in range(6)
                ],
            )

    def test_weight_bounds(self):
        from api.models import PortfolioAggregateRequest
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            PortfolioAggregateRequest(
                funds=[
                    {"scheme_code": 1, "weight": -0.1},
                    {"scheme_code": 2, "weight": 1.1},
                ],
            )

    def test_rolling_window_bounds(self):
        from api.models import PortfolioAggregateRequest
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            PortfolioAggregateRequest(
                funds=[
                    {"scheme_code": 1, "weight": 0.5},
                    {"scheme_code": 2, "weight": 0.5},
                ],
                rolling_window_points=5,  # too small
            )


if __name__ == "__main__":
    unittest.main()
