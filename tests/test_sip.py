from __future__ import annotations

import unittest
from datetime import date, timedelta
from typing import List

from backend.investment_analytics.sip import (
    monthly_contribution_dates,
    simulate_sip_window,
    rolling_sip_simulation,
    sip_return_distribution,
    sip_drawdown_summary,
    sip_consistency,
    sip_cost_impact,
    build_sip_insights,
)
from backend.investment_analytics.comparison import (
    align_multiple_series,
    alignment_quality,
)
from backend.investment_analytics.compiler import compile_insight
from backend.investment_analytics.errors import PolicyError
from backend.investment_analytics.mutual_funds import SeriesPoint


def _series(start: date, values: List[float], step_days: int = 1) -> List[SeriesPoint]:
    return [
        SeriesPoint(as_of=start + timedelta(days=i * step_days), value=v)
        for i, v in enumerate(values)
    ]


def _daily_dates(start: date, n: int) -> List[date]:
    return [start + timedelta(days=i) for i in range(n)]


def _monthly_dates(start: date, n_months: int) -> List[date]:
    """Generate first-of-month dates for n_months."""
    dates = []
    y, m = start.year, start.month
    for _ in range(n_months):
        dates.append(date(y, m, 1))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return dates


class TestMonthlyContributionDates(unittest.TestCase):

    def test_daily_series(self):
        dates = _daily_dates(date(2024, 1, 1), 90)
        indices = monthly_contribution_dates(dates)
        # Should get ~3 months (Jan, Feb, Mar)
        self.assertEqual(len(indices), 3)
        self.assertEqual(dates[indices[0]].month, 1)
        self.assertEqual(dates[indices[1]].month, 2)
        self.assertEqual(dates[indices[2]].month, 3)

    def test_monthly_series(self):
        dates = _monthly_dates(date(2024, 1, 1), 12)
        indices = monthly_contribution_dates(dates)
        self.assertEqual(len(indices), 12)

    def test_empty(self):
        self.assertEqual(monthly_contribution_dates([]), [])

    def test_single_date(self):
        indices = monthly_contribution_dates([date(2024, 6, 15)])
        self.assertEqual(len(indices), 1)


class TestSimulateSipWindow(unittest.TestCase):

    def test_basic_flat_nav(self):
        """Flat NAV → total return should be ~0%."""
        n_months = 12
        dates = _monthly_dates(date(2024, 1, 1), n_months)
        monthly_indices = list(range(n_months))
        aligned = {1: [100.0] * n_months}
        result = simulate_sip_window(
            monthly_indices, aligned, {1: 1.0}, 1000, 0, n_months,
        )
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["total_return_pct"], 0.0, places=1)
        self.assertEqual(result["total_invested"], 12000.0)
        self.assertAlmostEqual(result["final_value"], 12000.0, places=0)

    def test_rising_nav(self):
        """Steadily rising NAV → positive return."""
        n_months = 12
        dates = _monthly_dates(date(2024, 1, 1), n_months)
        monthly_indices = list(range(n_months))
        aligned = {1: [100 + i * 5 for i in range(n_months)]}
        result = simulate_sip_window(
            monthly_indices, aligned, {1: 1.0}, 1000, 0, n_months,
        )
        self.assertGreater(result["total_return_pct"], 0)

    def test_falling_nav(self):
        """Falling NAV → negative return, drawdown present."""
        n_months = 6
        monthly_indices = list(range(n_months))
        aligned = {1: [100, 90, 80, 70, 60, 50]}
        result = simulate_sip_window(
            monthly_indices, aligned, {1: 1.0}, 1000, 0, n_months,
        )
        self.assertLess(result["total_return_pct"], 0)
        self.assertLess(result["deepest_unrealized_drawdown_pct"], 0)
        self.assertGreater(result["negative_months"], 0)

    def test_two_funds_weighted(self):
        """Two funds with 60/40 split."""
        n_months = 6
        monthly_indices = list(range(n_months))
        aligned = {
            1: [100 + i * 2 for i in range(n_months)],
            2: [50 + i for i in range(n_months)],
        }
        result = simulate_sip_window(
            monthly_indices, aligned, {1: 0.6, 2: 0.4}, 10000, 0, n_months,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["total_invested"], 60000.0)

    def test_insufficient_months(self):
        monthly_indices = list(range(5))
        aligned = {1: [100.0] * 5}
        result = simulate_sip_window(
            monthly_indices, aligned, {1: 1.0}, 1000, 0, 10,
        )
        self.assertIsNone(result)

    def test_window_offset(self):
        """Window starting at month 2."""
        n_months = 8
        monthly_indices = list(range(n_months))
        aligned = {1: [100.0] * n_months}
        result = simulate_sip_window(
            monthly_indices, aligned, {1: 1.0}, 1000, 2, 5,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["window_months"], 5)
        self.assertEqual(result["total_invested"], 5000.0)


class TestRollingSipSimulation(unittest.TestCase):

    def test_basic(self):
        dates = _monthly_dates(date(2020, 1, 1), 48)
        aligned = {1: [100 + i * 0.5 for i in range(48)]}
        sim = rolling_sip_simulation(dates, aligned, {1: 1.0}, 1000, 12, 1)
        self.assertGreater(sim["window_count"], 0)
        self.assertEqual(sim["window_months"], 12)
        # Should have 48 - 12 + 1 = 37 windows
        self.assertEqual(sim["window_count"], 37)

    def test_step_months(self):
        dates = _monthly_dates(date(2020, 1, 1), 48)
        aligned = {1: [100.0] * 48}
        sim = rolling_sip_simulation(dates, aligned, {1: 1.0}, 1000, 12, 3)
        # 48 - 12 = 36, /3 + 1 = 13
        self.assertEqual(sim["window_count"], 13)

    def test_too_short(self):
        dates = _monthly_dates(date(2024, 1, 1), 5)
        aligned = {1: [100.0] * 5}
        sim = rolling_sip_simulation(dates, aligned, {1: 1.0}, 1000, 12, 1)
        self.assertEqual(sim["window_count"], 0)


class TestSipReturnDistribution(unittest.TestCase):

    def test_basic(self):
        outcomes = [
            {"total_return_pct": 10.0},
            {"total_return_pct": -5.0},
            {"total_return_pct": 15.0},
            {"total_return_pct": 3.0},
            {"total_return_pct": 8.0},
        ]
        dist = sip_return_distribution(outcomes)
        self.assertEqual(dist["window_count"], 5)
        self.assertEqual(dist["min_return_pct"], -5.0)
        self.assertEqual(dist["max_return_pct"], 15.0)
        self.assertIsNotNone(dist["median_return_pct"])
        self.assertIsNotNone(dist["p25_return_pct"])
        self.assertIsNotNone(dist["p75_return_pct"])

    def test_empty(self):
        dist = sip_return_distribution([])
        self.assertEqual(dist["window_count"], 0)
        self.assertIsNone(dist["median_return_pct"])

    def test_single_outcome(self):
        dist = sip_return_distribution([{"total_return_pct": 7.5}])
        self.assertEqual(dist["median_return_pct"], 7.5)
        self.assertEqual(dist["min_return_pct"], 7.5)


class TestSipDrawdownSummary(unittest.TestCase):

    def test_basic(self):
        outcomes = [
            {"deepest_unrealized_drawdown_pct": -5.0, "negative_months": 2,
             "longest_negative_streak_months": 2},
            {"deepest_unrealized_drawdown_pct": -10.0, "negative_months": 4,
             "longest_negative_streak_months": 3},
        ]
        dd = sip_drawdown_summary(outcomes)
        self.assertEqual(dd["extreme_deepest_drawdown_pct"], -10.0)
        self.assertEqual(dd["maximum_negative_months"], 4)
        self.assertEqual(dd["maximum_longest_negative_streak"], 3)

    def test_empty(self):
        dd = sip_drawdown_summary([])
        self.assertIsNone(dd["median_deepest_drawdown_pct"])


class TestSipConsistency(unittest.TestCase):

    def test_all_positive(self):
        outcomes = [{"total_return_pct": r} for r in [5.0, 10.0, 15.0, 20.0]]
        cons = sip_consistency(outcomes)
        self.assertEqual(cons["positive_pct"], 100.0)
        self.assertEqual(cons["above_threshold_pct"], 100.0)  # all 4 >= 5%

    def test_mixed(self):
        outcomes = [{"total_return_pct": r} for r in [-2.0, 1.0, 6.0]]
        cons = sip_consistency(outcomes, threshold_pct=5.0)
        self.assertAlmostEqual(cons["positive_pct"], 66.67, places=1)
        self.assertAlmostEqual(cons["above_threshold_pct"], 33.33, places=1)

    def test_empty(self):
        cons = sip_consistency([])
        self.assertEqual(cons["window_count"], 0)
        self.assertIsNone(cons["positive_pct"])


class TestSipCostImpact(unittest.TestCase):

    def test_basic(self):
        cost = sip_cost_impact({1: 1.5, 2: 0.5}, {1: 0.6, 2: 0.4}, 10000, 36)
        # Weighted TER = 0.6*1.5 + 0.4*0.5 = 1.1%
        self.assertAlmostEqual(cost["weighted_expense_ratio_pct"], 1.1, places=2)
        self.assertEqual(cost["total_invested"], 360000.0)
        self.assertGreater(cost["estimated_cost_drag"], 0)

    def test_zero_ter(self):
        cost = sip_cost_impact({1: 0.0}, {1: 1.0}, 1000, 12)
        self.assertEqual(cost["estimated_cost_drag"], 0.0)

    def test_single_fund(self):
        cost = sip_cost_impact({1: 2.0}, {1: 1.0}, 5000, 24)
        self.assertEqual(cost["weighted_expense_ratio_pct"], 2.0)
        self.assertGreater(cost["estimated_cost_drag"], 0)


class TestBuildSipInsights(unittest.TestCase):

    def _make_inputs(self, n_daily=365 * 5, num_funds=2):
        d = date(2019, 1, 1)
        dates = _daily_dates(d, n_daily)
        fund_series = {}
        names = {}
        expense = {}
        weights = {}
        w = round(1.0 / num_funds, 4)
        for i in range(num_funds):
            code = 100 + i
            pts = [SeriesPoint(as_of=dates[j], value=100 * (1 + 0.08 + i * 0.02) ** (j / 365))
                   for j in range(n_daily)]
            fund_series[code] = pts
            names[code] = f"Fund {i + 1}"
            expense[code] = 1.0 + i * 0.5
            weights[code] = w

        common_dates, aligned = align_multiple_series(fund_series)
        quality = alignment_quality(common_dates, fund_series)
        return common_dates, aligned, names, expense, weights, quality

    def test_produces_insights(self):
        dates, aligned, names, expense, weights, quality = self._make_inputs()
        insights, meta = build_sip_insights(
            dates, aligned, names, expense, weights, 10000, quality,
            window_months=36, step_months=1,
        )
        self.assertGreater(len(insights), 0)
        self.assertGreater(meta["window_count"], 0)
        for raw in insights:
            self.assertEqual(raw["type"], "diagnostic")

    def test_all_compile(self):
        """All insights pass compiler + language linter."""
        dates, aligned, names, expense, weights, quality = self._make_inputs()
        insights, meta = build_sip_insights(
            dates, aligned, names, expense, weights, 10000, quality,
            window_months=36, step_months=1,
        )
        compiled = []
        for raw in insights:
            compiled.append(compile_insight(raw))
        self.assertGreater(len(compiled), 0)
        for c in compiled:
            self.assertEqual(c["template"], "diagnostic")

    def test_single_fund(self):
        dates, aligned, names, expense, weights, quality = self._make_inputs(num_funds=1)
        insights, meta = build_sip_insights(
            dates, aligned, names, expense, weights, 5000, quality,
            window_months=24, step_months=1,
        )
        self.assertGreater(len(insights), 0)

    def test_five_funds(self):
        dates, aligned, names, expense, weights, quality = self._make_inputs(num_funds=5)
        insights, meta = build_sip_insights(
            dates, aligned, names, expense, weights, 10000, quality,
            window_months=36, step_months=1,
        )
        self.assertGreater(len(insights), 0)
        for raw in insights:
            compile_insight(raw)  # should not raise

    def test_too_few_dates(self):
        d = date(2024, 1, 1)
        dates = [d]
        aligned = {1: [100]}
        quality = {
            "aligned_points": 1, "max_series_points": 1,
            "relative_completeness": 1.0, "calendar_density": 1.0,
            "history_years": 0.0,
        }
        insights, meta = build_sip_insights(
            dates, aligned, {1: "A"}, {1: 1.0}, {1: 1.0}, 1000, quality,
        )
        self.assertEqual(insights, [])

    def test_short_history_no_windows(self):
        """3 months of data, 36 month window → no outcomes."""
        d = date(2024, 1, 1)
        dates = _daily_dates(d, 90)
        pts = [SeriesPoint(as_of=dates[i], value=100 + i * 0.1) for i in range(90)]
        common_dates, aligned = align_multiple_series({1: pts})
        quality = alignment_quality(common_dates, {1: pts})
        insights, meta = build_sip_insights(
            common_dates, aligned, {1: "Short"}, {1: 0.0}, {1: 1.0}, 1000, quality,
            window_months=36,
        )
        self.assertEqual(meta["window_count"], 0)
        self.assertEqual(insights, [])

    def test_evidence_degrades_with_low_fund(self):
        """Low per-fund evidence degrades SIP insights."""
        dates, aligned, names, expense, weights, quality = self._make_inputs()
        insights, _ = build_sip_insights(
            dates, aligned, names, expense, weights, 10000, quality,
            window_months=36, step_months=1,
            per_fund_evidence=["High", "Low"],
        )
        for raw in insights:
            # Cost insight has hardcoded "Low", others should be degraded to "Low"
            self.assertEqual(raw["evidence_strength"], "Low")

    def test_no_cost_insight_when_zero_ter(self):
        """No cost insight when all expense ratios are 0."""
        dates, aligned, names, _, weights, quality = self._make_inputs()
        zero_expense = {code: 0.0 for code in names}
        insights, _ = build_sip_insights(
            dates, aligned, names, zero_expense, weights, 10000, quality,
            window_months=36, step_months=1,
        )
        # Should have 4 insights (no cost)
        self.assertEqual(len(insights), 4)


class TestSipRequestValidation(unittest.TestCase):

    def test_valid_request(self):
        from api.models import SipRequest, SipFundEntry
        req = SipRequest(
            funds=[SipFundEntry(scheme_code=100, weight=0.6),
                   SipFundEntry(scheme_code=200, weight=0.4)],
            monthly_amount=10000,
        )
        self.assertEqual(len(req.funds), 2)
        self.assertEqual(req.monthly_amount, 10000)

    def test_single_fund(self):
        from api.models import SipRequest, SipFundEntry
        req = SipRequest(
            funds=[SipFundEntry(scheme_code=100, weight=1.0)],
            monthly_amount=5000,
        )
        self.assertEqual(len(req.funds), 1)

    def test_too_many_funds(self):
        from api.models import SipRequest, SipFundEntry
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            SipRequest(
                funds=[SipFundEntry(scheme_code=i, weight=0.2) for i in range(6)],
                monthly_amount=10000,
            )

    def test_zero_amount(self):
        from api.models import SipRequest, SipFundEntry
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            SipRequest(
                funds=[SipFundEntry(scheme_code=100, weight=1.0)],
                monthly_amount=0,
            )

    def test_window_too_short(self):
        from api.models import SipRequest, SipFundEntry
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            SipRequest(
                funds=[SipFundEntry(scheme_code=100, weight=1.0)],
                monthly_amount=1000,
                rolling_window_months=3,  # min is 6
            )


if __name__ == "__main__":
    unittest.main()
