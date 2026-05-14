"""Direct unit tests for backend.investment_analytics.ranking math.

Before this file, ranking.py was at 22% coverage and the math was
only exercised indirectly via portfolio_health integration paths.
This file:

- Plugs the coverage gap with direct tests on every helper.
- Converts the deep-audit DISMISSALS from agent reports into proofs
  (test_dismissal_*) — if the audit's "false positive" call was
  wrong, one of these tests will fail.
- Probes edge cases that the per-category live tests can't reach
  without network: degenerate inputs, tie-breakers, dedup collisions.

Naming: tests for behaviors that COULD have been bugs but aren't
are prefixed `test_dismissal_` — they document the audit decision
in code so future devs can re-audit.
"""
from __future__ import annotations

import math
import unittest
from datetime import date

from backend.investment_analytics import ranking as rk
from backend.investment_analytics.ranking import (
    FundMetrics,
    MIN_ALIGNED_POINTS,
    ROLLING_STEP_DAYS,
    ROLLING_WINDOW_DAYS,
)
from backend.data_discovery.registry import SchemeEntry


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _fm(code, **kw):
    """Build a FundMetrics with defaults; override any field by kw."""
    base = dict(
        scheme_code=code, fund_name=f"F{code}", fund_house="X",
        excess_return_pct=2.0, max_drawdown_pct=-15.0, consistency_pct=55.0,
        volatility_pct=12.0, downside_capture_ratio=1.0,
        fund_cagr_pct=10.0, benchmark_cagr_pct=8.0,
        aligned_points=1500, history_years=8.0, drawdown_trough_date=None,
    )
    base.update(kw)
    return FundMetrics(**base)


def _scheme(code, name):
    return SchemeEntry(
        scheme_code=code, scheme_name=name,
        scheme_category="Equity Scheme - Large Cap Fund",
        fund_house="X",
    )


def _trending_navs(start_iso: str, n_points: int, daily_drift: float = 0.0005,
                   start_nav: float = 100.0):
    """Build a list of {date, nav} records spaced one trading day apart.
    Returns deterministic, monotonic-trend series — no randomness so
    metric outputs are predictable."""
    from datetime import timedelta
    d = date.fromisoformat(start_iso)
    out, nav = [], start_nav
    for _ in range(n_points):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        out.append({"date": d.isoformat(), "nav": round(nav, 4)})
        nav *= (1.0 + daily_drift)
        d += timedelta(days=1)
    return out


# ─────────────────────────────────────────────────────────────────────
# _annualized_return
# ─────────────────────────────────────────────────────────────────────

class AnnualizedReturnTests(unittest.TestCase):

    def test_doubling_in_one_year(self) -> None:
        # 100 → 200 in 1y → 100% CAGR
        self.assertAlmostEqual(rk._annualized_return(100, 200, 1.0), 1.0, places=4)

    def test_doubling_in_two_years(self) -> None:
        # 100 → 200 in 2y → sqrt(2) - 1 ≈ 0.4142
        self.assertAlmostEqual(
            rk._annualized_return(100, 200, 2.0), math.sqrt(2) - 1, places=4
        )

    def test_zero_years_returns_zero(self) -> None:
        self.assertEqual(rk._annualized_return(100, 200, 0.0), 0.0)

    def test_negative_years_returns_zero(self) -> None:
        self.assertEqual(rk._annualized_return(100, 200, -1.0), 0.0)

    def test_zero_start_value_returns_zero(self) -> None:
        self.assertEqual(rk._annualized_return(0, 200, 1.0), 0.0)

    def test_total_loss_returns_minus_one(self) -> None:
        # 100 → 0 in 5y → -100% annualized.
        # KNOWN AUDIT FINDING (deferred): mutual_funds._annualized_return
        # in the LEGACY pipeline returns 0.0 instead of -1.0 for total
        # loss. Verifying ranking.py's version returns the mathematically
        # correct value at literal zero.
        self.assertAlmostEqual(rk._annualized_return(100, 0.0, 5), -1.0, places=4)


# ─────────────────────────────────────────────────────────────────────
# _years_between
# ─────────────────────────────────────────────────────────────────────

class YearsBetweenTests(unittest.TestCase):

    def test_one_year(self) -> None:
        # 2024 is a leap year, so Jan 1 2024 → Jan 1 2025 is 366 days.
        # Divide by 365.25 (the function's denominator).
        self.assertAlmostEqual(
            rk._years_between(date(2024, 1, 1), date(2025, 1, 1)),
            366 / 365.25, places=4,
        )

    def test_one_year_non_leap(self) -> None:
        # 2023 is NOT a leap year. 365 days.
        self.assertAlmostEqual(
            rk._years_between(date(2023, 1, 1), date(2024, 1, 1)),
            365 / 365.25, places=4,
        )

    def test_zero_for_same_date(self) -> None:
        self.assertEqual(rk._years_between(date(2024, 1, 1), date(2024, 1, 1)), 0.0)

    def test_negative_when_end_before_start(self) -> None:
        # No guard — caller's responsibility. Caller in _compute_metrics
        # uses first/last of sorted aligned list, so this can't happen
        # there. Document the contract.
        self.assertLess(rk._years_between(date(2025, 1, 1), date(2024, 1, 1)), 0)


# ─────────────────────────────────────────────────────────────────────
# _align_to_common_dates
# ─────────────────────────────────────────────────────────────────────

class AlignToCommonDatesTests(unittest.TestCase):

    def test_intersects_on_common_dates_only(self) -> None:
        fund = [{"date": "2024-01-01", "nav": 100},
                {"date": "2024-01-02", "nav": 101},
                {"date": "2024-01-03", "nav": 102}]
        bench = [{"date": "2024-01-01", "nav": 50},
                 {"date": "2024-01-03", "nav": 52}]
        aligned = rk._align_to_common_dates(fund, bench)
        self.assertEqual(len(aligned), 2)
        self.assertEqual(aligned[0], (date(2024, 1, 1), 100, 50))
        self.assertEqual(aligned[1], (date(2024, 1, 3), 102, 52))

    def test_output_is_sorted_by_date(self) -> None:
        fund = [{"date": "2024-01-03", "nav": 102},
                {"date": "2024-01-01", "nav": 100},
                {"date": "2024-01-02", "nav": 101}]
        bench = [{"date": d["date"], "nav": 50.0} for d in fund]
        aligned = rk._align_to_common_dates(fund, bench)
        dates = [a[0] for a in aligned]
        self.assertEqual(dates, sorted(dates))

    def test_empty_fund_returns_empty(self) -> None:
        self.assertEqual(rk._align_to_common_dates([], [{"date": "2024-01-01", "nav": 1}]), [])

    def test_no_overlap_returns_empty(self) -> None:
        fund = [{"date": "2024-01-01", "nav": 100}]
        bench = [{"date": "2024-02-01", "nav": 50}]
        self.assertEqual(rk._align_to_common_dates(fund, bench), [])

    def test_duplicate_bench_dates_keeps_last(self) -> None:
        """Documents current behavior: the dict comprehension in
        _align_to_common_dates keeps the LAST nav value for a duplicate
        benchmark date. mfapi.in shouldn't emit duplicates, but if it
        does, the behavior is well-defined (not crashed)."""
        fund = [{"date": "2024-01-01", "nav": 100}]
        bench = [{"date": "2024-01-01", "nav": 50},
                 {"date": "2024-01-01", "nav": 99}]
        aligned = rk._align_to_common_dates(fund, bench)
        self.assertEqual(aligned[0][2], 99)  # last value wins


# ─────────────────────────────────────────────────────────────────────
# _compute_metrics — end-to-end on synthetic data
# ─────────────────────────────────────────────────────────────────────

class ComputeMetricsTests(unittest.TestCase):

    def test_returns_none_below_min_aligned_points(self) -> None:
        fund = _trending_navs("2024-01-01", MIN_ALIGNED_POINTS - 1)
        bench = _trending_navs("2024-01-01", MIN_ALIGNED_POINTS - 1)
        result = rk._compute_metrics(1, "F1", "X", fund, bench)
        self.assertIsNone(result)

    def test_dismissal_rolling_window_works_at_min_aligned_points(self) -> None:
        """AUDIT DISMISSAL PROOF: agent claimed the rolling-window
        loop would compute zero windows at len(aligned) == 252.
        Verifying that at MIN_ALIGNED_POINTS=700 (the actual gate),
        the loop computes a sensible number of windows."""
        fund = _trending_navs("2018-01-01", MIN_ALIGNED_POINTS + 50, daily_drift=0.0005)
        bench = _trending_navs("2018-01-01", MIN_ALIGNED_POINTS + 50, daily_drift=0.0003)
        result = rk._compute_metrics(1, "F1", "X", fund, bench)
        self.assertIsNotNone(result)
        # With ~750 points, ROLLING_WINDOW_DAYS=252, ROLLING_STEP_DAYS=5,
        # we expect ~ (750-252)/5 ≈ 99 windows. So consistency must be
        # 100% (fund is drifting faster than benchmark every window) or
        # at least non-zero — NOT the 0.0 fallback.
        self.assertGreater(result.consistency_pct, 0.0,
                           "rolling-window must compute real consistency at MIN gate")

    def test_uptrending_fund_beats_uptrending_benchmark(self) -> None:
        """Fund drifts 0.05%/day, bench drifts 0.03%/day. Fund's CAGR
        must exceed benchmark's CAGR. Consistency must be ~100%."""
        fund = _trending_navs("2018-01-01", 1500, daily_drift=0.0005)
        bench = _trending_navs("2018-01-01", 1500, daily_drift=0.0003)
        result = rk._compute_metrics(1, "F1", "X", fund, bench)
        self.assertIsNotNone(result)
        self.assertGreater(result.excess_return_pct, 0)
        self.assertGreater(result.consistency_pct, 90.0)

    def test_drawdown_detection(self) -> None:
        # Build a NAV series that crashes mid-way and recovers, so
        # max_drawdown_pct should be clearly negative.
        from datetime import timedelta
        d = date(2018, 1, 1)
        fund, nav = [], 100.0
        for i in range(1500):
            while d.weekday() >= 5:
                d += timedelta(days=1)
            fund.append({"date": d.isoformat(), "nav": round(nav, 4)})
            # rise for first 500, crash to 40% at point 800, recover.
            if i < 500:
                nav *= 1.001
            elif i < 800:
                nav *= 0.995
            else:
                nav *= 1.0008
            d += timedelta(days=1)
        bench = _trending_navs("2018-01-01", 1500, daily_drift=0.0003)
        result = rk._compute_metrics(1, "F1", "X", fund, bench)
        self.assertIsNotNone(result)
        self.assertLess(result.max_drawdown_pct, -10.0)
        self.assertIsNotNone(result.drawdown_trough_date)


# ─────────────────────────────────────────────────────────────────────
# _dominates — truth table
# ─────────────────────────────────────────────────────────────────────

class DominatesTests(unittest.TestCase):

    def test_wins_all_five_dominates(self) -> None:
        a = _fm(1, excess_return_pct=5, max_drawdown_pct=-10, consistency_pct=70,
                volatility_pct=10, downside_capture_ratio=0.8)
        b = _fm(2, excess_return_pct=2, max_drawdown_pct=-20, consistency_pct=50,
                volatility_pct=15, downside_capture_ratio=1.1)
        self.assertTrue(rk._dominates(a, b))
        self.assertFalse(rk._dominates(b, a))

    def test_wins_three_of_five_dominates(self) -> None:
        # a wins on excess, dd, cons; loses on vol, downside.
        a = _fm(1, excess_return_pct=5, max_drawdown_pct=-10, consistency_pct=70,
                volatility_pct=15, downside_capture_ratio=1.2)
        b = _fm(2, excess_return_pct=2, max_drawdown_pct=-20, consistency_pct=50,
                volatility_pct=10, downside_capture_ratio=0.8)
        self.assertTrue(rk._dominates(a, b))
        self.assertFalse(rk._dominates(b, a))

    def test_wins_two_of_five_does_not_dominate(self) -> None:
        a = _fm(1, excess_return_pct=5, max_drawdown_pct=-10, consistency_pct=50,
                volatility_pct=15, downside_capture_ratio=1.2)
        b = _fm(2, excess_return_pct=2, max_drawdown_pct=-20, consistency_pct=60,
                volatility_pct=10, downside_capture_ratio=0.8)
        # a wins excess, dd; loses cons, vol, downside.
        self.assertFalse(rk._dominates(a, b))

    def test_ties_dont_count_as_wins(self) -> None:
        """Strict-inequality semantics: equal values give neither side
        a win. Two identical funds: neither dominates the other."""
        a = _fm(1)
        b = _fm(2)  # same defaults
        self.assertFalse(rk._dominates(a, b))
        self.assertFalse(rk._dominates(b, a))

    def test_dismissal_dominates_is_anti_symmetric(self) -> None:
        """AUDIT DISMISSAL PROOF: under strict inequality, no two funds
        can BOTH dominate each other. If a wins k of 5, b wins (5-k-ties)
        of 5. So if a dominates (>=3 wins), b has <=2 wins (cannot
        dominate). Demonstrating across a random spread."""
        import random
        rng = random.Random(7)
        for _ in range(200):
            a = _fm(1,
                    excess_return_pct=rng.uniform(-5, 10),
                    max_drawdown_pct=rng.uniform(-50, -5),
                    consistency_pct=rng.uniform(0, 100),
                    volatility_pct=rng.uniform(1, 30),
                    downside_capture_ratio=rng.uniform(0.5, 1.5))
            b = _fm(2,
                    excess_return_pct=rng.uniform(-5, 10),
                    max_drawdown_pct=rng.uniform(-50, -5),
                    consistency_pct=rng.uniform(0, 100),
                    volatility_pct=rng.uniform(1, 30),
                    downside_capture_ratio=rng.uniform(0.5, 1.5))
            if rk._dominates(a, b):
                self.assertFalse(rk._dominates(b, a),
                                 "both directions dominate — relation is broken")


# ─────────────────────────────────────────────────────────────────────
# _compute_dominance — ordering + tie-breakers
# ─────────────────────────────────────────────────────────────────────

class ComputeDominanceTests(unittest.TestCase):

    def test_top_dominator_is_first(self) -> None:
        a = _fm(1, excess_return_pct=5, max_drawdown_pct=-10, consistency_pct=70,
                volatility_pct=10, downside_capture_ratio=0.8)
        b = _fm(2, excess_return_pct=2, max_drawdown_pct=-15, consistency_pct=55,
                volatility_pct=12, downside_capture_ratio=1.0)
        c = _fm(3, excess_return_pct=0, max_drawdown_pct=-25, consistency_pct=40,
                volatility_pct=18, downside_capture_ratio=1.3)
        result = rk._compute_dominance([c, b, a])
        self.assertEqual(result[0][0].scheme_code, 1)  # a dominates everything
        self.assertEqual(result[-1][0].scheme_code, 3)  # c is dominated by all

    def test_dominance_counts_are_correct(self) -> None:
        a = _fm(1, excess_return_pct=5, max_drawdown_pct=-10, consistency_pct=70,
                volatility_pct=10, downside_capture_ratio=0.8)
        b = _fm(2, excess_return_pct=2, max_drawdown_pct=-15, consistency_pct=55,
                volatility_pct=12, downside_capture_ratio=1.0)
        c = _fm(3, excess_return_pct=0, max_drawdown_pct=-25, consistency_pct=40,
                volatility_pct=18, downside_capture_ratio=1.3)
        result = rk._compute_dominance([a, b, c])
        # a dominates b and c (wins everything) → count 2
        # b dominates c (wins 4/5: ret, dd, cons, vol, ds — actually 5/5)
        # c dominates none
        self.assertEqual(result[0][1], 2)
        self.assertEqual(result[2][1], 0)

    def test_dismissal_tie_breakers_produce_deterministic_order(self) -> None:
        """AUDIT DISMISSAL PROOF: agent claimed tie-breaker ordering is
        'brittle' — but it's by design. Two funds with identical
        dominance count must produce the SAME order across runs, AND
        the order must reflect documented tie-break semantics
        (consistency desc → drawdown desc → volatility asc)."""
        # Build two funds with the same dominance count (both dominate
        # one out of two others) — then a tie-break must pick one.
        a = _fm(1, excess_return_pct=3.0, max_drawdown_pct=-15, consistency_pct=70,
                volatility_pct=12, downside_capture_ratio=1.0)
        b = _fm(2, excess_return_pct=3.0, max_drawdown_pct=-15, consistency_pct=60,
                volatility_pct=12, downside_capture_ratio=1.0)
        weak = _fm(3, excess_return_pct=-1, max_drawdown_pct=-25, consistency_pct=30,
                   volatility_pct=20, downside_capture_ratio=1.5)
        r1 = rk._compute_dominance([a, b, weak])
        r2 = rk._compute_dominance([b, a, weak])  # different input order
        # Same scheme codes at each rank, regardless of input order.
        self.assertEqual(
            [f[0].scheme_code for f in r1],
            [f[0].scheme_code for f in r2],
        )
        # Tie-breaker: higher consistency wins → a (cons=70) before b (cons=60)
        self.assertEqual(r1[0][0].scheme_code, 1)


# ─────────────────────────────────────────────────────────────────────
# _label_strengths_weaknesses
# ─────────────────────────────────────────────────────────────────────

class LabelStrengthsWeaknessesTests(unittest.TestCase):

    def test_returns_empty_for_single_fund(self) -> None:
        f = _fm(1)
        s, w = rk._label_strengths_weaknesses(f, [f])
        self.assertEqual(s, [])
        self.assertEqual(w, [])

    def test_top_third_gets_strengths(self) -> None:
        top = _fm(1, excess_return_pct=10, max_drawdown_pct=-5, consistency_pct=90,
                  volatility_pct=8, downside_capture_ratio=0.6)
        peers = [
            _fm(2, excess_return_pct=2),
            _fm(3, excess_return_pct=1),
            _fm(4, excess_return_pct=0),
        ]
        all_funds = [top] + peers
        s, _ = rk._label_strengths_weaknesses(top, all_funds)
        # Top must have at least one strength bullet.
        self.assertGreater(len(s), 0)
        self.assertTrue(any("Higher returns" in line for line in s))

    def test_bottom_third_gets_weaknesses(self) -> None:
        bottom = _fm(1, excess_return_pct=-5, max_drawdown_pct=-40, consistency_pct=15,
                     volatility_pct=25, downside_capture_ratio=1.5)
        peers = [_fm(i + 2, excess_return_pct=5) for i in range(3)]
        s, w = rk._label_strengths_weaknesses(bottom, [bottom] + peers)
        self.assertGreater(len(w), 0)
        self.assertTrue(any("Lower returns" in line for line in w))


# ─────────────────────────────────────────────────────────────────────
# _deduplicate_variants
# ─────────────────────────────────────────────────────────────────────

class DeduplicateVariantsTests(unittest.TestCase):

    def test_prefers_growth_over_dividend(self) -> None:
        funds = [
            _scheme(100, "HDFC Top 100 Direct Plan - Dividend Reinvestment"),
            _scheme(101, "HDFC Top 100 Direct Plan - Growth"),
            _scheme(102, "HDFC Top 100 Direct Plan - IDCW Payout"),
        ]
        deduped = rk._deduplicate_variants(funds)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].scheme_code, 101)

    def test_single_variant_passes_through(self) -> None:
        funds = [_scheme(100, "HDFC Top 100 Direct Plan - Growth")]
        self.assertEqual(rk._deduplicate_variants(funds), funds)

    def test_no_growth_falls_back_to_first(self) -> None:
        funds = [
            _scheme(100, "HDFC Top 100 Direct Plan - Dividend Reinvestment"),
            _scheme(101, "HDFC Top 100 Direct Plan - IDCW Payout"),
        ]
        deduped = rk._deduplicate_variants(funds)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].scheme_code, 100)

    def test_distinct_funds_not_collapsed(self) -> None:
        funds = [
            _scheme(100, "HDFC Top 100 Direct Plan - Growth"),
            _scheme(200, "ICICI Bluechip Direct Plan - Growth"),
        ]
        self.assertEqual(len(rk._deduplicate_variants(funds)), 2)


# ─────────────────────────────────────────────────────────────────────
# _confidence_level
# ─────────────────────────────────────────────────────────────────────

class ConfidenceLevelTests(unittest.TestCase):

    def test_high_at_ten_years_exact(self) -> None:
        self.assertEqual(rk._confidence_level(10.0), "High")

    def test_high_above_ten_years(self) -> None:
        self.assertEqual(rk._confidence_level(15.0), "High")

    def test_medium_at_five_years_exact(self) -> None:
        self.assertEqual(rk._confidence_level(5.0), "Medium")

    def test_medium_below_ten(self) -> None:
        self.assertEqual(rk._confidence_level(9.99), "Medium")

    def test_low_below_five(self) -> None:
        self.assertEqual(rk._confidence_level(4.99), "Low")
        self.assertEqual(rk._confidence_level(0.0), "Low")


class LegacyAnnualizedReturnTests(unittest.TestCase):
    """The legacy mutual_funds._annualized_return previously returned
    0.0 for total loss (period_return == -1.0). That mislabels a
    wipeout as 'no return' in downstream displays. Fixed: now returns
    -1.0 for the total-loss boundary."""

    def test_legacy_total_loss_returns_minus_one(self) -> None:
        from backend.investment_analytics.mutual_funds import _annualized_return
        self.assertAlmostEqual(_annualized_return(100, 0, 5), -1.0, places=4)

    def test_legacy_normal_loss(self) -> None:
        from backend.investment_analytics.mutual_funds import _annualized_return
        # 100 → 50 in 2 years → 1/sqrt(2) - 1 ≈ -0.293
        self.assertAlmostEqual(_annualized_return(100, 50, 2),
                               (0.5) ** 0.5 - 1, places=4)

    def test_legacy_normal_gain(self) -> None:
        from backend.investment_analytics.mutual_funds import _annualized_return
        self.assertAlmostEqual(_annualized_return(100, 200, 1), 1.0, places=4)

    def test_legacy_zero_years(self) -> None:
        from backend.investment_analytics.mutual_funds import _annualized_return
        self.assertEqual(_annualized_return(100, 200, 0), 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
