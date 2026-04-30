"""Regression tests for backend.investment_analytics.portfolio_health.

Covers the safe-peer-selection path that previously crashed with
UnboundLocalError when the first holding was classified Weak/Neutral
(peer_points / peer_metrics referenced before assignment).
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.investment_analytics import portfolio_health as ph
from backend.investment_analytics.ranking import (
    CategoryRanking,
    FundMetrics,
    RankedFund,
)


def _make_fund(code: int, *, vol: float, dd: float, cons: float, ret: float,
               aligned_points: int = 1500, history_years: float = 8.0) -> FundMetrics:
    return FundMetrics(
        scheme_code=code,
        fund_name=f"Fund {code}",
        fund_house=f"AMC {code}",
        excess_return_pct=ret,
        max_drawdown_pct=dd,
        consistency_pct=cons,
        volatility_pct=vol,
        downside_capture_ratio=0.95,
        fund_cagr_pct=12.0,
        benchmark_cagr_pct=10.0,
        aligned_points=aligned_points,
        history_years=history_years,
        drawdown_trough_date=None,
    )


def _ranking_with_held_at_bottom() -> CategoryRanking:
    """Five-fund ranking where scheme_code 105 sits at the bottom (Weak)."""
    funds = [
        _make_fund(101, vol=10, dd=-12, cons=70, ret=4.0),
        _make_fund(102, vol=11, dd=-13, cons=65, ret=3.5),
        _make_fund(103, vol=12, dd=-14, cons=60, ret=3.0),
        _make_fund(104, vol=13, dd=-15, cons=55, ret=2.5),
        _make_fund(105, vol=22, dd=-32, cons=20, ret=-2.0,
                   aligned_points=900, history_years=4.0),
    ]
    ranked = [
        RankedFund(
            rank=i + 1,
            fund=fund,
            dominance_count=4 - i,
            total_peers=5,
            confidence_level="High" if fund.history_years >= 10 else "Medium" if fund.history_years >= 5 else "Low",
            strengths=[],
            weaknesses=[],
        )
        for i, fund in enumerate(funds)
    ]
    return CategoryRanking(
        category="Equity Scheme - Large Cap Fund",
        benchmark_name="Nifty 100",
        benchmark_code=999,
        benchmark_fallback=False,
        ranked=ranked,
        excluded=[],
        computed_at="2026-04-30T00:00:00+00:00",
        total_funds_in_category=5,
    )


class _Scheme:
    def __init__(self, code: int, name: str, category: str, house: str) -> None:
        self.scheme_code = code
        self.scheme_name = name
        self.scheme_category = category
        self.fund_house = house


class WeakFirstHoldingTests(unittest.TestCase):
    """First holding is Weak — must not crash, must produce alternatives."""

    def test_weak_first_holding_does_not_crash(self) -> None:
        registry = [_Scheme(105, "Held Fund Direct Plan - Growth",
                            "Equity Scheme - Large Cap Fund", "AMC 105")]
        ranking = _ranking_with_held_at_bottom()

        with patch.object(ph, "load_registry", return_value=registry), \
             patch.object(ph, "_get_or_rank_equity", return_value=ranking), \
             patch.object(ph, "_get_or_rank_debt", return_value=None):
            # Must not raise UnboundLocalError.
            result = ph.check_portfolio_health(
                scheme_codes=[105],
                weights=None,
                registry_path="ignored",
            )

        self.assertEqual(len(result.holdings), 1)
        h = result.holdings[0]
        self.assertEqual(h.scheme_code, 105)
        self.assertEqual(h.status, "Weak")
        self.assertEqual(h.action, "Review")
        # Safe peer selection should have produced alternatives from peers 101-104.
        self.assertGreater(len(h.alternatives), 0,
                           "Weak status with viable peers must yield alternatives")
        # And populate the personal-comparison gaps block.
        self.assertGreater(len(h.your_fund_gaps), 0,
                           "your_fund_gaps must be populated when alternatives exist")

    def test_weak_first_holding_followed_by_strong_does_not_leak_state(self) -> None:
        """Cross-iteration leak: peer_points/peer_metrics must NOT carry between funds.

        Holding A (Weak) in category X; holding B (Strong) in different category Y.
        Holding B's safe-peer code path must use B's own category data, not A's.
        Regression for the silent-corruption path.
        """
        registry = [
            _Scheme(105, "Held Weak Direct Plan - Growth",
                    "Equity Scheme - Large Cap Fund", "AMC 105"),
            _Scheme(201, "Held Strong Direct Plan - Growth",
                    "Equity Scheme - Mid Cap Fund", "AMC 201"),
        ]

        weak_ranking = _ranking_with_held_at_bottom()

        # Build a separate ranking for category Y where 201 ranks #1 (Strong).
        funds_y = [
            _make_fund(201, vol=14, dd=-18, cons=72, ret=5.0),
            _make_fund(202, vol=15, dd=-20, cons=60, ret=4.0),
            _make_fund(203, vol=16, dd=-22, cons=55, ret=3.0),
            _make_fund(204, vol=17, dd=-24, cons=50, ret=2.0),
        ]
        ranked_y = [
            RankedFund(
                rank=i + 1, fund=fund, dominance_count=3 - i, total_peers=4,
                confidence_level="High", strengths=[], weaknesses=[],
            )
            for i, fund in enumerate(funds_y)
        ]
        strong_ranking = CategoryRanking(
            category="Equity Scheme - Mid Cap Fund",
            benchmark_name="Nifty Midcap 150",
            benchmark_code=998,
            benchmark_fallback=False,
            ranked=ranked_y,
            excluded=[],
            computed_at="2026-04-30T00:00:00+00:00",
            total_funds_in_category=4,
        )

        def _ranker(category: str, _registry_path: str):
            if category == "Equity Scheme - Large Cap Fund":
                return weak_ranking
            if category == "Equity Scheme - Mid Cap Fund":
                return strong_ranking
            return None

        with patch.object(ph, "load_registry", return_value=registry), \
             patch.object(ph, "_get_or_rank_equity", side_effect=_ranker), \
             patch.object(ph, "_get_or_rank_debt", return_value=None):
            result = ph.check_portfolio_health(
                scheme_codes=[105, 201],
                weights=None,
                registry_path="ignored",
            )

        self.assertEqual(len(result.holdings), 2)
        weak, strong = result.holdings
        self.assertEqual(weak.status, "Weak")
        self.assertEqual(strong.status, "Strong")
        # Strong holding must NOT inherit Weak holding's peer_points (different categories).
        # Concrete check: Strong's data_quality_flags should evaluate against its OWN
        # category's peers — fund 201 has 1500 aligned_points vs peers in Mid Cap, so
        # no severe-gap flag should fire (its peers also have 1500).
        severe_flags = [f for f in strong.data_quality_flags
                        if f.get("severity") == "severe"]
        self.assertEqual(severe_flags, [],
                         "Strong holding must not be flagged severe based on prior fund's peer set")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
