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


class MaterialImprovementGateTests(unittest.TestCase):
    """P4: alternatives must beat held on >=3 metrics AND show one moderate+ improvement."""

    def _ranking_with_marginal_and_material_peers(self) -> CategoryRanking:
        # Held fund (rank 5) is the bottom of the pack.
        held = _make_fund(105, vol=15.0, dd=-20.0, cons=50.0, ret=0.5)
        # Peer 101: rank 1, materially better (excess return +5pp, consistency +20pp).
        material = _make_fund(101, vol=11.0, dd=-12.0, cons=70.0, ret=5.0)
        # Peer 102: rank 2, only marginally better (+0.5pp return, +2pp consistency, etc.)
        marginal = _make_fund(102, vol=14.5, dd=-19.5, cons=52.0, ret=1.0)
        # Peer 103: rank 3, beats held on 2 metrics but is dominated overall on the rest.
        partial = _make_fund(103, vol=14.0, dd=-25.0, cons=48.0, ret=0.7)
        # Peer 104: rank 4, slightly better on volatility only.
        weak = _make_fund(104, vol=14.8, dd=-22.0, cons=49.0, ret=0.4)

        all_funds = [material, marginal, partial, weak, held]
        ranked = []
        for i, fund in enumerate(all_funds):
            ranked.append(RankedFund(
                rank=i + 1, fund=fund, dominance_count=4 - i, total_peers=5,
                confidence_level="High", strengths=[], weaknesses=[],
            ))
        return CategoryRanking(
            category="Equity Scheme - Large Cap Fund",
            benchmark_name="Nifty 100", benchmark_code=999, benchmark_fallback=False,
            ranked=ranked, excluded=[],
            computed_at="2026-04-30T00:00:00+00:00",
            total_funds_in_category=5,
        )

    def test_marginal_alternative_is_filtered_out(self) -> None:
        registry = [_Scheme(105, "Held Direct Plan - Growth",
                            "Equity Scheme - Large Cap Fund", "AMC 105")]
        ranking = self._ranking_with_marginal_and_material_peers()

        with patch.object(ph, "load_registry", return_value=registry), \
             patch.object(ph, "_get_or_rank_equity", return_value=ranking), \
             patch.object(ph, "_get_or_rank_debt", return_value=None):
            result = ph.check_portfolio_health(
                scheme_codes=[105], weights=None, registry_path="ignored",
            )

        h = result.holdings[0]
        self.assertEqual(h.status, "Weak")
        # Material peer (101) survives the gate; marginal peer (102) does not.
        alt_codes = {a["scheme_code"] for a in h.alternatives}
        self.assertIn(101, alt_codes, "Materially better peer must be surfaced")
        self.assertNotIn(102, alt_codes, "Marginal peer must be filtered (P4 gate)")

    def test_justification_carries_magnitude(self) -> None:
        registry = [_Scheme(105, "Held Direct Plan - Growth",
                            "Equity Scheme - Large Cap Fund", "AMC 105")]
        ranking = self._ranking_with_marginal_and_material_peers()

        with patch.object(ph, "load_registry", return_value=registry), \
             patch.object(ph, "_get_or_rank_equity", return_value=ranking), \
             patch.object(ph, "_get_or_rank_debt", return_value=None):
            result = ph.check_portfolio_health(
                scheme_codes=[105], weights=None, registry_path="ignored",
            )

        h = result.holdings[0]
        self.assertGreater(len(h.alternatives), 0)
        for alt in h.alternatives:
            self.assertIn("metrics", alt, "alt must expose full metric set for UI-7")
            for j in alt["justification"]:
                self.assertIn("reason", j)
                self.assertIn("magnitude", j)
                self.assertIn(j["magnitude"], {"small", "moderate", "large"})

    def test_no_alternative_when_only_marginal_peers_exist(self) -> None:
        """Held vs all-marginal peers — gate should drop everyone, action_note explains."""
        # Held at rank 5; peers 101-104 only marginally better on each metric.
        held = _make_fund(105, vol=15.0, dd=-20.0, cons=50.0, ret=2.0)
        peers = [
            _make_fund(101, vol=14.5, dd=-19.0, cons=52.0, ret=2.5),
            _make_fund(102, vol=14.7, dd=-19.5, cons=51.0, ret=2.3),
            _make_fund(103, vol=14.9, dd=-19.8, cons=50.5, ret=2.1),
            _make_fund(104, vol=14.95, dd=-19.9, cons=50.2, ret=2.05),
        ]
        ranked = [
            RankedFund(rank=i + 1, fund=f, dominance_count=4 - i, total_peers=5,
                       confidence_level="High", strengths=[], weaknesses=[])
            for i, f in enumerate(peers + [held])
        ]
        ranking = CategoryRanking(
            category="Equity Scheme - Large Cap Fund",
            benchmark_name="Nifty 100", benchmark_code=999, benchmark_fallback=False,
            ranked=ranked, excluded=[],
            computed_at="2026-04-30T00:00:00+00:00",
            total_funds_in_category=5,
        )
        registry = [_Scheme(105, "Held Direct Plan - Growth",
                            "Equity Scheme - Large Cap Fund", "AMC 105")]
        with patch.object(ph, "load_registry", return_value=registry), \
             patch.object(ph, "_get_or_rank_equity", return_value=ranking), \
             patch.object(ph, "_get_or_rank_debt", return_value=None):
            result = ph.check_portfolio_health(
                scheme_codes=[105], weights=None, registry_path="ignored",
            )

        h = result.holdings[0]
        self.assertEqual(len(h.alternatives), 0,
                         "All peers marginal — gate must drop them")
        # OBS-2: when peers existed but the material gate filtered them all,
        # the action_note distinguishes "no materially better" from
        # "data limitations". Held is Weak in this fixture, so we expect
        # the Weak-specific phrasing.
        self.assertEqual(
            h.action_note,
            "No materially better peer in this category — current ranking position is comparable to top peers",
        )


class ActionNoteFraming(unittest.TestCase):
    """OBS-1 / OBS-2: action_note must distinguish data-limit cases from
    no-materially-better-peer cases, and Weak / Neutral phrasing must
    match the holding's status."""

    def test_neutral_with_comparable_peers_says_comparable(self) -> None:
        """OBS-1: Neutral holding + only marginally-better peers must
        emit a 'comparable to peers' note, NOT 'data limitations'."""
        cat = "Equity Scheme - Large Cap Fund"
        funds = [
            _make_fund(801, vol=11.5, dd=-14.5, cons=53, ret=2.6),
            _make_fund(802, vol=11.7, dd=-14.7, cons=52, ret=2.5),
            _make_fund(803, vol=12.0, dd=-15.0, cons=51, ret=2.3),  # held: rank 3
            _make_fund(804, vol=12.3, dd=-15.5, cons=50, ret=2.0),
            _make_fund(805, vol=12.5, dd=-15.8, cons=49, ret=1.8),
        ]
        ranked = [
            RankedFund(rank=i + 1, fund=f, dominance_count=4 - i, total_peers=5,
                       confidence_level="High", strengths=[], weaknesses=[])
            for i, f in enumerate(funds)
        ]
        ranking = CategoryRanking(
            category=cat, benchmark_name="Nifty 100", benchmark_code=999,
            benchmark_fallback=False, ranked=ranked, excluded=[],
            computed_at="2026-04-30T00:00:00+00:00", total_funds_in_category=5,
        )
        registry = [_Scheme(803, "Held Direct Plan - Growth", cat, "AMC 803")]
        with patch.object(ph, "load_registry", return_value=registry), \
             patch.object(ph, "_get_or_rank_equity", return_value=ranking), \
             patch.object(ph, "_get_or_rank_debt", return_value=None):
            result = ph.check_portfolio_health(
                scheme_codes=[803], weights=None, registry_path="ignored",
            )
        h = result.holdings[0]
        self.assertEqual(h.status, "Neutral")
        self.assertEqual(len(h.alternatives), 0)
        self.assertIn("comparable", h.action_note.lower())
        self.assertNotIn("data limitations", h.action_note.lower())

    def test_weak_with_all_low_conf_peers_says_data_limitations(self) -> None:
        """OBS-2: Weak holding + every peer Low-conf must say
        'data limitations', NOT 'no materially better peer'."""
        cat = "Equity Scheme - Mid Cap Fund"
        # Held has full history so its peer-percentile is computed,
        # but every peer has <5y history → all peers Low-confidence.
        held = _make_fund(901, vol=20, dd=-30, cons=15, ret=-3.0,
                          history_years=8.0)
        peers = [
            _make_fund(902 + i, vol=15 + i, dd=-18, cons=55 - i,
                       ret=4 - i, history_years=3.0)
            for i in range(4)
        ]
        # 902 = rank 1, ... 905 = rank 4, held 901 = rank 5 (Weak)
        ordered = peers + [held]
        ranked = [
            RankedFund(
                rank=i + 1, fund=f, dominance_count=4 - i, total_peers=5,
                confidence_level=("Low" if f.history_years < 5 else "High"),
                strengths=[], weaknesses=[],
            )
            for i, f in enumerate(ordered)
        ]
        ranking = CategoryRanking(
            category=cat, benchmark_name="Nifty Midcap 150", benchmark_code=998,
            benchmark_fallback=False, ranked=ranked, excluded=[],
            computed_at="2026-04-30T00:00:00+00:00", total_funds_in_category=5,
        )
        registry = [_Scheme(901, "Held Direct Plan - Growth", cat, "AMC 901")]
        with patch.object(ph, "load_registry", return_value=registry), \
             patch.object(ph, "_get_or_rank_equity", return_value=ranking), \
             patch.object(ph, "_get_or_rank_debt", return_value=None):
            result = ph.check_portfolio_health(
                scheme_codes=[901], weights=None, registry_path="ignored",
            )
        h = result.holdings[0]
        self.assertEqual(h.status, "Weak")
        self.assertEqual(len(h.alternatives), 0)
        self.assertIn("data limitations", h.action_note.lower())


class DecisionSummaryWeightTests(unittest.TestCase):
    """UI-1: decision summary entries must carry weight_pct and sort by it desc."""

    def _ranking_with_three_holdings(self) -> Tuple[List, CategoryRanking]:
        # Three funds: 201 Strong (rank 1), 202 Neutral (rank 3), 203 Weak (rank 5).
        funds = [
            _make_fund(201, vol=10, dd=-12, cons=72, ret=5.0),
            _make_fund(204, vol=11, dd=-13, cons=66, ret=4.0),
            _make_fund(202, vol=12, dd=-15, cons=55, ret=2.5),
            _make_fund(205, vol=13, dd=-17, cons=45, ret=1.0),
            _make_fund(203, vol=20, dd=-30, cons=20, ret=-3.0),
        ]
        ranked = [
            RankedFund(rank=i + 1, fund=f, dominance_count=4 - i, total_peers=5,
                       confidence_level="High", strengths=[], weaknesses=[])
            for i, f in enumerate(funds)
        ]
        ranking = CategoryRanking(
            category="Equity Scheme - Large Cap Fund",
            benchmark_name="Nifty 100", benchmark_code=999, benchmark_fallback=False,
            ranked=ranked, excluded=[],
            computed_at="2026-04-30T00:00:00+00:00",
            total_funds_in_category=5,
        )
        registry = [
            _Scheme(201, "Big Fund Direct Plan - Growth",
                    "Equity Scheme - Large Cap Fund", "AMC 201"),
            _Scheme(202, "Mid Fund Direct Plan - Growth",
                    "Equity Scheme - Large Cap Fund", "AMC 202"),
            _Scheme(203, "Tiny Fund Direct Plan - Growth",
                    "Equity Scheme - Large Cap Fund", "AMC 203"),
        ]
        return registry, ranking

    def test_decision_summary_entries_include_weight_pct(self) -> None:
        from backend.investment_analytics.portfolio_health import (
            check_portfolio_health, portfolio_health_to_dict,
        )
        registry, ranking = self._ranking_with_three_holdings()
        # Heavy on the Weak holding so it should sort first in Review.
        weights = {201: 0.15, 202: 0.25, 203: 0.60}
        with patch.object(ph, "load_registry", return_value=registry), \
             patch.object(ph, "_get_or_rank_equity", return_value=ranking), \
             patch.object(ph, "_get_or_rank_debt", return_value=None):
            result = check_portfolio_health(
                scheme_codes=[201, 202, 203], weights=weights, registry_path="ignored",
            )
        out = portfolio_health_to_dict(result, weights=weights)
        self.assertEqual(out["schema_version"], "v2")
        ds = out["decision_summary"]
        for bucket in ("Continue", "Monitor", "Review"):
            for entry in ds[bucket]:
                self.assertIn("weight_pct", entry)
        # Bucket totals exposed for the UI column header.
        self.assertIn("decision_summary_weight_pct", out)

    def test_decision_summary_is_sorted_by_weight_desc(self) -> None:
        from backend.investment_analytics.portfolio_health import (
            check_portfolio_health, portfolio_health_to_dict,
        )
        registry, ranking = self._ranking_with_three_holdings()
        weights = {201: 0.10, 202: 0.30, 203: 0.60}
        with patch.object(ph, "load_registry", return_value=registry), \
             patch.object(ph, "_get_or_rank_equity", return_value=ranking), \
             patch.object(ph, "_get_or_rank_debt", return_value=None):
            result = check_portfolio_health(
                scheme_codes=[201, 202, 203], weights=weights, registry_path="ignored",
            )
        out = portfolio_health_to_dict(result, weights=weights)
        # Within each non-empty bucket entries must be desc by weight_pct.
        for bucket in ("Continue", "Monitor", "Review"):
            entries = out["decision_summary"][bucket]
            for i in range(1, len(entries)):
                self.assertGreaterEqual(
                    entries[i - 1]["weight_pct"], entries[i]["weight_pct"],
                    f"{bucket} entries not sorted desc by weight_pct",
                )


class CoverageIntegrityTests(unittest.TestCase):
    """Item 1: capital-weighted analyzed coverage must classify the
    portfolio into full/partial/low confidence bands and emit a note
    that the UI can render above conclusions."""

    def _ranking(self, cat: str = "Equity Scheme - Large Cap Fund") -> CategoryRanking:
        funds = [
            _make_fund(901, vol=11, dd=-12, cons=70, ret=4.0),
            _make_fund(902, vol=12, dd=-13, cons=65, ret=3.0),
            _make_fund(903, vol=13, dd=-14, cons=58, ret=2.0),
            _make_fund(904, vol=14, dd=-15, cons=55, ret=1.5),
            _make_fund(905, vol=20, dd=-30, cons=20, ret=-2.0),
        ]
        ranked = [
            RankedFund(rank=i + 1, fund=f, dominance_count=4 - i, total_peers=5,
                       confidence_level="High", strengths=[], weaknesses=[])
            for i, f in enumerate(funds)
        ]
        return CategoryRanking(
            category=cat, benchmark_name="Nifty 100", benchmark_code=999,
            benchmark_fallback=False, ranked=ranked, excluded=[],
            computed_at="2026-04-30T00:00:00+00:00", total_funds_in_category=5,
        )

    def _run(self, scheme_codes, registry, weights=None):
        ranking = self._ranking()
        with patch.object(ph, "load_registry", return_value=registry), \
             patch.object(ph, "_get_or_rank_equity", return_value=ranking), \
             patch.object(ph, "_get_or_rank_debt", return_value=None), \
             patch.object(ph, "fetch_scheme_nav",
                          side_effect=RuntimeError("no NAV in test")):
            return ph.check_portfolio_health(
                scheme_codes=scheme_codes, weights=weights,
                registry_path="ignored",
            )

    def test_full_coverage_band_when_all_holdings_ranked(self) -> None:
        cat = "Equity Scheme - Large Cap Fund"
        registry = [_Scheme(905, "Held Direct Plan - Growth", cat, "AMC X")]
        result = self._run([905], registry)
        cov = result.coverage
        self.assertIsNotNone(cov)
        self.assertEqual(cov.confidence_band, "full")
        self.assertEqual(cov.analyzed_pct, 100.0)
        self.assertEqual(cov.not_ranked_pct, 0.0)
        self.assertEqual(cov.note, "")
        self.assertEqual(cov.affected_metrics, [])

    def test_partial_coverage_band_when_etf_is_minority(self) -> None:
        # 70/30: ranked equity fund + ETF (Not Ranked).
        cat = "Equity Scheme - Large Cap Fund"
        registry = [
            _Scheme(905, "Held Direct Plan - Growth", cat, "AMC X"),
            _Scheme(7777, "Some Index ETF Direct - Growth",
                    "Other Scheme - Index Funds", "AMC Y"),
        ]
        weights = {905: 0.65, 7777: 0.35}  # 65% ranked → "partial" (50-70)
        result = self._run([905, 7777], registry, weights=weights)
        cov = result.coverage
        self.assertEqual(cov.confidence_band, "partial")
        self.assertAlmostEqual(cov.analyzed_pct, 65.0, places=1)
        self.assertAlmostEqual(cov.not_ranked_pct, 35.0, places=1)
        self.assertIn("portfolio-level", cov.note.lower())
        self.assertGreater(len(cov.affected_metrics), 0)

    def test_low_coverage_band_when_etf_is_majority(self) -> None:
        cat = "Equity Scheme - Large Cap Fund"
        registry = [
            _Scheme(905, "Held Direct Plan - Growth", cat, "AMC X"),
            _Scheme(7777, "Index ETF Direct - Growth",
                    "Other Scheme - Index Funds", "AMC Y"),
        ]
        weights = {905: 0.30, 7777: 0.70}  # 30% ranked → "low"
        result = self._run([905, 7777], registry, weights=weights)
        cov = result.coverage
        self.assertEqual(cov.confidence_band, "low")
        self.assertLess(cov.analyzed_pct, 50.0)
        self.assertIn("misleading", cov.note.lower())

    def test_coverage_serialised_in_response(self) -> None:
        cat = "Equity Scheme - Large Cap Fund"
        registry = [
            _Scheme(905, "Held Direct Plan - Growth", cat, "AMC X"),
            _Scheme(7777, "ETF Direct - Growth",
                    "Other Scheme - Index Funds", "AMC Y"),
        ]
        weights = {905: 0.40, 7777: 0.60}
        result = self._run([905, 7777], registry, weights=weights)
        out = ph.portfolio_health_to_dict(result, weights=weights)
        self.assertIn("coverage", out)
        cov = out["coverage"]
        self.assertEqual(cov["confidence_band"], "low")
        self.assertEqual(cov["total_holdings"], 2)
        self.assertEqual(cov["analyzed_holdings"], 1)
        self.assertIn("affected_metrics", cov)


class CompressionLayerTests(unittest.TestCase):
    """A/B/C/D + top-ranked + Direct/Regular: signals are compressed,
    not advisory; overlapping signals dedup; low coverage suppresses
    portfolio-level conclusions."""

    def _ranking(self, cat="Equity Scheme - Large Cap Fund"):
        funds = [
            _make_fund(2001, vol=11, dd=-12, cons=70, ret=4.0),
            _make_fund(2002, vol=12, dd=-13, cons=65, ret=3.0),
            _make_fund(2003, vol=13, dd=-14, cons=58, ret=2.0),
            _make_fund(2004, vol=14, dd=-15, cons=55, ret=1.5),
            _make_fund(2005, vol=20, dd=-30, cons=20, ret=-2.0),
        ]
        ranked = [
            RankedFund(rank=i + 1, fund=f, dominance_count=4 - i, total_peers=5,
                       confidence_level="High", strengths=[], weaknesses=[])
            for i, f in enumerate(funds)
        ]
        return CategoryRanking(
            category=cat, benchmark_name="Nifty 100", benchmark_code=999,
            benchmark_fallback=False, ranked=ranked, excluded=[],
            computed_at="2026-04-30T00:00:00+00:00", total_funds_in_category=5,
        )

    def _run(self, scheme_codes, registry, weights=None):
        ranking = self._ranking()
        with patch.object(ph, "load_registry", return_value=registry), \
             patch.object(ph, "_get_or_rank_equity", return_value=ranking), \
             patch.object(ph, "_get_or_rank_debt", return_value=None), \
             patch.object(ph, "fetch_scheme_nav",
                          side_effect=RuntimeError("no NAV in test")):
            result = ph.check_portfolio_health(
                scheme_codes=scheme_codes, weights=weights,
                registry_path="ignored",
            )
        out = ph.portfolio_health_to_dict(result, weights=weights)
        return result, out

    def test_action_priority_picks_highest_weight_review(self) -> None:
        """A: heaviest Review wins, with severity tiebreaker over Monitor."""
        cat = "Equity Scheme - Large Cap Fund"
        registry = [
            _Scheme(2001, "Top Direct Plan - Growth", cat, "AMC X"),  # Strong
            _Scheme(2003, "Mid Direct Plan - Growth", cat, "AMC Y"),  # Neutral
            _Scheme(2005, "Weak Direct Plan - Growth", cat, "AMC Z"),  # Weak
        ]
        # Tiny Weak holding (10%), big Neutral (60%), small Strong (30%).
        # Severity wins: Weak Review picked even though Neutral has more weight.
        weights = {2001: 0.30, 2003: 0.60, 2005: 0.10}
        _, out = self._run([2001, 2003, 2005], registry, weights=weights)
        ap = out["action_priority"]
        self.assertIsNotNone(ap)
        self.assertEqual(ap["scheme_code"], 2005)  # the Review wins by severity
        self.assertEqual(ap["action"], "Review")
        self.assertIn("Address first:", ap["headline"])

    def test_action_priority_falls_back_to_heaviest_monitor(self) -> None:
        """If no Review exists, pick the heaviest Monitor."""
        cat = "Equity Scheme - Large Cap Fund"
        registry = [
            _Scheme(2001, "Top Direct Plan - Growth", cat, "AMC X"),
            _Scheme(2003, "Mid Direct Plan - Growth", cat, "AMC Y"),
        ]
        weights = {2001: 0.40, 2003: 0.60}  # 2003 (Neutral) is bigger
        _, out = self._run([2001, 2003], registry, weights=weights)
        ap = out["action_priority"]
        self.assertEqual(ap["action"], "Monitor")
        self.assertEqual(ap["scheme_code"], 2003)

    def test_action_priority_none_when_all_continue(self) -> None:
        """All Strong/Continue → no priority headline (nothing to do)."""
        cat = "Equity Scheme - Large Cap Fund"
        registry = [_Scheme(2001, "Top Direct Plan - Growth", cat, "AMC X")]
        _, out = self._run([2001], registry)
        self.assertIsNone(out["action_priority"])

    def test_portfolio_status_downgrades_under_low_coverage(self) -> None:
        """B: low coverage forces a coverage-aware label, suppressing
        'Well diversified' when half the portfolio wasn't analyzed."""
        cat = "Equity Scheme - Large Cap Fund"
        registry = [
            _Scheme(2001, "Top Direct Plan - Growth", cat, "AMC X"),
            _Scheme(8888, "ETF Direct - Growth",
                    "Other Scheme - Index Funds", "AMC Y"),
        ]
        weights = {2001: 0.30, 8888: 0.70}  # 30% ranked → low band
        _, out = self._run([2001, 8888], registry, weights=weights)
        self.assertEqual(out["coverage"]["confidence_band"], "low")
        self.assertIn("Coverage limited", out["portfolio_status"])

    def test_portfolio_conclusions_suppressed_under_low_coverage(self) -> None:
        """D: under low coverage, concentration/redundancy/correlation/
        exposure_gaps are suppressed — banner explains why."""
        cat = "Equity Scheme - Large Cap Fund"
        registry = [
            _Scheme(2001, "Top Direct Plan - Growth", cat, "AMC X"),
            _Scheme(8888, "ETF Direct - Growth",
                    "Other Scheme - Index Funds", "AMC Y"),
        ]
        weights = {2001: 0.30, 8888: 0.70}
        _, out = self._run([2001, 8888], registry, weights=weights)
        self.assertEqual(out["concentration"], [])
        self.assertEqual(out["redundancies"], [])
        self.assertEqual(out["correlations"], [])
        self.assertEqual(out["exposure_gaps"], [])
        # Top-ranked also suppressed under low coverage.
        self.assertEqual(out["top_ranked_by_category"], [])

    def test_top_ranked_per_held_category_non_advisory(self) -> None:
        """E: rank-1 fund per held category surfaces — but only for
        categories the user already holds, never new ones."""
        cat = "Equity Scheme - Large Cap Fund"
        registry = [_Scheme(2003, "Mid Direct Plan - Growth", cat, "AMC X")]
        _, out = self._run([2003], registry)
        rows = out["top_ranked_by_category"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["category"], cat)
        self.assertEqual(rows[0]["scheme_code"], 2001)  # rank 1
        # Held fund is rank 3; top-ranked is different fund.
        self.assertNotEqual(rows[0]["scheme_code"], 2003)

    def test_top_ranked_skipped_when_user_already_holds_top(self) -> None:
        """If the held fund IS already the top-ranked in its category,
        we don't duplicate the card via top_ranked_by_category."""
        cat = "Equity Scheme - Large Cap Fund"
        registry = [_Scheme(2001, "Top Direct Plan - Growth", cat, "AMC X")]
        _, out = self._run([2001], registry)
        self.assertEqual(out["top_ranked_by_category"], [])

    def test_regular_plan_detected_with_direct_sibling(self) -> None:
        """F: a Regular Plan holding is flagged with the Direct sibling's code."""
        cat = "Equity Scheme - Large Cap Fund"
        # Registry contains BOTH variants of the same scheme so the
        # base-name matcher can find the Direct sibling.
        registry = [
            _Scheme(3001, "Acme Bluechip Regular Plan - Growth", cat, "Acme MF"),
            _Scheme(3002, "Acme Bluechip Direct Plan - Growth", cat, "Acme MF"),
            _Scheme(2003, "Mid Direct Plan - Growth", cat, "AMC Y"),
        ]
        _, out = self._run([3001, 2003], registry)
        flags = out["plan_efficiency_flags"]
        self.assertEqual(len(flags), 1)
        f = flags[0]
        self.assertEqual(f["scheme_code"], 3001)
        self.assertTrue(f["is_regular_plan"])
        self.assertIsNotNone(f["direct_sibling"])
        self.assertEqual(f["direct_sibling"]["scheme_code"], 3002)
        self.assertIn("Plan choice is structural", f["message"])
        # No advisory verbs.
        for forbidden in ["should", "buy", "sell", "switch", "best", "recommend"]:
            self.assertNotIn(forbidden, f["message"].lower())

    def test_regular_plan_detected_without_direct_sibling(self) -> None:
        """If no Direct sibling exists in the registry, we still flag
        the Regular plan but direct_sibling is None."""
        cat = "Equity Scheme - Large Cap Fund"
        registry = [
            _Scheme(3001, "Acme Bluechip Regular Plan - Growth", cat, "Acme MF"),
        ]
        _, out = self._run([3001], registry)
        flags = out["plan_efficiency_flags"]
        self.assertEqual(len(flags), 1)
        self.assertIsNone(flags[0]["direct_sibling"])

    def test_direct_plan_holdings_not_flagged(self) -> None:
        cat = "Equity Scheme - Large Cap Fund"
        registry = [_Scheme(2001, "Top Direct Plan - Growth", cat, "AMC X")]
        _, out = self._run([2001], registry)
        self.assertEqual(out["plan_efficiency_flags"], [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
