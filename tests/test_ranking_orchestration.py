"""End-to-end orchestration tests for ranking.py's public functions.

Before this file, rank_category / rank_debt_category / rank_gold_funds
/ rank_all_assets were only exercised via portfolio_health integration
or via the live network. Both paths skip code branches that this file
covers explicitly:

- Excluded categories raise.
- Sparse registry (< 2 Direct Growth funds) raises.
- Benchmark fallback path (BENCHMARK_FALLBACK_CATEGORIES with
  insufficient benchmark history) triggers the secondary fetch.
- Self-benchmark (fund IS the benchmark) gets excluded with the
  correct reason.
- Individual fund fetch errors get isolated to the excluded list,
  don't abort the whole category.
- Fewer than 2 computed funds raises.

All tests mock fetch_scheme_nav so no network calls happen.
"""
from __future__ import annotations

import random
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from backend.data_discovery.registry import SchemeEntry, save_registry
from backend.investment_analytics import ranking as rk


# ─────────────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────────────

def _gen_nav_blob(
    n_points: int = 1500,
    start_nav: float = 100.0,
    daily_drift: float = 0.0005,
    daily_vol: float = 0.0,
    start_date: str = "2018-01-01",
    seed: int = 0,
) -> dict:
    """mfapi-style payload (newest-first) with deterministic NAVs."""
    rng = random.Random(seed)
    d = date.fromisoformat(start_date)
    out, nav = [], start_nav
    for _ in range(n_points):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        out.append({"date": d.strftime("%d-%m-%Y"), "nav": f"{nav:.4f}"})
        ret = daily_drift + (daily_vol * rng.gauss(0, 1) if daily_vol else 0)
        nav = max(0.01, nav * (1.0 + ret))
        d += timedelta(days=1)
    out.reverse()  # mfapi convention: newest first
    return {"meta": {}, "data": out}


def _scheme(code: int, name: str, category: str, house: str = "AMC X") -> SchemeEntry:
    return SchemeEntry(
        scheme_code=code, scheme_name=name,
        scheme_category=category, fund_house=house,
    )


def _persist_registry(entries: List[SchemeEntry], tmpdir: Path) -> str:
    """save_registry writes JSON we can hand to rank_category."""
    path = tmpdir / "schemes.json"
    save_registry(entries, path)
    return str(path)


class RankCategoryOrchestrationTests(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _registry(self, *, category: str, fund_codes: List[int]) -> str:
        entries = [
            _scheme(c, f"Test {c} Direct Plan - Growth", category)
            for c in fund_codes
        ]
        return _persist_registry(entries, self.tmpdir)

    def test_excluded_category_raises(self) -> None:
        path = self._registry(category="Equity Scheme - Sectoral/ Thematic",
                              fund_codes=[1, 2, 3])
        with self.assertRaises(ValueError) as ctx:
            rk.rank_category("Equity Scheme - Sectoral/ Thematic", path)
        self.assertIn("excluded", str(ctx.exception).lower())

    def test_fewer_than_two_funds_raises(self) -> None:
        path = self._registry(category="Equity Scheme - Large Cap Fund",
                              fund_codes=[1])
        with self.assertRaises(ValueError) as ctx:
            rk.rank_category("Equity Scheme - Large Cap Fund", path)
        self.assertIn("fewer than 2", str(ctx.exception))

    def test_happy_path_ranks_all_funds(self) -> None:
        path = self._registry(
            category="Equity Scheme - Large Cap Fund",
            fund_codes=[101, 102, 103, 104],
        )
        # Differentiated funds: 101 climbs fastest, 104 slowest.
        nav_blobs = {
            120716: _gen_nav_blob(daily_drift=0.0003, seed=0),  # benchmark
            101: _gen_nav_blob(daily_drift=0.0007, seed=1),
            102: _gen_nav_blob(daily_drift=0.0006, seed=2),
            103: _gen_nav_blob(daily_drift=0.0005, seed=3),
            104: _gen_nav_blob(daily_drift=0.0004, seed=4),
        }
        with patch.object(rk, "fetch_scheme_nav",
                          side_effect=lambda code: nav_blobs[code]):
            result = rk.rank_category("Equity Scheme - Large Cap Fund", path)
        self.assertEqual(len(result.ranked), 4)
        self.assertEqual(len(result.excluded), 0)
        self.assertEqual(result.benchmark_fallback, False)
        # Top-ranked should be the highest-drift fund (101).
        self.assertEqual(result.ranked[0].fund.scheme_code, 101)
        self.assertEqual(result.ranked[-1].fund.scheme_code, 104)

    def test_benchmark_fallback_when_primary_has_insufficient_history(self) -> None:
        """Categories in BENCHMARK_FALLBACK_CATEGORIES (e.g., Flexi Cap)
        try their mapped benchmark first; if it has insufficient data,
        fall back to Nifty 50 (DEFAULT_BENCHMARK)."""
        path = self._registry(
            category="Equity Scheme - Flexi Cap Fund",
            fund_codes=[201, 202],
        )
        # Map Flexi Cap to a primary benchmark whose data is too short.
        primary_bench_code = rk.CATEGORY_BENCHMARK_MAP[
            "Equity Scheme - Flexi Cap Fund"][0]
        default_bench_code = rk.DEFAULT_BENCHMARK[0]
        nav_blobs = {
            primary_bench_code: _gen_nav_blob(n_points=100),  # < MIN_ALIGNED_POINTS
            default_bench_code: _gen_nav_blob(daily_drift=0.0003, seed=10),
            201: _gen_nav_blob(daily_drift=0.0007, seed=11),
            202: _gen_nav_blob(daily_drift=0.0005, seed=12),
        }
        with patch.object(rk, "fetch_scheme_nav",
                          side_effect=lambda code: nav_blobs[code]):
            result = rk.rank_category("Equity Scheme - Flexi Cap Fund", path)
        self.assertTrue(result.benchmark_fallback)
        # Benchmark info should reflect the fallback.
        self.assertEqual(result.benchmark_code, default_bench_code)

    def test_fund_that_is_the_benchmark_is_excluded(self) -> None:
        bench_code = rk.CATEGORY_BENCHMARK_MAP[
            "Equity Scheme - Large Cap Fund"][0]
        # User somehow holds the benchmark index fund itself.
        path = self._registry(
            category="Equity Scheme - Large Cap Fund",
            fund_codes=[bench_code, 301, 302],
        )
        nav_blobs = {
            bench_code: _gen_nav_blob(daily_drift=0.0003, seed=20),
            301: _gen_nav_blob(daily_drift=0.0007, seed=21),
            302: _gen_nav_blob(daily_drift=0.0005, seed=22),
        }
        with patch.object(rk, "fetch_scheme_nav",
                          side_effect=lambda code: nav_blobs[code]):
            result = rk.rank_category("Equity Scheme - Large Cap Fund", path)
        excluded_codes = {e.scheme_code for e in result.excluded}
        self.assertIn(bench_code, excluded_codes)
        # The other two should rank.
        ranked_codes = {rf.fund.scheme_code for rf in result.ranked}
        self.assertEqual(ranked_codes, {301, 302})

    def test_fund_with_insufficient_data_is_excluded(self) -> None:
        path = self._registry(
            category="Equity Scheme - Large Cap Fund",
            fund_codes=[401, 402, 403],
        )
        bench_code = rk.CATEGORY_BENCHMARK_MAP[
            "Equity Scheme - Large Cap Fund"][0]
        nav_blobs = {
            bench_code: _gen_nav_blob(daily_drift=0.0003, seed=30),
            401: _gen_nav_blob(daily_drift=0.0007, seed=31),
            402: _gen_nav_blob(daily_drift=0.0005, seed=32),
            403: _gen_nav_blob(n_points=100, seed=33),  # too few points
        }
        with patch.object(rk, "fetch_scheme_nav",
                          side_effect=lambda code: nav_blobs[code]):
            result = rk.rank_category("Equity Scheme - Large Cap Fund", path)
        excluded_codes = {e.scheme_code for e in result.excluded}
        self.assertIn(403, excluded_codes)
        # 401 and 402 should still rank.
        ranked_codes = {rf.fund.scheme_code for rf in result.ranked}
        self.assertEqual(ranked_codes, {401, 402})

    def test_fund_with_fetch_error_is_excluded(self) -> None:
        path = self._registry(
            category="Equity Scheme - Large Cap Fund",
            fund_codes=[501, 502, 503],
        )
        bench_code = rk.CATEGORY_BENCHMARK_MAP[
            "Equity Scheme - Large Cap Fund"][0]
        nav_blobs = {
            bench_code: _gen_nav_blob(seed=40),
            501: _gen_nav_blob(daily_drift=0.0007, seed=41),
            502: _gen_nav_blob(daily_drift=0.0005, seed=42),
        }

        def fake_fetch(code):
            if code == 503:
                raise RuntimeError("simulated mfapi outage")
            return nav_blobs[code]

        with patch.object(rk, "fetch_scheme_nav", side_effect=fake_fetch):
            result = rk.rank_category("Equity Scheme - Large Cap Fund", path)
        excluded_codes = {e.scheme_code for e in result.excluded}
        self.assertIn(503, excluded_codes)
        # The error message should mention the failure cause.
        bad = next(e for e in result.excluded if e.scheme_code == 503)
        self.assertIn("error", bad.reason.lower())

    def test_fewer_than_two_computed_funds_raises(self) -> None:
        """If most funds fail to fetch, ranking is impossible."""
        path = self._registry(
            category="Equity Scheme - Large Cap Fund",
            fund_codes=[601, 602, 603],
        )
        bench_code = rk.CATEGORY_BENCHMARK_MAP[
            "Equity Scheme - Large Cap Fund"][0]
        nav_blobs = {
            bench_code: _gen_nav_blob(seed=50),
            601: _gen_nav_blob(daily_drift=0.0007, seed=51),
            # 602 and 603 fail.
        }

        def fake_fetch(code):
            if code in (602, 603):
                raise RuntimeError("no NAV")
            return nav_blobs[code]

        with patch.object(rk, "fetch_scheme_nav", side_effect=fake_fetch):
            with self.assertRaises(ValueError) as ctx:
                rk.rank_category("Equity Scheme - Large Cap Fund", path)
        self.assertIn("Only 1 funds had sufficient data", str(ctx.exception))

    def test_benchmark_insufficient_data_raises(self) -> None:
        path = self._registry(
            category="Equity Scheme - Large Cap Fund",
            fund_codes=[701, 702],
        )
        bench_code = rk.CATEGORY_BENCHMARK_MAP[
            "Equity Scheme - Large Cap Fund"][0]
        # Primary AND fallback benchmark have too few points.
        nav_blobs = {
            bench_code: _gen_nav_blob(n_points=100),
            rk.DEFAULT_BENCHMARK[0]: _gen_nav_blob(n_points=100),
            701: _gen_nav_blob(seed=60),
            702: _gen_nav_blob(seed=61),
        }
        with patch.object(rk, "fetch_scheme_nav",
                          side_effect=lambda code: nav_blobs[code]):
            with self.assertRaises(ValueError) as ctx:
                rk.rank_category("Equity Scheme - Large Cap Fund", path)
        self.assertIn("insufficient data", str(ctx.exception).lower())


class RankDebtCategoryOrchestrationTests(unittest.TestCase):
    """Debt ranking uses absolute metrics (no benchmark)."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_happy_path_ranks_all_debt_funds(self) -> None:
        entries = [
            _scheme(c, f"Debt {c} Direct Plan - Growth",
                    "Debt Scheme - Short Duration Fund")
            for c in [801, 802, 803]
        ]
        path = _persist_registry(entries, self.tmpdir)
        nav_blobs = {
            801: _gen_nav_blob(daily_drift=0.0003, seed=70),
            802: _gen_nav_blob(daily_drift=0.00025, seed=71),
            803: _gen_nav_blob(daily_drift=0.0002, seed=72),
        }
        with patch.object(rk, "fetch_scheme_nav",
                          side_effect=lambda code: nav_blobs[code]):
            result = rk.rank_debt_category(
                "Debt Scheme - Short Duration Fund", path)
        self.assertEqual(len(result.ranked), 3)
        self.assertEqual(len(result.excluded), 0)

    def test_fewer_than_two_debt_funds_raises(self) -> None:
        entries = [_scheme(801, "Debt 801 Direct Plan - Growth",
                           "Debt Scheme - Short Duration Fund")]
        path = _persist_registry(entries, self.tmpdir)
        with self.assertRaises(ValueError):
            rk.rank_debt_category("Debt Scheme - Short Duration Fund", path)


class RankAllAssetsOrchestrationTests(unittest.TestCase):
    """The top-level entry point that drives equity + debt + gold."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_partial_failure_does_not_abort_run(self) -> None:
        """If one equity category errors, others must still rank."""
        entries = [
            # Two large cap funds.
            _scheme(901, "X Large Direct Plan - Growth",
                    "Equity Scheme - Large Cap Fund"),
            _scheme(902, "Y Large Direct Plan - Growth",
                    "Equity Scheme - Large Cap Fund"),
            # Mid cap empty — should error.
            # Short Duration debt.
            _scheme(951, "X Debt Direct Plan - Growth",
                    "Debt Scheme - Short Duration Fund"),
            _scheme(952, "Y Debt Direct Plan - Growth",
                    "Debt Scheme - Short Duration Fund"),
        ]
        path = _persist_registry(entries, self.tmpdir)
        bench_lc = rk.CATEGORY_BENCHMARK_MAP[
            "Equity Scheme - Large Cap Fund"][0]
        nav_blobs = {
            bench_lc: _gen_nav_blob(seed=80),
            901: _gen_nav_blob(daily_drift=0.0007, seed=81),
            902: _gen_nav_blob(daily_drift=0.0005, seed=82),
            951: _gen_nav_blob(daily_drift=0.0003, seed=83),
            952: _gen_nav_blob(daily_drift=0.00025, seed=84),
        }

        def fake_fetch(code):
            return nav_blobs.get(code, {"data": []})

        with patch.object(rk, "fetch_scheme_nav", side_effect=fake_fetch):
            result = rk.rank_all_assets(
                registry_path=path, top_n=5,
                equity_categories=[
                    "Equity Scheme - Large Cap Fund",
                    "Equity Scheme - Mid Cap Fund",
                ],
                debt_categories=["Debt Scheme - Short Duration Fund"],
            )
        # Large Cap succeeded.
        self.assertIn("Equity Scheme - Large Cap Fund", result.equity)
        # Mid Cap should be in errors (no funds in registry → fewer than 2).
        self.assertIn("Equity Scheme - Mid Cap Fund", result.equity_errors)
        # Short Duration succeeded.
        self.assertIn("Debt Scheme - Short Duration Fund", result.debt)

    def test_serialisation_dict_shape(self) -> None:
        """all_assets_to_dict produces the shape the frontend reads."""
        entries = [
            _scheme(1001, "X Direct Plan - Growth",
                    "Equity Scheme - Large Cap Fund"),
            _scheme(1002, "Y Direct Plan - Growth",
                    "Equity Scheme - Large Cap Fund"),
        ]
        path = _persist_registry(entries, self.tmpdir)
        bench_lc = rk.CATEGORY_BENCHMARK_MAP[
            "Equity Scheme - Large Cap Fund"][0]
        nav_blobs = {
            bench_lc: _gen_nav_blob(seed=90),
            1001: _gen_nav_blob(daily_drift=0.0007, seed=91),
            1002: _gen_nav_blob(daily_drift=0.0005, seed=92),
        }

        def fake_fetch(code):
            if code in nav_blobs:
                return nav_blobs[code]
            raise RuntimeError("no NAV")

        with patch.object(rk, "fetch_scheme_nav", side_effect=fake_fetch):
            result = rk.rank_all_assets(
                registry_path=path, top_n=5,
                equity_categories=["Equity Scheme - Large Cap Fund"],
                debt_categories=[],
            )
        out = rk.all_assets_to_dict(result)
        # Frontend reads these top-level keys.
        for key in ("computed_at", "top_n", "summary", "equity", "debt", "gold"):
            self.assertIn(key, out)
        self.assertIn("Equity Scheme - Large Cap Fund", out["equity"]["categories"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
