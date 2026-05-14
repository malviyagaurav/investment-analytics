"""Tests for backend.jobs.watchlist.

Exercises the snapshot store + diff logic with mocked rank_*
functions so the test never hits the network.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.jobs import watchlist as wl
from backend.investment_analytics.ranking import (
    CategoryRanking, FundMetrics, RankedFund,
)


def _f(code: int, **kw) -> FundMetrics:
    base = dict(
        scheme_code=code,
        fund_name=f"Fund {code} Direct Plan - Growth",
        fund_house=f"AMC {code // 100}",
        excess_return_pct=2.0, max_drawdown_pct=-15.0, consistency_pct=55.0,
        volatility_pct=12.0, downside_capture_ratio=1.0,
        fund_cagr_pct=10.0, benchmark_cagr_pct=8.0,
        aligned_points=1500, history_years=8.0, drawdown_trough_date=None,
    )
    base.update(kw)
    return FundMetrics(**base)


def _ranking(category: str, codes_in_order: list, conf="High") -> CategoryRanking:
    funds = [_f(c, excess_return_pct=4.0 - i) for i, c in enumerate(codes_in_order)]
    ranked = [
        RankedFund(rank=i + 1, fund=f, dominance_count=4 - i, total_peers=5,
                   confidence_level=conf, strengths=[], weaknesses=[])
        for i, f in enumerate(funds)
    ]
    return CategoryRanking(
        category=category, benchmark_name="Nifty 100", benchmark_code=999,
        benchmark_fallback=False, ranked=ranked, excluded=[],
        computed_at="2026-05-06T00:00:00+00:00", total_funds_in_category=len(funds),
    )


class WatchlistConfigTests(unittest.TestCase):

    def test_load_config_defaults_when_file_missing(self) -> None:
        cfg = wl.load_config(Path("/nonexistent/categories.json"))
        self.assertIn("equity_categories", cfg)
        self.assertIn("debt_categories", cfg)
        self.assertGreater(len(cfg["equity_categories"]), 0)


class SnapshotStoreTests(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self._orig_dir = wl.SNAPSHOTS_DIR
        wl.SNAPSHOTS_DIR = Path(self.tmp.name) / "snapshots"

    def tearDown(self) -> None:
        wl.SNAPSHOTS_DIR = self._orig_dir
        self.tmp.cleanup()

    def test_save_and_load_roundtrip(self) -> None:
        cat = "Equity Scheme - Large Cap Fund"
        payload = {"category": cat, "snapshot_date": "2026-05-06",
                   "ranked": [{"scheme_code": 101, "fund_name": "X", "rank": 1,
                                "dominance": {"beats": 4, "of": 5}}]}
        wl.save_snapshot(cat, "2026-05-06", payload)
        roundtrip = wl.load_snapshot(cat, "2026-05-06")
        self.assertEqual(roundtrip, payload)

    def test_list_dates_sorted(self) -> None:
        cat = "Equity Scheme - Mid Cap Fund"
        for d in ("2026-05-04", "2026-05-06", "2026-05-05"):
            wl.save_snapshot(cat, d, {"snapshot_date": d, "ranked": []})
        self.assertEqual(
            wl.list_snapshot_dates(cat),
            ["2026-05-04", "2026-05-05", "2026-05-06"],
        )

    def test_latest_snapshot_returns_most_recent(self) -> None:
        cat = "Equity Scheme - Small Cap Fund"
        wl.save_snapshot(cat, "2026-05-04", {"snapshot_date": "2026-05-04",
                                              "ranked": [{"scheme_code": 1}]})
        wl.save_snapshot(cat, "2026-05-06", {"snapshot_date": "2026-05-06",
                                              "ranked": [{"scheme_code": 2}]})
        latest = wl.latest_snapshot(cat)
        self.assertIsNotNone(latest)
        d, snap = latest
        self.assertEqual(d, "2026-05-06")
        self.assertEqual(snap["ranked"][0]["scheme_code"], 2)

    def test_latest_snapshot_none_when_empty(self) -> None:
        self.assertIsNone(wl.latest_snapshot("nonexistent_category"))


class DiffSnapshotTests(unittest.TestCase):

    def _snap(self, date: str, codes_ranked: list) -> dict:
        return {
            "category": "Equity Scheme - Large Cap Fund",
            "snapshot_date": date,
            "ranked": [
                {"scheme_code": code, "fund_name": f"F{code}",
                 "rank": i + 1,
                 "dominance": {"beats": 4 - i, "of": 5}}
                for i, code in enumerate(codes_ranked)
            ],
        }

    def test_diff_no_changes(self) -> None:
        prev = self._snap("2026-05-05", [101, 102, 103])
        curr = self._snap("2026-05-06", [101, 102, 103])
        d = wl.diff_snapshots(prev, curr)
        self.assertEqual(d["new_funds"], [])
        self.assertEqual(d["dropped_funds"], [])
        self.assertEqual(d["rank_changes"], {})
        self.assertEqual(d["dominance_changes"], {})

    def test_diff_detects_new_and_dropped(self) -> None:
        prev = self._snap("2026-05-05", [101, 102, 103])
        curr = self._snap("2026-05-06", [102, 103, 104])
        d = wl.diff_snapshots(prev, curr)
        self.assertEqual([f["scheme_code"] for f in d["new_funds"]], [104])
        self.assertEqual([f["scheme_code"] for f in d["dropped_funds"]], [101])

    def test_diff_detects_rank_climbs_and_drops(self) -> None:
        prev = self._snap("2026-05-05", [101, 102, 103])
        # Curr: 102 climbed from rank 2 to rank 1; 101 dropped 1 → 2.
        curr = self._snap("2026-05-06", [102, 101, 103])
        d = wl.diff_snapshots(prev, curr)
        # 102: prev_rank=2, curr_rank=1, delta=+1 (improvement)
        self.assertEqual(d["rank_changes"][102]["delta"], 1)
        # 101: prev_rank=1, curr_rank=2, delta=-1 (regression)
        self.assertEqual(d["rank_changes"][101]["delta"], -1)

    def test_diff_dominance_changes(self) -> None:
        prev = self._snap("2026-05-05", [101, 102])
        curr = self._snap("2026-05-06", [101, 102])
        # Tweak dominance counts to simulate a metric shift.
        curr["ranked"][0]["dominance"]["beats"] = 5
        d = wl.diff_snapshots(prev, curr)
        self.assertIn(101, d["dominance_changes"])
        self.assertEqual(d["dominance_changes"][101]["delta"], 1)


class RunSnapshotTests(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self._orig_dir = wl.SNAPSHOTS_DIR
        wl.SNAPSHOTS_DIR = Path(self.tmp.name) / "snapshots"

    def tearDown(self) -> None:
        wl.SNAPSHOTS_DIR = self._orig_dir
        self.tmp.cleanup()

    def test_run_snapshot_writes_one_file_per_equity_category(self) -> None:
        cfg = {
            "equity_categories": [
                "Equity Scheme - Large Cap Fund",
                "Equity Scheme - Mid Cap Fund",
            ],
            "debt_categories": [],
            "include_gold": False,
            "top_n_to_persist": 5,
        }

        def fake_rank(category, _registry_path):
            return _ranking(category, [101, 102, 103, 104, 105])

        with patch.object(wl, "rank_category", side_effect=fake_rank), \
             patch.object(wl, "_validate_registry", return_value=None):
            summary = wl.run_snapshot(
                config=cfg, registry_path="ignored", snapshot_date="2026-05-06",
            )

        self.assertEqual(summary["Equity Scheme - Large Cap Fund"], "ok")
        self.assertEqual(summary["Equity Scheme - Mid Cap Fund"], "ok")
        for cat in cfg["equity_categories"]:
            snap = wl.load_snapshot(cat, "2026-05-06")
            self.assertIsNotNone(snap)
            self.assertEqual(snap["snapshot_kind"], "equity")
            self.assertEqual(snap["snapshot_date"], "2026-05-06")
            self.assertEqual(len(snap["ranked"]), 5)
            self.assertEqual(snap["showing_top_n"], 5)

    def test_run_snapshot_swallows_per_category_errors(self) -> None:
        cfg = {
            "equity_categories": [
                "Equity Scheme - Large Cap Fund",
                "Equity Scheme - Mid Cap Fund",
            ],
            "debt_categories": [],
            "include_gold": False,
            "top_n_to_persist": 5,
        }
        # Raise for one category, succeed for the other.

        def fake_rank(category, _path):
            if "Mid Cap" in category:
                raise ValueError("benchmark unavailable")
            return _ranking(category, [101, 102, 103])

        with patch.object(wl, "rank_category", side_effect=fake_rank), \
             patch.object(wl, "_validate_registry", return_value=None):
            summary = wl.run_snapshot(
                config=cfg, registry_path="ignored", snapshot_date="2026-05-06",
            )
        self.assertEqual(summary["Equity Scheme - Large Cap Fund"], "ok")
        self.assertIn("error:", summary["Equity Scheme - Mid Cap Fund"])
        # Successful category still got persisted; failing one didn't.
        self.assertIsNotNone(wl.load_snapshot("Equity Scheme - Large Cap Fund", "2026-05-06"))
        self.assertIsNone(wl.load_snapshot("Equity Scheme - Mid Cap Fund", "2026-05-06"))


class RegistryValidationTests(unittest.TestCase):
    """Audit fix: watchlist runner aborts early on missing / corrupt
    / undersized registry rather than producing N per-category errors
    that obscure the real problem."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self._orig_dir = wl.SNAPSHOTS_DIR
        wl.SNAPSHOTS_DIR = Path(self.tmp.name) / "snapshots"

    def tearDown(self) -> None:
        wl.SNAPSHOTS_DIR = self._orig_dir
        self.tmp.cleanup()

    def test_empty_registry_aborts_with_single_summary_entry(self) -> None:
        # Stub a registry loader that returns 0 schemes.
        with patch.object(wl, "_validate_registry",
                          return_value="registry is empty"):
            summary = wl.run_snapshot(
                config={"equity_categories": ["X"], "debt_categories": [],
                        "include_gold": False, "top_n_to_persist": 5},
                registry_path="ignored",
                snapshot_date="2026-05-06",
            )
        self.assertIn("__registry__", summary)
        self.assertIn("registry is empty", summary["__registry__"])
        # No per-category errors should be produced.
        self.assertEqual(len(summary), 1)

    def test_undersized_registry_aborts(self) -> None:
        with patch.object(wl, "_validate_registry",
                          return_value="registry has only 42 entries"):
            summary = wl.run_snapshot(
                config={"equity_categories": ["X", "Y"], "debt_categories": ["Z"],
                        "include_gold": True, "top_n_to_persist": 5},
                registry_path="ignored",
                snapshot_date="2026-05-06",
            )
        self.assertEqual(len(summary), 1)
        self.assertIn("__registry__", summary)

    def test_valid_registry_passes_through(self) -> None:
        # With _validate_registry returning None, the runner should
        # proceed to rank categories normally.
        from backend.investment_analytics.ranking import (
            CategoryRanking, FundMetrics, RankedFund,
        )
        with patch.object(wl, "_validate_registry", return_value=None), \
             patch.object(wl, "rank_category") as rc:
            # Build a tiny ranking object.
            def fake_rank(category, _path):
                funds = [
                    FundMetrics(
                        scheme_code=100 + i, fund_name=f"F{i}", fund_house="X",
                        excess_return_pct=4.0 - i, max_drawdown_pct=-12.0,
                        consistency_pct=70.0 - i * 5, volatility_pct=10.0,
                        downside_capture_ratio=1.0, fund_cagr_pct=10.0,
                        benchmark_cagr_pct=8.0, aligned_points=1500,
                        history_years=8.0, drawdown_trough_date=None,
                    ) for i in range(3)
                ]
                ranked = [
                    RankedFund(rank=i + 1, fund=f, dominance_count=2 - i,
                               total_peers=3, confidence_level="High",
                               strengths=[], weaknesses=[])
                    for i, f in enumerate(funds)
                ]
                return CategoryRanking(
                    category=category, benchmark_name="X", benchmark_code=1,
                    benchmark_fallback=False, ranked=ranked, excluded=[],
                    computed_at="2026-05-06T00:00:00+00:00",
                    total_funds_in_category=3,
                )
            rc.side_effect = fake_rank
            summary = wl.run_snapshot(
                config={"equity_categories": ["Equity Scheme - Large Cap Fund"],
                        "debt_categories": [], "include_gold": False,
                        "top_n_to_persist": 5},
                registry_path="ignored", snapshot_date="2026-05-06",
            )
        self.assertEqual(summary["Equity Scheme - Large Cap Fund"], "ok")
        self.assertNotIn("__registry__", summary)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
