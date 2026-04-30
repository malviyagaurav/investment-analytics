"""Tests for the hidden-overlap correlation detector (Slice 2 / P3).

Two layers:
- Pure-function tests for `_correlation_pairs_from_nav` using
  synthetic NAV histories with known relationships (perfect copy,
  scaled copy, anti-correlated, independent, insufficient overlap).
- Wrapper tests for `_compute_held_correlations` with `fetch_scheme_nav`
  patched so we never hit the network.
"""
from __future__ import annotations

import math
import unittest
from unittest.mock import patch

from backend.investment_analytics import portfolio_health as ph


def _generate_nav(start_date: str = "2020-01-01", days: int = 800,
                  seed: int = 0, drift: float = 0.0003,
                  vol: float = 0.01, base: float = 100.0):
    """Generate a synthetic NAV series as list[{date, nav}].
    Returns deterministic output for a given seed."""
    import random
    from datetime import date, timedelta
    rng = random.Random(seed)
    d = date.fromisoformat(start_date)
    series = []
    nav = base
    for _ in range(days):
        # weekday-only, simple business-day generator
        while d.weekday() >= 5:
            d += timedelta(days=1)
        ret = drift + vol * (rng.gauss(0, 1))
        nav = max(0.01, nav * (1.0 + ret))
        series.append({"date": d.isoformat(), "nav": round(nav, 4)})
        d += timedelta(days=1)
    return series


def _scaled_copy(series, factor: float = 1.5):
    """Create a NAV series whose returns are `factor` × the original
    series' returns, on the same dates. Pearson correlation should be 1.0."""
    out = []
    base = series[0]["nav"]
    out.append({"date": series[0]["date"], "nav": base})
    for i in range(1, len(series)):
        prev_orig = series[i - 1]["nav"]
        curr_orig = series[i]["nav"]
        ret = (curr_orig / prev_orig) - 1.0
        new_ret = factor * ret
        prev_new = out[-1]["nav"]
        out.append({"date": series[i]["date"], "nav": max(0.01, prev_new * (1 + new_ret))})
    return out


def _anti_copy(series):
    """Returns negated. Correlation should be -1.0."""
    return _scaled_copy(series, factor=-1.0)


class CorrelationPureFunctionTests(unittest.TestCase):

    def test_two_independent_series_below_threshold(self) -> None:
        a = _generate_nav(seed=1)
        b = _generate_nav(seed=42)  # independent draws
        pairs = ph._correlation_pairs_from_nav({1: a, 2: b}, threshold=0.85)
        self.assertEqual(pairs, [],
                         "Independent series must not flag as high overlap")

    def test_perfectly_correlated_pair_is_flagged(self) -> None:
        a = _generate_nav(seed=7)
        b = _scaled_copy(a, factor=1.0)  # identical returns
        pairs = ph._correlation_pairs_from_nav({1: a, 2: b}, threshold=0.85)
        self.assertEqual(len(pairs), 1)
        self.assertGreaterEqual(pairs[0]["correlation"], 0.99)
        self.assertEqual(pairs[0]["fund_a_code"], 1)
        self.assertEqual(pairs[0]["fund_b_code"], 2)

    def test_scaled_copy_still_correlates_perfectly(self) -> None:
        # Pearson is scale-invariant.
        a = _generate_nav(seed=11)
        b = _scaled_copy(a, factor=2.0)
        pairs = ph._correlation_pairs_from_nav({1: a, 2: b}, threshold=0.85)
        self.assertEqual(len(pairs), 1)
        self.assertGreaterEqual(pairs[0]["correlation"], 0.99)

    def test_anti_correlated_below_threshold(self) -> None:
        # Threshold is for HIGH positive correlation; anti-correlation
        # should NOT flag because gilt-vs-equity behavior is structurally
        # different and the user shouldn't be alarmed by it.
        a = _generate_nav(seed=2)
        b = _anti_copy(a)
        pairs = ph._correlation_pairs_from_nav({1: a, 2: b}, threshold=0.85)
        self.assertEqual(pairs, [])

    def test_insufficient_overlap_returns_empty(self) -> None:
        # Below MIN_CORRELATION_DAYS (252) of common dates → no pairs.
        a = _generate_nav(start_date="2024-01-01", days=100, seed=3)
        b = _generate_nav(start_date="2024-01-01", days=100, seed=5)
        pairs = ph._correlation_pairs_from_nav({1: a, 2: b}, threshold=0.85)
        self.assertEqual(pairs, [])

    def test_results_sorted_by_correlation_desc(self) -> None:
        # Three funds: 1 and 2 are identical, 1 and 3 are scaled,
        # 2 and 3 also similar. We expect at least 2 pairs over threshold,
        # sorted highest-first.
        a = _generate_nav(seed=21)
        b = _scaled_copy(a, factor=1.0)        # ρ ≈ 1.0
        c = _scaled_copy(a, factor=0.9)        # ρ ≈ 1.0 (scale-invariant)
        pairs = ph._correlation_pairs_from_nav({1: a, 2: b, 3: c}, threshold=0.85)
        self.assertGreaterEqual(len(pairs), 2)
        for i in range(1, len(pairs)):
            self.assertGreaterEqual(
                pairs[i - 1]["correlation"], pairs[i]["correlation"],
                "Pairs must be sorted by correlation desc",
            )

    def test_pair_carries_common_day_count(self) -> None:
        a = _generate_nav(seed=33, days=500)
        b = _scaled_copy(a, factor=1.0)
        pairs = ph._correlation_pairs_from_nav({1: a, 2: b}, threshold=0.85)
        self.assertEqual(len(pairs), 1)
        # 500 input points → ~499 returns over identical date set.
        self.assertGreater(pairs[0]["common_days"], 100)


class CorrelationWrapperTests(unittest.TestCase):
    """`_compute_held_correlations` is the wrapper that fetches NAV.
    Patch fetch_scheme_nav so the test never hits the network."""

    def test_wrapper_fetches_each_code_and_returns_pairs(self) -> None:
        a = _generate_nav(seed=51)
        b = _scaled_copy(a, factor=1.0)
        nav_responses = {
            101: {"data": [{"date": _to_iso(r["date"]), "nav": r["nav"]} for r in a]},
            102: {"data": [{"date": _to_iso(r["date"]), "nav": r["nav"]} for r in b]},
        }

        def _fake_fetch(code):
            return nav_responses[code]

        with patch.object(ph, "fetch_scheme_nav", side_effect=_fake_fetch), \
             patch.object(ph, "_convert_nav_to_records",
                          side_effect=lambda data: [{"date": d["date"], "nav": d["nav"]} for d in data]):
            pairs = ph._compute_held_correlations([101, 102], threshold=0.85)
        self.assertEqual(len(pairs), 1)
        self.assertGreaterEqual(pairs[0]["correlation"], 0.99)

    def test_wrapper_swallows_individual_fetch_errors(self) -> None:
        a = _generate_nav(seed=61)
        b = _scaled_copy(a, factor=1.0)

        def _fake_fetch(code):
            if code == 999:
                raise RuntimeError("simulated network error")
            return {"data": [{"date": r["date"], "nav": r["nav"]}
                             for r in (a if code == 1 else b)]}

        with patch.object(ph, "fetch_scheme_nav", side_effect=_fake_fetch), \
             patch.object(ph, "_convert_nav_to_records",
                          side_effect=lambda data: [{"date": d["date"], "nav": d["nav"]} for d in data]):
            # 999 fails, 1 and 2 succeed → pair (1, 2) still computed.
            pairs = ph._compute_held_correlations([1, 999, 2], threshold=0.85)
        self.assertEqual(len(pairs), 1)
        self.assertEqual({pairs[0]["fund_a_code"], pairs[0]["fund_b_code"]}, {1, 2})


def _to_iso(s: str) -> str:
    """Helper: pass-through; series already ISO."""
    return s


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
