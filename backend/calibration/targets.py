"""Registry of calibratable thresholds with per-target substrate fetchers.

Each target carries:
  - canonical_id              — stable string identifier
  - substrate_fetcher         — takes (window, cache_dir, ...) and returns
                                a list of (value, weight=1.0) raw samples
                                for that window. The runner applies the
                                confidence weighting separately.
  - default_percentile        — recommended quantile of the sample set
  - description               — operator-facing string

Day-one registry: ONE target (HIGH_CORRELATION_THRESHOLD). Adding
targets is mechanical follow-up — register the canonical_id in
CALIBRATION_TARGETS (config.py) and add an entry here.

## HIGH_CORRELATION_THRESHOLD substrate

For a regime window [start, end]:
  1. Load the deep-history equity universe (funds with ≥
     MIN_FUND_HISTORY_POINTS_FOR_CALIBRATION NAV observations in the
     listed CALIBRATION_EQUITY_UNIVERSE_CATEGORIES).
  2. For each fund: slice its NAV series to [start, end].
  3. For each pair (a, b): compute Pearson correlation on the
     aligned daily returns within the window. Skip pairs with fewer
     than MIN_ALIGNED_POINTS_FOR_CORRELATION aligned observations
     (low statistical authority).
  4. Take the ABSOLUTE value (correlation thresholds operate on |r|).
  5. Return list of (abs_correlation, 1.0) — uniform raw weight.
     The runner multiplies each sample's weight by the regime_summary's
     classification_confidence at aggregation time.

The substrate fetcher is pure: same window + same cache → same
samples. Replayable end-to-end.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.calibration.config import (
    CALIBRATION_EQUITY_UNIVERSE_CATEGORIES,
    DEFAULT_PERCENTILE_BY_TARGET,
    MIN_FUND_HISTORY_POINTS_FOR_CALIBRATION,
)


ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CACHE_DIR = ROOT / "data" / "cache"
DEFAULT_REGISTRY_PATH = ROOT / "data" / "registry" / "schemes.json"


# A pair must have at least this many aligned daily returns within
# the regime window for its correlation to count as a sample. Below
# this, the correlation estimate is statistically unstable —
# excluded BEFORE confidence weighting because no amount of
# weighting rescues a 5-observation correlation.
MIN_ALIGNED_POINTS_FOR_CORRELATION = 30


# ── Helpers ──────────────────────────────────────────────────────────


def _parse_dd_mm_yyyy(s: str) -> Optional[date]:
    try:
        return datetime.strptime(s.strip(), "%d-%m-%Y").date()
    except (ValueError, AttributeError):
        return None


def _parse_iso(s: str) -> date:
    return datetime.fromisoformat(s).date()


def _load_scheme_navs(
    cache_dir: Path, scheme_code: int,
) -> List[Tuple[date, float]]:
    path = cache_dir / f"{scheme_code}.json"
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    data = raw.get("response", {}).get("data", []) or []
    out: List[Tuple[date, float]] = []
    for entry in data:
        d = _parse_dd_mm_yyyy(entry.get("date", ""))
        if d is None:
            continue
        try:
            nav = float(entry.get("nav", ""))
        except (ValueError, TypeError):
            continue
        if nav <= 0:
            continue
        out.append((d, nav))
    out.sort(key=lambda x: x[0])
    return out


def _scheme_category(cache_dir: Path, scheme_code: int) -> Optional[str]:
    path = cache_dir / f"{scheme_code}.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return raw.get("response", {}).get("meta", {}).get("scheme_category")


def _enumerate_deep_history_equity_universe(
    cache_dir: Path,
) -> List[int]:
    """Scan the cache for funds meeting the calibration universe
    criteria: scheme_category in CALIBRATION_EQUITY_UNIVERSE_CATEGORIES
    AND ≥ MIN_FUND_HISTORY_POINTS_FOR_CALIBRATION observations.

    Returns sorted list of scheme_codes. Sorted for determinism: the
    pair-enumeration order downstream depends on this ordering, so
    two calibration runs over the same cache produce the same sample
    sequence.
    """
    if not cache_dir.exists():
        return []
    universe: List[int] = []
    for path in cache_dir.glob("*.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        meta = raw.get("response", {}).get("meta", {})
        category = meta.get("scheme_category")
        if category not in CALIBRATION_EQUITY_UNIVERSE_CATEGORIES:
            continue
        data = raw.get("response", {}).get("data", []) or []
        if len(data) < MIN_FUND_HISTORY_POINTS_FOR_CALIBRATION:
            continue
        try:
            scheme_code = int(path.stem)
        except ValueError:
            continue
        universe.append(scheme_code)
    return sorted(universe)


def _slice_window(
    series: List[Tuple[date, float]], start: date, end: date,
) -> List[Tuple[date, float]]:
    return [(d, v) for d, v in series if start <= d <= end]


def _daily_returns(window: List[Tuple[date, float]]) -> Dict[date, float]:
    """Date-keyed daily simple returns within the window. Same key
    structure across funds enables alignment-by-date intersection."""
    out: Dict[date, float] = {}
    for i in range(1, len(window)):
        prev_date, prev_nav = window[i - 1]
        cur_date, cur_nav = window[i]
        if prev_nav > 0:
            out[cur_date] = (cur_nav / prev_nav) - 1.0
    return out


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    if denom <= 0:
        return None
    return cov / denom


# ── Substrate fetchers ───────────────────────────────────────────────


def _fetch_correlation_samples(
    window_start: str,
    window_end: str,
    cache_dir: Path,
) -> List[Tuple[float, float]]:
    """Substrate fetcher for HIGH_CORRELATION_THRESHOLD.

    Returns list of (|pearson_r|, 1.0) for every pair in the deep-
    history equity universe whose pair-wise aligned-return overlap
    inside [window_start, window_end] meets
    MIN_ALIGNED_POINTS_FOR_CORRELATION.

    The returned weight is always 1.0 — the calibration runner
    multiplies by the regime_summary's classification_confidence at
    aggregation time. Keeping confidence-weighting OUT of the fetcher
    keeps the fetcher pure-per-window and lets a future weighting
    scheme swap in without touching the substrate code.
    """
    start = _parse_iso(window_start)
    end = _parse_iso(window_end)
    if end < start:
        return []

    universe = _enumerate_deep_history_equity_universe(cache_dir)
    if len(universe) < 2:
        return []

    # Precompute date-keyed daily returns per fund inside the window.
    returns_by_code: Dict[int, Dict[date, float]] = {}
    for code in universe:
        series = _load_scheme_navs(cache_dir, code)
        window = _slice_window(series, start, end)
        returns_by_code[code] = _daily_returns(window)

    samples: List[Tuple[float, float]] = []
    for i, a in enumerate(universe):
        ra = returns_by_code.get(a, {})
        if not ra:
            continue
        a_keys = set(ra.keys())
        for b in universe[i + 1:]:
            rb = returns_by_code.get(b, {})
            if not rb:
                continue
            common = sorted(a_keys & set(rb.keys()))
            if len(common) < MIN_ALIGNED_POINTS_FOR_CORRELATION:
                continue
            xs = [ra[d] for d in common]
            ys = [rb[d] for d in common]
            r = _pearson(xs, ys)
            if r is None:
                continue
            samples.append((abs(r), 1.0))
    return samples


# ── Registry ─────────────────────────────────────────────────────────


SubstrateFetcher = Callable[[str, str, Path], List[Tuple[float, float]]]


@dataclass(frozen=True)
class CalibrationTarget:
    """One entry in the calibration target registry."""
    canonical_id:        str
    substrate_fetcher:   SubstrateFetcher
    default_percentile:  float
    description:         str


REGISTERED_TARGETS: Dict[str, CalibrationTarget] = {
    "portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD": CalibrationTarget(
        canonical_id="portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
        substrate_fetcher=_fetch_correlation_samples,
        default_percentile=DEFAULT_PERCENTILE_BY_TARGET[
            "portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD"
        ],
        description=(
            "Absolute Pearson correlation between deep-history equity "
            "funds in CALIBRATION_EQUITY_UNIVERSE_CATEGORIES, sampled "
            "per regime window. p95 recommendation = the value above "
            "which the top 5% of cross-asset correlations sit, per "
            "regime — operationally the right threshold for "
            "redundancy detection in portfolio_health."
        ),
    ),
}


def get_target(canonical_id: str) -> CalibrationTarget:
    if canonical_id not in REGISTERED_TARGETS:
        raise KeyError(
            f"unknown calibration target: {canonical_id!r}. "
            f"Registered: {sorted(REGISTERED_TARGETS)}"
        )
    return REGISTERED_TARGETS[canonical_id]
