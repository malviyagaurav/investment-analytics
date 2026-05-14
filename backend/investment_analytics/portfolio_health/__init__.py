"""Portfolio Health Check — evaluate user holdings against category peers.

For each fund the user holds, this module:
1. Identifies the fund's AMFI category
2. Runs or reuses the category ranking (equity or debt)
3. Finds where the fund places among its peers
4. Tags: Strong / Neutral / Weak
5. Assigns action: Continue / Monitor / Review
6. Lists higher-ranked alternatives with justification
7. Detects portfolio-level mistakes, redundancy, exposure gaps

No advisory language. No BUY/SELL. Factual peer comparison only.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Response schema bumped from v1 to v2 for Slice 1:
# - decision_summary entries gain weight_pct
# - alternatives.justification is now List[{reason, magnitude, metric}]
#   (was List[str])
# - alternatives must pass _alternative_is_material gate
PORTFOLIO_HEALTH_SCHEMA_VERSION = "v2"

from backend.data_discovery.fetch import fetch_scheme_nav, _convert_nav_to_records
from backend.data_discovery.registry import SchemeEntry, load_registry
from backend.investment_analytics.ranking import (
    ALL_RANKABLE_CATEGORIES,
    DEBT_CATEGORIES,
    DEBT_RISK_TAGS,
    DEFAULT_DEBT_CATEGORIES,
    EXCLUDED_CATEGORIES,
    RANKABLE_CATEGORIES,
    CategoryRanking,
    DebtCategoryRanking,
    RankedFund,
    _confidence_level,
    rank_category,
    rank_debt_category,
    ranking_to_dict,
    debt_ranking_to_dict,
)

logger = logging.getLogger("investment_analytics.portfolio_health")


# ── Status thresholds ──────────────────────────────────────────
# Top 25% = Strong (only if confidence ≠ Low), Bottom 25% = Weak, Middle 50% = Neutral

def _fund_status(rank: int, total: int, confidence: str) -> str:
    """Determine fund status based on rank position and confidence level.

    Strong requires confidence ≠ Low (insufficient data cannot
    reliably confirm strong standing).
    """
    if total < 2:
        return "Insufficient peers"
    pct = rank / total  # lower = better (rank 1 = best)
    if pct <= 0.25:
        if confidence == "Low":
            return "Neutral"  # downgrade: insufficient data to confirm Strong
        return "Strong"
    elif pct <= 0.75:
        return "Neutral"
    else:
        return "Weak"


# ── Time-horizon estimation ────────────────────────────────────

def _horizon_tag(volatility_pct: float, max_drawdown_pct: float, asset_class: str) -> str:
    """Estimate suitable holding horizon from volatility and drawdown.

    Returns "Short-term", "Mid-term", or "Long-term" — purely data-driven,
    not a recommendation.
    """
    if asset_class == "equity":
        # Equity thresholds (annualized)
        if volatility_pct > 22 or max_drawdown_pct < -30:
            return "Long-term"
        elif volatility_pct > 14 or max_drawdown_pct < -18:
            return "Mid-term"
        else:
            return "Short-term"
    else:
        # Debt thresholds (typically lower volatility)
        if volatility_pct > 6 or max_drawdown_pct < -8:
            return "Long-term"
        elif volatility_pct > 2.5 or max_drawdown_pct < -3:
            return "Mid-term"
        else:
            return "Short-term"


# ── Action engine ──────────────────────────────────────────────

def _assign_action(status: str, has_alternatives: bool) -> Tuple[str, str]:
    """Assign portfolio action and optional note based on status.

    Returns (action, note) where action is Continue/Monitor/Review
    and note is an optional factual observation.
    """
    if status == "Strong":
        return "Continue", ""
    elif status == "Neutral":
        return "Monitor", ""
    elif status == "Weak":
        if has_alternatives:
            return "Review", "Higher-ranked alternatives available"
        return "Review", ""
    else:
        # Not Ranked / Insufficient peers
        return "Monitor", "Could not be ranked"


# ── Switch justification ──────────────────────────────────────

# Material-improvement thresholds for the alternative-filter gate.
# Calibrated to "would a thoughtful investor actually feel this delta
# in net returns over a 3-5y horizon?" Conservative on equity (high
# cross-sectional dispersion) and tighter on debt (compressed yields).
# delta_for_moderate must be exceeded for a metric to be "material";
# delta_for_large for a "large" magnitude tag.
_ALT_THRESHOLDS = {
    "equity": {
        # key: (moderate_delta, large_delta) — units = same as the metric
        "excess_return_pct":      (1.5,  3.0),   # +pp CAGR vs benchmark
        "consistency_pct":        (10.0, 20.0),  # +pp rolling-window wins
        "max_drawdown_pct":       (5.0,  10.0),  # pp shallower (alt - held > 0)
        "volatility_pct":         (1.5,  3.0),   # pp lower
        "downside_capture_ratio": (0.10, 0.20),  # ratio — lower better
    },
    "debt": {
        "cagr_pct":               (0.75, 1.5),
        "consistency_pct":        (5.0,  10.0),
        "max_drawdown_pct":       (1.5,  3.0),
        "volatility_pct":         (0.5,  1.0),
        "risk_adj_return":        (0.5,  1.0),
    },
}


def _improvement_magnitude(metric_key: str, delta: float, asset_class: str) -> str:
    """Bucket a per-metric improvement delta into 'large' / 'moderate' / 'small'.

    delta MUST be expressed as 'alt-better-than-held' in the metric's
    natural direction (positive = better). Returns 'none' if the alt is
    not better, 'small' if better but below the moderate threshold.
    """
    if delta <= 0:
        return "none"
    bands = _ALT_THRESHOLDS.get(asset_class, {}).get(metric_key)
    if bands is None:
        return "small"
    moderate, large = bands
    if delta >= large:
        return "large"
    if delta >= moderate:
        return "moderate"
    return "small"


def _signed_delta(metric_key: str, held_val: float, alt_val: float) -> float:
    """Return alt-vs-held delta in the metric's natural 'better' direction.

    Positive delta means alt is better than held.
    """
    # Metrics where higher is better.
    if metric_key in {"excess_return_pct", "consistency_pct", "cagr_pct",
                      "risk_adj_return"}:
        return alt_val - held_val
    # Drawdown is stored as negative; less-negative = shallower = better.
    if metric_key == "max_drawdown_pct":
        return alt_val - held_val
    # Lower-is-better.
    if metric_key in {"volatility_pct", "downside_capture_ratio"}:
        return held_val - alt_val
    return 0.0


_EQUITY_BULLETS: Tuple[Tuple[str, str], ...] = (
    ("consistency_pct",        "Better consistency vs benchmark"),
    ("max_drawdown_pct",       "Shallower drawdowns"),
    ("volatility_pct",         "Lower volatility"),
    ("downside_capture_ratio", "Better downside protection"),
    ("excess_return_pct",      "Historically higher return than peers"),
)
_DEBT_BULLETS: Tuple[Tuple[str, str], ...] = (
    ("cagr_pct",         "Historically higher return"),
    ("volatility_pct",   "Lower volatility"),
    ("max_drawdown_pct", "Shallower drawdowns"),
    ("risk_adj_return",  "Better risk-adjusted return"),
)


def _build_justification(
    held_metrics: Dict[str, Any],
    alt_metrics: Dict[str, Any],
    asset_class: str,
) -> List[Dict[str, str]]:
    """Build factual justification for why an alternative ranks higher.

    Returns list of {"reason": ..., "magnitude": small/moderate/large}.
    No percentage claims, no predictions. UI maps magnitude to display.
    """
    bullets = _EQUITY_BULLETS if asset_class == "equity" else _DEBT_BULLETS
    out: List[Dict[str, str]] = []
    for key, label in bullets:
        h = held_metrics.get(key, 0) or 0
        a = alt_metrics.get(key, 0) or 0
        delta = _signed_delta(key, h, a)
        if delta <= 0:
            continue
        out.append({
            "reason": label,
            "magnitude": _improvement_magnitude(key, delta, asset_class),
            "metric": key,
        })
    return out


def _alternative_is_material(
    held_metrics: Dict[str, Any],
    alt_metrics: Dict[str, Any],
    asset_class: str,
) -> bool:
    """Gate: an alternative must beat held on >=3 metrics AND show at
    least one moderate-or-large improvement to be surfaced.

    Without this gate the system would suggest rank-N+1 funds whose
    advantage is statistically indistinguishable from rank-N — pure
    churn-encouragement. The 3-of-5 wins requirement mirrors the
    pairwise-dominance rule already used in ranking.py.
    """
    bullets = _EQUITY_BULLETS if asset_class == "equity" else _DEBT_BULLETS
    wins = 0
    has_material = False
    for key, _label in bullets:
        h = held_metrics.get(key, 0) or 0
        a = alt_metrics.get(key, 0) or 0
        delta = _signed_delta(key, h, a)
        if delta > 0:
            wins += 1
            mag = _improvement_magnitude(key, delta, asset_class)
            if mag in ("moderate", "large"):
                has_material = True
    return wins >= 3 and has_material


def _build_your_fund_gaps(
    held_metrics: Dict[str, Any],
    top_alt_metrics: Dict[str, Any],
    asset_class: str,
) -> List[str]:
    """Build personal comparison: what YOUR fund is weaker at vs the top alternative.

    Inverse of _build_justification — framed from user's fund perspective.
    """
    gaps: List[str] = []

    if asset_class == "equity":
        h_cons = held_metrics.get("consistency_pct", 0) or 0
        a_cons = top_alt_metrics.get("consistency_pct", 0) or 0
        if a_cons > h_cons:
            gaps.append("Lower consistency than top-ranked peer")

        h_dd = held_metrics.get("max_drawdown_pct", 0) or 0
        a_dd = top_alt_metrics.get("max_drawdown_pct", 0) or 0
        if a_dd > h_dd:
            gaps.append("Deeper drawdowns than top-ranked peer")

        h_vol = held_metrics.get("volatility_pct", 0) or 0
        a_vol = top_alt_metrics.get("volatility_pct", 0) or 0
        if a_vol < h_vol:
            gaps.append("Higher volatility than top-ranked peer")
    else:
        h_cagr = held_metrics.get("cagr_pct", 0) or 0
        a_cagr = top_alt_metrics.get("cagr_pct", 0) or 0
        if a_cagr > h_cagr:
            gaps.append("Lower historical return than top-ranked peer")

        h_vol = held_metrics.get("volatility_pct", 0) or 0
        a_vol = top_alt_metrics.get("volatility_pct", 0) or 0
        if a_vol < h_vol:
            gaps.append("Higher volatility than top-ranked peer")

        h_dd = held_metrics.get("max_drawdown_pct", 0) or 0
        a_dd = top_alt_metrics.get("max_drawdown_pct", 0) or 0
        if a_dd > h_dd:
            gaps.append("Deeper drawdowns than top-ranked peer")

    return gaps


# ── Portfolio mistake detector ─────────────────────────────────

def _detect_mistakes(holdings: List["FundHealthResult"]) -> List[Dict[str, Any]]:
    """Detect portfolio-level structural issues."""
    mistakes: List[Dict[str, Any]] = []

    ranked_holdings = [h for h in holdings if h.status != "Not Ranked"]

    # 1. Over-diversification: >8 ranked funds
    if len(ranked_holdings) > 8:
        mistakes.append({
            "type": "over_diversification",
            "severity": "moderate",
            "message": f"Portfolio has {len(ranked_holdings)} ranked holdings — potential over-diversification",
        })

    # 2. Category crowding: >3 funds in same category
    cat_counts: Dict[str, int] = {}
    for h in holdings:
        cat_counts[h.category] = cat_counts.get(h.category, 0) + 1
    for cat, count in cat_counts.items():
        if count > 3:
            mistakes.append({
                "type": "category_crowding",
                "severity": "high",
                "message": f"{count} funds in {_short_category(cat)} — consider if all are needed",
                "category": cat,
            })

    # 3. AMC concentration: ≥2 funds from same AMC in same category
    amc_cat_counts: Dict[Tuple[str, str], int] = {}
    for h in holdings:
        key = (h.fund_house, h.category)
        amc_cat_counts[key] = amc_cat_counts.get(key, 0) + 1
    for (amc, cat), count in amc_cat_counts.items():
        if count >= 2:
            # Shorten AMC name for display
            amc_short = amc.split(" Mutual Fund")[0] if " Mutual Fund" in amc else amc
            mistakes.append({
                "type": "amc_concentration",
                "severity": "moderate",
                "message": f"{count} funds from {amc_short} in {_short_category(cat)} — same AMC concentration",
                "fund_house": amc,
                "category": cat,
            })

    return mistakes


# ── Redundancy detector ───────────────────────────────────────

def _detect_redundancy(holdings: List["FundHealthResult"]) -> List[Dict[str, Any]]:
    """Detect holding pairs that behave similarly (potential redundancy)."""
    redundancies: List[Dict[str, Any]] = []
    ranked = [h for h in holdings if h.rank > 0 and h.metrics]

    for i in range(len(ranked)):
        for j in range(i + 1, len(ranked)):
            a, b = ranked[i], ranked[j]
            if a.category != b.category:
                continue

            # Same category — check rank proximity and metric similarity
            rank_diff = abs(a.rank - b.rank)
            if rank_diff > 3:
                continue

            # Check volatility and drawdown similarity
            a_vol = a.metrics.get("volatility_pct", 0) or 0
            b_vol = b.metrics.get("volatility_pct", 0) or 0
            a_dd = abs(a.metrics.get("max_drawdown_pct", 0) or 0)
            b_dd = abs(b.metrics.get("max_drawdown_pct", 0) or 0)

            vol_close = abs(a_vol - b_vol) < max(a_vol, b_vol, 1) * 0.3
            dd_close = abs(a_dd - b_dd) < max(a_dd, b_dd, 1) * 0.3

            if vol_close and dd_close:
                a_name = a.fund_name.replace(" - Direct Plan", "").replace(" Direct Plan", "")
                b_name = b.fund_name.replace(" - Direct Plan", "").replace(" Direct Plan", "")
                redundancies.append({
                    "fund_a": {"scheme_code": a.scheme_code, "fund_name": a_name, "rank": a.rank},
                    "fund_b": {"scheme_code": b.scheme_code, "fund_name": b_name, "rank": b.rank},
                    "category": _short_category(a.category),
                    "message": "Funds show similar behavior in same category (potential overlap)",
                })

    return redundancies


# ── Exposure gap detector ─────────────────────────────────────

# Major categories users typically need
_MAJOR_EQUITY_CATEGORIES = [
    "Equity Scheme - Large Cap Fund",
    "Equity Scheme - Mid Cap Fund",
    "Equity Scheme - Small Cap Fund",
    "Equity Scheme - Flexi Cap Fund",
]

_MAJOR_DEBT_CATEGORIES = [
    "Debt Scheme - Short Duration Fund",
    "Debt Scheme - Corporate Bond Fund",
    "Debt Scheme - Liquid Fund",
]


def _detect_exposure_gaps(holdings: List["FundHealthResult"]) -> List[Dict[str, Any]]:
    """Detect major asset categories with no exposure."""
    held_categories = {h.category for h in holdings}
    has_equity = any(h.asset_class == "equity" for h in holdings)
    has_debt = any(h.asset_class == "debt" for h in holdings)

    gaps: List[Dict[str, Any]] = []

    # Only flag equity gaps if user has equity (otherwise it's a debt-only portfolio, which is fine)
    if has_equity:
        eq_gaps = [_short_category(c) for c in _MAJOR_EQUITY_CATEGORIES if c not in held_categories]
        if eq_gaps:
            gaps.append({
                "asset_class": "equity",
                "missing": eq_gaps,
                "message": "No exposure to: " + ", ".join(eq_gaps),
            })

    # Flag missing debt only if user has no debt at all
    if has_equity and not has_debt:
        gaps.append({
            "asset_class": "debt",
            "missing": ["Any debt category"],
            "message": "Portfolio has no debt exposure",
        })

    # Flag missing equity if user has only debt
    if has_debt and not has_equity:
        gaps.append({
            "asset_class": "equity",
            "missing": ["Any equity category"],
            "message": "Portfolio has no equity exposure",
        })

    return gaps


# ── Data quality & outlier detection ──────────────────────────

def _data_quality_flags(
    history_years: float,
    aligned_points: int,
    confidence: str,
    peer_points: List[int],
) -> List[Dict[str, str]]:
    """Flag data quality concerns with severity levels.

    Severity: mild / moderate / severe.
    Uses existing signals — no new computation.
    """
    flags: List[Dict[str, str]] = []

    if confidence == "Low":
        flags.append({
            "severity": "moderate",
            "message": f"Limited history ({history_years:.1f}y) \u2014 metrics less reliable",
        })

    # Compare aligned points to peer median
    if peer_points:
        sorted_pts = sorted(peer_points)
        median_pts = sorted_pts[len(sorted_pts) // 2]
        if median_pts > 0:
            ratio = aligned_points / median_pts
            if ratio < 0.4:
                flags.append({
                    "severity": "severe",
                    "message": "Significant data gaps vs peers \u2014 interpret cautiously",
                })
            elif ratio < 0.6:
                flags.append({
                    "severity": "moderate",
                    "message": "Partial data gaps \u2014 some metrics may be less reliable",
                })
            elif ratio < 0.8:
                flags.append({
                    "severity": "mild",
                    "message": "Slightly less data than most peers in this category",
                })

    return flags


# ── Hidden-overlap (return correlation across held funds) ─────

# Threshold for flagging a pair as "high overlap". Indian large-cap
# active funds typically run 0.93-0.97 against each other and against
# Nifty 100; cross-category large+flexi+multi often runs 0.90+.
# 0.85 picks up the cross-category cases where the "different category"
# label gives a false sense of diversification, without over-firing on
# obviously-different holdings (e.g., equity vs gilt fund typically <0.3).
HIGH_CORRELATION_THRESHOLD = 0.85

# Minimum aligned days needed for a correlation estimate to be meaningful.
# 252 = ~1 trading year. Below that, correlation is noisy.
MIN_CORRELATION_DAYS = 252


def _pearson(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation. Returns 0.0 on degenerate inputs."""
    n = len(xs)
    if n != len(ys) or n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sxx * syy) ** 0.5
    if den <= 0:
        return 0.0
    return sxy / den


def _correlation_pairs_from_nav(
    nav_by_code: Dict[int, List[Dict[str, Any]]],
    threshold: float = HIGH_CORRELATION_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Pure function: given NAV records per scheme code, return high-
    correlation pairs.

    Aligns on common dates across ALL inputs (intersection — strict),
    computes per-fund daily returns, then pairwise Pearson correlation.
    Returns pairs whose correlation >= threshold, with the actual corr
    value and number of common days so the UI can disclose both.
    """
    if len(nav_by_code) < 2:
        return []
    nav_maps: Dict[int, Dict[str, float]] = {}
    for code, records in nav_by_code.items():
        if not records:
            continue
        nav_maps[code] = {r["date"]: r["nav"] for r in records if r.get("nav", 0) > 0}
    if len(nav_maps) < 2:
        return []
    common_dates_set = set.intersection(*(set(m.keys()) for m in nav_maps.values()))
    if len(common_dates_set) < MIN_CORRELATION_DAYS:
        return []
    sorted_dates = sorted(common_dates_set)
    returns_by_code: Dict[int, List[float]] = {}
    for code, m in nav_maps.items():
        rets: List[float] = []
        prev = m[sorted_dates[0]]
        for d in sorted_dates[1:]:
            curr = m[d]
            if prev > 0:
                rets.append((curr / prev) - 1.0)
            else:
                rets.append(0.0)
            prev = curr
        returns_by_code[code] = rets
    pairs: List[Dict[str, Any]] = []
    codes = list(returns_by_code.keys())
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            a, b = codes[i], codes[j]
            corr = _pearson(returns_by_code[a], returns_by_code[b])
            if corr >= threshold:
                pairs.append({
                    "fund_a_code": a,
                    "fund_b_code": b,
                    "correlation": round(corr, 3),
                    "common_days": len(sorted_dates),
                })
    pairs.sort(key=lambda p: -p["correlation"])
    return pairs


def _compute_held_correlations(
    scheme_codes: List[int],
    threshold: float = HIGH_CORRELATION_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Fetch cached NAV histories for the held funds and return the
    pairs whose return-correlation exceeds the threshold.

    Side-effect free except for cache warming via fetch_scheme_nav.
    Errors per fund are swallowed — partial results are still useful.
    """
    if len(scheme_codes) < 2:
        return []
    nav_by_code: Dict[int, List[Dict[str, Any]]] = {}
    for code in scheme_codes:
        try:
            raw = fetch_scheme_nav(code)
            records = _convert_nav_to_records(raw.get("data", []))
            if records:
                nav_by_code[code] = records
        except Exception as exc:
            logger.warning("Correlation fetch failed for scheme %s: %s", code, exc)
    return _correlation_pairs_from_nav(nav_by_code, threshold=threshold)


def _outlier_flags(
    metrics: Dict[str, Any],
    peer_metrics_list: List[Dict[str, Any]],
    asset_class: str,
) -> List[str]:
    """Flag outlier behavior vs category peers with context-aware messages.

    Uses z-score check (>2 std devs from mean).
    """
    flags: List[str] = []
    if len(peer_metrics_list) < 5:
        return flags  # too few peers to detect outliers

    checks = []
    if asset_class == "equity":
        checks = [
            ("excess_return_pct", "high"),
            ("volatility_pct", "low"),
        ]
    else:
        checks = [
            ("cagr_pct", "high"),
            ("volatility_pct", "low"),
        ]

    for key, direction in checks:
        fund_val = metrics.get(key)
        if fund_val is None:
            continue
        peer_vals = [m.get(key) for m in peer_metrics_list if m.get(key) is not None]
        if len(peer_vals) < 5:
            continue
        mean = sum(peer_vals) / len(peer_vals)
        variance = sum((v - mean) ** 2 for v in peer_vals) / len(peer_vals)
        std = variance ** 0.5
        if std == 0:
            continue
        z = (fund_val - mean) / std
        if direction == "high" and z > 2.0:
            if key in ("excess_return_pct", "cagr_pct"):
                flags.append("Unusually high returns vs peers \u2014 may include short-term spikes")
            else:
                flags.append(f"Unusually high {key} vs peers \u2014 interpret cautiously")
        elif direction == "low" and z < -2.0:
            if key == "volatility_pct":
                flags.append("Unusually stable data vs peers \u2014 verify data quality")
            else:
                flags.append(f"Unusually low {key} vs peers \u2014 interpret cautiously")

    return flags


# ── Data structures ────────────────────────────────────────────

@dataclass
class FundHealthResult:
    """Health check result for a single holding."""
    scheme_code: int
    fund_name: str
    fund_house: str
    category: str
    asset_class: str  # "equity" or "debt"
    rank: int
    total_in_category: int
    status: str  # Strong / Neutral / Weak / Not Ranked
    confidence_level: str
    history_years: float
    horizon: str  # Short-term / Mid-term / Long-term / Unknown
    action: str  # Continue / Monitor / Review
    action_note: str  # optional factual observation
    strengths: List[str]
    weaknesses: List[str]
    metrics: Dict[str, Any]
    alternatives: List[Dict[str, Any]]  # top 3 from same category (if Weak/Neutral)
    data_quality_flags: List[Dict[str, str]] = field(default_factory=list)
    outlier_flags: List[str] = field(default_factory=list)
    your_fund_gaps: List[str] = field(default_factory=list)  # personal comparison items
    benchmark_name: str = ""  # benchmark used for ranking (equity only)


@dataclass
class ConcentrationWarning:
    """Warns about category/risk concentration."""
    category: str
    count: int
    weight_pct: float
    message: str


# Coverage Integrity Layer — extracted to .coverage submodule.
# Re-exported below for backward compatibility of imports.
from backend.investment_analytics.portfolio_health.coverage import (
    COVERAGE_FULL_PCT,
    COVERAGE_PARTIAL_PCT,
    CoverageReport,
    _build_coverage_report,
)


@dataclass
class PortfolioHealthResult:
    """Complete portfolio health check result."""
    holdings: List[FundHealthResult]
    not_found: List[Dict[str, Any]]  # scheme codes not in registry
    concentration: List[ConcentrationWarning]
    mistakes: List[Dict[str, Any]]
    redundancies: List[Dict[str, Any]]
    exposure_gaps: List[Dict[str, Any]]
    risk_summary: Dict[str, Any]
    portfolio_status: str  # Well diversified / Moderately concentrated / Highly concentrated
    computed_at: str
    correlations: List[Dict[str, Any]] = field(default_factory=list)
    correlation_threshold: float = HIGH_CORRELATION_THRESHOLD
    coverage: Optional[CoverageReport] = None
    action_priority: Optional[Dict[str, Any]] = None
    structural_priority: Optional[Dict[str, Any]] = None
    top_ranked_by_category: List[Dict[str, Any]] = field(default_factory=list)
    plan_efficiency_flags: List[Dict[str, Any]] = field(default_factory=list)


# ── Core logic ─────────────────────────────────────────────────

# Cache category rankings within a single health check call
_ranking_cache: Dict[str, Any] = {}


def _get_or_rank_equity(category: str, registry_path: str) -> Optional[CategoryRanking]:
    """Get cached equity ranking or compute it."""
    key = f"eq:{category}"
    if key in _ranking_cache:
        return _ranking_cache[key]
    try:
        result = rank_category(category, registry_path)
        _ranking_cache[key] = result
        return result
    except Exception as exc:
        logger.warning("Failed to rank equity category %s: %s", category, exc)
        _ranking_cache[key] = None
        return None


def _get_or_rank_debt(category: str, registry_path: str) -> Optional[DebtCategoryRanking]:
    """Get cached debt ranking or compute it."""
    key = f"dt:{category}"
    if key in _ranking_cache:
        return _ranking_cache[key]
    try:
        result = rank_debt_category(category, registry_path)
        _ranking_cache[key] = result
        return result
    except Exception as exc:
        logger.warning("Failed to rank debt category %s: %s", category, exc)
        _ranking_cache[key] = None
        return None


def _find_fund_in_ranking(
    scheme_code: int,
    ranked_funds: List[RankedFund],
) -> Optional[RankedFund]:
    """Find a specific fund in a ranking result by scheme code."""
    for rf in ranked_funds:
        if rf.fund.scheme_code == scheme_code:
            return rf
    return None


def _build_alternatives(
    ranked_funds: List[RankedFund],
    exclude_code: int,
    asset_class: str,
    held_metrics: Optional[Dict[str, Any]] = None,
    max_n: int = 3,
    unsafe_codes: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Get top N alternatives from the same category, excluding the held fund.

    Skips funds with Low confidence, funds in `unsafe_codes` (severe data
    quality / extreme outlier), and funds whose advantage over the held
    fund is not material per `_alternative_is_material`. Without the
    materiality gate the function would suggest rank-N+1 funds with
    statistically-indistinguishable improvements, encouraging churn.

    When held_metrics is omitted (the legacy path used for Not-Ranked
    holdings where we have no metrics to compare against), the gate is
    skipped — we just show the top-of-category as a reference list.

    Returns (alts, filter_stats) where filter_stats counts WHY peers
    were dropped: {"low_conf", "unsafe", "immaterial", "considered"}.
    The caller uses these counts to emit an accurate action_note when
    `alts` is empty — distinguishing "data thin" from "no peer is
    materially better" (OBS-2).
    """
    if unsafe_codes is None:
        unsafe_codes = set()
    alts: List[Dict[str, Any]] = []
    stats: Dict[str, int] = {
        "low_conf": 0,    # peers dropped because confidence == Low
        "unsafe": 0,      # peers dropped because of severe DQ / outlier
        "immaterial": 0,  # peers dropped by the material-improvement gate
        "considered": 0,  # peers that survived data filters (gate-eligible)
    }
    for rf in ranked_funds:
        if rf.fund.scheme_code == exclude_code:
            continue
        if rf.confidence_level == "Low":
            stats["low_conf"] += 1
            continue
        if rf.fund.scheme_code in unsafe_codes:
            stats["unsafe"] += 1
            continue

        stats["considered"] += 1
        alt_metrics_full = _metrics_for_display(rf.fund, asset_class)

        # Materiality gate — only when we can compare against the held fund.
        if held_metrics and not _alternative_is_material(
            held_metrics, alt_metrics_full, asset_class
        ):
            stats["immaterial"] += 1
            continue

        alt: Dict[str, Any] = {
            "rank": rf.rank,
            "scheme_code": rf.fund.scheme_code,
            "fund_name": rf.fund.fund_name,
            "fund_house": rf.fund.fund_house,
            "confidence_level": rf.confidence_level,
        }
        # Expose the full metric set so the UI can render side-by-side
        # comparisons (UI-7) without re-fetching peer data.
        alt["metrics"] = alt_metrics_full
        if asset_class == "equity":
            alt["excess_return_pct"] = rf.fund.excess_return_pct
            alt["consistency_pct"] = rf.fund.consistency_pct
        else:
            alt["cagr_pct"] = rf.fund.fund_cagr_pct
            alt["volatility_pct"] = rf.fund.volatility_pct

        # Justification: why this alternative ranks higher (with magnitude).
        if held_metrics:
            alt["justification"] = _build_justification(held_metrics, alt_metrics_full, asset_class)
        else:
            alt["justification"] = []

        alts.append(alt)
        if len(alts) >= max_n:
            break
    return alts, stats


def _metrics_for_display(fund: Any, asset_class: str) -> Dict[str, Any]:
    """Extract display-ready metrics from a FundMetrics object."""
    if asset_class == "equity":
        return {
            "excess_return_pct": fund.excess_return_pct,
            "consistency_pct": fund.consistency_pct,
            "max_drawdown_pct": fund.max_drawdown_pct,
            "volatility_pct": fund.volatility_pct,
            "downside_capture_ratio": fund.downside_capture_ratio,
        }
    else:
        return {
            "cagr_pct": fund.fund_cagr_pct,
            "volatility_pct": fund.volatility_pct,
            "max_drawdown_pct": fund.max_drawdown_pct,
            "consistency_pct": fund.consistency_pct,
            "risk_adj_return": fund.downside_capture_ratio,
        }


def check_portfolio_health(
    scheme_codes: List[int],
    weights: Optional[Dict[int, float]],
    registry_path: str,
) -> PortfolioHealthResult:
    """Run health check for a list of user holdings.

    Args:
        scheme_codes: List of AMFI scheme codes the user holds.
        weights: Optional dict of scheme_code → weight (0-1). If None, equal weight.
        registry_path: Path to the schemes.json registry.

    Returns:
        PortfolioHealthResult with per-fund status, alternatives, and risk view.
    """
    global _ranking_cache
    _ranking_cache = {}  # fresh cache per call

    registry = load_registry(Path(registry_path))
    code_to_scheme: Dict[int, SchemeEntry] = {s.scheme_code: s for s in registry}

    # Default equal weights if not provided
    if not weights:
        w = 1.0 / len(scheme_codes) if scheme_codes else 0.0
        weights = {code: w for code in scheme_codes}

    # Normalize weights to sum to 1. If the caller supplied weights
    # that sum to zero (every holding marked 0.0, or only NaN/None
    # entries), fall back to equal weights — otherwise downstream
    # capital-weighted views silently render every metric as zero
    # and the user sees "no concentration / no priority" with no
    # explanation that the input was malformed.
    total_weight = sum(weights.get(c, 0) for c in scheme_codes)
    if total_weight > 0:
        weights = {c: weights.get(c, 0) / total_weight for c in scheme_codes}
    elif scheme_codes:
        eq_w = 1.0 / len(scheme_codes)
        weights = {c: eq_w for c in scheme_codes}
        logger.warning(
            "Supplied weights sum to zero; falling back to equal weights "
            "for %d holdings.", len(scheme_codes),
        )

    holdings: List[FundHealthResult] = []
    not_found: List[Dict[str, Any]] = []
    category_weights: Dict[str, float] = {}
    # Snapshot of which CategoryRanking object was used for which category,
    # so the top-ranked-per-category collector doesn't have to re-fetch.
    rankings_by_category: Dict[str, Any] = {}

    for code in scheme_codes:
        scheme = code_to_scheme.get(code)
        if not scheme:
            not_found.append({
                "scheme_code": code,
                "reason": "Scheme code not found in AMFI registry",
            })
            continue

        category = scheme.scheme_category
        fund_weight = weights.get(code, 0)

        # Track category concentration
        category_weights[category] = category_weights.get(category, 0) + fund_weight

        # Detect ETFs — not supported for health check ranking
        _name_lower = scheme.scheme_name.lower()
        _cat_lower = category.lower()
        if "etf" in _name_lower or "etf" in _cat_lower or "exchange traded" in _cat_lower:
            holdings.append(FundHealthResult(
                scheme_code=code,
                fund_name=scheme.scheme_name,
                fund_house=scheme.fund_house,
                category=category,
                asset_class="equity",
                rank=0,
                total_in_category=0,
                status="Not Ranked",
                confidence_level="Unknown",
                history_years=0,
                horizon="Unknown",
                action="Monitor",
                action_note="ETF evaluation not supported yet",
                strengths=[],
                weaknesses=["ETF evaluation not supported yet"],
                metrics={},
                alternatives=[],
            ))
            continue

        # Determine asset class
        asset_class = "equity"
        if category in DEBT_CATEGORIES or category.startswith("Debt Scheme"):
            asset_class = "debt"

        # Skip excluded categories
        if category in EXCLUDED_CATEGORIES:
            holdings.append(FundHealthResult(
                scheme_code=code,
                fund_name=scheme.scheme_name,
                fund_house=scheme.fund_house,
                category=category,
                asset_class=asset_class,
                rank=0,
                total_in_category=0,
                status="Not Ranked",
                confidence_level="Unknown",
                history_years=0,
                horizon="Unknown",
                action="Monitor",
                action_note="Category excluded from ranking",
                strengths=[],
                weaknesses=["Category excluded from ranking (too heterogeneous)"],
                metrics={},
                alternatives=[],
            ))
            continue

        # Run category ranking
        ranking: Any = None
        if asset_class == "equity":
            ranking = _get_or_rank_equity(category, registry_path)
        else:
            ranking = _get_or_rank_debt(category, registry_path)
        if ranking is not None:
            rankings_by_category[category] = ranking

        if ranking is None:
            holdings.append(FundHealthResult(
                scheme_code=code,
                fund_name=scheme.scheme_name,
                fund_house=scheme.fund_house,
                category=category,
                asset_class=asset_class,
                rank=0,
                total_in_category=0,
                status="Not Ranked",
                confidence_level="Unknown",
                history_years=0,
                horizon="Unknown",
                action="Monitor",
                action_note="Could not be ranked",
                strengths=[],
                weaknesses=["Could not rank this category"],
                metrics={},
                alternatives=[],
            ))
            continue

        # Find this fund in the ranking
        found = _find_fund_in_ranking(code, ranking.ranked)

        if found is None:
            # Fund exists in registry but was excluded from ranking
            holdings.append(FundHealthResult(
                scheme_code=code,
                fund_name=scheme.scheme_name,
                fund_house=scheme.fund_house,
                category=category,
                asset_class=asset_class,
                rank=0,
                total_in_category=ranking.total_funds_in_category,
                status="Not Ranked",
                confidence_level="Unknown",
                history_years=0,
                horizon="Unknown",
                action="Monitor",
                action_note="Could not be ranked",
                strengths=[],
                weaknesses=["Excluded from ranking (insufficient data or not Direct Growth)"],
                metrics={},
                alternatives=_build_alternatives(ranking.ranked, code, asset_class)[0],
            ))
            continue

        total_ranked = len(ranking.ranked)
        conf = found.confidence_level
        status = _fund_status(found.rank, total_ranked, conf)

        # Compute metrics and horizon
        metrics = _metrics_for_display(found.fund, asset_class)
        vol = metrics.get("volatility_pct", 0) or 0
        dd = metrics.get("max_drawdown_pct", 0) or 0
        horizon = _horizon_tag(vol, dd, asset_class)

        # Peer sets computed per-iteration, scoped to THIS fund's category.
        # Must be defined before safe-peer-selection or dq/outlier flags use them.
        peer_points: List[int] = [rf.fund.aligned_points for rf in ranking.ranked]
        peer_metrics: List[Dict[str, Any]] = [
            _metrics_for_display(rf.fund, asset_class) for rf in ranking.ranked
        ]

        # Build alternatives for Weak and Neutral funds (with justification)
        alts: List[Dict[str, Any]] = []
        your_gaps: List[str] = []
        if status in ("Weak", "Neutral"):
            # ── Safe peer selection: exclude severe data quality + outlier peers ──
            unsafe_peer_codes: set = set()
            for rf in ranking.ranked:
                if rf.fund.scheme_code == code:
                    continue
                rf_pts = rf.fund.aligned_points
                # Severe data quality: <40% of peer median points
                if peer_points:
                    sorted_pts = sorted(peer_points)
                    median_pts = sorted_pts[len(sorted_pts) // 2]
                    if median_pts > 0 and rf_pts < median_pts * 0.4:
                        unsafe_peer_codes.add(rf.fund.scheme_code)
                        continue
                # Outlier check on this peer
                rf_m = _metrics_for_display(rf.fund, asset_class)
                rf_ol = _outlier_flags(rf_m, peer_metrics, asset_class)
                if rf_ol:
                    unsafe_peer_codes.add(rf.fund.scheme_code)

            alts, alt_filter_stats = _build_alternatives(
                ranking.ranked, code, asset_class,
                held_metrics=metrics, unsafe_codes=unsafe_peer_codes,
            )
            # Personal comparison: what YOUR fund is weaker at vs #1 alternative
            if alts:
                top_alt_fund = _find_fund_in_ranking(alts[0]["scheme_code"], ranking.ranked)
                if top_alt_fund:
                    top_alt_m = _metrics_for_display(top_alt_fund.fund, asset_class)
                    your_gaps = _build_your_fund_gaps(metrics, top_alt_m, asset_class)
        else:
            alt_filter_stats = {"low_conf": 0, "unsafe": 0, "immaterial": 0, "considered": 0}

        # Assign action
        action, action_note = _assign_action(status, bool(alts))

        # If Weak/Neutral but no alternatives surfaced, distinguish WHY:
        # - data thin (every peer Low-conf or unsafe DQ/outlier)
        # - no materially better peer existed (gate filtered everyone)
        # OBS-1 / OBS-2: the previous single message conflated both cases.
        if status in ("Weak", "Neutral") and not alts:
            considered = alt_filter_stats.get("considered", 0)
            if considered == 0:
                # No peer survived data filters — true data limitation.
                action_note = (
                    "No reliable comparison peer available due to data limitations in this category"
                )
            else:
                # Peers existed but none cleared the material-improvement gate.
                if status == "Neutral":
                    action_note = (
                        "Holding is comparable to peers — no materially better alternative in this category"
                    )
                else:
                    action_note = (
                        "No materially better peer in this category — current ranking position is comparable to top peers"
                    )

        # Get benchmark name for trust block (equity only)
        bench_name = ""
        if asset_class == "equity" and hasattr(ranking, "benchmark_name"):
            bench_name = ranking.benchmark_name

        # ── Data quality flags ──
        # peer_points / peer_metrics already computed above this iteration.
        dq_flags = _data_quality_flags(
            found.fund.history_years,
            found.fund.aligned_points,
            conf,
            peer_points,
        )

        # ── Outlier flags ──
        ol_flags = _outlier_flags(metrics, peer_metrics, asset_class)

        holdings.append(FundHealthResult(
            scheme_code=code,
            fund_name=found.fund.fund_name,
            fund_house=found.fund.fund_house,
            category=category,
            asset_class=asset_class,
            rank=found.rank,
            total_in_category=total_ranked,
            status=status,
            confidence_level=conf,
            history_years=found.fund.history_years,
            horizon=horizon,
            action=action,
            action_note=action_note,
            strengths=found.strengths,
            weaknesses=found.weaknesses,
            metrics=metrics,
            alternatives=alts,
            data_quality_flags=dq_flags,
            outlier_flags=ol_flags,
            your_fund_gaps=your_gaps,
            benchmark_name=bench_name,
        ))

    # ── Concentration analysis ──
    concentration: List[ConcentrationWarning] = []
    for cat, w in sorted(category_weights.items(), key=lambda x: -x[1]):
        count = sum(1 for h in holdings if h.category == cat)
        w_pct = round(w * 100, 1)
        if w_pct > 40:
            concentration.append(ConcentrationWarning(
                category=cat,
                count=count,
                weight_pct=w_pct,
                message=f"{w_pct}% in {_short_category(cat)} ({count} fund{'s' if count > 1 else ''}) — high concentration",
            ))
        elif w_pct > 25:
            concentration.append(ConcentrationWarning(
                category=cat,
                count=count,
                weight_pct=w_pct,
                message=f"{w_pct}% in {_short_category(cat)} ({count} fund{'s' if count > 1 else ''}) — moderate concentration",
            ))

    # ── Risk summary ──
    risk_summary = _build_risk_summary(holdings, weights)

    # ── Coverage Integrity Layer ──
    # Capital-weighted share of holdings that got a real peer rank.
    # "Not Ranked" holdings (ETFs, hybrids, sectoral, insufficient data,
    # registry misses) contribute to not_ranked_pct, NOT analyzed_pct.
    coverage = _build_coverage_report(holdings, weights or {})

    # ── Portfolio-level analysis ──
    mistakes = _detect_mistakes(holdings)
    redundancies = _detect_redundancy(holdings)
    exposure_gaps = _detect_exposure_gaps(holdings)

    # ── Slice 2 / P3: hidden-overlap correlation across all held funds.
    # Computed on the actual held set (not just ranked ones) so a user
    # who holds 5 large-caps under different category labels still gets
    # the "these aren't really diversified" signal. Only ranks-needed
    # holdings are considered (skip ETFs / unsupported categories whose
    # NAV fetch path may behave differently).
    rankable_codes = [h.scheme_code for h in holdings
                      if h.status != "Not Ranked" and h.scheme_code in code_to_scheme]
    correlations = _compute_held_correlations(rankable_codes)

    # ── Top-ranked visibility per held category (non-advisory) ──
    # For each category the user actually holds, surface the rank-1
    # fund's identity. Pure information ("top-ranked in category"),
    # not a recommendation. Suppressed when coverage band == "low".
    top_ranked_by_category = _collect_top_ranked_per_category(holdings, rankings_by_category)

    # ── Direct vs Regular plan structural flag ──
    # Detect held funds whose name lacks "Direct" — the Direct sibling
    # of the same scheme has a structurally lower expense ratio.
    # Surfaced as a separate signal, not as Continue/Monitor/Review.
    plan_efficiency_flags = _detect_regular_plan_holdings(holdings, registry)
    # Attach capital-weight % to each flag so a small Regular plan and
    # a large Regular plan are visually distinguishable in the UI.
    _resolved_weights = _resolve_weights(holdings, weights or {})
    _eq_w = 1.0 / (len(holdings) or 1)
    for f in plan_efficiency_flags:
        f["weight_pct"] = round(
            _resolved_weights.get(f["scheme_code"], _eq_w) * 100, 1
        )

    # Structural priority — heaviest Regular plan holding. Distinct
    # from action_priority (peer-rank verdict). UI renders both.
    structural_priority = _build_structural_priority(
        plan_efficiency_flags, holdings, weights or {},
    )

    _ranking_cache = {}  # clear cache

    # ── Portfolio status label (concentration- AND coverage-aware) ──
    portfolio_status = _portfolio_status_label(
        concentration, mistakes, exposure_gaps, coverage,
    )

    # ── Coverage-gated suppression (item D) ──
    # When coverage is "low", portfolio-level conclusions are based on
    # a minority of capital and may be misleading. Drop the conclusions
    # rather than show possibly-misleading rows. The coverage banner
    # already explains why.
    if coverage and coverage.confidence_band == "low":
        concentration_out: List[ConcentrationWarning] = []
        redundancies_out: List[Dict[str, Any]] = []
        exposure_gaps_out: List[Dict[str, Any]] = []
        correlations_out: List[Dict[str, Any]] = []
        # Keep mistakes (over-diversification, AMC concentration,
        # category crowding) — those operate on counts, not capital
        # share, and remain factually true regardless of coverage.
    else:
        # Cross-detector dedup (item C): collapse same fund-pair signals
        # so the user sees one row per root cause rather than three.
        redundancies_out, correlations_out, mistakes = _dedup_overlap_signals(
            redundancies, correlations, mistakes, holdings,
        )
        concentration_out = concentration
        exposure_gaps_out = exposure_gaps

    # ── Action-priority headline (item A) ──
    # One-line "address this first" picked by capital weight, then
    # severity. Pure pointer, not a recommendation.
    action_priority = _build_action_priority(holdings, weights or {})

    return PortfolioHealthResult(
        holdings=holdings,
        not_found=not_found,
        concentration=concentration_out,
        mistakes=mistakes,
        redundancies=redundancies_out,
        exposure_gaps=exposure_gaps_out,
        risk_summary=risk_summary,
        portfolio_status=portfolio_status,
        computed_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        correlations=correlations_out,
        correlation_threshold=HIGH_CORRELATION_THRESHOLD,
        coverage=coverage,
        action_priority=action_priority,
        structural_priority=structural_priority,
        top_ranked_by_category=top_ranked_by_category,
        plan_efficiency_flags=plan_efficiency_flags,
    )


# _build_coverage_report now lives in .coverage and is re-exported above.


def _portfolio_status_label(
    concentration: List[ConcentrationWarning],
    mistakes: List[Dict[str, Any]],
    exposure_gaps: List[Dict[str, Any]],
    coverage: Optional[CoverageReport] = None,
) -> str:
    """Derive a single portfolio structure label from existing signals.

    Uses only data already computed — no invented math. When coverage
    is low, the diversification verdict is suppressed in favour of a
    coverage-aware label, because asserting "Well diversified" while
    half the capital wasn't analyzed is exactly the false-confidence
    pattern Item 1 was designed to prevent.
    """
    if coverage and coverage.confidence_band == "low":
        return "Coverage limited — partial diversification view"
    high_conc = any(c.weight_pct > 40 for c in concentration)
    mod_conc = any(c.weight_pct > 25 for c in concentration)
    over_div = any(m.get("type") == "over_diversification" for m in mistakes)
    crowded = any(m.get("type") == "category_crowding" for m in mistakes)

    if high_conc:
        return "Highly concentrated"
    if over_div:
        return "Over-diversified"
    if mod_conc or crowded:
        return "Some concentration present"
    if coverage and coverage.confidence_band == "partial":
        return "Well diversified (partial coverage)"
    return "Well diversified"


def _resolve_weights(
    holdings: List[FundHealthResult],
    weights: Dict[int, float],
) -> Dict[int, float]:
    """Resolve per-holding weights, normalising user-supplied weights and
    falling back to equal share when none are provided. Used by the
    priority + flag enrichment helpers below."""
    n = len(holdings) or 1
    eq_weight = 1.0 / n
    held_codes = {h.scheme_code for h in holdings}
    held_weight_total = sum(weights.get(c, 0) for c in held_codes) or 0.0
    if held_weight_total > 0:
        return {c: weights.get(c, 0) / held_weight_total for c in held_codes}
    return {c: eq_weight for c in held_codes}


def _build_action_priority(
    holdings: List[FundHealthResult],
    weights: Dict[int, float],
) -> Optional[Dict[str, Any]]:
    """Pick the single highest-impact action the user should look at first.

    Capital-weighted; severity tiebreaker (Review > Monitor > nothing).
    Pure pointer — names a position the user already holds, not a fund
    to acquire. Returns None when no Review/Monitor exists.
    """
    if not holdings:
        return None
    resolved = _resolve_weights(holdings, weights)
    eq_weight = 1.0 / (len(holdings) or 1)

    # Severity rank: Review = 2, Monitor = 1, Continue = 0.
    severity_rank = {"Review": 2, "Monitor": 1, "Continue": 0}
    candidates = []
    for h in holdings:
        sev = severity_rank.get(h.action, 0)
        if sev == 0:
            continue
        candidates.append((sev, resolved.get(h.scheme_code, eq_weight), h))
    if not candidates:
        return None
    # Highest severity wins; within severity, highest weight wins.
    candidates.sort(key=lambda t: (-t[0], -t[1]))
    sev, w, h = candidates[0]
    return {
        "scheme_code": h.scheme_code,
        "fund_name": h.fund_name,
        "action": h.action,
        "weight_pct": round(w * 100, 1),
        "category_short": _short_category(h.category),
        "headline": (
            f"Address first: {h.fund_name} — {h.action} "
            f"({round(w * 100, 1)}% of portfolio, {_short_category(h.category)})"
        ),
    }


def _build_structural_priority(
    plan_efficiency_flags: List[Dict[str, Any]],
    holdings: List[FundHealthResult],
    weights: Dict[int, float],
) -> Optional[Dict[str, Any]]:
    """Pick the heaviest STRUCTURAL inefficiency (currently: Regular plan
    holdings). Distinct from action_priority because plan choice is not
    a peer-rank verdict — it's a fixed cost leak that exists even on a
    rank-1 fund.

    Returns None when no Regular plan holdings exist. The UI renders
    this alongside action_priority; whichever has more weight gets
    visual primacy. We deliberately do NOT collapse the two axes —
    "your top fund is on the wrong plan" and "your bottom fund is
    a peer-rank Review" are different problems that deserve
    different headlines.
    """
    if not plan_efficiency_flags:
        return None
    resolved = _resolve_weights(holdings, weights)
    eq_weight = 1.0 / (len(holdings) or 1)
    by_code = {h.scheme_code: h for h in holdings}
    candidates = []
    for f in plan_efficiency_flags:
        code = f["scheme_code"]
        h = by_code.get(code)
        if h is None:
            continue
        w = resolved.get(code, eq_weight)
        candidates.append((w, f, h))
    if not candidates:
        return None
    candidates.sort(key=lambda t: -t[0])
    w, flag, h = candidates[0]
    pct = round(w * 100, 1)
    return {
        "scheme_code": h.scheme_code,
        "fund_name": h.fund_name,
        "weight_pct": pct,
        "category_short": _short_category(h.category),
        "type": "regular_plan",
        "direct_sibling_code": (
            None if flag.get("direct_sibling") is None
            else flag["direct_sibling"]["scheme_code"]
        ),
        "headline": (
            f"Plan inefficiency: {h.fund_name} is on the Regular plan "
            f"({pct}% of portfolio). Direct plan variant has a structurally "
            f"lower expense ratio."
        ),
    }


def _primary_metric_gap(
    held_metrics: Dict[str, Any],
    top_metrics: Dict[str, Any],
    asset_class: str,
) -> Optional[Dict[str, Any]]:
    """Pick the single largest improvement metric and return its delta
    + magnitude bucket. Used to quantify the gap between a held fund
    and the rank-1 fund in its category, so "top-ranked exists" turns
    from a label into a quantified observation."""
    bullets = _EQUITY_BULLETS if asset_class == "equity" else _DEBT_BULLETS
    best: Optional[Dict[str, Any]] = None
    for key, label in bullets:
        h_val = held_metrics.get(key, 0) or 0
        t_val = top_metrics.get(key, 0) or 0
        delta = _signed_delta(key, h_val, t_val)
        if delta <= 0:
            continue
        magnitude = _improvement_magnitude(key, delta, asset_class)
        rank = {"large": 3, "moderate": 2, "small": 1}.get(magnitude, 0)
        if best is None or rank > best["_rank"]:
            best = {
                "metric": key,
                "label": label,
                "held_value": round(h_val, 2),
                "top_value": round(t_val, 2),
                "delta": round(abs(delta), 2),
                "magnitude": magnitude,
                "_rank": rank,
            }
    if best is None:
        return None
    best.pop("_rank", None)
    return best


def _collect_top_ranked_per_category(
    holdings: List[FundHealthResult],
    rankings_by_category: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """For each category the user holds, surface the rank-1 fund's name.

    Strictly informational. Limited to categories actually held — we do
    NOT introduce categories the user doesn't own, which would cross
    into "consider this fund" territory.

    `rankings_by_category` maps category name -> CategoryRanking /
    DebtCategoryRanking. Built by the caller from the per-holding
    rankings that were already fetched, so this function does not
    re-fetch.
    """
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for h in holdings:
        if h.status == "Not Ranked":
            continue
        if h.category in seen:
            continue
        seen.add(h.category)
        ranking = rankings_by_category.get(h.category)
        if not ranking or not getattr(ranking, "ranked", None):
            continue
        top = ranking.ranked[0]
        if top.confidence_level == "Low":
            continue
        # Skip if the top-ranked is the user's own holding — saying
        # "your fund is already top-ranked" duplicates their card.
        held_in_cat = {x.scheme_code for x in holdings if x.category == h.category}
        if top.fund.scheme_code in held_in_cat:
            continue
        # Compute the gap between the user's held fund(s) in this
        # category and the rank-1 fund. "Top-ranked exists" is
        # informational; "top-ranked beats yours by 2.7pp on excess
        # return" is decision-relevant. Quantification is non-advisory.
        top_metrics = _metrics_for_display(top.fund, h.asset_class)
        vs_holdings: List[Dict[str, Any]] = []
        for held in holdings:
            if held.category != h.category:
                continue
            if not held.metrics:
                continue
            gap = _primary_metric_gap(held.metrics, top_metrics, held.asset_class)
            if gap is None:
                continue
            # Determine whether the gap clears the material-improvement
            # gate (same logic the alternatives filter uses).
            material = _alternative_is_material(
                held.metrics, top_metrics, held.asset_class,
            )
            vs_holdings.append({
                "held_scheme_code": held.scheme_code,
                "held_fund_name": held.fund_name,
                "primary_delta": gap,
                "is_material": material,
            })

        out.append({
            "category": h.category,
            "category_short": _short_category(h.category),
            "scheme_code": top.fund.scheme_code,
            "fund_name": top.fund.fund_name,
            "fund_house": top.fund.fund_house,
            "confidence_level": top.confidence_level,
            "vs_holdings": vs_holdings,
        })
    return out


def _detect_regular_plan_holdings(
    holdings: List[FundHealthResult],
    registry: List[SchemeEntry],
) -> List[Dict[str, Any]]:
    """Detect held funds that are NOT Direct Plans and find the Direct sibling.

    Direct Plan vs Regular Plan is a structural fact: same fund, same
    portfolio, same returns BEFORE expenses — but Regular pays an
    intermediary commission baked into TER. Direct typically runs
    0.5-1.5pp/yr lower.

    The flag carries the held scheme + the Direct sibling code if found.
    Phrased as observation, not action: "Plan choice is structural,
    distinct from peer ranking." UI surfaces this on the holding card.

    Uses the registry's scheme_name (canonical AMFI label) — not
    FundHealthResult.fund_name, which can be a downstream-derived
    string and may not preserve the "Regular Plan" / "Direct Plan"
    suffix verbatim.
    """
    flags: List[Dict[str, Any]] = []
    by_code = {s.scheme_code: s for s in registry}
    by_base = _build_base_name_index(registry)
    for h in holdings:
        scheme = by_code.get(h.scheme_code)
        if scheme is None:
            continue
        name = scheme.scheme_name or ""
        if not name:
            continue
        if "Direct" in name:
            continue  # already on the efficient plan
        base = _scheme_base_name(name).lower()
        # Find the Direct sibling — same base name with "Direct" present.
        sibling = None
        for entry in by_base.get(base, []):
            if "Direct" in entry.scheme_name and "Growth" in entry.scheme_name:
                sibling = entry
                break
        flag = {
            "scheme_code": h.scheme_code,
            "fund_name": h.fund_name,
            "category_short": _short_category(h.category),
            "is_regular_plan": True,
            "direct_sibling": (
                None if sibling is None else {
                    "scheme_code": sibling.scheme_code,
                    "fund_name": sibling.scheme_name,
                }
            ),
            "message": (
                "This holding is the Regular plan. The Direct plan "
                "variant of the same scheme has a structurally lower "
                "expense ratio (typically 0.5-1.5 percentage points "
                "per year). Plan choice is structural, distinct from "
                "peer ranking."
            ),
        }
        flags.append(flag)
    return flags


def _build_base_name_index(
    registry: List[SchemeEntry],
) -> Dict[str, List[SchemeEntry]]:
    """Group registry entries by their base name (plan/option suffixes
    stripped) for Direct/Regular sibling lookup."""
    index: Dict[str, List[SchemeEntry]] = {}
    for entry in registry:
        base = _scheme_base_name(entry.scheme_name).lower()
        index.setdefault(base, []).append(entry)
    return index


_VARIANT_SUFFIXES_RE = re.compile(
    r"\s*-?\s*(?:Direct|Regular)\s*Plan\s*-?\s*(?:Growth|Dividend|IDCW|Payout|Reinvestment)?.*$",
    re.IGNORECASE,
)


def _scheme_base_name(name: str) -> str:
    """Strip "Direct/Regular Plan - Growth/Dividend/..." suffixes so two
    variants of the same scheme map to the same key."""
    return _VARIANT_SUFFIXES_RE.sub("", name).strip()


def _dedup_overlap_signals(
    redundancies: List[Dict[str, Any]],
    correlations: List[Dict[str, Any]],
    mistakes: List[Dict[str, Any]],
    holdings: List[FundHealthResult],
) -> tuple:
    """Collapse signals that describe the same root cause (same fund pair
    or same category) into one row.

    A redundancy + a correlation + a category_crowding entry can all be
    triggered by the same two funds in the same category. Showing all
    three is noise, not insight.
    """
    # Build a key for each redundancy + correlation: ordered fund-code pair.
    def _pair_key(a: int, b: int) -> tuple:
        return (a, b) if a < b else (b, a)

    correlated_pairs = {
        _pair_key(c["fund_a_code"], c["fund_b_code"]): c
        for c in correlations
    }
    deduped_redundancies: List[Dict[str, Any]] = []
    for r in redundancies:
        key = _pair_key(r["fund_a"]["scheme_code"], r["fund_b"]["scheme_code"])
        if key in correlated_pairs:
            # Correlation is a stronger signal (return-based) than
            # metric-similarity redundancy — drop the redundancy row.
            continue
        deduped_redundancies.append(r)

    # category_crowding: if every fund in the crowded category is part of
    # a flagged correlation pair, the crowding row is redundant with the
    # correlation rows. Otherwise keep it.
    by_cat: Dict[str, List[FundHealthResult]] = {}
    for h in holdings:
        by_cat.setdefault(h.category, []).append(h)
    cat_codes_in_corr: Dict[str, set] = {}
    for c in correlations:
        a_code = c["fund_a_code"]
        b_code = c["fund_b_code"]
        for h in holdings:
            if h.scheme_code in (a_code, b_code):
                cat_codes_in_corr.setdefault(h.category, set()).add(h.scheme_code)

    deduped_mistakes: List[Dict[str, Any]] = []
    for m in mistakes:
        if m.get("type") == "category_crowding":
            cat = m.get("category") or ""
            cat_holdings = by_cat.get(cat, [])
            corr_codes = cat_codes_in_corr.get(cat, set())
            cat_codes = {h.scheme_code for h in cat_holdings}
            if cat_codes and cat_codes.issubset(corr_codes):
                # Every crowded fund is already in a correlation row.
                continue
        deduped_mistakes.append(m)

    return deduped_redundancies, correlations, deduped_mistakes


def _filter_top_ranked_by_coverage(
    rows: List[Dict[str, Any]],
    coverage: Optional[CoverageReport],
) -> List[Dict[str, Any]]:
    """Suppress top-ranked visibility entirely when coverage is low —
    declaring 'top-ranked in your categories' on the back of a minority
    of capital is the false-confidence pattern Item 1 was meant to stop.
    Tag with a partial-coverage note when band is partial."""
    if not rows:
        return []
    if coverage and coverage.confidence_band == "low":
        return []
    if coverage and coverage.confidence_band == "partial":
        return [
            {**r, "coverage_note": "based on partial portfolio coverage"}
            for r in rows
        ]
    return rows


def _enrich_correlations(
    pairs: List[Dict[str, Any]],
    holdings: List["FundHealthResult"],
    weights: Optional[Dict[int, float]] = None,
) -> List[Dict[str, Any]]:
    """Attach fund names + categories AND per-fund + combined capital
    weight to each correlation pair so the UI can rank pairs by
    capital impact (a 50%/40% pair matters more than a 5%/3% pair
    even at identical ρ). Drops pairs whose codes are not present in
    `holdings` (defensive)."""
    by_code = {h.scheme_code: h for h in holdings}
    resolved = _resolve_weights(holdings, weights or {})
    eq_w = 1.0 / (len(holdings) or 1)

    def _w(code: int) -> float:
        return resolved.get(code, eq_w)

    enriched: List[Dict[str, Any]] = []
    for p in pairs:
        a = by_code.get(p["fund_a_code"])
        b = by_code.get(p["fund_b_code"])
        if not a or not b:
            continue
        a_w = round(_w(a.scheme_code) * 100, 1)
        b_w = round(_w(b.scheme_code) * 100, 1)
        combined_w = round(a_w + b_w, 1)
        enriched.append({
            "fund_a": {
                "scheme_code": a.scheme_code,
                "fund_name": a.fund_name,
                "category_short": _short_category(a.category),
                "weight_pct": a_w,
            },
            "fund_b": {
                "scheme_code": b.scheme_code,
                "fund_name": b.fund_name,
                "category_short": _short_category(b.category),
                "weight_pct": b_w,
            },
            "correlation": p["correlation"],
            "common_days": p["common_days"],
            "cross_category": a.category != b.category,
            "combined_weight_pct": combined_w,
            "message": (
                "Funds move together (correlation "
                + str(p["correlation"])
                + (" across different categories" if a.category != b.category
                   else " in the same category")
                + f"; combined {combined_w}% of portfolio"
                + ") — actual diversification is lower than category labels suggest"
            ),
        })
    # Sort by combined capital first, correlation second — a heavy
    # pair at 0.86 is more decision-relevant than a tiny pair at 0.99.
    enriched.sort(key=lambda x: (-x["combined_weight_pct"], -x["correlation"]))
    return enriched


def _short_category(cat: str) -> str:
    """Shorten category name for display."""
    return (cat
            .replace("Equity Scheme - ", "")
            .replace("Debt Scheme - ", "")
            .replace(" Fund", ""))


def _build_risk_summary(
    holdings: List[FundHealthResult],
    weights: Dict[int, float],
) -> Dict[str, Any]:
    """Build portfolio-level risk summary."""
    if not holdings:
        return {"equity_pct": 0, "debt_pct": 0, "other_pct": 0, "risk_level": "Unknown"}

    eq_w = sum(weights.get(h.scheme_code, 0) for h in holdings if h.asset_class == "equity")
    dt_w = sum(weights.get(h.scheme_code, 0) for h in holdings if h.asset_class == "debt")
    other_w = max(0, 1.0 - eq_w - dt_w)

    # Simple risk level based on equity allocation
    if eq_w >= 0.8:
        risk_level = "High"
    elif eq_w >= 0.5:
        risk_level = "Medium-High"
    elif eq_w >= 0.3:
        risk_level = "Medium"
    elif eq_w >= 0.1:
        risk_level = "Low-Medium"
    else:
        risk_level = "Low"

    # Count statuses
    strong_count = sum(1 for h in holdings if h.status == "Strong")
    avg_count = sum(1 for h in holdings if h.status == "Neutral")
    weak_count = sum(1 for h in holdings if h.status == "Weak")
    nr_count = sum(1 for h in holdings if h.status == "Not Ranked")
    low_conf_count = sum(1 for h in holdings if h.confidence_level == "Low")

    return {
        "equity_pct": round(eq_w * 100, 1),
        "debt_pct": round(dt_w * 100, 1),
        "other_pct": round(other_w * 100, 1),
        "risk_level": risk_level,
        "strong_count": strong_count,
        "neutral_count": avg_count,
        "weak_count": weak_count,
        "not_ranked_count": nr_count,
        "low_confidence_count": low_conf_count,
        "total_holdings": len(holdings),
    }


# ── Serialization ──────────────────────────────────────────────

def portfolio_health_to_dict(
    result: PortfolioHealthResult,
    weights: Optional[Dict[int, float]] = None,
) -> dict:
    """Serialize for API response.

    `weights` (scheme_code -> 0..1) lets the decision_summary carry
    weight_pct per entry so the frontend can prioritize Review items by
    actual capital impact instead of input order. If omitted, all
    weights default to equal share.
    """
    # Resolve weights — default to equal share if not provided.
    n_holdings = len(result.holdings) or 1
    eq_weight = 1.0 / n_holdings
    if weights:
        # Normalize to sum 1 across the holdings actually evaluated.
        held_codes = {h.scheme_code for h in result.holdings}
        held_weight_total = sum(weights.get(c, 0) for c in held_codes) or 0
        if held_weight_total > 0:
            resolved = {c: weights.get(c, 0) / held_weight_total for c in held_codes}
        else:
            resolved = {c: eq_weight for c in held_codes}
    else:
        resolved = {h.scheme_code: eq_weight for h in result.holdings}

    # Build decision summary (grouped by action) — entries carry
    # weight_pct so the UI can sort by capital impact.
    decision_summary: Dict[str, List[Dict[str, Any]]] = {
        "Continue": [],
        "Monitor": [],
        "Review": [],
    }
    decision_weight_pct: Dict[str, float] = {"Continue": 0.0, "Monitor": 0.0, "Review": 0.0}
    for h in result.holdings:
        w_pct = round(resolved.get(h.scheme_code, eq_weight) * 100, 1)
        entry = {
            "scheme_code": h.scheme_code,
            "fund_name": h.fund_name,
            "category_short": _short_category(h.category),
            "action_note": h.action_note,
            "weight_pct": w_pct,
        }
        bucket = h.action if h.action in decision_summary else "Monitor"
        decision_summary[bucket].append(entry)
        decision_weight_pct[bucket] = round(decision_weight_pct[bucket] + w_pct, 1)

    # Sort each bucket by weight_pct descending so highest-capital-
    # impact action shows first.
    for bucket in decision_summary:
        decision_summary[bucket].sort(key=lambda e: -e["weight_pct"])

    return {
        "computed_at": result.computed_at,
        "schema_version": PORTFOLIO_HEALTH_SCHEMA_VERSION,
        "total_holdings": len(result.holdings),
        "not_found_count": len(result.not_found),
        "decision_summary": decision_summary,
        "decision_summary_weight_pct": decision_weight_pct,
        "holdings": [
            {
                "scheme_code": h.scheme_code,
                "fund_name": h.fund_name,
                "fund_house": h.fund_house,
                "category": h.category,
                "category_short": _short_category(h.category),
                "asset_class": h.asset_class,
                "rank": h.rank,
                "total_in_category": h.total_in_category,
                "status": h.status,
                "action": h.action,
                "action_note": h.action_note,
                "confidence_level": h.confidence_level,
                "history_years": h.history_years,
                "benchmark_name": h.benchmark_name,
                "strengths": h.strengths,
                "weaknesses": h.weaknesses,
                "metrics": h.metrics,
                "alternatives": h.alternatives,
                "horizon": h.horizon,
                "risk_tag": DEBT_RISK_TAGS.get(h.category, None) if h.asset_class == "debt" else None,
                "data_quality_flags": h.data_quality_flags,
                "outlier_flags": h.outlier_flags,
                "your_fund_gaps": h.your_fund_gaps,
            }
            for h in result.holdings
        ],
        "not_found": result.not_found,
        "concentration": [
            {
                "category": c.category,
                "category_short": _short_category(c.category),
                "count": c.count,
                "weight_pct": c.weight_pct,
                "message": c.message,
            }
            for c in result.concentration
        ],
        "mistakes": result.mistakes,
        "redundancies": result.redundancies,
        "exposure_gaps": result.exposure_gaps,
        "correlations": _enrich_correlations(
            result.correlations, result.holdings, weights,
        ),
        "correlation_threshold": result.correlation_threshold,
        "coverage": (
            None if result.coverage is None else {
                "total_holdings": result.coverage.total_holdings,
                "analyzed_holdings": result.coverage.analyzed_holdings,
                "analyzed_pct": result.coverage.analyzed_pct,
                "not_ranked_pct": result.coverage.not_ranked_pct,
                "confidence_band": result.coverage.confidence_band,
                "note": result.coverage.note,
                "affected_metrics": result.coverage.affected_metrics,
            }
        ),
        "action_priority": result.action_priority,
        "structural_priority": result.structural_priority,
        "top_ranked_by_category": _filter_top_ranked_by_coverage(
            result.top_ranked_by_category, result.coverage,
        ),
        "plan_efficiency_flags": result.plan_efficiency_flags,
        "risk_summary": result.risk_summary,
        "portfolio_status": result.portfolio_status,
        "no_major_issues": (
            len(result.mistakes) == 0
            and len(result.redundancies) == 0
            and len(result.exposure_gaps) == 0
            and len(result.concentration) == 0
            and len(result.correlations) == 0
        ),
        "data_as_of": result.computed_at[:10],
        "limitations": [
            "Health check is based on historical peer ranking — not predictive.",
            "Only currently active funds are ranked (survivorship bias).",
            "Status reflects peer position, not absolute quality.",
            "Strong requires both top-25% rank and sufficient data history.",
            "Actions (Continue/Monitor/Review) reflect data signals, not financial advice.",
            "Alternatives shown for context — not recommendations.",
            "Potential overlap between holdings is not measured — actual diversification may differ.",
            "Redundancy detection uses metric similarity, not portfolio correlation.",
            "Exposure gaps are observations, not allocation guidance.",
            "Debt metrics are NAV-based. Credit quality and duration not captured.",
            "ETFs are not evaluated — NAV-based ranking does not apply to exchange-traded instruments.",
            "Horizon tags are estimated from historical volatility/drawdown — not a holding-period recommendation.",
        ],
    }
