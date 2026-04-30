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
) -> List[Dict[str, Any]]:
    """Get top N alternatives from the same category, excluding the held fund.

    Skips funds with Low confidence, funds in `unsafe_codes` (severe data
    quality / extreme outlier), and funds whose advantage over the held
    fund is not material per `_alternative_is_material`. Without the
    materiality gate the function would suggest rank-N+1 funds with
    statistically-indistinguishable improvements, encouraging churn.

    When held_metrics is omitted (the legacy path used for Not-Ranked
    holdings where we have no metrics to compare against), the gate is
    skipped — we just show the top-of-category as a reference list.
    """
    if unsafe_codes is None:
        unsafe_codes = set()
    alts: List[Dict[str, Any]] = []
    for rf in ranked_funds:
        if rf.fund.scheme_code == exclude_code:
            continue
        if rf.confidence_level == "Low":
            continue
        if rf.fund.scheme_code in unsafe_codes:
            continue
        alt_metrics_full = _metrics_for_display(rf.fund, asset_class)

        # Materiality gate — only when we can compare against the held fund.
        if held_metrics and not _alternative_is_material(
            held_metrics, alt_metrics_full, asset_class
        ):
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
    return alts


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

    # Normalize weights to sum to 1
    total_weight = sum(weights.get(c, 0) for c in scheme_codes)
    if total_weight > 0:
        weights = {c: weights.get(c, 0) / total_weight for c in scheme_codes}

    holdings: List[FundHealthResult] = []
    not_found: List[Dict[str, Any]] = []
    category_weights: Dict[str, float] = {}

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
                alternatives=_build_alternatives(ranking.ranked, code, asset_class),
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

            alts = _build_alternatives(
                ranking.ranked, code, asset_class,
                held_metrics=metrics, unsafe_codes=unsafe_peer_codes,
            )
            # Personal comparison: what YOUR fund is weaker at vs #1 alternative
            if alts:
                top_alt_fund = _find_fund_in_ranking(alts[0]["scheme_code"], ranking.ranked)
                if top_alt_fund:
                    top_alt_m = _metrics_for_display(top_alt_fund.fund, asset_class)
                    your_gaps = _build_your_fund_gaps(metrics, top_alt_m, asset_class)

        # Assign action
        action, action_note = _assign_action(status, bool(alts))

        # If Weak/Neutral but no safe alternatives available, note it
        if status in ("Weak", "Neutral") and not alts:
            action_note = "No reliable comparison peer available due to data limitations in this category"

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

    # ── Portfolio-level analysis ──
    mistakes = _detect_mistakes(holdings)
    redundancies = _detect_redundancy(holdings)
    exposure_gaps = _detect_exposure_gaps(holdings)

    _ranking_cache = {}  # clear cache

    # ── Portfolio status label (concentration-based) ──
    portfolio_status = _portfolio_status_label(concentration, mistakes, exposure_gaps)

    return PortfolioHealthResult(
        holdings=holdings,
        not_found=not_found,
        concentration=concentration,
        mistakes=mistakes,
        redundancies=redundancies,
        exposure_gaps=exposure_gaps,
        risk_summary=risk_summary,
        portfolio_status=portfolio_status,
        computed_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    )


def _portfolio_status_label(
    concentration: List[ConcentrationWarning],
    mistakes: List[Dict[str, Any]],
    exposure_gaps: List[Dict[str, Any]],
) -> str:
    """Derive a single portfolio structure label from existing signals.

    Uses only data already computed — no invented math.
    """
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
    return "Well diversified"


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
        "risk_summary": result.risk_summary,
        "portfolio_status": result.portfolio_status,
        "no_major_issues": (
            len(result.mistakes) == 0
            and len(result.redundancies) == 0
            and len(result.exposure_gaps) == 0
            and len(result.concentration) == 0
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
