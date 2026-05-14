"""Alternative-selection gate, justification, and metric-gap helpers.

Encapsulates the "is this peer materially better than the held fund?"
discipline. Without the gate here, the system would surface rank-N+1
funds whose advantage is statistically indistinguishable from the
held fund — pure churn-encouragement. Every threshold lives in
`_ALT_THRESHOLDS`, every comparison goes through `_signed_delta`
which knows each metric's natural "better" direction.

Dependency direction: this module is a leaf — it depends only on
stdlib + ranking.FundMetrics (forward-referenced via TYPE_CHECKING)
+ RankedFund (same). It must never import from __init__.py or any
other portfolio_health submodule.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from backend.investment_analytics.ranking import FundMetrics, RankedFund  # noqa: F401


# ─────────────────────────────────────────────────────────────────────
# Threshold table — single source of truth for "what counts as
# material improvement" per metric per asset class. Conservative on
# equity (high cross-sectional dispersion); tighter on debt
# (compressed yields).
# Each entry: (delta_for_moderate, delta_for_large) in the metric's
# natural units. delta_for_moderate must be exceeded for a metric
# improvement to count toward the materiality gate;
# delta_for_large drives the "large" magnitude UI tag.
# ─────────────────────────────────────────────────────────────────────
_ALT_THRESHOLDS = {
    "equity": {
        "excess_return_pct":      (1.5,  3.0),
        "consistency_pct":        (10.0, 20.0),
        "max_drawdown_pct":       (5.0,  10.0),
        "volatility_pct":         (1.5,  3.0),
        "downside_capture_ratio": (0.10, 0.20),
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
    """Bucket a per-metric improvement delta into 'large' / 'moderate'
    / 'small'. delta MUST be expressed as 'alt-better-than-held' in
    the metric's natural direction (positive = better). Returns
    'none' if the alt is not better, 'small' if better but below the
    moderate threshold."""
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
    """Return alt-vs-held delta in the metric's natural 'better'
    direction. Positive delta means alt is better than held.

    The sign convention is metric-specific because some metrics are
    higher-is-better (returns) and some are lower-is-better
    (volatility, downside capture). max_drawdown_pct is stored as a
    negative number; less-negative = shallower = better.
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


# Bullet ordering matters — UI renders justification bullets in the
# order given here. Most-decision-relevant first.
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

    Returns list of {"reason": ..., "magnitude": small/moderate/large,
    "metric": <key>}. No percentage claims, no predictions.
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
    """Gate: an alternative must beat held on >=3 of 5 metrics AND
    show at least one moderate-or-large improvement to be surfaced.

    Without this gate the system would suggest rank-N+1 funds whose
    advantage is statistically indistinguishable from rank-N — pure
    churn-encouragement. The 3-of-5 wins requirement mirrors the
    pairwise-dominance rule already used in ranking.
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
    """Personal-comparison: what the user's fund is weaker at vs the
    top-ranked alternative. Inverse framing of _build_justification —
    points the gap at the holding, not the peer."""
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


def _metrics_for_display(fund: Any, asset_class: str) -> Dict[str, Any]:
    """Bridge function: convert a FundMetrics dataclass to the
    display-shape dict that alternatives + correlation + comparison
    consume. Pure projection — no computation, no formatting choices
    beyond the field rename for debt's risk_adj_return."""
    if asset_class == "equity":
        return {
            "excess_return_pct": fund.excess_return_pct,
            "consistency_pct": fund.consistency_pct,
            "max_drawdown_pct": fund.max_drawdown_pct,
            "volatility_pct": fund.volatility_pct,
            "downside_capture_ratio": fund.downside_capture_ratio,
        }
    return {
        "cagr_pct": fund.fund_cagr_pct,
        "volatility_pct": fund.volatility_pct,
        "max_drawdown_pct": fund.max_drawdown_pct,
        "consistency_pct": fund.consistency_pct,
        "risk_adj_return": fund.downside_capture_ratio,
    }


def _build_alternatives(
    ranked_funds: List["RankedFund"],
    exclude_code: int,
    asset_class: str,
    held_metrics: Optional[Dict[str, Any]] = None,
    max_n: int = 3,
    unsafe_codes: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Get top N alternatives from the same category, excluding the
    held fund.

    Skips funds with Low confidence, funds in `unsafe_codes` (severe
    data quality / extreme outlier), and funds whose advantage over
    the held fund is not material per `_alternative_is_material`.

    When held_metrics is omitted (Not-Ranked holdings with no
    metrics to compare against), the gate is skipped — we just show
    the top-of-category as a reference list.

    Returns (alts, filter_stats) where filter_stats counts WHY peers
    were dropped: {"low_conf", "unsafe", "immaterial", "considered"}.
    The caller uses these counts to emit an accurate action_note
    when `alts` is empty — distinguishing "data thin" from "no peer
    is materially better" (OBS-2).
    """
    if unsafe_codes is None:
        unsafe_codes = set()
    alts: List[Dict[str, Any]] = []
    stats: Dict[str, int] = {
        "low_conf": 0,
        "unsafe": 0,
        "immaterial": 0,
        "considered": 0,
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
        alt["metrics"] = alt_metrics_full
        if asset_class == "equity":
            alt["excess_return_pct"] = rf.fund.excess_return_pct
            alt["consistency_pct"] = rf.fund.consistency_pct
        else:
            alt["cagr_pct"] = rf.fund.fund_cagr_pct
            alt["volatility_pct"] = rf.fund.volatility_pct

        if held_metrics:
            alt["justification"] = _build_justification(
                held_metrics, alt_metrics_full, asset_class)
        else:
            alt["justification"] = []

        alts.append(alt)
        if len(alts) >= max_n:
            break
    return alts, stats


def _primary_metric_gap(
    held_metrics: Dict[str, Any],
    top_metrics: Dict[str, Any],
    asset_class: str,
) -> Optional[Dict[str, Any]]:
    """Pick the single largest improvement metric and return its
    delta + magnitude bucket. Used to quantify the gap between a
    held fund and the rank-1 fund in its category — converts the
    informational "top-ranked exists" surface into a quantified
    observation."""
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
