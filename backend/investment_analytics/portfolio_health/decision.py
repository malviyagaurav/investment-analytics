"""Decision engine — per-fund + portfolio-level status, action, summary.

Orchestrator + helpers for the Portfolio Health Check. Given a list of
held scheme codes (and optional weights), this module:
  - Resolves each holding against the AMFI registry
  - Runs (or reuses) the category peer ranking
  - Tags each holding Strong/Neutral/Weak/Not Ranked
  - Assigns Continue/Monitor/Review with a non-advisory note
  - Detects portfolio-level signals: mistakes, redundancy, exposure
    gaps, hidden-overlap correlations, plan inefficiency, action +
    structural priorities

Discipline:
- No advisory language. No BUY/SELL. Status reflects peer position,
  not absolute quality. Action reflects data signals, not advice.
- The orchestrator stays a leaf consumer: it calls helpers from
  sibling submodules (coverage, alternatives, correlation, structural,
  _util) and never re-imports __init__ in a way that would form a
  cycle.

Dependency direction (acyclic):
  decision.py imports from:
    - stdlib (datetime, pathlib, typing, logging)
    - data_discovery (SchemeEntry, load_registry — eager import for
      types; load_registry call goes through the lazy parent-package
      seam so tests' patch.object(ph, "load_registry", ...) reach us)
    - ranking (CategoryRanking, DebtCategoryRanking, RankedFund,
      DEBT_CATEGORIES, EXCLUDED_CATEGORIES, rank_category,
      rank_debt_category — eager; no test currently patches these
      via the ph.* namespace, only the _get_or_rank_* wrappers)
    - portfolio_health._util  (_short_category, _resolve_weights)
    - portfolio_health.coverage  (_build_coverage_report)
    - portfolio_health.alternatives  (_build_alternatives,
      _metrics_for_display, _build_your_fund_gaps,
      _alternative_is_material, _primary_metric_gap)
    - portfolio_health.correlation  (HIGH_CORRELATION_THRESHOLD,
      _compute_held_correlations)
    - portfolio_health.structural  (_detect_regular_plan_holdings,
      _build_structural_priority)
    - portfolio_health.models  (FundHealthResult,
      ConcentrationWarning, PortfolioHealthResult)

Lazy-import seams (refactor-stability):
  Inside check_portfolio_health, the following names are resolved
  via `from backend.investment_analytics import portfolio_health
  as _ph` at call time, NOT bound at module load:
    - _ph.load_registry           — tests patch this via patch.object(ph, ...)
    - _ph._get_or_rank_equity     — tests patch this via patch.object(ph, ...)
    - _ph._get_or_rank_debt       — tests patch this via patch.object(ph, ...)

  Reason: `from X import Y` creates a separate binding in the
  importing module. patch.object(ph, "Y", ...) rebinds Y on the
  parent package only — an eager import here would call the
  unpatched original. The lazy seam re-resolves the name from the
  parent package on each call.

  See feedback memory: feedback_refactor_lazy_import_seam.md.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.data_discovery.registry import SchemeEntry
from backend.investment_analytics.ranking import (
    CategoryRanking,
    DebtCategoryRanking,
    RankedFund,
    DEBT_CATEGORIES,
    EXCLUDED_CATEGORIES,
    rank_category,
    rank_debt_category,
)

from backend.investment_analytics.portfolio_health._util import (
    _resolve_weights,
    _short_category,
)
from backend.investment_analytics.portfolio_health.coverage import (
    CoverageReport,
    _build_coverage_report,
)
from backend.investment_analytics.portfolio_health.alternatives import (
    _alternative_is_material,
    _build_alternatives,
    _build_your_fund_gaps,
    _metrics_for_display,
    _primary_metric_gap,
)
from backend.investment_analytics.portfolio_health.correlation import (
    HIGH_CORRELATION_THRESHOLD,
    _compute_held_correlations,
)
from backend.investment_analytics.portfolio_health.structural import (
    _build_structural_priority,
    _detect_regular_plan_holdings,
)
from backend.investment_analytics.portfolio_health.models import (
    ConcentrationWarning,
    FundHealthResult,
    PortfolioHealthResult,
)


logger = logging.getLogger("investment_analytics.portfolio_health.decision")


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
            "message": f"Limited history ({history_years:.1f}y) — metrics less reliable",
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
                    "message": "Significant data gaps vs peers — interpret cautiously",
                })
            elif ratio < 0.6:
                flags.append({
                    "severity": "moderate",
                    "message": "Partial data gaps — some metrics may be less reliable",
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
                flags.append("Unusually high returns vs peers — may include short-term spikes")
            else:
                flags.append(f"Unusually high {key} vs peers — interpret cautiously")
        elif direction == "low" and z < -2.0:
            if key == "volatility_pct":
                flags.append("Unusually stable data vs peers — verify data quality")
            else:
                flags.append(f"Unusually low {key} vs peers — interpret cautiously")

    return flags


# ── Core logic ─────────────────────────────────────────────────

# Cache category rankings within a single health check call. Module-
# level state; check_portfolio_health resets it on entry/exit.
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

    # SEAM: resolve load_registry + _get_or_rank_equity + _get_or_rank_debt
    # via the parent package at call time. Tests patch these names on
    # `portfolio_health` (the package) via patch.object(ph, "name", ...);
    # an eager `from ... import` here would bind to the originals and
    # silently bypass those patches. Refactor-stability test seam.
    from backend.investment_analytics import portfolio_health as _ph

    registry = _ph.load_registry(Path(registry_path))
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

        # Run category ranking. SEAM: route through _ph so tests that
        # patch ph._get_or_rank_equity / ph._get_or_rank_debt reach us.
        ranking: Any = None
        if asset_class == "equity":
            ranking = _ph._get_or_rank_equity(category, registry_path)
        else:
            ranking = _ph._get_or_rank_debt(category, registry_path)
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
