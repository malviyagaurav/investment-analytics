from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from .compiler import compile_insights
from .errors import PolicyError
from .lineage import LICENSE_USER, make_source
from .mutual_funds import analyze_mutual_fund


DEFAULT_TOP_N = 3
EVIDENCE_ORDER = {"Low": 0, "Medium": 1, "High": 2}
EVIDENCE_LABELS = ["Low", "Medium", "High"]
MAX_AGGREGATED_LIMITATIONS = 8
JURISDICTION_KEYS = {"user_country", "asset_market", "serving_entity", "subject_token"}
MF_ANALYZER_VERSION = "mf_v2"
PORTFOLIO_AGGREGATOR_VERSION = "portfolio_v1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _evidence_from_rows(row_count: int) -> str:
    if row_count >= 10:
        return "High"
    if row_count >= 3:
        return "Medium"
    return "Low"


def _hhi(weights: list[float]) -> float:
    return round(sum(w * w for w in weights), 6)


def _effective_holdings(hhi: float) -> float:
    if hhi <= 0:
        return 0.0
    return round(1.0 / hhi, 2)


def analyze_portfolio(payload: dict[str, Any]) -> list[dict[str, Any]]:
    holdings = payload.get("holdings") or []
    if not isinstance(holdings, list):
        holdings = []

    normalization_events: list[dict[str, Any]] = []
    raw_by_name: dict[str, tuple[str, float]] = {}
    duplicate_names: set[str] = set()

    for holding in holdings:
        if not isinstance(holding, dict):
            continue
        market_value = float(holding.get("market_value", 0) or 0)
        if market_value <= 0:
            continue
        asset_class = str(holding.get("asset_class", "Unknown")).strip() or "Unknown"
        name = str(holding.get("name", holding.get("symbol", "Unknown"))).strip() or "Unknown"
        if name in raw_by_name:
            duplicate_names.add(name)
            existing_class, existing_value = raw_by_name[name]
            raw_by_name[name] = (existing_class, existing_value + market_value)
        else:
            raw_by_name[name] = (asset_class, market_value)

    if duplicate_names:
        normalization_events.append({
            "type": "duplicate_holdings_merged",
            "names": sorted(duplicate_names),
        })

    values_by_asset: dict[str, float] = defaultdict(float)
    values_by_holding: dict[str, float] = {}
    total_value = 0.0
    valid_count = len(raw_by_name)

    for name, (asset_class, market_value) in raw_by_name.items():
        values_by_asset[asset_class] += market_value
        values_by_holding[name] = market_value
        total_value += market_value

    timestamp = _now_iso()
    source = make_source("user_submitted_portfolio", timestamp, LICENSE_USER)

    if normalization_events:
        data_completeness = "Medium"
    else:
        data_completeness = "High" if valid_count > 0 else "Low"

    if total_value <= 0:
        return compile_insights(
            [
                {
                    "type": "diagnostic",
                    "observation": "No positive market value was found in the submitted holdings.",
                    "why_it_matters": "Diagnostics require market values to calculate exposure and concentration.",
                    "supporting_data": {"holding_count": len(holdings), "source": source},
                    "evidence_strength": "Low",
                    "data_completeness": "Low",
                    "limitations": ["Only submitted holdings are evaluated."],
                    "unavailable_components": ["exposure_breakdown", "concentration_breakdown"],
                }
            ]
        )

    evidence = _evidence_from_rows(valid_count)
    holding_weights = {
        name: round(value / total_value, 6)
        for name, value in values_by_holding.items()
    }
    sorted_holdings = sorted(holding_weights.items(), key=lambda item: item[1], reverse=True)

    asset_weights = {
        asset: round(value / total_value, 6)
        for asset, value in values_by_asset.items()
    }
    sorted_assets = sorted(asset_weights.items(), key=lambda item: item[1], reverse=True)

    # --- Exposure by asset class ---
    largest_asset_name, largest_asset_weight = sorted_assets[0]
    largest_asset_pct = round(largest_asset_weight * 100, 2)
    exposure_by_class = [
        {"asset_class": asset, "weight": weight}
        for asset, weight in sorted_assets
    ]
    asset_breakdown_pct = {
        asset: round(weight * 100, 2)
        for asset, weight in sorted_assets
    }

    base_limitations = ["Classification is based on provided asset_class labels."]
    if normalization_events:
        base_limitations.insert(0, "Duplicate holding names were merged by summing market values.")

    insights: list[dict[str, Any]] = [
        {
            "type": "diagnostic",
            "observation": f"{largest_asset_name} exposure is {largest_asset_pct:.2f}% of submitted market value.",
            "why_it_matters": "A large exposure in one asset class can amplify portfolio volatility.",
            "supporting_data": {
                "total_market_value": round(total_value, 2),
                "asset_breakdown_pct": asset_breakdown_pct,
                "exposure_by_class": exposure_by_class,
                "distinct_classes": len(values_by_asset),
                "normalization_events": normalization_events,
                "source": source,
            },
            "evidence_strength": evidence,
            "data_completeness": data_completeness,
            "limitations": base_limitations + [
                "No look-through holdings; category overlap may exist.",
            ],
            "unavailable_components": [],
        },
    ]

    # --- Top-N concentration ---
    top_n = int(payload.get("top_n", DEFAULT_TOP_N) or DEFAULT_TOP_N)
    top_n = min(top_n, len(sorted_holdings))
    top_entries = sorted_holdings[:top_n]
    top_weight = round(sum(w for _, w in top_entries), 6)
    tail_weight = round(1.0 - top_weight, 6)

    insights.append(
        {
            "type": "diagnostic",
            "observation": f"Top {top_n} holdings represent {round(top_weight * 100, 2):.2f}% of submitted market value.",
            "why_it_matters": "Concentration in a small number of holdings can amplify the impact of individual position outcomes.",
            "supporting_data": {
                "top_n": top_n,
                "top_n_weight": round(top_weight, 6),
                "tail_weight": round(tail_weight, 6),
                "top_holdings": [
                    {"name": name, "weight": round(w, 6)} for name, w in top_entries
                ],
                "contributing_assets": [
                    {"name": name, "weight": round(w, 6)} for name, w in top_entries
                ],
                "source": source,
            },
            "evidence_strength": evidence,
            "data_completeness": data_completeness,
            "limitations": [
                "Top-N reflects concentration among largest positions only.",
                "Does not account for correlation between assets.",
            ],
            "unavailable_components": ["look_through_fund_holdings"],
        },
    )

    # --- HHI + effective holdings ---
    all_weights = [w for _, w in sorted_holdings]
    holdings_hhi = _hhi(all_weights)
    eff_holdings = _effective_holdings(holdings_hhi)

    insights.append(
        {
            "type": "diagnostic",
            "observation": f"Portfolio HHI is {holdings_hhi:.4f} with an effective holding count of {eff_holdings:.2f}.",
            "why_it_matters": "HHI measures weight concentration across all positions; effective count indicates diversification breadth.",
            "supporting_data": {
                "hhi": holdings_hhi,
                "effective_holdings": eff_holdings,
                "holdings_count": valid_count,
                "source": source,
            },
            "evidence_strength": evidence,
            "data_completeness": data_completeness,
            "limitations": [
                "HHI measures weight concentration only; ignores correlation and underlying holdings overlap.",
            ],
            "unavailable_components": [],
        },
    )

    # --- Class-level HHI (overlap proxy) ---
    class_weights = [w for _, w in sorted_assets]
    class_hhi = _hhi(class_weights)

    insights.append(
        {
            "type": "diagnostic",
            "observation": f"Asset class HHI is {class_hhi:.4f} across {len(values_by_asset)} distinct classes.",
            "why_it_matters": "Class-level concentration indicates how diversified the portfolio is across asset categories.",
            "supporting_data": {
                "class_hhi": class_hhi,
                "exposure_by_class": exposure_by_class,
                "distinct_classes": len(values_by_asset),
                "source": source,
            },
            "evidence_strength": evidence,
            "data_completeness": data_completeness,
            "limitations": [
                "Overlap proxy uses asset_class only; does not reflect underlying security overlap.",
                "Within-class diversification is not captured by this metric.",
            ],
            "unavailable_components": [],
        },
    )

    # --- Horizon benchmark comparison (existing) ---
    profile = payload.get("profile") or {}
    horizon_years = float(profile.get("horizon_years", 0) or 0)
    if horizon_years > 0:
        equity_pct = round(sum(
            weight * 100
            for asset, weight in asset_weights.items()
            if asset.lower() in {"equity", "stock", "stocks", "etf", "mutual fund"}
        ), 2)
        insights.append(
            {
                "type": "benchmark_comparison",
                "observation": f"Equity-linked exposure is {equity_pct:.2f}% for the submitted horizon of {horizon_years:.1f} years.",
                "benchmark": {
                    "name": "Selected comparison framework",
                    "methodology": "Compares submitted exposure with configurable horizon bands. No action is derived.",
                    "source": "internal_config:user_adjustable",
                },
                "supporting_data": {
                    "equity_linked_pct": round(equity_pct, 2),
                    "horizon_years": horizon_years,
                    "source": source,
                },
                "evidence_strength": "Low",
                "data_completeness": data_completeness,
                "limitations": ["Comparison framework is user-adjustable.", "No suitability decision is produced."],
                "unavailable_components": ["full_cashflow_profile", "tax_lot_detail"],
            }
        )

    return compile_insights(insights)


# ---------------------------------------------------------------------------
# Portfolio-with-funds aggregation (Option A — thin composition layer)
# ---------------------------------------------------------------------------


def _normalize_limitation(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"[^\w\s]", "", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _worst_evidence(levels: list[str]) -> str:
    if not levels:
        return "Low"
    return EVIDENCE_LABELS[min(EVIDENCE_ORDER.get(level, 0) for level in levels)]


def _fund_evidence(mf_result: dict[str, Any]) -> str:
    non_suppressed = mf_result.get("insights", [])
    if not non_suppressed:
        return "Low"
    levels = [
        item.get("payload", {}).get("evidence_strength", "Low")
        for item in non_suppressed
    ]
    return _worst_evidence(levels)


def _fund_completeness(mf_result: dict[str, Any]) -> str:
    non_suppressed = mf_result.get("insights", [])
    if not non_suppressed:
        return "Low"
    levels = [
        item.get("payload", {}).get("data_completeness", "Low")
        for item in non_suppressed
    ]
    return _worst_evidence(levels)


def _aggregate_limitations(
    fund_results: list[dict[str, Any]],
    fund_names: list[str],
    fund_weights: list[float],
) -> tuple[list[str], list[dict[str, Any]]]:
    seen_normalized: set[str] = set()
    aggregated: list[str] = []
    sources: list[dict[str, Any]] = []
    seen_sources: set[str] = set()

    for idx, result in enumerate(fund_results):
        name = fund_names[idx]
        weight = fund_weights[idx]
        if name not in seen_sources:
            seen_sources.add(name)
            sources.append({"name": name, "weight": round(weight, 6)})
        for insight in result.get("insights", []):
            for lim in insight.get("payload", {}).get("limitations", []):
                normalized = _normalize_limitation(lim)
                if normalized not in seen_normalized and len(aggregated) < MAX_AGGREGATED_LIMITATIONS:
                    seen_normalized.add(normalized)
                    aggregated.append(lim)

    return aggregated, sources


def analyze_portfolio_with_funds(payload: dict[str, Any]) -> dict[str, Any]:
    funds = payload.get("funds") or []
    if not isinstance(funds, list) or not funds:
        raise PolicyError(
            "data_validation_error",
            "At least one fund entry is required.",
            {"fund_count": 0},
        )

    # Validate and prepare
    fund_names: list[str] = []
    fund_values: list[float] = []
    mf_payloads: list[dict[str, Any]] = []

    for idx, fund in enumerate(funds):
        if not isinstance(fund, dict):
            raise PolicyError(
                "data_validation_error",
                f"Fund entry at index {idx} must be an object.",
                {"fund_index": idx},
            )
        name = str(fund.get("name", "")).strip()
        if not name:
            raise PolicyError(
                "data_validation_error",
                f"Fund entry at index {idx} has an empty name.",
                {"fund_index": idx},
            )
        market_value = float(fund.get("market_value", 0) or 0)
        if market_value <= 0:
            raise PolicyError(
                "data_validation_error",
                f"Fund entry at index {idx} has non-positive market value.",
                {"fund_index": idx, "fund_name": name},
            )
        mf_payload = fund.get("mf_payload")
        if not isinstance(mf_payload, dict):
            raise PolicyError(
                "data_validation_error",
                f"Fund entry at index {idx} has invalid mf_payload.",
                {"fund_index": idx, "fund_name": name},
            )
        # Jurisdiction safeguard: reject if mf_payload carries jurisdiction keys
        present_jurisdiction = JURISDICTION_KEYS.intersection(mf_payload.keys())
        if present_jurisdiction:
            raise PolicyError(
                "data_validation_error",
                f"mf_payload at index {idx} must not contain jurisdiction fields.",
                {"fund_index": idx, "fund_name": name, "rejected_keys": sorted(present_jurisdiction)},
            )
        fund_names.append(name)
        fund_values.append(market_value)
        mf_payloads.append(mf_payload)

    total_value = sum(fund_values)
    fund_weights = [v / total_value for v in fund_values]

    # Run MF analyzer per fund
    mf_results: list[dict[str, Any]] = []
    for idx, mf_payload in enumerate(mf_payloads):
        try:
            result = analyze_mutual_fund(mf_payload)
        except PolicyError as exc:
            raise PolicyError(
                "mf_analysis_failed",
                f"MF analysis failed for fund at index {idx}.",
                {
                    "fund_index": idx,
                    "fund_name": fund_names[idx],
                    "underlying_reason": exc.message,
                    "underlying_code": exc.code,
                    "underlying_details": exc.details,
                },
            ) from exc
        mf_results.append(result)

    # --- Aggregate ---
    timestamp = _now_iso()
    source = make_source("portfolio_with_funds_aggregation", timestamp, LICENSE_USER)

    # Per-fund evidence + completeness
    per_fund_evidence = [_fund_evidence(r) for r in mf_results]
    per_fund_completeness = [_fund_completeness(r) for r in mf_results]
    portfolio_evidence = _worst_evidence(per_fund_evidence)
    portfolio_completeness = _worst_evidence(per_fund_completeness)

    # Evidence distribution by weight bucket
    evidence_buckets: dict[str, float] = defaultdict(float)
    for weight, ev in zip(fund_weights, per_fund_evidence):
        evidence_buckets[ev] += weight
    evidence_distribution = [
        {"bucket": label, "weight": round(evidence_buckets.get(label, 0.0), 6)}
        for label in EVIDENCE_LABELS
        if evidence_buckets.get(label, 0.0) > 0
    ]

    contributing_evidence = [
        {"name": name, "weight": round(w, 6), "evidence": ev}
        for name, w, ev in zip(fund_names, fund_weights, per_fund_evidence)
    ]

    # Limitation aggregation
    aggregated_limitations, limitation_sources = _aggregate_limitations(
        mf_results, fund_names, fund_weights,
    )

    # Expense aggregation
    expense_ratios: list[float] = []
    for mf_payload in mf_payloads:
        expense_ratios.append(float(mf_payload.get("expense_ratio_pct", 0.0) or 0.0) / 100.0)

    weighted_ter = sum(w * e for w, e in zip(fund_weights, expense_ratios))
    min_ter = min(expense_ratios) if expense_ratios else 0.0
    max_ter = max(expense_ratios) if expense_ratios else 0.0

    all_horizons: set[float] = set()
    for mf_payload in mf_payloads:
        ei = mf_payload.get("expense_impact") or {}
        for h in ei.get("horizons_years", [3, 5, 10]):
            all_horizons.add(float(h))
    sorted_horizons = sorted(all_horizons)

    expense_impact_rows: list[dict[str, Any]] = []
    for years in sorted_horizons:
        low_drag = (1.0 - max_ter) ** years
        high_drag = (1.0 - min_ter) ** years
        expense_impact_rows.append({
            "years": years,
            "range": [round(low_drag, 6), round(high_drag, 6)],
        })

    contributing_expense = [
        {"name": name, "weight": round(w, 6), "expense_ratio": round(e, 6)}
        for name, w, e in zip(fund_names, fund_weights, expense_ratios)
    ]

    # Attribution snapshot
    sorted_by_weight = sorted(
        zip(fund_names, fund_weights), key=lambda item: item[1], reverse=True,
    )
    weights_snapshot = [
        {"name": name, "weight": round(w, 6)}
        for name, w in sorted_by_weight
    ]

    # MF version capture
    mf_versions = [
        {"name": name, "analyzer_version": MF_ANALYZER_VERSION}
        for name in fund_names
    ]

    # --- Build insights ---
    insights: list[dict[str, Any]] = []

    # 1) Evidence & quality floor
    insights.append({
        "type": "diagnostic",
        "observation": f"Portfolio evidence level is {portfolio_evidence} based on {len(funds)} contributing funds.",
        "why_it_matters": "Portfolio-level evidence reflects the weakest data quality among contributing funds.",
        "supporting_data": {
            "portfolio_evidence": portfolio_evidence,
            "data_completeness": portfolio_completeness,
            "evidence_distribution": evidence_distribution,
            "contributing_assets": contributing_evidence,
            "fund_count": len(funds),
            "mf_versions": mf_versions,
            "source": source,
        },
        "evidence_strength": portfolio_evidence,
        "data_completeness": portfolio_completeness,
        "limitations": [
            "Portfolio evidence reflects the lowest evidence level among contributing funds.",
            "Weight distribution by evidence indicates how much capital is associated with each evidence level.",
        ],
        "unavailable_components": [],
    })

    # 2) Limitation aggregation
    insights.append({
        "type": "diagnostic",
        "observation": f"{len(aggregated_limitations)} distinct limitations identified across {len(funds)} funds.",
        "why_it_matters": "Aggregated limitations surface data and methodology constraints from all contributing funds.",
        "supporting_data": {
            "aggregated_limitations": aggregated_limitations,
            "aggregated_limitations_count": len(aggregated_limitations),
            "sources": limitation_sources,
            "source": source,
        },
        "evidence_strength": portfolio_evidence,
        "data_completeness": portfolio_completeness,
        "limitations": [
            "Limitations are deduplicated by normalized form.",
            f"Capped at {MAX_AGGREGATED_LIMITATIONS} entries.",
        ],
        "unavailable_components": [],
    })

    # 3) Expense aggregation
    insights.append({
        "type": "cost_tax",
        "scenario_a": {
            "name": "Weighted TER 0.00%",
            "terminal_value_proxy": 1.0,
        },
        "scenario_b": {
            "name": f"Weighted TER {round(weighted_ter * 100, 2):.2f}%",
            "terminal_value_proxy_range": [
                round(expense_impact_rows[-1]["range"][0], 6) if expense_impact_rows else 1.0,
                round(expense_impact_rows[-1]["range"][1], 6) if expense_impact_rows else 1.0,
            ],
        },
        "assumptions": {
            "calculation_kind": "cost",
            "weighted_expense_ratio": round(weighted_ter, 6),
            "min_expense_ratio": round(min_ter, 6),
            "max_expense_ratio": round(max_ter, 6),
            "horizons": expense_impact_rows,
            "compounding": "annual constant TER drag factor",
            "units": "drag_factor",
            "tax_year": "not_applicable",
            "residency": "not_applicable",
            "rates": {},
            "holding_period": max(sorted_horizons) if sorted_horizons else 0,
            "source": source,
        },
        "estimated_impact": {
            "range": [
                round(expense_impact_rows[-1]["range"][0], 6) if expense_impact_rows else 1.0,
                round(expense_impact_rows[-1]["range"][1], 6) if expense_impact_rows else 1.0,
            ],
            "units": "drag_factor",
        },
        "evidence_strength": portfolio_evidence,
        "data_completeness": portfolio_completeness,
        "limitations": [
            "Expense impact is a hypothetical calculation based on stated expense ratios and compounding assumptions.",
            "Does not account for taxes, fees outside TER, or changes in expense over time.",
            "Consult a qualified professional",
        ],
        "unavailable_components": ["tax_lot_detail"],
    })

    # 4) Attribution snapshot
    insights.append({
        "type": "diagnostic",
        "observation": f"Portfolio comprises {len(funds)} funds with a combined value of {round(total_value, 2)}.",
        "why_it_matters": "Attribution shows the composition used for aggregation, enabling traceability.",
        "supporting_data": {
            "total_value": round(total_value, 2),
            "weights": weights_snapshot,
            "source": source,
        },
        "evidence_strength": portfolio_evidence,
        "data_completeness": portfolio_completeness,
        "limitations": [
            "Weights are based on provided market values.",
        ],
        "unavailable_components": [],
    })

    compiled = compile_insights(insights)

    return {
        "insights": compiled,
        "per_fund_results": [
            {
                "fund_name": name,
                "weight": round(w, 6),
                "evidence": ev,
                "completeness": comp,
                "insight_count": len(r.get("insights", [])),
                "suppressed_count": len(r.get("suppressed_insights", [])),
            }
            for name, w, ev, comp, r in zip(
                fund_names, fund_weights, per_fund_evidence, per_fund_completeness, mf_results,
            )
        ],
        "aggregation_metadata": {
            "schema_version": "v1",
            "aggregator_version": PORTFOLIO_AGGREGATOR_VERSION,
            "mf_analyzer_version": MF_ANALYZER_VERSION,
            "fund_count": len(funds),
            "portfolio_evidence": portfolio_evidence,
            "portfolio_completeness": portfolio_completeness,
        },
    }

