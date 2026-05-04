from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

try:
    from fastapi import FastAPI, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc

from backend.investment_analytics.analyzers import analyze_portfolio, analyze_portfolio_with_funds
from backend.investment_analytics.audit import append_audit_record, hash_payload, verify_audit_chain
from backend.investment_analytics.errors import PolicyError
from backend.investment_analytics.etf import analyze_etf
from backend.investment_analytics.jurisdiction import JurisdictionContext, evaluate_jurisdiction
from backend.investment_analytics.mutual_funds import analyze_mutual_fund, load_mutual_fund_csv, SeriesPoint
from backend.investment_analytics.scenarios import list_standard_scenarios, resolve_scenario_definition
from backend.data_ingestion.ingest import ingest_mf_from_csv
from backend.data_ingestion.schema_map import KNOWN_MAPPINGS
from api.models import (
    AllAssetsRankRequest,
    CategoryRankRequest,
    CompareRequest,
    DiscoverFetchRequest,
    ETFAnalysisRequest,
    FromSourceRequest,
    JurisdictionRequest,
    MultiCategoryRankRequest,
    MutualFundAnalysisRequest,
    PortfolioAggregateRequest,
    PortfolioAnalyticsRequest,
    PortfolioEvaluateRequest,
    PortfolioHealthRequest,
    PortfolioWithFundsRequest,
    ScenarioRunRequest,
    SipRequest,
)


ROOT = Path(__file__).resolve().parent.parent
AUDIT_PATH = ROOT / "data" / "audit" / "audit.jsonl"
REGISTRY_PATH = ROOT / "data" / "registry" / "schemes.json"

# ── Discovery layer (lazy-loaded) ──────────────────────────────
from backend.data_discovery.registry import (
    SchemeEntry,
    load_registry,
    parse_navall_text,
    save_registry,
)
from backend.data_discovery.search import search_schemes
from backend.data_discovery.fetch import (
    fetch_navall_text,
    fetch_scheme_nav,
    ingest_from_mfapi,
)
from backend.investment_analytics.comparison import (
    align_multiple_series,
    alignment_quality,
    build_comparison_insights,
)
from backend.investment_analytics.ranking import rank_category, ranking_to_dict, rank_all_categories, multi_ranking_to_dict, rank_all_assets, all_assets_to_dict
from backend.investment_analytics.portfolio_health import check_portfolio_health, portfolio_health_to_dict
from backend.investment_analytics.sip import build_sip_insights
from backend.investment_analytics.portfolio import build_portfolio_insights
from backend.investment_analytics.evaluation import evaluate_portfolio

# Scheme registry — loaded from cached JSON if available
_scheme_registry: list[SchemeEntry] = []

MF_SOURCE_REGISTRY: dict[tuple[str, str], str] = {
    ("csv_sample", "messy_mf"): "data/sample/messy_mf_nav.csv",
    ("csv_sample", "clean_mf"): "data/sample/mutual_fund_nav.csv",
    ("csv_sample", "amfi_mf"): "data/sample/amfi_nav.csv",
    ("csv_sample", "alt_mf"): "data/sample/alt_mf_nav.csv",
    ("csv_sample", "etf_price"): "data/sample/etf_price.csv",
    # Stress-test datasets
    ("csv_sample", "stress_sparse"): "data/sample/stress_sparse.csv",
    ("csv_sample", "stress_gaps"): "data/sample/stress_gaps.csv",
    ("csv_sample", "stress_spike"): "data/sample/stress_spike.csv",
    ("csv_sample", "stress_flat"): "data/sample/stress_flat.csv",
    ("csv_sample", "stress_duplicates"): "data/sample/stress_duplicates.csv",
    ("csv_sample", "stress_missing"): "data/sample/stress_missing.csv",
    ("csv_sample", "stress_conflicting"): "data/sample/stress_conflicting.csv",
    ("csv_sample", "stress_misaligned"): "data/sample/stress_misaligned_fund.csv",
    ("csv_sample", "stress_negatives"): "data/sample/stress_negatives.csv",
    ("csv_sample", "stress_unsorted"): "data/sample/stress_unsorted.csv",
}

app = FastAPI(title="Investment Analytics Engine", version="0.1.0")
app.add_middleware(GZipMiddleware, minimum_size=1000)
# CORS: allow any localhost / 127.0.0.1 port in dev so the dynamic-port
# launcher (api.server) doesn't break a separate frontend dev server.
# The SPA mounted at "/" is same-origin and doesn't need CORS at all;
# this regex only matters for cross-origin dev tools.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("analytics")


@app.middleware("http")
async def request_id_middleware(request: Request, call_next) -> Response:
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    start = time.monotonic()
    response = await call_next(request)
    elapsed_ms = round((time.monotonic() - start) * 1000, 1)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_id=%s method=%s path=%s status=%s elapsed_ms=%s",
        request_id, request.method, request.url.path,
        response.status_code, elapsed_ms,
    )
    return response


def _request_id(request: Request) -> str:
    return getattr(getattr(request, "state", None), "request_id", "unknown")


@app.exception_handler(PolicyError)
async def policy_error_handler(request: Request, exc: PolicyError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "request_id": _request_id(request),
            "reason": exc.code.upper(),
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "request_id": _request_id(request),
            "reason": "REQUEST_VALIDATION_ERROR",
            "message": "Request payload failed strict input validation.",
            "details": {"errors": exc.errors()},
        },
    )


@app.get("/health")
def health() -> dict:
    audit_ok = verify_audit_chain(AUDIT_PATH)
    data_dir_ok = (ROOT / "data" / "sample").is_dir()
    if not data_dir_ok:
        status = "unhealthy"
    elif not audit_ok:
        status = "degraded"
    else:
        status = "ok"
    return {
        "status": status,
        "audit_chain_valid": audit_ok,
        "data_directory_accessible": data_dir_ok,
        "registered_sources": len(MF_SOURCE_REGISTRY),
    }


@app.post("/policy/jurisdiction")
def jurisdiction_policy(payload: JurisdictionRequest) -> dict:
    gate = evaluate_jurisdiction(
        JurisdictionContext(
            user_country=payload.user_country,
            asset_market=payload.asset_market,
            serving_entity=payload.serving_entity,
        )
    )
    append_audit_record(
        AUDIT_PATH,
        {
            "event_type": "jurisdiction_gate",
            "subject_token": payload.subject_token,
            "gate": gate,
        },
    )
    return gate


@app.get("/scenarios/standard")
def standard_scenarios() -> list[dict[str, Any]]:
    return list_standard_scenarios()


@app.post("/analytics/portfolio")
def portfolio_analytics(payload: PortfolioAnalyticsRequest) -> dict:
    payload_dict = payload.model_dump()
    gate = evaluate_jurisdiction(
        JurisdictionContext(
            user_country=payload.user_country,
            asset_market=payload.asset_market,
            serving_entity=payload.serving_entity,
        )
    )
    if not gate["features"]["analytics"]:
        return {
            "gate": gate,
            "insights": [],
            "message": "Analytics are disabled until supported jurisdiction context is present.",
        }

    insights = analyze_portfolio(payload_dict)
    append_audit_record(
        AUDIT_PATH,
        {
            "event_type": "portfolio_analytics",
            "subject_token": payload.subject_token,
            "regime_flags": gate["features"],
            "templates_rendered": [item["template"] for item in insights],
        },
    )
    return {"gate": gate, "insights": insights}


@app.post("/analytics/portfolio-with-funds")
def portfolio_with_funds_analytics(payload: PortfolioWithFundsRequest) -> dict:
    gate = evaluate_jurisdiction(
        JurisdictionContext(
            user_country=payload.user_country,
            asset_market=payload.asset_market,
            serving_entity=payload.serving_entity,
        )
    )
    if not gate["features"]["analytics"]:
        return {
            "gate": gate,
            "insights": [],
            "per_fund_results": [],
            "aggregation_metadata": {},
            "message": "Analytics are disabled until supported jurisdiction context is present.",
        }

    payload_dict = payload.model_dump()
    result = analyze_portfolio_with_funds(payload_dict)
    append_audit_record(
        AUDIT_PATH,
        {
            "event_type": "portfolio_with_funds",
            "subject_token": payload.subject_token,
            "regime_flags": gate["features"],
            "schema_version": result["aggregation_metadata"]["schema_version"],
            "aggregator_version": result["aggregation_metadata"]["aggregator_version"],
            "mf_analyzer_version": result["aggregation_metadata"]["mf_analyzer_version"],
            "fund_count": result["aggregation_metadata"]["fund_count"],
            "portfolio_evidence": result["aggregation_metadata"]["portfolio_evidence"],
            "templates_rendered": [item["template"] for item in result["insights"]],
        },
    )
    return {"gate": gate, **result}


@app.post("/analytics/mutual-fund")
def mutual_fund_analytics(payload: MutualFundAnalysisRequest) -> dict:
    payload_dict = payload.model_dump()
    gate = evaluate_jurisdiction(
        JurisdictionContext(
            user_country=payload.user_country,
            asset_market=payload.asset_market,
            serving_entity=payload.serving_entity,
        )
    )
    if not gate["features"]["analytics"]:
        return {
            "gate": gate,
            "insights": [],
            "suppressed_insights": [],
            "message": "Analytics are disabled until supported jurisdiction context is present.",
        }

    result = analyze_mutual_fund(payload_dict)
    append_audit_record(
        AUDIT_PATH,
        {
            "event_type": "mutual_fund_analytics",
            "subject_token": payload.subject_token,
            "input_hash": hash_payload(payload_dict),
            "regime_flags": gate["features"],
            "templates_rendered": [item["template"] for item in result["insights"]],
            "suppressed_count": len(result["suppressed_insights"]),
            "data_quality": result["data_quality"],
        },
    )
    return {"gate": gate, **result}


@app.post("/analytics/etf")
def etf_analytics(payload: ETFAnalysisRequest) -> dict:
    payload_dict = payload.model_dump()
    gate = evaluate_jurisdiction(
        JurisdictionContext(
            user_country=payload.user_country,
            asset_market=payload.asset_market,
            serving_entity=payload.serving_entity,
        )
    )
    if not gate["features"]["analytics"]:
        return {
            "gate": gate,
            "insights": [],
            "suppressed_insights": [],
            "message": "Analytics are disabled until supported jurisdiction context is present.",
        }

    result = analyze_etf(payload_dict)
    append_audit_record(
        AUDIT_PATH,
        {
            "event_type": "etf_analytics",
            "subject_token": payload.subject_token,
            "input_hash": hash_payload(payload_dict),
            "regime_flags": gate["features"],
            "templates_rendered": [item["template"] for item in result["insights"]],
            "suppressed_count": len(result["suppressed_insights"]),
            "data_quality": result["data_quality"],
            "schema_version": "v1",
            "analyzer_version": "etf_v1",
        },
    )
    return {"gate": gate, **result}


@app.post("/analytics/mutual-fund/from-source")
def mutual_fund_from_source(payload: FromSourceRequest) -> dict:
    key = (payload.source, payload.symbol)
    source_path = MF_SOURCE_REGISTRY.get(key)
    if source_path is None:
        raise PolicyError(
            "ingestion_source_error",
            f"Unknown source/symbol: {payload.source}/{payload.symbol}",
            {
                "source": payload.source,
                "symbol": payload.symbol,
                "available": [
                    f"{s}/{sym}" for s, sym in MF_SOURCE_REGISTRY
                ],
            },
        )

    gate = evaluate_jurisdiction(
        JurisdictionContext(
            user_country=payload.user_country,
            asset_market=payload.asset_market,
            serving_entity=payload.serving_entity,
        )
    )
    if not gate["features"]["analytics"]:
        return {
            "gate": gate,
            "insights": [],
            "suppressed_insights": [],
            "ingestion_report": {},
            "message": "Analytics are disabled until supported jurisdiction context is present.",
        }

    # Resolve schema mapping (if provided)
    mapping = None
    if payload.schema_mapping is not None:
        mapping = KNOWN_MAPPINGS.get(payload.schema_mapping)
        if mapping is None:
            raise PolicyError(
                "ingestion_schema_error",
                f"Unknown schema mapping: '{payload.schema_mapping}'. "
                f"Available: {list(KNOWN_MAPPINGS.keys())}",
                {
                    "requested": payload.schema_mapping,
                    "available": list(KNOWN_MAPPINGS.keys()),
                },
            )

    mf_payload, ingestion_report = ingest_mf_from_csv(
        ROOT,
        source_path,
        fund_name=payload.fund_name,
        benchmark_name=payload.benchmark_name,
        category=payload.category,
        expense_ratio_pct=payload.expense_ratio_pct,
        mapping=mapping,
        rolling_window_points=payload.rolling_window_points,
        rolling_step_points=payload.rolling_step_points,
        rolling_min_windows=payload.rolling_min_windows,
    )

    result = analyze_mutual_fund(mf_payload)

    append_audit_record(
        AUDIT_PATH,
        {
            "event_type": "mutual_fund_from_source",
            "subject_token": payload.subject_token,
            "source": payload.source,
            "symbol": payload.symbol,
            "input_hash": hash_payload(mf_payload),
            "regime_flags": gate["features"],
            "templates_rendered": [i["template"] for i in result["insights"]],
            "suppressed_count": len(result["suppressed_insights"]),
            "data_quality": result["data_quality"],
            "ingestion_summary": {
                "fund_points": ingestion_report["fund_series"]["output_points"],
                "benchmark_points": ingestion_report["benchmark_series"]["output_points"],
                "fund_rejected": ingestion_report["fund_series"]["rejected_count"],
                "benchmark_rejected": ingestion_report["benchmark_series"]["rejected_count"],
                "limitations_count": len(ingestion_report["ingestion_limitations"]),
            },
            "schema_version": "v1",
            "analyzer_version": "mf_v2",
        },
    )
    return {"gate": gate, **result, "ingestion_report": ingestion_report}


@app.get("/analytics/mutual-fund/demo")
def mutual_fund_demo() -> dict:
    payload = load_mutual_fund_csv(ROOT / "data" / "sample" / "mutual_fund_nav.csv")
    payload.update({"user_country": "IN", "asset_market": "IN", "serving_entity": "local_demo"})
    gate = evaluate_jurisdiction(
        JurisdictionContext(user_country="IN", asset_market="IN", serving_entity="local_demo")
    )
    result = analyze_mutual_fund(payload)
    append_audit_record(
        AUDIT_PATH,
        {
            "event_type": "mutual_fund_demo",
            "subject_token": "demo",
            "input_hash": hash_payload(payload),
            "regime_flags": gate["features"],
            "templates_rendered": [item["template"] for item in result["insights"]],
            "suppressed_count": len(result["suppressed_insights"]),
            "data_quality": result["data_quality"],
        },
    )
    return {"gate": gate, **result}


@app.post("/scenarios/run")
def run_scenario(payload: ScenarioRunRequest) -> dict:
    definition = resolve_scenario_definition(payload.scenario_definition.model_dump())
    impact_pct = float(definition.get("params", {}).get("market_return_pct", 0) or 0)
    portfolio_value = payload.portfolio_value
    impact = portfolio_value * impact_pct / 100.0
    from backend.investment_analytics.compiler import compile_insight

    insight = compile_insight(
        {
            "type": "scenario",
            "scenario_definition": definition,
            "assumptions": {
                "portfolio_value": portfolio_value,
                "calculation": "Linear exposure proxy supplied by user inputs.",
            },
            "projected_impact": {
                "range": [round(impact * 0.9, 2), round(impact * 1.1, 2)],
                "units": "base_currency",
            },
            "sensitivity": ["Impact range changes with portfolio value and scenario parameters."],
            "evidence_strength": "Low",
            "data_completeness": "Medium",
            "limitations": ["Scenario output is a calculation, not a prescribed action."],
            "unavailable_components": ["full_position_level_beta"],
        }
    )
    append_audit_record(
        AUDIT_PATH,
        {
            "event_type": "scenario_run",
            "subject_token": payload.subject_token,
            "scenario_kind": definition["kind"],
        },
    )
    return {"insight": insight}


# ── Discovery endpoints ────────────────────────────────────────

@app.post("/discover/refresh-registry")
def refresh_registry() -> dict:
    """Fetch NAVAll.txt from AMFI and rebuild the local scheme registry."""
    global _scheme_registry
    try:
        raw = fetch_navall_text()
    except Exception as exc:
        raise PolicyError(
            "registry_fetch_error",
            f"Failed to fetch AMFI scheme list: {exc}",
            {"url": "https://portal.amfiindia.com/spages/NAVAll.txt"},
        )
    _scheme_registry = parse_navall_text(raw)
    save_registry(_scheme_registry, REGISTRY_PATH)
    return {
        "status": "ok",
        "schemes_loaded": len(_scheme_registry),
    }


@app.get("/discover/search")
def discover_search(
    q: str = "",
    max_results: int = 20,
    category: Optional[str] = None,
    fund_house: Optional[str] = None,
) -> dict:
    """Search AMFI mutual fund schemes by name, code, or fund house."""
    global _scheme_registry
    # Lazy-load from cached file if registry is empty
    if not _scheme_registry:
        _scheme_registry = load_registry(REGISTRY_PATH)
    if not _scheme_registry:
        return {
            "status": "empty",
            "message": "Registry not loaded. Call POST /discover/refresh-registry first.",
            "results": [],
        }
    results = search_schemes(
        _scheme_registry,
        query=q,
        max_results=max_results,
        category_filter=category,
        fund_house_filter=fund_house,
    )
    return {
        "status": "ok",
        "query": q,
        "count": len(results),
        "results": results,
    }


@app.get("/discover/registry-status")
def registry_status() -> dict:
    """Check whether the scheme registry is loaded and how many entries it has."""
    global _scheme_registry
    if not _scheme_registry:
        _scheme_registry = load_registry(REGISTRY_PATH)
    return {
        "loaded": len(_scheme_registry) > 0,
        "count": len(_scheme_registry),
        "cache_path": str(REGISTRY_PATH),
        "cache_exists": REGISTRY_PATH.exists(),
    }


@app.post("/discover/fetch-and-analyze")
def discover_fetch_and_analyze(payload: DiscoverFetchRequest) -> dict:
    """Fetch real NAV data for a scheme from mfapi.in, run through analyzer."""
    gate = evaluate_jurisdiction(
        JurisdictionContext(
            user_country=payload.user_country,
            asset_market=payload.asset_market,
            serving_entity=payload.serving_entity,
        )
    )
    if not gate["features"]["analytics"]:
        return {
            "gate": gate,
            "insights": [],
            "suppressed_insights": [],
            "ingestion_report": {},
            "message": "Analytics are disabled until supported jurisdiction context is present.",
        }

    try:
        mf_payload, ingestion_report = ingest_from_mfapi(
            scheme_code=payload.scheme_code,
            fund_name=payload.fund_name,
            category=payload.category,
            expense_ratio_pct=payload.expense_ratio_pct,
            rolling_window_points=payload.rolling_window_points,
            rolling_step_points=payload.rolling_step_points,
            rolling_min_windows=payload.rolling_min_windows,
        )
    except Exception as exc:
        raise PolicyError(
            "mfapi_fetch_error",
            f"Failed to fetch NAV data for scheme {payload.scheme_code}: {exc}",
            {"scheme_code": payload.scheme_code},
        )

    result = analyze_mutual_fund(mf_payload)

    append_audit_record(
        AUDIT_PATH,
        {
            "event_type": "discover_fetch_and_analyze",
            "subject_token": payload.subject_token,
            "scheme_code": payload.scheme_code,
            "input_hash": hash_payload(mf_payload),
            "regime_flags": gate["features"],
            "templates_rendered": [i["template"] for i in result["insights"]],
            "suppressed_count": len(result["suppressed_insights"]),
            "data_quality": result["data_quality"],
            "ingestion_summary": {
                "fund_points": ingestion_report["fund_series"]["output_points"],
                "benchmark_points": ingestion_report["benchmark_series"]["output_points"],
                "fund_rejected": ingestion_report["fund_series"]["rejected_count"],
                "benchmark_rejected": ingestion_report["benchmark_series"]["rejected_count"],
                "limitations_count": len(ingestion_report["ingestion_limitations"]),
            },
            "schema_version": "v1",
            "analyzer_version": "mf_v2",
        },
    )
    return {"gate": gate, **result, "ingestion_report": ingestion_report}


@app.post("/analytics/compare")
def compare_funds(payload: CompareRequest) -> dict:
    """Compare 2-5 mutual funds side-by-side on common dates."""
    gate = evaluate_jurisdiction(
        JurisdictionContext(
            user_country=payload.user_country,
            asset_market=payload.asset_market,
            serving_entity=payload.serving_entity,
        )
    )
    if not gate["features"]["analytics"]:
        return {
            "gate": gate,
            "funds": [],
            "comparison_insights": [],
            "message": "Analytics are disabled until supported jurisdiction context is present.",
        }

    # Fetch NAV for each fund
    fund_series_map: dict[int, list[SeriesPoint]] = {}
    fund_names: dict[int, str] = {}
    fund_expense_ratios: dict[int, float] = {}
    per_fund_results: list[dict] = []
    ingestion_reports: dict[int, dict] = {}

    for entry in payload.funds:
        code = entry.scheme_code
        try:
            mf_payload, ing_report = ingest_from_mfapi(
                scheme_code=code,
                fund_name=entry.fund_name or f"Scheme {code}",
                category=entry.category,
                expense_ratio_pct=entry.expense_ratio_pct,
                rolling_window_points=payload.rolling_window_points,
                rolling_step_points=payload.rolling_step_points,
                rolling_min_windows=1,
            )
        except Exception as exc:
            raise PolicyError(
                "mfapi_fetch_error",
                f"Failed to fetch NAV for scheme {code}: {exc}",
                {"scheme_code": code},
            )

        # Run per-fund MF analysis
        result = analyze_mutual_fund(mf_payload)

        fund_name = mf_payload.get("fund_name", str(code))
        fund_names[code] = fund_name
        fund_expense_ratios[code] = entry.expense_ratio_pct

        # Build SeriesPoint list from the fund NAV for alignment
        from datetime import date as date_type
        points = []
        for pt in mf_payload["fund"]:
            try:
                d = date_type.fromisoformat(pt["date"])
                points.append(SeriesPoint(as_of=d, value=pt["nav"]))
            except (ValueError, KeyError):
                continue
        fund_series_map[code] = points

        per_fund_results.append({
            "scheme_code": code,
            "name": fund_name,
            "category": entry.category,
            "insights": result["insights"],
            "data_quality": result["data_quality"],
        })
        ingestion_reports[code] = ing_report

    # Align across all funds
    common_dates, aligned_values = align_multiple_series(fund_series_map)

    if len(common_dates) < 2:
        raise PolicyError(
            "comparison_alignment_error",
            "Selected funds have fewer than 2 common observation dates.",
            {
                "scheme_codes": [e.scheme_code for e in payload.funds],
                "per_fund_points": {code: len(pts) for code, pts in fund_series_map.items()},
            },
        )

    quality = alignment_quality(common_dates, fund_series_map)

    # Build comparison insights (evidence = worst-case across funds + alignment)
    per_fund_ev = [
        r["data_quality"].get("evidence_strength", "Low")
        for r in per_fund_results
    ]
    from backend.investment_analytics.compiler import compile_insight
    raw_insights = build_comparison_insights(
        dates=common_dates,
        aligned_values=aligned_values,
        fund_names=fund_names,
        fund_expense_ratios=fund_expense_ratios,
        quality=quality,
        window_points=payload.rolling_window_points,
        step_points=payload.rolling_step_points,
        per_fund_evidence=per_fund_ev,
    )

    compiled_comparison: list[dict] = []
    suppressed_comparison: list[dict] = []
    for raw in raw_insights:
        try:
            compiled_comparison.append(compile_insight(raw))
        except PolicyError as exc:
            suppressed_comparison.append({
                "type": raw.get("type"),
                "reason": exc.code,
                "details": exc.details,
            })

    append_audit_record(
        AUDIT_PATH,
        {
            "event_type": "compare_funds",
            "subject_token": payload.subject_token,
            "scheme_codes": [e.scheme_code for e in payload.funds],
            "aligned_points": quality["aligned_points"],
            "comparison_templates": [c["template"] for c in compiled_comparison],
            "suppressed_count": len(suppressed_comparison),
        },
    )

    return {
        "gate": gate,
        "funds": per_fund_results,
        "comparison_insights": compiled_comparison,
        "suppressed_comparison_insights": suppressed_comparison,
        "alignment_quality": quality,
        "ingestion_reports": {
            str(code): report for code, report in ingestion_reports.items()
        },
    }


@app.post("/analytics/sip")
def simulate_sip(payload: SipRequest) -> dict:
    """Simulate rolling historical SIP outcomes across 1-5 funds."""
    # Validate weights sum to ~1.0
    total_weight = sum(f.weight for f in payload.funds)
    if abs(total_weight - 1.0) > 0.01:
        raise PolicyError(
            "weight_sum_error",
            f"Fund weights must sum to 1.0, got {total_weight:.4f}",
            {"total_weight": round(total_weight, 4)},
        )

    gate = evaluate_jurisdiction(
        JurisdictionContext(
            user_country=payload.user_country,
            asset_market=payload.asset_market,
            serving_entity=payload.serving_entity,
        )
    )
    if not gate["features"]["analytics"]:
        return {
            "gate": gate,
            "funds": [],
            "sip_insights": [],
            "message": "Analytics are disabled until supported jurisdiction context is present.",
        }

    # Fetch NAV for each fund
    fund_series_map: dict[int, list[SeriesPoint]] = {}
    fund_names: dict[int, str] = {}
    fund_expense_ratios: dict[int, float] = {}
    weights: dict[int, float] = {}
    per_fund_results: list[dict] = []

    for entry in payload.funds:
        code = entry.scheme_code
        try:
            mf_payload, ing_report = ingest_from_mfapi(
                scheme_code=code,
                fund_name=entry.fund_name or f"Scheme {code}",
                category=entry.category,
                expense_ratio_pct=entry.expense_ratio_pct,
                rolling_window_points=4,
                rolling_step_points=1,
                rolling_min_windows=1,
            )
        except Exception as exc:
            raise PolicyError(
                "mfapi_fetch_error",
                f"Failed to fetch NAV for scheme {code}: {exc}",
                {"scheme_code": code},
            )

        result = analyze_mutual_fund(mf_payload)

        fund_name = mf_payload.get("fund_name", str(code))
        fund_names[code] = fund_name
        fund_expense_ratios[code] = entry.expense_ratio_pct
        weights[code] = entry.weight

        from datetime import date as date_type
        points = []
        for pt in mf_payload["fund"]:
            try:
                d = date_type.fromisoformat(pt["date"])
                points.append(SeriesPoint(as_of=d, value=pt["nav"]))
            except (ValueError, KeyError):
                continue
        fund_series_map[code] = points

        per_fund_results.append({
            "scheme_code": code,
            "name": fund_name,
            "category": entry.category,
            "insights": result["insights"],
            "data_quality": result["data_quality"],
        })

    # Align across all funds
    common_dates, aligned_values = align_multiple_series(fund_series_map)

    if len(common_dates) < 2:
        raise PolicyError(
            "sip_alignment_error",
            "Selected funds have fewer than 2 common observation dates.",
            {
                "scheme_codes": [e.scheme_code for e in payload.funds],
                "per_fund_points": {code: len(pts) for code, pts in fund_series_map.items()},
            },
        )

    quality = alignment_quality(common_dates, fund_series_map)

    # Build SIP insights (evidence = worst-case across funds + alignment)
    per_fund_ev = [
        r["data_quality"].get("evidence_strength", "Low")
        for r in per_fund_results
    ]
    from backend.investment_analytics.compiler import compile_insight
    raw_insights, sim_meta = build_sip_insights(
        dates=common_dates,
        aligned_values=aligned_values,
        fund_names=fund_names,
        fund_expense_ratios=fund_expense_ratios,
        weights=weights,
        monthly_amount=payload.monthly_amount,
        quality=quality,
        window_months=payload.rolling_window_months,
        step_months=payload.step_months,
        per_fund_evidence=per_fund_ev,
    )

    compiled_sip: list[dict] = []
    suppressed_sip: list[dict] = []
    for raw in raw_insights:
        try:
            compiled_sip.append(compile_insight(raw))
        except PolicyError as exc:
            suppressed_sip.append({
                "type": raw.get("type"),
                "reason": exc.code,
                "details": exc.details,
            })

    append_audit_record(
        AUDIT_PATH,
        {
            "event_type": "simulate_sip",
            "subject_token": payload.subject_token,
            "scheme_codes": [e.scheme_code for e in payload.funds],
            "monthly_amount": payload.monthly_amount,
            "window_months": payload.rolling_window_months,
            "sip_templates": [c["template"] for c in compiled_sip],
            "suppressed_count": len(suppressed_sip),
        },
    )

    return {
        "gate": gate,
        "funds": per_fund_results,
        "sip_insights": compiled_sip,
        "suppressed_sip_insights": suppressed_sip,
        "simulation_meta": sim_meta,
        "alignment_quality": quality,
    }


@app.post("/analytics/portfolio-aggregate")
def aggregate_portfolio(payload: PortfolioAggregateRequest) -> dict:
    """Aggregate 2-5 funds into a combined portfolio and analyze behavior."""
    # Validate weights sum to ~1.0
    total_weight = sum(f.weight for f in payload.funds)
    if abs(total_weight - 1.0) > 0.01:
        raise PolicyError(
            "weight_sum_error",
            f"Fund weights must sum to 1.0, got {total_weight:.4f}",
            {"total_weight": round(total_weight, 4)},
        )

    gate = evaluate_jurisdiction(
        JurisdictionContext(
            user_country=payload.user_country,
            asset_market=payload.asset_market,
            serving_entity=payload.serving_entity,
        )
    )
    if not gate["features"]["analytics"]:
        return {
            "gate": gate,
            "funds": [],
            "portfolio_insights": [],
            "message": "Analytics are disabled until supported jurisdiction context is present.",
        }

    # Fetch NAV for each fund
    fund_series_map: dict[int, list[SeriesPoint]] = {}
    fund_names: dict[int, str] = {}
    weights: dict[int, float] = {}
    per_fund_results: list[dict] = []

    for entry in payload.funds:
        code = entry.scheme_code
        try:
            mf_payload, ing_report = ingest_from_mfapi(
                scheme_code=code,
                fund_name=entry.fund_name or f"Scheme {code}",
                category=entry.category,
                expense_ratio_pct=entry.expense_ratio_pct,
                rolling_window_points=4,
                rolling_step_points=1,
                rolling_min_windows=1,
            )
        except Exception as exc:
            raise PolicyError(
                "mfapi_fetch_error",
                f"Failed to fetch NAV for scheme {code}: {exc}",
                {"scheme_code": code},
            )

        result = analyze_mutual_fund(mf_payload)

        fund_name = mf_payload.get("fund_name", str(code))
        fund_names[code] = fund_name
        weights[code] = entry.weight

        from datetime import date as date_type
        points = []
        for pt in mf_payload["fund"]:
            try:
                d = date_type.fromisoformat(pt["date"])
                points.append(SeriesPoint(as_of=d, value=pt["nav"]))
            except (ValueError, KeyError):
                continue
        fund_series_map[code] = points

        per_fund_results.append({
            "scheme_code": code,
            "name": fund_name,
            "category": entry.category,
            "insights": result["insights"],
            "data_quality": result["data_quality"],
        })

    # Align across all funds
    common_dates, aligned_values = align_multiple_series(fund_series_map)

    if len(common_dates) < 2:
        raise PolicyError(
            "portfolio_alignment_error",
            "Selected funds have fewer than 2 common observation dates.",
            {
                "scheme_codes": [e.scheme_code for e in payload.funds],
                "per_fund_points": {code: len(pts) for code, pts in fund_series_map.items()},
            },
        )

    quality = alignment_quality(common_dates, fund_series_map)

    # Build portfolio insights (evidence = worst-case across funds + alignment)
    per_fund_ev = [
        r["data_quality"].get("evidence_strength", "Low")
        for r in per_fund_results
    ]
    from backend.investment_analytics.compiler import compile_insight
    raw_insights = build_portfolio_insights(
        dates=common_dates,
        aligned_values=aligned_values,
        fund_names=fund_names,
        weights=weights,
        quality=quality,
        rolling_window_points=payload.rolling_window_points,
        rolling_step_points=payload.rolling_step_points,
        per_fund_evidence=per_fund_ev,
    )

    compiled_portfolio: list[dict] = []
    suppressed_portfolio: list[dict] = []
    for raw in raw_insights:
        try:
            compiled_portfolio.append(compile_insight(raw))
        except PolicyError as exc:
            suppressed_portfolio.append({
                "type": raw.get("type"),
                "reason": exc.code,
                "details": exc.details,
            })

    append_audit_record(
        AUDIT_PATH,
        {
            "event_type": "aggregate_portfolio",
            "subject_token": payload.subject_token,
            "scheme_codes": [e.scheme_code for e in payload.funds],
            "aligned_points": quality["aligned_points"],
            "portfolio_templates": [c["template"] for c in compiled_portfolio],
            "suppressed_count": len(suppressed_portfolio),
        },
    )

    return {
        "gate": gate,
        "funds": per_fund_results,
        "portfolio_insights": compiled_portfolio,
        "suppressed_portfolio_insights": suppressed_portfolio,
        "alignment_quality": quality,
    }


@app.post("/analytics/portfolio-evaluate")
def evaluate_portfolio_endpoint(payload: PortfolioEvaluateRequest) -> dict:
    """Aggregate portfolio and evaluate against user-defined constraints."""
    # Validate weights sum to ~1.0
    total_weight = sum(f.weight for f in payload.funds)
    if abs(total_weight - 1.0) > 0.01:
        raise PolicyError(
            "weight_sum_error",
            f"Fund weights must sum to 1.0, got {total_weight:.4f}",
            {"total_weight": round(total_weight, 4)},
        )

    gate = evaluate_jurisdiction(
        JurisdictionContext(
            user_country=payload.user_country,
            asset_market=payload.asset_market,
            serving_entity=payload.serving_entity,
        )
    )
    if not gate["features"]["analytics"]:
        return {
            "gate": gate,
            "evaluation": None,
            "message": "Analytics are disabled until supported jurisdiction context is present.",
        }

    # Fetch + analyze per fund (same as aggregate)
    fund_series_map: dict[int, list[SeriesPoint]] = {}
    fund_names: dict[int, str] = {}
    weights: dict[int, float] = {}
    per_fund_results: list[dict] = []

    for entry in payload.funds:
        code = entry.scheme_code
        try:
            mf_payload, ing_report = ingest_from_mfapi(
                scheme_code=code,
                fund_name=entry.fund_name or f"Scheme {code}",
                category=entry.category,
                expense_ratio_pct=entry.expense_ratio_pct,
                rolling_window_points=4,
                rolling_step_points=1,
                rolling_min_windows=1,
            )
        except Exception as exc:
            raise PolicyError(
                "mfapi_fetch_error",
                f"Failed to fetch NAV for scheme {code}: {exc}",
                {"scheme_code": code},
            )

        result = analyze_mutual_fund(mf_payload)
        fund_name = mf_payload.get("fund_name", str(code))
        fund_names[code] = fund_name
        weights[code] = entry.weight

        from datetime import date as date_type
        points = []
        for pt in mf_payload["fund"]:
            try:
                d = date_type.fromisoformat(pt["date"])
                points.append(SeriesPoint(as_of=d, value=pt["nav"]))
            except (ValueError, KeyError):
                continue
        fund_series_map[code] = points

        per_fund_results.append({
            "scheme_code": code,
            "name": fund_name,
            "data_quality": result["data_quality"],
        })

    common_dates, aligned_values = align_multiple_series(fund_series_map)
    if len(common_dates) < 2:
        raise PolicyError(
            "portfolio_alignment_error",
            "Selected funds have fewer than 2 common observation dates.",
            {"scheme_codes": [e.scheme_code for e in payload.funds]},
        )

    quality = alignment_quality(common_dates, fund_series_map)
    per_fund_ev = [
        r["data_quality"].get("evidence_strength", "Low")
        for r in per_fund_results
    ]

    raw_insights = build_portfolio_insights(
        dates=common_dates,
        aligned_values=aligned_values,
        fund_names=fund_names,
        weights=weights,
        quality=quality,
        rolling_window_points=payload.rolling_window_points,
        rolling_step_points=payload.rolling_step_points,
        per_fund_evidence=per_fund_ev,
    )

    # Build user constraints dict (only non-None fields)
    user_constraints = {
        k: v for k, v in payload.constraints.dict().items()
        if v is not None
    }

    evaluation = evaluate_portfolio(raw_insights, user_constraints or None)

    append_audit_record(
        AUDIT_PATH,
        {
            "event_type": "evaluate_portfolio",
            "subject_token": payload.subject_token,
            "scheme_codes": [e.scheme_code for e in payload.funds],
            "constraints": user_constraints,
            "verdict": evaluation["summary"]["verdict"],
            "failed_checks": [
                c["name"] for c in evaluation["checks"] if c["status"] == "FAIL"
            ],
            "flag_count": evaluation["summary"]["flag_count"],
        },
    )

    return {
        "gate": gate,
        "evaluation": evaluation,
        "alignment_quality": quality,
    }


# ── Category Ranking ────────────────────────────────────────────

@app.post("/analytics/rank-category")
def analytics_rank_category(payload: CategoryRankRequest) -> dict:
    """Rank all Direct Growth funds in a category using pairwise dominance."""
    gate = evaluate_jurisdiction(
        JurisdictionContext(
            user_country=payload.user_country,
            asset_market=payload.asset_market,
            serving_entity=payload.serving_entity,
        )
    )
    if not gate["features"]["analytics"]:
        return {
            "gate": gate,
            "ranking": None,
            "message": "Analytics are disabled until supported jurisdiction context is present.",
        }

    try:
        result = rank_category(
            category=payload.category,
            registry_path=str(REGISTRY_PATH),
        )
    except ValueError as exc:
        raise PolicyError(
            "ranking_error",
            str(exc),
            {"category": payload.category},
        )

    append_audit_record(
        AUDIT_PATH,
        {
            "event_type": "rank_category",
            "subject_token": payload.subject_token,
            "category": payload.category,
            "ranked_count": len(result.ranked),
            "excluded_count": len(result.excluded),
            "benchmark_code": result.benchmark_code,
            "benchmark_fallback": result.benchmark_fallback,
            "schema_version": "v1",
        },
    )

    return {"gate": gate, "ranking": ranking_to_dict(result)}


# ── Multi-Category Ranking ──────────────────────────────────────

@app.post("/analytics/rank-all-categories")
def analytics_rank_all_categories(payload: MultiCategoryRankRequest) -> dict:
    """Rank top funds across multiple categories independently."""
    gate = evaluate_jurisdiction(
        JurisdictionContext(
            user_country=payload.user_country,
            asset_market=payload.asset_market,
            serving_entity=payload.serving_entity,
        )
    )
    if not gate["features"]["analytics"]:
        return {
            "gate": gate,
            "multi_ranking": None,
            "message": "Analytics are disabled until supported jurisdiction context is present.",
        }

    try:
        result = rank_all_categories(
            registry_path=str(REGISTRY_PATH),
            top_n=payload.top_n,
            categories=payload.categories if payload.categories else None,
        )
    except ValueError as exc:
        raise PolicyError(
            "multi_ranking_error",
            str(exc),
            {},
        )

    append_audit_record(
        AUDIT_PATH,
        {
            "event_type": "rank_all_categories",
            "subject_token": payload.subject_token,
            "top_n": payload.top_n,
            "categories_ranked": len(result.categories),
            "categories_failed": len(result.errors),
            "schema_version": "v1",
        },
    )

    return {"gate": gate, "multi_ranking": multi_ranking_to_dict(result)}


# ── Full Investment View (All Assets) ───────────────────────────

@app.post("/analytics/rank-all-assets")
def analytics_rank_all_assets(payload: AllAssetsRankRequest) -> dict:
    """Rank top funds across all asset classes independently."""
    gate = evaluate_jurisdiction(
        JurisdictionContext(
            user_country=payload.user_country,
            asset_market=payload.asset_market,
            serving_entity=payload.serving_entity,
        )
    )
    if not gate["features"]["analytics"]:
        return {
            "gate": gate,
            "all_assets": None,
            "message": "Analytics are disabled until supported jurisdiction context is present.",
        }

    try:
        result = rank_all_assets(
            registry_path=str(REGISTRY_PATH),
            top_n=payload.top_n,
        )
    except ValueError as exc:
        raise PolicyError("all_assets_error", str(exc), {})

    append_audit_record(
        AUDIT_PATH,
        {
            "event_type": "rank_all_assets",
            "subject_token": payload.subject_token,
            "top_n": payload.top_n,
            "equity_ranked": len(result.equity),
            "debt_ranked": len(result.debt),
            "gold_ranked": result.gold is not None,
            "schema_version": "v1",
        },
    )

    return {"gate": gate, "all_assets": all_assets_to_dict(result)}


@app.post("/analytics/portfolio-health")
def analytics_portfolio_health(payload: PortfolioHealthRequest) -> dict:
    """Check health of user's existing portfolio holdings."""
    gate = evaluate_jurisdiction(
        JurisdictionContext(
            user_country=payload.user_country,
            asset_market=payload.asset_market,
            serving_entity=payload.serving_entity,
        )
    )
    if not gate["features"]["analytics"]:
        return {
            "gate": gate,
            "health": None,
            "message": "Analytics are disabled until supported jurisdiction context is present.",
        }

    try:
        result = check_portfolio_health(
            scheme_codes=payload.scheme_codes,
            weights=payload.weights,
            registry_path=str(REGISTRY_PATH),
        )
    except ValueError as exc:
        raise PolicyError("portfolio_health_error", str(exc), {})

    append_audit_record(
        AUDIT_PATH,
        {
            "event_type": "portfolio_health_check",
            "subject_token": payload.subject_token,
            "holdings_count": len(payload.scheme_codes),
            "schema_version": "v1",
        },
    )

    return {"gate": gate, "health": portfolio_health_to_dict(result, payload.weights)}


frontend_dir = ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

if __name__ == "__main__":
    from api.server import run
    run()
