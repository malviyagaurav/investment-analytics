"""Fetch historical NAV data from mfapi.in and convert to pipeline-ready format.

Uses https://api.mfapi.in/mf/{scheme_code} which returns JSON with full
historical NAV data.  Converts DD-MM-YYYY dates to ISO and builds
the same normalized structures that the CSV ingestion pipeline produces.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, date
from typing import Any, List, Optional, Tuple

import httpx

from backend.data_ingestion.normalize import (
    NormalizedPoint,
    SeriesNormalization,
    normalize_series,
)
from backend.data_ingestion.validate import build_ingestion_limitations
from backend.data_ingestion.impact_links import build_impact_links
from backend.investment_analytics.lineage import LICENSE_REDIS, make_source
from backend.data_discovery.cache import get_cached_nav, put_cached_nav

logger = logging.getLogger("data_discovery.fetch")

MFAPI_BASE = "https://api.mfapi.in/mf"
NAVALL_URL = "https://portal.amfiindia.com/spages/NAVAll.txt"

# Request timeout (seconds)
REQUEST_TIMEOUT = 30.0

# Max points to ingest (safety guard)
MAX_NAV_POINTS = 10_000

# ── Benchmark index fund mapping ──
# Maps AMFI scheme categories to index fund scheme codes (Direct Growth plans).
# Each entry: (scheme_code, benchmark_display_name)
# These are passively managed index funds whose NAV closely tracks the index.

CATEGORY_BENCHMARK_MAP: dict[str, tuple[int, str]] = {
    # Large Cap → Nifty 50
    "Equity Scheme - Large Cap Fund": (120716, "UTI Nifty 50 Index Fund"),
    "Equity Scheme - Dividend Yield Fund": (120716, "UTI Nifty 50 Index Fund"),
    # Large & Mid Cap → Nifty LargeMidcap 250
    "Equity Scheme - Large & Mid Cap Fund": (152482, "ICICI Pru Nifty LargeMidcap 250 Index Fund"),
    # Mid Cap → Nifty Midcap 150
    "Equity Scheme - Mid Cap Fund": (151724, "HDFC Nifty Midcap 150 Index Fund"),
    # Small Cap → Nifty Smallcap 250
    "Equity Scheme - Small Cap Fund": (149283, "ICICI Pru Nifty Smallcap 250 Index Fund"),
    # Multi / Flexi / ELSS / Value / Focused / Sectoral → Nifty 500
    "Equity Scheme - Multi Cap Fund": (153161, "ICICI Pru Nifty 500 Index Fund"),
    "Equity Scheme - Flexi Cap Fund": (153161, "ICICI Pru Nifty 500 Index Fund"),
    "Equity Scheme - ELSS": (153161, "ICICI Pru Nifty 500 Index Fund"),
    "Equity Scheme - Value Fund": (153161, "ICICI Pru Nifty 500 Index Fund"),
    "Equity Scheme - Focused Fund": (153161, "ICICI Pru Nifty 500 Index Fund"),
    "Equity Scheme - Sectoral/ Thematic": (153161, "ICICI Pru Nifty 500 Index Fund"),
    "Equity Scheme - Contra": (153161, "ICICI Pru Nifty 500 Index Fund"),
}

# Default benchmark when category is unknown or not in the map
DEFAULT_BENCHMARK: tuple[int, str] = (120716, "UTI Nifty 50 Index Fund")


def _parse_dd_mm_yyyy(date_str: str) -> Optional[date]:
    """Parse DD-MM-YYYY to date object."""
    try:
        return datetime.strptime(date_str.strip(), "%d-%m-%Y").date()
    except (ValueError, AttributeError):
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def fetch_scheme_nav(scheme_code: int) -> dict:
    """Fetch historical NAV from mfapi.in (with cache).

    Returns the raw JSON response as a dict.
    Raises httpx.HTTPStatusError on non-2xx.
    """
    cached = get_cached_nav(scheme_code)
    if cached is not None:
        return cached

    url = f"{MFAPI_BASE}/{scheme_code}"
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        resp = client.get(url)
        resp.raise_for_status()
    data = resp.json()
    put_cached_nav(scheme_code, data)
    return data


def fetch_navall_text() -> str:
    """Fetch NAVAll.txt from AMFI portal."""
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        resp = client.get(NAVALL_URL)
        resp.raise_for_status()
    return resp.text


def _convert_nav_to_records(
    nav_data: List[dict],
) -> List[dict]:
    """Convert mfapi.in nav entries to canonical records.

    Input format:  [{"date": "DD-MM-YYYY", "nav": "string_number"}, ...]
    Output format: [{"date": "YYYY-MM-DD", "nav": float}, ...]

    mfapi returns newest-first; we reverse to oldest-first.
    """
    records: list[dict] = []
    for entry in nav_data:
        parsed_date = _parse_dd_mm_yyyy(entry.get("date", ""))
        if parsed_date is None:
            continue
        try:
            nav_val = float(entry["nav"])
        except (ValueError, KeyError, TypeError):
            continue
        if nav_val <= 0:
            continue
        records.append({"date": parsed_date.isoformat(), "nav": nav_val})

    # Reverse: oldest first
    records.reverse()

    # Safety cap
    if len(records) > MAX_NAV_POINTS:
        records = records[-MAX_NAV_POINTS:]

    return records


def _resolve_benchmark(category: str) -> tuple[int, str]:
    """Look up the benchmark index fund for a given AMFI category."""
    return CATEGORY_BENCHMARK_MAP.get(category, DEFAULT_BENCHMARK)


def ingest_from_mfapi(
    scheme_code: int,
    fund_name: str = "AMFI Fund",
    category: str = "Unknown",
    expense_ratio_pct: float = 0.0,
    **analysis_params: Any,
) -> Tuple[dict, dict]:
    """Fetch NAV from mfapi.in and build analyzer payload + ingestion report.

    The fund series is the historical NAV.
    The benchmark is a real index fund NAV fetched from mfapi, mapped
    from the fund's AMFI category.  If the benchmark fetch fails, falls
    back to self-benchmark with an explicit limitation flag.

    Returns (analyzer_payload, ingestion_report) — same shape as ingest_mf_from_csv.
    """
    raw = fetch_scheme_nav(scheme_code)

    meta = raw.get("meta", {})
    nav_data = raw.get("data", [])

    if not nav_data:
        raise ValueError(f"No NAV data returned for scheme {scheme_code}")

    # Use meta to enrich names if not provided or empty
    if (not fund_name or fund_name == "AMFI Fund") and meta.get("scheme_name"):
        fund_name = meta["scheme_name"]
    if (not category or category == "Unknown") and meta.get("scheme_category"):
        category = meta["scheme_category"]

    records = _convert_nav_to_records(nav_data)

    if len(records) < 2:
        raise ValueError(
            f"Insufficient NAV data for scheme {scheme_code}: {len(records)} points"
        )

    # Build canonical fund records for normalize_series
    fund_records = [{"date": r["date"], "nav": r["nav"]} for r in records]

    # ── Fetch real benchmark ──
    bench_code, bench_display_name = _resolve_benchmark(category)
    self_benchmark = False

    if bench_code == scheme_code:
        # Fund IS an index fund being used as its own benchmark — fall back
        self_benchmark = True
        bench_records = [{"date": r["date"], "value": r["nav"]} for r in records]
        benchmark_name = f"{fund_name} (self)"
        bench_source_tag = f"mfapi:{scheme_code}:self_benchmark"
    else:
        try:
            bench_raw = fetch_scheme_nav(bench_code)
            bench_nav = bench_raw.get("data", [])
            if not bench_nav:
                raise ValueError("Empty benchmark NAV")
            bench_records_raw = _convert_nav_to_records(bench_nav)
            if len(bench_records_raw) < 2:
                raise ValueError("Insufficient benchmark NAV")
            bench_records = [
                {"date": r["date"], "value": r["nav"]} for r in bench_records_raw
            ]
            benchmark_name = bench_display_name
            bench_source_tag = f"mfapi:{bench_code}"
        except Exception as exc:
            logger.warning(
                "Benchmark fetch failed for %s (code %d): %s — falling back to self-benchmark",
                bench_display_name, bench_code, exc,
            )
            self_benchmark = True
            bench_records = [{"date": r["date"], "value": r["nav"]} for r in records]
            benchmark_name = f"{fund_name} (self)"
            bench_source_tag = f"mfapi:{scheme_code}:self_benchmark"

    fund_norm = normalize_series(fund_records, "date", "nav", "fund")
    bench_norm = normalize_series(bench_records, "date", "value", "benchmark")

    timestamp = _now_iso()
    fund_source = make_source(
        f"mfapi:{scheme_code}", timestamp, LICENSE_REDIS,
    )
    benchmark_source = make_source(
        bench_source_tag, timestamp, LICENSE_REDIS,
    )

    ingestion_limitations = build_ingestion_limitations(fund_norm, bench_norm)
    if self_benchmark:
        ingestion_limitations.append(
            "Benchmark is self-referencing (same NAV used as benchmark). "
            "Relative performance metrics compare fund to itself."
        )

    payload: dict[str, Any] = {
        "fund_name": fund_name,
        "benchmark_name": benchmark_name,
        "category": category,
        "expense_ratio_pct": expense_ratio_pct,
        "fund_source": fund_source,
        "benchmark_source": benchmark_source,
        "fund": [
            {"date": p.as_of.isoformat(), "nav": p.value}
            for p in fund_norm.points
        ],
        "benchmark": [
            {"date": p.as_of.isoformat(), "value": p.value}
            for p in bench_norm.points
        ],
    }

    for key in (
        "rolling_window_points", "rolling_step_points", "rolling_min_windows",
        "expense_impact",
    ):
        if key in analysis_params:
            payload[key] = analysis_params[key]

    ingestion_report: dict[str, Any] = {
        "source": "mfapi",
        "scheme_code": scheme_code,
        "meta": meta,
        "ingestion_timestamp": timestamp,
        "license": LICENSE_REDIS,
        "fund_series": fund_norm.metadata,
        "benchmark_series": bench_norm.metadata,
        "fund_events": fund_norm.events,
        "benchmark_events": bench_norm.events,
        "fund_rejected": fund_norm.rejected,
        "benchmark_rejected": bench_norm.rejected,
        "fund_anomalies": fund_norm.anomalies,
        "benchmark_anomalies": bench_norm.anomalies,
        "ingestion_limitations": ingestion_limitations,
        "impact_links": build_impact_links(
            fund_meta=fund_norm.metadata,
            bench_meta=bench_norm.metadata,
            fund_anomalies=fund_norm.anomalies,
            bench_anomalies=bench_norm.anomalies,
            fund_events=fund_norm.events,
            bench_events=bench_norm.events,
        ),
    }

    return payload, ingestion_report
