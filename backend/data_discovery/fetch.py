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

logger = logging.getLogger("data_discovery.fetch")

MFAPI_BASE = "https://api.mfapi.in/mf"
NAVALL_URL = "https://portal.amfiindia.com/spages/NAVAll.txt"

# Request timeout (seconds)
REQUEST_TIMEOUT = 30.0

# Max points to ingest (safety guard)
MAX_NAV_POINTS = 10_000


def _parse_dd_mm_yyyy(date_str: str) -> Optional[date]:
    """Parse DD-MM-YYYY to date object."""
    try:
        return datetime.strptime(date_str.strip(), "%d-%m-%Y").date()
    except (ValueError, AttributeError):
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def fetch_scheme_nav(scheme_code: int) -> dict:
    """Fetch historical NAV from mfapi.in.

    Returns the raw JSON response as a dict.
    Raises httpx.HTTPStatusError on non-2xx.
    """
    url = f"{MFAPI_BASE}/{scheme_code}"
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        resp = client.get(url)
        resp.raise_for_status()
    return resp.json()


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


def ingest_from_mfapi(
    scheme_code: int,
    fund_name: str = "AMFI Fund",
    category: str = "Unknown",
    expense_ratio_pct: float = 0.0,
    **analysis_params: Any,
) -> Tuple[dict, dict]:
    """Fetch NAV from mfapi.in and build analyzer payload + ingestion report.

    The fund series is the historical NAV.
    The benchmark is a synthetic clone (same NAV as fund) since we don't
    have a separate benchmark from mfapi.  This is explicitly flagged
    as a limitation in the ingestion report.

    Returns (analyzer_payload, ingestion_report) — same shape as ingest_mf_from_csv.
    """
    raw = fetch_scheme_nav(scheme_code)

    meta = raw.get("meta", {})
    nav_data = raw.get("data", [])

    if not nav_data:
        raise ValueError(f"No NAV data returned for scheme {scheme_code}")

    # Use meta to enrich names if not provided
    if fund_name == "AMFI Fund" and meta.get("scheme_name"):
        fund_name = meta["scheme_name"]
    if category == "Unknown" and meta.get("scheme_category"):
        category = meta["scheme_category"]

    records = _convert_nav_to_records(nav_data)

    if len(records) < 2:
        raise ValueError(
            f"Insufficient NAV data for scheme {scheme_code}: {len(records)} points"
        )

    # Build canonical fund records for normalize_series
    fund_records = [{"date": r["date"], "nav": r["nav"]} for r in records]
    # Benchmark = same data (self-benchmark), flagged as limitation
    bench_records = [{"date": r["date"], "value": r["nav"]} for r in records]

    fund_norm = normalize_series(fund_records, "date", "nav", "fund")
    bench_norm = normalize_series(bench_records, "date", "value", "benchmark")

    timestamp = _now_iso()
    fund_source = make_source(
        f"mfapi:{scheme_code}", timestamp, LICENSE_REDIS,
    )
    benchmark_source = make_source(
        f"mfapi:{scheme_code}:self_benchmark", timestamp, LICENSE_REDIS,
    )

    ingestion_limitations = build_ingestion_limitations(fund_norm, bench_norm)
    # Add explicit self-benchmark limitation
    ingestion_limitations.append(
        "Benchmark is self-referencing (same NAV used as benchmark). "
        "Relative performance metrics compare fund to itself."
    )

    payload: dict[str, Any] = {
        "fund_name": fund_name,
        "benchmark_name": f"{fund_name} (self)",
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
