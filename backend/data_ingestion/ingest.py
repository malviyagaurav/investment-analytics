from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.investment_analytics.lineage import LICENSE_REDIS, make_source

from .normalize import normalize_series
from .schema_map import (
    CANONICAL_BENCHMARK_VALUE,
    CANONICAL_DATE,
    CANONICAL_FUND_VALUE,
    ColumnMapping,
)
from .sources import load_csv
from .validate import build_ingestion_limitations, validate_series
from .impact_links import build_impact_links


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ingest_mf_from_csv(
    project_root: Path,
    source_path: str,
    fund_name: str = "Ingested Fund",
    benchmark_name: str = "Ingested Benchmark",
    category: str = "Unknown",
    expense_ratio_pct: float = 0.0,
    license_value: str = LICENSE_REDIS,
    mapping: ColumnMapping | None = None,
    **analysis_params: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ingest MF data from CSV, normalize, validate, build analyzer payload.

    If *mapping* is provided, source columns are remapped to canonical names
    before normalization.  Otherwise the CSV must use default column names.

    Returns (analyzer_payload, ingestion_report).
    """
    fund_records, benchmark_records, file_meta = load_csv(
        project_root,
        source_path,
        mapping=mapping,
    )

    fund_norm = normalize_series(
        fund_records, CANONICAL_DATE, CANONICAL_FUND_VALUE, "fund",
    )
    bench_norm = normalize_series(
        benchmark_records, CANONICAL_DATE, CANONICAL_BENCHMARK_VALUE, "benchmark",
    )

    validate_series(fund_norm, "fund", min_points=2)
    validate_series(bench_norm, "benchmark", min_points=2)

    timestamp = _now_iso()
    fund_source = make_source(f"ingested:{source_path}", timestamp, license_value)
    benchmark_source = make_source(f"ingested:{source_path}", timestamp, license_value)

    ingestion_limitations = build_ingestion_limitations(fund_norm, bench_norm)

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
        "expense_impact", "expected_window_span_days",
    ):
        if key in analysis_params:
            payload[key] = analysis_params[key]

    ingestion_report: dict[str, Any] = {
        "source": "csv",
        "source_path": source_path,
        "ingestion_timestamp": timestamp,
        "license": license_value,
        "mapping_label": mapping.label if mapping else "direct",
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
