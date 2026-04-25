"""Schema mapping layer for ingesting CSV files with varying column layouts.

Maps external column names → canonical internal names so that the rest
of the pipeline (normalization, validation, analysis) never sees format
variation.

Design principles:
    - Explicit mapping only.  No auto-detection, no heuristics.
    - Validation at load time.  Missing columns fail immediately.
    - Immutable after construction.  Frozen dataclass.
"""
from __future__ import annotations
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.investment_analytics.errors import PolicyError


@dataclass(frozen=True)
class ColumnMapping:
    """Maps source CSV columns to canonical internal names.

    Attributes:
        date_column:      Name of the date column in the source CSV.
        fund_column:      Name of the fund value column in the source CSV.
        benchmark_column: Name of the benchmark value column in the source CSV.
        date_format:      strptime format string for dates, or None for ISO default.
        label:            Human-readable format name (e.g. "AMFI NAV").
    """
    date_column: str
    fund_column: str
    benchmark_column: str
    date_format: str | None = None
    label: str = "custom"


# Canonical internal key names — the rest of the pipeline uses these.
CANONICAL_DATE = "date"
CANONICAL_FUND_VALUE = "fund_nav"
CANONICAL_BENCHMARK_VALUE = "benchmark_nav"


def validate_mapping(mapping: ColumnMapping, available_columns: list[str]) -> None:
    """Raise PolicyError if required columns are missing from the CSV header."""
    required = {
        "date_column": mapping.date_column,
        "fund_column": mapping.fund_column,
        "benchmark_column": mapping.benchmark_column,
    }
    missing = [
        f"'{col}' ({role.replace('_column', '')})"
        for role, col in required.items()
        if col not in available_columns
    ]
    if missing:
        raise PolicyError(
            "ingestion_schema_error",
            f"Column(s) not found in CSV: {', '.join(missing)}. "
            f"Available columns: {available_columns}. "
            f"Check schema_mapping='{mapping.label}'.",
            {
                "missing": missing,
                "available": available_columns,
                "mapping_label": mapping.label,
            },
        )


def apply_mapping(
    rows: list[dict[str, str]],
    mapping: ColumnMapping,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remap raw CSV rows to canonical key names.

    Returns (fund_records, benchmark_records) where each record uses
    canonical keys: ``date``, ``fund_nav``, ``benchmark_nav``.
    """
    fund_records: list[dict[str, Any]] = []
    benchmark_records: list[dict[str, Any]] = []

    for row in rows:
        date_val = row.get(mapping.date_column, "")
        fund_val = row.get(mapping.fund_column, "")
        bench_val = row.get(mapping.benchmark_column, "")

        fund_records.append({
            CANONICAL_DATE: date_val,
            CANONICAL_FUND_VALUE: fund_val,
        })
        benchmark_records.append({
            CANONICAL_DATE: date_val,
            CANONICAL_BENCHMARK_VALUE: bench_val,
        })

    return fund_records, benchmark_records


# ── Pre-configured mappings for known formats ─────────────────

DEFAULT_MAPPING = ColumnMapping(
    date_column="date",
    fund_column="fund_nav",
    benchmark_column="benchmark_nav",
    date_format=None,
    label="default",
)

AMFI_NAV_MAPPING = ColumnMapping(
    date_column="Date",
    fund_column="NAV",
    benchmark_column="Index Value",
    date_format="%d-%m-%Y",
    label="AMFI NAV",
)

ETF_PRICE_MAPPING = ColumnMapping(
    date_column="Trade Date",
    fund_column="Close Price",
    benchmark_column="Index Close",
    date_format=None,
    label="ETF price",
)

ALT_MF_MAPPING = ColumnMapping(
    date_column="Date",
    fund_column="NAV",
    benchmark_column="Benchmark NAV",
    date_format="%d-%m-%Y",
    label="alternate MF",
)

# Registry: name → mapping
KNOWN_MAPPINGS: dict[str, ColumnMapping] = {
    "default": DEFAULT_MAPPING,
    "amfi_nav": AMFI_NAV_MAPPING,
    "etf_price": ETF_PRICE_MAPPING,
    "alt_mf": ALT_MF_MAPPING,
}
