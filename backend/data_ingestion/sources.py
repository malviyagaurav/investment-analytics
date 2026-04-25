from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from backend.investment_analytics.errors import PolicyError

from .schema_map import (
    CANONICAL_BENCHMARK_VALUE,
    CANONICAL_DATE,
    CANONICAL_FUND_VALUE,
    ColumnMapping,
    DEFAULT_MAPPING,
    apply_mapping,
    validate_mapping,
)

_ALLOWED_ROOTS: tuple[str, ...] = ("data/",)
MAX_CSV_ROWS = 10_000


def _safe_resolve(base: Path, relative: str) -> Path:
    """Resolve a path safely, blocking traversal outside the project root."""
    resolved = (base / relative).resolve()
    base_resolved = base.resolve()
    if not str(resolved).startswith(str(base_resolved)):
        raise PolicyError(
            "ingestion_security_error",
            "Path traversal detected.",
            {"requested": relative},
        )
    if not resolved.exists():
        raise PolicyError(
            "ingestion_source_error",
            f"Source file not found: {relative}",
            {"path": relative},
        )
    return resolved


def load_csv(
    project_root: Path,
    relative_path: str,
    date_col: str = "date",
    fund_value_col: str = "fund_nav",
    benchmark_value_col: str = "benchmark_nav",
    mapping: ColumnMapping | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load fund and benchmark records from a CSV file.

    If *mapping* is provided, it overrides ``date_col``, ``fund_value_col``,
    and ``benchmark_value_col`` — source column names are remapped to
    canonical internal keys so the rest of the pipeline is format-agnostic.

    Returns (fund_records, benchmark_records, file_metadata).
    """
    allowed = any(relative_path.startswith(root) for root in _ALLOWED_ROOTS)
    if not allowed:
        raise PolicyError(
            "ingestion_security_error",
            f"Source path must be under allowed directories: {_ALLOWED_ROOTS}",
            {"requested": relative_path},
        )

    path = _safe_resolve(project_root, relative_path)

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise PolicyError(
                "ingestion_source_error",
                "CSV file has no header row.",
                {"path": relative_path},
            )
        for row in reader:
            rows.append(dict(row))
            if len(rows) > MAX_CSV_ROWS:
                raise PolicyError(
                    "ingestion_size_error",
                    f"CSV exceeds maximum of {MAX_CSV_ROWS:,} data rows.",
                    {"path": relative_path, "max_rows": MAX_CSV_ROWS},
                )

    if not rows:
        raise PolicyError(
            "ingestion_source_error",
            "CSV file has no data rows.",
            {"path": relative_path},
        )

    columns = list(rows[0].keys()) if rows else []

    if mapping is not None:
        validate_mapping(mapping, columns)
        fund_records, benchmark_records = apply_mapping(rows, mapping)
    else:
        fund_records = [
            {date_col: row.get(date_col, ""), fund_value_col: row.get(fund_value_col, "")}
            for row in rows
        ]
        benchmark_records = [
            {date_col: row.get(date_col, ""), benchmark_value_col: row.get(benchmark_value_col, "")}
            for row in rows
        ]

    file_metadata = {
        "source_file": relative_path,
        "total_rows": len(rows),
        "columns": columns,
        "mapping_label": mapping.label if mapping else "direct",
    }

    return fund_records, benchmark_records, file_metadata
