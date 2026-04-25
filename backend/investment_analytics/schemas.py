from __future__ import annotations

from typing import Any

from .errors import PolicyError


EVIDENCE_VALUES = {"High", "Medium", "Low"}
COMPLETENESS_VALUES = {"High", "Medium", "Low"}

COMMON_FIELDS = {
    "type",
    "evidence_strength",
    "data_completeness",
    "limitations",
    "unavailable_components",
}

TEMPLATE_SCHEMAS: dict[str, set[str]] = {
    "diagnostic": COMMON_FIELDS
    | {
        "observation",
        "why_it_matters",
        "supporting_data",
    },
    "benchmark_comparison": COMMON_FIELDS
    | {
        "observation",
        "benchmark",
        "supporting_data",
    },
    "scenario": COMMON_FIELDS
    | {
        "scenario_definition",
        "assumptions",
        "projected_impact",
        "sensitivity",
    },
    "cost_tax": COMMON_FIELDS
    | {
        "scenario_a",
        "scenario_b",
        "assumptions",
        "estimated_impact",
    },
}


def _assert_keys(payload: dict[str, Any], allowed: set[str]) -> None:
    actual = set(payload.keys())
    missing = allowed - actual
    extra = actual - allowed
    if missing or extra:
        raise PolicyError(
            "schema_violation",
            "Insight does not match an allowlisted template.",
            {"missing": sorted(missing), "extra": sorted(extra)},
        )


def _assert_enum(value: Any, allowed: set[str], path: str) -> None:
    if value not in allowed:
        raise PolicyError(
            "schema_violation",
            f"{path} must be one of {sorted(allowed)}.",
            {"path": path, "value": value},
        )


def _assert_list(value: Any, path: str) -> None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise PolicyError(
            "schema_violation",
            f"{path} must be a list of strings.",
            {"path": path, "value": value},
        )


def _assert_dict(value: Any, path: str) -> None:
    if not isinstance(value, dict):
        raise PolicyError(
            "schema_violation",
            f"{path} must be an object.",
            {"path": path, "value": value},
        )


def validate_template_schema(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise PolicyError("schema_violation", "Insight payload must be an object.")

    insight_type = payload.get("type")
    if insight_type not in TEMPLATE_SCHEMAS:
        raise PolicyError(
            "schema_violation",
            "Insight type is not allowlisted.",
            {"type": insight_type, "allowed": sorted(TEMPLATE_SCHEMAS)},
        )

    _assert_keys(payload, TEMPLATE_SCHEMAS[str(insight_type)])
    _assert_enum(payload.get("evidence_strength"), EVIDENCE_VALUES, "evidence_strength")
    _assert_enum(payload.get("data_completeness"), COMPLETENESS_VALUES, "data_completeness")
    _assert_list(payload.get("limitations"), "limitations")
    _assert_list(payload.get("unavailable_components"), "unavailable_components")

    if insight_type in {"diagnostic", "benchmark_comparison"}:
        _assert_dict(payload.get("supporting_data"), "supporting_data")

    if insight_type == "benchmark_comparison":
        benchmark = payload.get("benchmark")
        _assert_dict(benchmark, "benchmark")
        for key in ("name", "methodology", "source"):
            if not benchmark.get(key):
                raise PolicyError(
                    "schema_violation",
                    "Benchmark comparison requires name, methodology, and source.",
                    {"missing_key": key},
                )

    if insight_type == "scenario":
        scenario_definition = payload.get("scenario_definition")
        projected_impact = payload.get("projected_impact")
        _assert_dict(scenario_definition, "scenario_definition")
        _assert_dict(payload.get("assumptions"), "assumptions")
        _assert_dict(projected_impact, "projected_impact")
        _assert_list(payload.get("sensitivity"), "sensitivity")
        if scenario_definition.get("kind") not in {"standard", "user"}:
            raise PolicyError(
                "schema_violation",
                "Scenario kind must be standard or user.",
                {"kind": scenario_definition.get("kind")},
            )
        if "range" not in projected_impact or "units" not in projected_impact:
            raise PolicyError(
                "schema_violation",
                "Scenario projected_impact requires range and units.",
            )

    if insight_type == "cost_tax":
        _assert_dict(payload.get("scenario_a"), "scenario_a")
        _assert_dict(payload.get("scenario_b"), "scenario_b")
        _assert_dict(payload.get("assumptions"), "assumptions")
        _assert_dict(payload.get("estimated_impact"), "estimated_impact")
        assumptions = payload["assumptions"]
        for key in ("tax_year", "residency", "rates", "holding_period"):
            if key not in assumptions:
                raise PolicyError(
                    "schema_violation",
                    "Cost/tax output requires tax year, residency, rates, and holding period.",
                    {"missing_key": key},
                )
        if "Consult a qualified professional" not in payload["limitations"]:
            raise PolicyError(
                "schema_violation",
                "Cost/tax output must include professional consultation limitation.",
            )

