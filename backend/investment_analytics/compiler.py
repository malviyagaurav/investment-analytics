from __future__ import annotations

from copy import deepcopy
from typing import Any

from .language_policy import assert_language_allowed
from .lineage import assert_renderable_lineage
from .schemas import validate_template_schema


TEMPLATE_LABELS: dict[str, list[str]] = {
    "diagnostic": [
        "Observation",
        "Why it matters",
        "Supporting data",
        "Evidence Strength",
        "Data Completeness",
        "Limitations",
        "Unavailable Components",
    ],
    "benchmark_comparison": [
        "Observation",
        "Methodology / Source",
        "Supporting data",
        "Evidence Strength",
        "Data Completeness",
        "Limitations",
        "Unavailable Components",
    ],
    "scenario": [
        "Scenario definition",
        "Assumptions",
        "Projected impact",
        "Sensitivity notes",
        "Evidence Strength",
        "Data Completeness",
        "Limitations",
        "Unavailable Components",
    ],
    "cost_tax": [
        "Scenario A",
        "Scenario B",
        "Assumptions",
        "Estimated impact",
        "Evidence Strength",
        "Data Completeness",
        "Limitations",
        "Unavailable Components",
    ],
}


def compile_insight(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and compile an insight without rewriting any text."""
    compiled = deepcopy(payload)
    validate_template_schema(compiled)
    assert_renderable_lineage(compiled)
    assert_language_allowed(compiled)
    return {
        "template": compiled["type"],
        "labels": TEMPLATE_LABELS[compiled["type"]],
        "payload": compiled,
    }


def compile_insights(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [compile_insight(payload) for payload in payloads]

