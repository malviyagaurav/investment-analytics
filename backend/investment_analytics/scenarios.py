from __future__ import annotations

from typing import Any

from .errors import PolicyError


STANDARD_SCENARIOS: dict[str, dict[str, Any]] = {
    "market_down_20": {
        "label": "Market -20% shock",
        "params": {"market_return_pct": -20.0},
    },
    "rates_up_1": {
        "label": "Rates +1% parallel shift",
        "params": {"rate_shift_bps": 100},
    },
    "fee_delta_50bps": {
        "label": "Expense ratio +0.5% differential",
        "params": {"fee_delta_bps": 50},
    },
    "tracking_error_widens_50bps": {
        "label": "Tracking error widens by 50 bps",
        "params": {"tracking_error_delta_bps": 50},
    },
}


def list_standard_scenarios() -> list[dict[str, Any]]:
    return [{"id": key, **value} for key, value in STANDARD_SCENARIOS.items()]


def resolve_scenario_definition(definition: dict[str, Any]) -> dict[str, Any]:
    kind = definition.get("kind")
    if kind == "standard":
        scenario_id = definition.get("id")
        if scenario_id not in STANDARD_SCENARIOS:
            raise PolicyError(
                "scenario_violation",
                "Standard scenario is not allowlisted.",
                {"id": scenario_id},
            )
        return {"kind": "standard", "id": scenario_id, **STANDARD_SCENARIOS[scenario_id]}
    if kind == "user":
        params = definition.get("params")
        if not isinstance(params, dict) or not params:
            raise PolicyError(
                "scenario_violation",
                "User-authored scenarios require explicit params.",
                {"definition": definition},
            )
        return {"kind": "user", "params": params}
    raise PolicyError(
        "scenario_violation",
        "Scenario kind must be standard or user.",
        {"kind": kind},
    )

