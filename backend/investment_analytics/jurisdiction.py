from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_JURISDICTIONS = {
    "IN": {
        "name": "India",
        "primary_regulator": "Securities and Exchange Board of India",
    },
    "US": {
        "name": "United States",
        "primary_regulator": "U.S. Securities and Exchange Commission / state regulators",
    },
}


@dataclass(frozen=True)
class JurisdictionContext:
    user_country: str
    asset_market: str
    serving_entity: str


def _normalize_country(value: str) -> str:
    normalized = (value or "").strip().upper()
    aliases = {
        "INDIA": "IN",
        "USA": "US",
        "UNITED STATES": "US",
        "UNITED STATES OF AMERICA": "US",
    }
    return aliases.get(normalized, normalized)


def evaluate_jurisdiction(context: JurisdictionContext) -> dict:
    user_country = _normalize_country(context.user_country)
    asset_market = _normalize_country(context.asset_market)
    supported = user_country in SUPPORTED_JURISDICTIONS and asset_market in SUPPORTED_JURISDICTIONS
    strict = not supported
    regulator = SUPPORTED_JURISDICTIONS.get(user_country, {}).get("primary_regulator", "unsupported")
    return {
        "user_country": user_country,
        "asset_market": asset_market,
        "serving_entity": context.serving_entity,
        "supported": supported,
        "regulator": regulator,
        "mode": "analytics_mode",
        "strict_constraints": strict,
        "features": {
            "analytics": supported,
            "ra_mode": False,
            "advisory_mode": False,
            "ra_risk_features": False,
        },
        "reason": (
            "Jurisdiction and market are whitelisted."
            if supported
            else "Unsupported or incomplete jurisdiction context. Sensitive pages are disabled."
        ),
    }

