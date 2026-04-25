from __future__ import annotations

from typing import Any

from .errors import PolicyError


LICENSE_REDIS = "redistributable"
LICENSE_RESTRICTED = "restricted"
LICENSE_USER = "user_supplied"
KNOWN_LICENSES = {LICENSE_REDIS, LICENSE_RESTRICTED, LICENSE_USER}


def most_restrictive_license(licenses: list[str]) -> str:
    normalized = [item for item in licenses if item]
    if LICENSE_RESTRICTED in normalized:
        return LICENSE_RESTRICTED
    if LICENSE_USER in normalized:
        return LICENSE_USER
    return LICENSE_REDIS


def collect_licenses(value: Any) -> list[str]:
    licenses: list[str] = []
    if isinstance(value, dict):
        license_value = value.get("license")
        if isinstance(license_value, str):
            normalized = license_value.strip().lower()
            if normalized in KNOWN_LICENSES:
                licenses.append(normalized)
        for child in value.values():
            licenses.extend(collect_licenses(child))
    elif isinstance(value, list):
        for child in value:
            licenses.extend(collect_licenses(child))
    return licenses


def assert_renderable_lineage(value: Any) -> None:
    licenses = collect_licenses(value)
    if most_restrictive_license(licenses) == LICENSE_RESTRICTED:
        raise PolicyError(
            "restricted_lineage",
            "Metric has restricted data lineage and cannot be rendered.",
            {"licenses": licenses},
        )


def make_source(
    source: str,
    timestamp: str,
    license_value: str = LICENSE_REDIS,
    lineage: list[dict] | None = None,
) -> dict:
    normalized = license_value.strip().lower()
    if normalized not in KNOWN_LICENSES:
        raise PolicyError(
            "unknown_license",
            f"Unknown license flag: {license_value}",
            {"source": source},
        )
    return {
        "source": source,
        "timestamp": timestamp,
        "license": normalized,
        "lineage": lineage or [],
    }

