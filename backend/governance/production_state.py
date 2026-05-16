"""Snapshot of production constants at decision time.

A governance_decision binds the operator's declaration to the
EXACT production value that was active at the moment of decision.
This is forensically load-bearing: an "approve at value 0.85"
record is meaningless if production silently moved to 0.90 between
the decision and the next audit pass.

## Discipline

  - This module READS production constants. It NEVER writes them.
  - The byte_hash is a sha256 over the canonical-JSON
    serialization of the value. Numeric values, strings, lists,
    dicts are all hashable through this path; binary blobs are
    not — production thresholds are scalar today, the layer
    intentionally does NOT generalize to opaque bytes.
  - The source_attestation is an engineered string identifying
    WHERE the value lives in the codebase (module + symbol).
    Bumping the source path is a real change and should surface
    on replay — but the surfacing happens via byte_hash
    divergence, not via attestation text comparison.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Dict


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _byte_hash(value: Any) -> str:
    canonical = _canonical_json(value)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_high_correlation_threshold() -> Any:
    # Local import so reading the registry does not cascade module
    # imports through the portfolio_health package at governance
    # module-load time.
    from backend.investment_analytics.portfolio_health.correlation import (
        HIGH_CORRELATION_THRESHOLD,
    )
    return HIGH_CORRELATION_THRESHOLD


# Registry of production constants the governance layer is allowed
# to snapshot. The key is the canonical target id (matches
# CALIBRATION_TARGETS); the value is a (reader, source_attestation)
# tuple. The reader is invoked at decision time to capture the
# CURRENT production value; the attestation is a static descriptor.
#
# Adding a target here is a real governance extension: it widens
# the surface that governance can declare against. Reviewed at the
# same gate as adding to CALIBRATION_TARGETS.
_PRODUCTION_REGISTRY: Dict[str, Dict[str, Any]] = {
    "portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD": {
        "reader":              _read_high_correlation_threshold,
        "source_attestation":  (
            "backend.investment_analytics.portfolio_health.correlation"
            ".HIGH_CORRELATION_THRESHOLD"
        ),
    },
}


def supported_targets() -> frozenset:
    return frozenset(_PRODUCTION_REGISTRY.keys())


def snapshot_production_state(target_canonical_id: str) -> Dict[str, Any]:
    """Capture the live production value for a canonical target.

    Returns a dict suitable for embedding under
    ``production_state_at_decision`` in the governance_decision
    payload:

      {
        "target_canonical_id": "...",
        "current_value":        <python scalar>,
        "value_byte_hash":      "<sha256 hex>",
        "source_attestation":   "<dotted path>",
      }

    Raises KeyError when the target is not registered for
    production snapshotting (a refusal_reason at the eligibility
    layer maps this to target_unsupported).
    """
    entry = _PRODUCTION_REGISTRY.get(target_canonical_id)
    if entry is None:
        raise KeyError(
            f"target {target_canonical_id!r} is not registered for "
            f"production-state snapshotting. Registered: "
            f"{sorted(_PRODUCTION_REGISTRY)}"
        )
    reader: Callable[[], Any] = entry["reader"]
    value = reader()
    return {
        "target_canonical_id": target_canonical_id,
        "current_value":       value,
        "value_byte_hash":     _byte_hash(value),
        "source_attestation":  entry["source_attestation"],
    }
