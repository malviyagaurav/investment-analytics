"""ExperimentConfig — frozen, content-addressable, governance-validated.

A config is the complete specification of an experiment: which
parameterized function to call, with what data inputs, with what
parameter overrides, classified by methodology provenance and
experimental intent, with explicit lineage to any prior experiments
whose outputs informed this one.

Two governance dimensions are enforced at construction:

  1. ``methodology_kind`` — engineered_variant (reasoned parameters)
     vs data_driven_variant (parameters derived from prior measurement).
     A data_driven_variant MUST cite at least one ``derived_from_run_ids``
     entry. This is the anti-evidence-laundering control: a heuristic
     cannot masquerade as measured by simply switching the label.

  2. ``experiment_status`` — the experimental intent. Used by
     downstream promotion gating (Step 15): an exploratory experiment
     can never auto-promote; a shadow_candidate can; a
     calibration_candidate must be tied to a calibration report.

``config_fingerprint`` is the content-addressed identity of the
experiment: sha256 over canonical JSON of the semantic fields. Two
engineers running the same logical experiment converge on the same
fingerprint regardless of comments, rationale, or any other
non-semantic metadata.

``non_semantic_metadata`` is the explicit escape hatch for free text
(rationale, links, notes). It is recorded into the audit trail for
forensic value but is EXCLUDED from:
  - config_fingerprint
  - replay equality semantics
  - calibration / reliability derivation paths
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Tuple


# Methodology provenance classification. Required on every config —
# no default. Forces the caller to commit to a classification.
METHODOLOGY_KINDS: frozenset = frozenset({
    "engineered_variant",     # parameters chosen by reasoning
    "data_driven_variant",    # parameters derived from prior measurement
})


# Experimental intent classification. Required on every config —
# no default. Used for promotion gating (Step 15) and recursive-
# depth controls (Step 14 reliability scoring).
EXPERIMENT_STATUSES: frozenset = frozenset({
    "exploratory",            # ad-hoc tuning; never auto-promotable
    "validation",             # verifying a specific hypothesis
    "shadow_candidate",       # eligible for shadow run prior to promotion
    "calibration_candidate",  # produced by / feeding a calibration cycle
})


CONFIG_SCHEMA_VERSION = "v1"


def _freeze_mapping(value: Any) -> Any:
    """Recursively convert dicts → MappingProxyType, lists → tuples.

    Prevents post-construction mutation of nested data. Frozen
    dataclasses freeze attribute assignment but NOT the underlying
    objects — without this, a caller could mutate a dict inside a
    'frozen' config and silently change its fingerprint.
    """
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze_mapping(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_mapping(v) for v in value)
    return value


def _to_plain(value: Any) -> Any:
    """Inverse of _freeze_mapping for serialization. MappingProxyType
    is not JSON-serializable; we unwrap to plain dicts/lists for
    canonical encoding and downstream consumers."""
    if isinstance(value, MappingProxyType):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_plain(v) for v in value]
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _to_plain(value),
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    )


@dataclass(frozen=True)
class ExperimentConfig:
    """Frozen, content-addressable specification of an experiment.

    Fields included in ``config_fingerprint`` (semantic identity):
      - target, target_inputs, param_overrides
      - methodology_kind, experiment_status
      - derived_from_run_ids
      - schema_version

    Fields EXCLUDED from fingerprint (non-semantic):
      - non_semantic_metadata (rationale, notes, links — forensic only)
    """
    target: str
    target_inputs: Mapping[str, Any]
    param_overrides: Mapping[str, Any]
    methodology_kind: str
    experiment_status: str
    derived_from_run_ids: Tuple[str, ...] = ()
    non_semantic_metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        # Validate closed enums first — fail fast on misclassification.
        if self.methodology_kind not in METHODOLOGY_KINDS:
            raise ValueError(
                f"methodology_kind must be one of {sorted(METHODOLOGY_KINDS)}, "
                f"got {self.methodology_kind!r}"
            )
        if self.experiment_status not in EXPERIMENT_STATUSES:
            raise ValueError(
                f"experiment_status must be one of {sorted(EXPERIMENT_STATUSES)}, "
                f"got {self.experiment_status!r}"
            )

        # Anti-evidence-laundering invariant: data_driven_variant
        # cannot exist without a citation chain.
        if (self.methodology_kind == "data_driven_variant"
                and not self.derived_from_run_ids):
            raise ValueError(
                "data_driven_variant requires non-empty derived_from_run_ids — "
                "data-derived configs must cite the prior measurements they came from"
            )

        # Freeze nested collections so the fingerprint cannot drift via
        # post-construction mutation. Uses object.__setattr__ because
        # frozen dataclasses forbid normal attribute assignment.
        object.__setattr__(self, "target_inputs",
                           _freeze_mapping(dict(self.target_inputs)))
        object.__setattr__(self, "param_overrides",
                           _freeze_mapping(dict(self.param_overrides)))
        object.__setattr__(self, "non_semantic_metadata",
                           _freeze_mapping(dict(self.non_semantic_metadata)))
        object.__setattr__(self, "derived_from_run_ids",
                           tuple(self.derived_from_run_ids))

    def _semantic_dict(self) -> dict:
        """Plain-dict view of the fingerprint-relevant fields.
        ``non_semantic_metadata`` is intentionally absent."""
        return {
            "schema_version":       self.schema_version,
            "target":               self.target,
            "target_inputs":        _to_plain(self.target_inputs),
            "param_overrides":      _to_plain(self.param_overrides),
            "methodology_kind":     self.methodology_kind,
            "experiment_status":    self.experiment_status,
            "derived_from_run_ids": list(self.derived_from_run_ids),
        }

    @property
    def config_fingerprint(self) -> str:
        """sha256 hex over canonical-JSON of semantic fields. Excludes
        non_semantic_metadata so prose changes never affect identity."""
        return hashlib.sha256(
            _canonical_json(self._semantic_dict()).encode("utf-8")
        ).hexdigest()

    def to_payload(self) -> dict:
        """Full config view (including non_semantic_metadata) for
        embedding in evidence payloads. The fingerprint stays
        semantic-only; the payload retains the prose for forensics."""
        out = self._semantic_dict()
        out["config_fingerprint"] = self.config_fingerprint
        out["non_semantic_metadata"] = _to_plain(self.non_semantic_metadata)
        return out
