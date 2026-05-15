"""Experiment runner — emits ``experiment_run`` evidence rows.

The runner takes a frozen ExperimentConfig + a baseline run_id,
validates the lineage chain, computes recursive derivation depth,
captures the production methodology snapshot, runs the parameterized
function, and emits exactly one ``experiment_run`` row + one evidence
file via the existing emit_evidence path.

## Side-effect discipline (load-bearing)

Running an experiment:
  - DOES emit one experiment_run audit row + evidence file
  - DOES NOT mutate METHODOLOGY_VERSIONS
  - DOES NOT touch production snapshot/sidecar state
  - DOES NOT auto-promote any parameter change

Promotion is Step 15's concern, gated on human review of accumulated
experiment evidence. Step 9 is the substrate only.

## Production isolation in the payload

Two separate fields appear in every emitted payload:
  - ``production_methodology_versions`` — what production believed at
    experiment time (snapshot of METHODOLOGY_VERSIONS)
  - ``experiment_overrides`` — what THIS experiment temporarily varied
    (the param_overrides applied)

Replay / drift / calibration tooling must consult both to distinguish
"what production believed" from "what we varied" without inferring it
from lineage.

## Derivation depth

Computed deterministically at runner time:
  - 0 for a config with empty derived_from_run_ids (engineered root)
  - max(parent depths) + 1 otherwise

Parents that are NOT experiment_run rows contribute 0 to the max
(they're external inputs, not derived experiments). Parents that
cannot be found in the audit chain → ValueError before any compute.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Optional

from backend.evidence.store import emit_evidence
from backend.evidence.replay import find_record_by_run_id
from backend.experiments.config import ExperimentConfig
from backend.experiments.registry import (
    REGISTERED_PARAMETERIZED_FUNCS,
    registry_contract,
)
from backend.investment_analytics.methodology import current_methodology

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_AUDIT_PATH = ROOT / "data" / "audit" / "audit.jsonl"
DEFAULT_REGISTRY_PATH = ROOT / "data" / "registry" / "schemes.json"

EXPERIMENT_AUDIT_SCHEMA_VERSION = "v1"


def _validate_param_overrides(config: ExperimentConfig) -> None:
    """Disallow any param key not in the target's allowed_param_keys.

    Protects against typos that would otherwise be silently ignored
    (since the swap is opt-in per key) AND against experimenters
    widening the override surface via the config.
    """
    entry = REGISTERED_PARAMETERIZED_FUNCS[config.target]
    bad_keys = set(config.param_overrides.keys()) - entry.allowed_param_keys
    if bad_keys:
        raise ValueError(
            f"experiment param_overrides contains keys not in the "
            f"allowed set for target {config.target!r}: {sorted(bad_keys)}. "
            f"Allowed: {sorted(entry.allowed_param_keys)}"
        )


def _resolve_derivation_depth(
    audit_path: Path,
    derived_from_run_ids: Iterable[str],
) -> int:
    """Walk one level back through derived_from lineage and compute:

        depth = 0 if no parents
        depth = max(parent.derivation_depth or 0) + 1 otherwise

    Validates that every cited run_id exists in the chain. Parents
    that aren't experiment_run rows contribute 0 to the max — they're
    external inputs, not derived experiments. Missing run_id → raises.
    """
    parent_ids = list(derived_from_run_ids)
    if not parent_ids:
        return 0
    max_parent_depth = 0
    for parent_id in parent_ids:
        record = find_record_by_run_id(audit_path, parent_id)
        if record is None:
            raise ValueError(
                f"derived_from_run_ids cites unknown run_id {parent_id!r} — "
                f"not found in audit chain at {audit_path}"
            )
        event = record.get("event", {})
        parent_depth = (event.get("derivation_depth") or 0
                        if event.get("evidence_kind") == "experiment_run"
                        else 0)
        if parent_depth > max_parent_depth:
            max_parent_depth = parent_depth
    return max_parent_depth + 1


def run_experiment(
    config: ExperimentConfig,
    baseline_run_id: str,
    *,
    audit_path: Optional[Path] = None,
    registry_path: Optional[Path] = None,
) -> dict:
    """Validate, run, and emit a single experiment_run evidence event.

    Args:
      config:          frozen ExperimentConfig (validated at construction).
      baseline_run_id: the production run this experiment varies from;
                       becomes parent_run_id on the envelope.
      audit_path:      path to audit.jsonl. Defaults to live data/audit/.
      registry_path:   registry passed to the parameterized callable.
                       Defaults to live data/registry/schemes.json.

    Returns the audit record dict from emit_evidence.

    Raises:
      KeyError    — target not in REGISTERED_PARAMETERIZED_FUNCS.
      ValueError  — param_overrides keys outside allowed set,
                    derived_from_run_ids missing from chain,
                    baseline_run_id missing from chain.
    """
    audit_path = audit_path or DEFAULT_AUDIT_PATH
    registry_path = registry_path or DEFAULT_REGISTRY_PATH

    if config.target not in REGISTERED_PARAMETERIZED_FUNCS:
        raise KeyError(
            f"unknown experiment target: {config.target!r}. "
            f"Registered: {sorted(REGISTERED_PARAMETERIZED_FUNCS)}"
        )

    _validate_param_overrides(config)

    # Baseline must exist in the chain — parent_run_id can't dangle.
    baseline_record = find_record_by_run_id(audit_path, baseline_run_id)
    if baseline_record is None:
        raise ValueError(
            f"baseline_run_id {baseline_run_id!r} not found in audit chain "
            f"at {audit_path}"
        )

    # Recursive derivation depth — computed BEFORE compute so a missing
    # derived_from cite fails before any work happens.
    derivation_depth = _resolve_derivation_depth(
        audit_path, config.derived_from_run_ids,
    )

    contract = registry_contract(config.target)
    production_methodology = current_methodology()

    # ── Execute parameterized callable ──────────────────────────────
    entry = REGISTERED_PARAMETERIZED_FUNCS[config.target]
    call_kwargs = dict(config.target_inputs)
    # registry_path is a structural input every registered callable
    # accepts. If the config didn't supply one, fall back to the
    # runner's registry_path. The config's own choice wins (lets
    # experiments target a frozen registry snapshot in the future).
    call_kwargs.setdefault("registry_path", str(registry_path))
    call_kwargs.update(config.param_overrides)
    result = entry.callable(**call_kwargs)
    output_payload = entry.serializer(result)

    # ── Build the audit_event + payload ─────────────────────────────
    audit_event = {
        "event_type":                    "experiment_run",
        "target":                        config.target,
        "config_fingerprint":            config.config_fingerprint,
        "methodology_kind":              config.methodology_kind,
        "experiment_status":             config.experiment_status,
        "derivation_depth":              derivation_depth,
        "derived_from_run_ids_count":    len(config.derived_from_run_ids),
        "registry_contract_fingerprint": contract["fingerprint"],
        "schema_version":                EXPERIMENT_AUDIT_SCHEMA_VERSION,
    }

    payload = {
        "config":                            config.to_payload(),
        "output":                            output_payload,
        "production_methodology_versions":   production_methodology,
        "experiment_overrides":              dict(config.param_overrides),
        "derivation_depth":                  derivation_depth,
        "registry_contract":                 contract,
        "baseline_run_id":                   baseline_run_id,
        "schema_version":                    EXPERIMENT_AUDIT_SCHEMA_VERSION,
    }

    return emit_evidence(
        audit_log_path=audit_path,
        evidence_kind="experiment_run",
        audit_event=audit_event,
        payload=payload,
        parent_run_id=baseline_run_id,
    )


def find_experiment_runs(
    audit_path: Optional[Path] = None,
    *,
    target: Optional[str] = None,
    experiment_status: Optional[str] = None,
    methodology_kind: Optional[str] = None,
) -> List[dict]:
    """Scan the audit chain for experiment_run rows, optionally
    filtered by target / status / methodology_kind.

    Returns the list of inner event dicts (not full audit records).
    Linear scan — appropriate for single-machine bounded chain size.
    """
    audit_path = audit_path or DEFAULT_AUDIT_PATH
    if not audit_path.exists():
        return []
    matches: List[dict] = []
    with audit_path.open("r", encoding="utf-8") as h:
        for line in h:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = record.get("event", {})
            if event.get("evidence_kind") != "experiment_run":
                continue
            if target is not None and event.get("target") != target:
                continue
            if (experiment_status is not None
                    and event.get("experiment_status") != experiment_status):
                continue
            if (methodology_kind is not None
                    and event.get("methodology_kind") != methodology_kind):
                continue
            matches.append(event)
    return matches
