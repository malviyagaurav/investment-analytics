"""Experiment framework — Step 9.

Activates the ``experiment_run`` evidence_kind (reserved in EVIDENCE_KINDS
since Step 3) so that parameter variations on production computations
can be:

  - frozen and content-addressed (ExperimentConfig.config_fingerprint),
  - lineage-linked via parent_run_id + derived_from_run_ids,
  - replayed via Step 7's replay machinery,
  - and emitted as first-class evidence (never silently mutating
    production state).

Phase 2 anchor: this layer answers the question "can the system
*justify* believing what it said?" — the substrate on which Steps
10 (regime detection), 11 (calibration), 14 (reliability scoring),
and 15 (controlled self-improvement) are built.

## Anti-evidence-laundering controls (load-bearing)

  - ``methodology_kind`` distinguishes engineered_variant (parameters
    chosen by reasoning) from data_driven_variant (parameters derived
    from a prior measurement). The latter REQUIRES a non-empty
    derived_from_run_ids list — caught at config construction.
  - ``experiment_status`` distinguishes exploratory tuning from
    validation / shadow_candidate / calibration_candidate so promotion
    gating (Step 15) can refuse to promote an exploratory experiment.
  - ``derivation_depth`` is computed deterministically by the runner
    so recursive evidentiary promotion has cheap visibility.
  - ``registry_contract_fingerprint`` (with full contract components)
    is recorded at experiment time so replay can surface WHAT changed
    if the parameterized surface drifts.
  - ``production_methodology_versions`` is recorded separately from
    ``experiment_overrides`` so future replay / drift tooling can
    cleanly distinguish "what production believed" from "what this
    experiment temporarily varied."
  - ``non_semantic_metadata`` (rationale, etc.) is explicitly excluded
    from config_fingerprint, replay equality, and any downstream
    reasoning. Free text never silently encodes assumptions.

## Scope (day-one, Step 9)

ONE registered parameterized function: ``rank_category`` with two
overridable parameters (``MIN_ALIGNED_POINTS``, ``ROLLING_WINDOW_DAYS``).
Demonstrates the full chain end-to-end. Adding more registered
functions / parameters is mechanical follow-up; the spine is what
Step 9 establishes.
"""
from __future__ import annotations

from backend.experiments.config import (
    EXPERIMENT_STATUSES,
    METHODOLOGY_KINDS,
    ExperimentConfig,
)
from backend.experiments.registry import (
    REGISTERED_PARAMETERIZED_FUNCS,
    registry_contract,
)
from backend.experiments.runner import (
    find_experiment_runs,
    run_experiment,
)

__all__ = [
    "EXPERIMENT_STATUSES",
    "METHODOLOGY_KINDS",
    "ExperimentConfig",
    "REGISTERED_PARAMETERIZED_FUNCS",
    "registry_contract",
    "run_experiment",
    "find_experiment_runs",
]
