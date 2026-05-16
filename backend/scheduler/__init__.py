"""Scheduler — Step 12.

Deterministic orchestration of the evidence-producing CLIs from a
single launchd/cron entry. The scheduler:

  - acquires a non-blocking cross-process lock at
    ``data/scheduler/.lock`` (fcntl on POSIX, msvcrt on Windows, via
    ``backend.investment_analytics._locking``)
  - performs a pre-flight audit chain integrity check
  - invokes each sub-job in DAG order as a FRESH Python subprocess
  - aggregates per-sub-job outcomes (ok / refused / failed /
    skipped_dependency_failed)
  - emits a single audit-only ``scheduled_run`` event recording
    the cadence-tick provenance
  - releases the lock

## Load-bearing discipline

  - subprocess-per-sub-job, NEVER in-process invocation
    (keeps ``_git_head_sha`` honest across the cadence)
  - sequential execution, NEVER parallel
    (preserves replay determinism + matches audit append lock)
  - NO retries
    (failures become tomorrow's refusal; retries hide drift)
  - audit-only event, NEVER a new EVIDENCE_KIND
    (orchestration metadata, not new evidence)
  - lock conflicts logged to a dedicated sidecar + stderr,
    NEVER to the audit chain (no orchestration happened)
  - NEVER mutates production thresholds, methodology versions,
    or any source-of-truth config file

## Scope (day-one)

ONE cadence: ``daily_weekday_evening``
SUB-JOBS:
  watchlist_run
    → regimes_emit_recent_window  (depends_on: watchlist_run)
        → calibration_correlation  (depends_on: regimes_emit_recent_window)
  audit_verify  (always runs, regardless of upstream)
"""
from __future__ import annotations

from backend.scheduler.config import (
    DEFAULT_AUDIT_PATH,
    DEFAULT_LOCK_CONFLICT_LOG,
    DEFAULT_LOCK_PATH,
    DEFAULT_SCHEDULER_DIR,
    SCHEDULED_RUN_OVERALL_OUTCOMES,
    SCHEDULED_RUN_SCHEMA_VERSION,
    SUB_JOB_OUTCOMES,
    SubJob,
)
from backend.scheduler.dag import (
    CADENCE_REGISTRY,
    DAILY_WEEKDAY_EVENING_DAG,
    get_dag,
    resolve_argv_templates,
)
from backend.scheduler.runner import (
    find_scheduled_runs,
    run_cadence,
)

__all__ = [
    "CADENCE_REGISTRY",
    "DAILY_WEEKDAY_EVENING_DAG",
    "DEFAULT_AUDIT_PATH",
    "DEFAULT_LOCK_CONFLICT_LOG",
    "DEFAULT_LOCK_PATH",
    "DEFAULT_SCHEDULER_DIR",
    "SCHEDULED_RUN_OVERALL_OUTCOMES",
    "SCHEDULED_RUN_SCHEMA_VERSION",
    "SUB_JOB_OUTCOMES",
    "SubJob",
    "find_scheduled_runs",
    "get_dag",
    "resolve_argv_templates",
    "run_cadence",
]
