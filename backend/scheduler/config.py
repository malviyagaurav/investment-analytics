"""Scheduler configuration: SubJob declaration + cadence registry +
filesystem paths.

The day-one scheduler runs ONE cadence (``daily_weekday_evening``)
at the existing weekday 21:30 IST slot, replacing the watchlist
launchd/cron entry. Sub-jobs are sequential, each in a fresh
subprocess (preserves ``_git_head_sha`` freshness per sub-job, the
single biggest constraint surfaced by the Step 12 audit).

Closed concepts live here so other modules don't drift:
  - SUB_JOB_OUTCOMES (closed enum used in scheduled_run payloads)
  - SCHEDULED_RUN_OVERALL_OUTCOMES (closed enum, aggregate)
  - CADENCES (closed set of supported cadence IDs)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet, Tuple


ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_AUDIT_PATH        = ROOT / "data" / "audit" / "audit.jsonl"
DEFAULT_REGISTRY_PATH     = ROOT / "data" / "registry" / "schemes.json"
DEFAULT_CACHE_DIR         = ROOT / "data" / "cache"
DEFAULT_SCHEDULER_DIR     = ROOT / "data" / "scheduler"
DEFAULT_LOCK_PATH         = DEFAULT_SCHEDULER_DIR / ".lock"
DEFAULT_LOCK_CONFLICT_LOG = DEFAULT_SCHEDULER_DIR / "last_lock_conflict.json"


SCHEDULED_RUN_SCHEMA_VERSION = "v1"


# Closed enum: per-sub-job outcome categories surfaced in the
# scheduled_run audit row. ``refused`` is distinct from ``failed``
# — a sub-job that exits with a typed refusal exit code (calibration
# returning insufficient_substrate, regimes returning indeterminate)
# is honest evidence-bearing behavior, not a failure.
SUB_JOB_OUTCOMES: FrozenSet[str] = frozenset({
    "ok",
    "refused",
    "failed",
    "skipped_dependency_failed",
    "skipped_lock_pre_flight",
})


# Aggregate outcome for the whole scheduled_run row.
#   all_ok     — every sub-job ok
#   partial    — at least one refused or skipped; nothing failed
#   any_failed — at least one failed
SCHEDULED_RUN_OVERALL_OUTCOMES: FrozenSet[str] = frozenset({
    "all_ok",
    "partial",
    "any_failed",
})


# Closed set of cadence IDs. Adding a cadence is a code-level
# extension that lands a new entry here + a new DAG in dag.py.
# We intentionally do NOT support free-text or operator-supplied
# cadence ids — every recurring schedule the system honors is
# defined in code under governance review.
CADENCES: FrozenSet[str] = frozenset({
    "daily_weekday_evening",
})


@dataclass(frozen=True)
class SubJob:
    """Declaration of one sub-job within a cadence DAG.

    Fields:
      name: stable identifier surfaced in scheduled_run.sub_jobs[*].name.
            Must be unique within the DAG.

      argv: arguments passed to a fresh Python subprocess
            (``[sys.executable, *argv]``). Tokens of the form
            ``"<placeholder>"`` are substituted at invoke time from
            the cadence's template context (see runner.py).

      depends_on: names of sub-jobs that must succeed (outcome=ok)
            OR cleanly refuse (outcome=refused) BEFORE this job
            runs. An upstream ``failed`` (or its own
            ``skipped_dependency_failed``) propagates through any
            transitive descendant that has
            ``skip_on_dependency_failure=True``.

      skip_on_dependency_failure: when True, this sub-job becomes
            ``skipped_dependency_failed`` if any upstream is
            ``failed`` / ``skipped_dependency_failed``. When False,
            this sub-job runs regardless (used by audit.verify —
            integrity check should never be skipped by upstream
            failure).

      timeout_sec: hard timeout on the subprocess. Exceeding it
            counts as ``failed`` and is recorded with that outcome.

      refusal_exit_codes: NON-ZERO exit codes that the runner treats
            as ``refused`` (typed honest refusal) rather than
            ``failed``. Exit code 0 is ALWAYS ``ok``; everything
            else not in this set is ``failed``. Default empty —
            sub-jobs that only know success/failure don't need to
            declare anything.
    """
    name:                        str
    argv:                        Tuple[str, ...]
    depends_on:                  Tuple[str, ...] = ()
    skip_on_dependency_failure:  bool = True
    timeout_sec:                 int = 600
    refusal_exit_codes:          FrozenSet[int] = field(default_factory=frozenset)
