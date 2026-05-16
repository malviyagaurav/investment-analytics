"""Day-one execution DAG: the ``daily_weekday_evening`` cadence.

Hardcoded sequential pipeline. New sub-jobs are added by extending
this file under code review (CLOSED registry, NOT operator
configurable).

## Day-one DAG

  watchlist_run
    └── regimes_emit_recent_window  (depends on: watchlist_run)
          └── calibration_correlation  (depends on: regimes_emit_recent_window)

  audit_verify  (always runs — depends_on=[]; final integrity gate)

## Deliberate deferrals (Step 12 audit findings)

  - regimes.drift  — requires ≥ 2 regime_summary rows; live chain
    currently has zero. Adding it before substrate exists would
    emit "insufficient_substrate"-style refusals every day.
  - experiments.run — no calibration_candidate experiments are
    registered yet.
  - Periodic replay audits — separate cadence concern. The integrity
    gate audit.verify covers chain validity; targeted replay is a
    future cadence (e.g., weekly random replay of N evidence rows).

## Template substitution

Argv tokens of the form ``"<placeholder>"`` are substituted at
invoke time from the cadence's template context. Today's templates:

  <yesterday>          — yesterday's date (UTC), YYYY-MM-DD
  <rolling_1y_start>   — 365 days before <yesterday>, YYYY-MM-DD

The orchestrator (runner.py) resolves these BEFORE invoking the
subprocess. Sub-jobs receive fully-resolved argv — they never see
placeholders.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Callable, Dict, List, Tuple

from backend.scheduler.config import SubJob


CALIBRATION_TARGET_HIGH_CORRELATION = (
    "portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD"
)


# Tokens substituted in argv at invoke time. Each template is a
# pure function of "today" (a date object) → string. Today is
# computed once per cadence tick; all sub-jobs in the tick see
# the same date context.
TEMPLATE_RESOLVERS: Dict[str, Callable[[date], str]] = {
    "<yesterday>":        lambda today: (today - timedelta(days=1)).isoformat(),
    "<rolling_1y_start>": lambda today: (today - timedelta(days=365)).isoformat(),
}


DAILY_WEEKDAY_EVENING_DAG: Tuple[SubJob, ...] = (
    SubJob(
        name="watchlist_run",
        argv=("-m", "backend.jobs.watchlist", "run"),
        depends_on=(),
        skip_on_dependency_failure=False,
        timeout_sec=900,
        refusal_exit_codes=frozenset(),
    ),
    SubJob(
        name="regimes_emit_recent_window",
        argv=(
            "-m", "backend.regimes", "emit",
            "--start", "<rolling_1y_start>",
            "--end",   "<yesterday>",
        ),
        depends_on=("watchlist_run",),
        skip_on_dependency_failure=True,
        timeout_sec=180,
        # `regimes emit` exits 0 on success. The `regimes classify`
        # subcommand exits 2 for indeterminate, but `emit` itself
        # always exits 0 if the row is appended. Empty refusal set.
        refusal_exit_codes=frozenset(),
    ),
    SubJob(
        name="calibration_correlation",
        argv=(
            "-m", "backend.calibration", "calibrate",
            "--target", CALIBRATION_TARGET_HIGH_CORRELATION,
        ),
        depends_on=("regimes_emit_recent_window",),
        skip_on_dependency_failure=True,
        timeout_sec=600,
        # `calibrate` exits 0 for a recommendation, 2 for any typed
        # refusal. Both are honest evidence-bearing outcomes; the
        # day-one chain has no substrate so 2 is the expected exit.
        refusal_exit_codes=frozenset({2}),
    ),
    SubJob(
        name="audit_verify",
        argv=("-m", "backend.audit", "verify"),
        # depends_on=() AND skip_on_dependency_failure=False →
        # always runs as the final integrity gate, regardless of
        # what happened upstream.
        depends_on=(),
        skip_on_dependency_failure=False,
        timeout_sec=60,
        # `audit verify` exits 0 valid, 2 partial/empty, 3 invalid/
        # unverifiable. 2 is operationally noteworthy but not a
        # scheduler failure — the chain may have legacy orphans.
        # 3 IS a failure (chain integrity broken).
        refusal_exit_codes=frozenset({2}),
    ),
)


CADENCE_REGISTRY: Dict[str, Tuple[SubJob, ...]] = {
    "daily_weekday_evening": DAILY_WEEKDAY_EVENING_DAG,
}


# ── Validation (runs at module import time) ─────────────────────────


def _validate_dag(name: str, dag: Tuple[SubJob, ...]) -> None:
    """Ensure (a) names are unique, (b) every depends_on cites an
    existing name AND (c) no cycles. The DAG is small and hardcoded;
    these checks are cheap and catch typos at import time rather
    than during a cadence tick where they would be much more
    expensive to surface."""
    names = [j.name for j in dag]
    if len(names) != len(set(names)):
        raise ValueError(
            f"cadence {name!r} has duplicate sub-job names: {names}"
        )
    name_set = set(names)
    for job in dag:
        for dep in job.depends_on:
            if dep not in name_set:
                raise ValueError(
                    f"cadence {name!r} sub-job {job.name!r} cites "
                    f"unknown dependency {dep!r}"
                )
    # Cycle detection — simple BFS from each node.
    by_name = {j.name: j for j in dag}
    for start in names:
        visited = set()
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for dep in by_name[cur].depends_on:
                if dep == start:
                    raise ValueError(
                        f"cadence {name!r} contains a cycle involving {start!r}"
                    )
                stack.append(dep)


for _cadence_name, _dag in CADENCE_REGISTRY.items():
    _validate_dag(_cadence_name, _dag)


# ── Helpers ──────────────────────────────────────────────────────────


def resolve_argv_templates(
    argv: Tuple[str, ...], today: date,
) -> Tuple[str, ...]:
    """Substitute ``"<placeholder>"`` tokens in argv. Tokens not in
    TEMPLATE_RESOLVERS pass through unchanged — sub-jobs are allowed
    to have positional/option args that aren't templated."""
    return tuple(
        TEMPLATE_RESOLVERS[t](today) if t in TEMPLATE_RESOLVERS else t
        for t in argv
    )


def get_dag(cadence_id: str) -> Tuple[SubJob, ...]:
    if cadence_id not in CADENCE_REGISTRY:
        raise KeyError(
            f"unknown cadence_id: {cadence_id!r}. "
            f"Registered: {sorted(CADENCE_REGISTRY)}"
        )
    return CADENCE_REGISTRY[cadence_id]
