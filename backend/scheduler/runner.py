"""Scheduler orchestrator: acquire lock, invoke sub-jobs as fresh
subprocesses in DAG order, emit a single audit-only ``scheduled_run``
event recording the cadence-tick provenance.

## Load-bearing invariants

  1. EACH sub-job runs in a NEW Python subprocess. This is what
     keeps ``_git_head_sha`` (lifetime-cached in provenance.py)
     honest across the cadence — a ``git pull`` between sub-jobs
     is correctly reflected in subsequent evidence.

  2. The scheduler holds a single cross-process exclusive lock
     (via ``backend.investment_analytics._locking``: fcntl.flock on
     POSIX, msvcrt.locking on Windows) on ``data/scheduler/.lock``
     for the duration of the tick. Concurrent invocations refuse
     cleanly with a structured log entry under
     ``data/scheduler/last_lock_conflict.json`` AND a human-readable
     stderr line — visible in scheduler output without polluting
     the audit chain.

  3. A pre-flight ``audit.verify`` is run before the DAG. The chain
     must be in a state ∈ {valid, partial_failure, empty} to start.
     ``invalid``/``unverifiable`` chains cause refusal — the system
     does not pile more events on top of a broken chain.

  4. Refusal vs failure: a sub-job that exits 0 is ``ok``, in its
     declared ``refusal_exit_codes`` is ``refused``, anything else
     is ``failed``. Downstream propagation: ``refused`` is honest
     so descendants still run; ``failed`` makes descendants with
     ``skip_on_dependency_failure=True`` become
     ``skipped_dependency_failed``.

  5. Sequential execution. No parallel sub-jobs. Preserves replay
     determinism + matches the audit append lock discipline.

  6. NO retries. If a sub-job fails, tomorrow's cadence is the
     remediation. Retries hide intermittent failures and risk
     duplicate evidence.

## scheduled_run audit emission

Audit-only event (``evidence_kind=null``) emitted ONCE per
successful lock-acquired cadence tick. NOT emitted when the
scheduler refused to start (lock conflict / chain unverifiable) —
those are visible via the dedicated lock-conflict log / stderr.

The orchestrator records:
  - cadence_id, started_at, completed_at, duration_ms
  - sub_jobs[*]: name, argv, outcome, exit_code, duration_ms,
    emitted_run_ids (chain-delta scan), stderr_tail on failure
  - overall_outcome aggregate
  - audit_chain_valid (post-cadence integrity check)

``parent_run_id`` on the envelope points at the prior
``scheduled_run`` row when one exists, threading orchestration
ticks into a typed chain.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.investment_analytics._locking import (
    acquire_exclusive_nonblocking,
    release as release_exclusive,
)
from backend.investment_analytics.audit import (
    append_audit_record,
    verify_audit_chain_multi,
)
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
from backend.scheduler.dag import get_dag, resolve_argv_templates


# Outcome buckets — kept here so the runner is the single authority
# on how exit codes / exceptions map to typed outcomes.
class _Outcome:
    OK                          = "ok"
    REFUSED                     = "refused"
    FAILED                      = "failed"
    SKIPPED_DEPENDENCY_FAILED   = "skipped_dependency_failed"


# Default invoker — replaced in tests by a deterministic stub. Must
# return ``(exit_code, stderr_tail, duration_ms)``. A ``None`` exit
# code means timeout / subprocess crash — also counts as failed.
InvokerResult = Tuple[Optional[int], str, int]
Invoker = Callable[[Tuple[str, ...], int, Path], InvokerResult]


def _subprocess_invoker(
    argv: Tuple[str, ...], timeout_sec: int, cwd: Path,
) -> InvokerResult:
    """Spawn ``[sys.executable, *argv]`` as a subprocess. Captures
    stderr (returns the last ~2000 chars on failure). Timeout maps
    to (None, "<timeout>", ms)."""
    start = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, *argv],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        duration_ms = int((time.monotonic() - start) * 1000)
        stderr_tail = (result.stderr or "")[-2000:]
        return result.returncode, stderr_tail, duration_ms
    except subprocess.TimeoutExpired:
        duration_ms = int((time.monotonic() - start) * 1000)
        return None, f"timeout after {timeout_sec}s", duration_ms
    except Exception as exc:  # noqa: BLE001
        duration_ms = int((time.monotonic() - start) * 1000)
        return None, f"subprocess error: {exc!r}", duration_ms


# ── Lock discipline ──────────────────────────────────────────────────


def _record_lock_conflict(
    lock_path: Path, conflict_log_path: Path,
) -> Dict[str, Any]:
    """Write a structured lock-conflict record. Operationally visible
    via the file + via stderr (caller emits the stderr line).

    The audit chain is intentionally NOT touched — no orchestration
    happened, so no ``scheduled_run`` row is appropriate. The
    forensic record is a single-writer sidecar file plus stderr."""
    holding_pid_raw: Optional[str] = None
    try:
        if lock_path.exists():
            # read_text on Path applies universal-newlines by
            # default; on Windows that means CRLF→LF transparent
            # normalization. Since the lock file only ever holds a
            # numeric PID + trailing LF, the strip() below catches
            # both forms uniformly.
            holding_pid_raw = lock_path.read_text(encoding="utf-8").strip() or None
    except OSError:
        holding_pid_raw = None
    record: Dict[str, Any] = {
        "schema_version":      "v1",
        "kind":                "scheduler_lock_conflict",
        "requesting_pid":      os.getpid(),
        "holding_pid_in_lock": holding_pid_raw,
        "timestamp":           datetime.now(timezone.utc)
                                       .replace(microsecond=0).isoformat(),
        "lock_path":           str(lock_path),
    }
    conflict_log_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = conflict_log_path.with_suffix(conflict_log_path.suffix + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True),
                   encoding="utf-8")
    tmp.replace(conflict_log_path)
    return record


def _acquire_scheduler_lock(lock_path: Path) -> Optional[int]:
    """Non-blocking cross-process lock acquisition via the typed
    locking adapter. Returns the file descriptor on success or
    ``None`` on contention. Kernel releases the lock if the holding
    process dies, so a crashed scheduler does NOT leave a stuck
    lock — true on POSIX (fcntl) and on Windows (msvcrt).

    Lock semantics: exclusive cross-process, non-blocking (try-
    acquire), maps to fcntl.LOCK_EX | LOCK_NB on POSIX and
    msvcrt.LK_NBLCK on Windows. Failure to acquire is reported as
    ``None`` (the caller's typed ``refused_lock`` outcome), never as
    a soft fallback that would let two schedulers run.

    On Windows ``msvcrt.locking`` locks a byte range at the current
    file position; the adapter seeks to a sparse offset
    (``_LOCK_OFFSET`` in backend.investment_analytics._locking,
    post-A1 `04ed947`) so all processes contend on the same well-
    known byte AND the lock never overlaps real file content. The
    truncate-and-write-PID below seeks back to byte 0 first so the
    forensic PID lands at the start of the file rather than at the
    sparse lock offset; on POSIX the seek is semantically a no-op
    because fcntl.flock ignores byte offsets."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY, 0o644)
    # Wrap the adapter call so a propagating exception from
    # ``acquire_exclusive_nonblocking`` (e.g., a non-EAGAIN OSError
    # such as EBADF / EINTR / ENOLCK) does not leak the open fd.
    # Pre-portability code closed the fd before re-raising; this
    # block restores that exact cleanup discipline.
    try:
        acquired = acquire_exclusive_nonblocking(fd)
    except BaseException:
        os.close(fd)
        raise
    if not acquired:
        os.close(fd)
        return None
    # The locking adapter leaves the file position at _LOCK_OFFSET
    # (a sparse offset on Windows, post-A1 `04ed947`). Seek back to
    # byte 0 BEFORE the PID write so the forensic PID lands at the
    # start of the file rather than at the sparse offset, which on
    # Windows would create a 4-EiB-sized sparse file and confuse
    # shutil.rmtree during test/operator cleanup. POSIX is unaffected:
    # fcntl.flock ignores byte offsets, so the seek is a no-op there
    # in semantic effect.
    os.lseek(fd, 0, os.SEEK_SET)
    # Truncate + write PID for forensic visibility (not used for locking).
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    return fd


def _release_scheduler_lock(fd: int) -> None:
    try:
        release_exclusive(fd)
    finally:
        os.close(fd)


# ── Pre-flight integrity gate ────────────────────────────────────────


def _chain_state_for_pre_flight(audit_path: Path) -> str:
    """Returns the verify_audit_chain_multi overall_status. The
    scheduler refuses to start when status is invalid or
    unverifiable; valid / partial_failure / empty all proceed.
    partial_failure tolerates legacy orphan chains the operator
    already knows about; empty is the day-one state."""
    diag = verify_audit_chain_multi(audit_path.parent)
    return str(diag.get("overall_status", "unverifiable"))


# ── Main orchestrator ────────────────────────────────────────────────


def run_cadence(
    cadence_id: str,
    *,
    audit_path: Optional[Path] = None,
    lock_path: Optional[Path] = None,
    lock_conflict_log_path: Optional[Path] = None,
    today: Optional[date] = None,
    invoker: Optional[Invoker] = None,
    cwd: Optional[Path] = None,
    emit_audit: bool = True,
) -> Dict[str, Any]:
    """Run one cadence tick.

    Returns a dict describing the outcome — same shape as the
    emitted scheduled_run payload PLUS a top-level
    ``scheduler_status`` field for the caller (CLI / tests) to
    branch on:

      scheduler_status: one of
        "completed"             — DAG ran end-to-end; audit row emitted
        "refused_lock"          — lock conflict; NO audit row emitted
        "refused_chain_invalid" — pre-flight chain check failed; NO row
        "refused_unknown_cadence" — bad cadence_id; NO row

    The audit row (when emitted) has parent_run_id pointing at the
    prior scheduled_run, threading cadence ticks into a typed chain.
    """
    audit_path             = audit_path             or DEFAULT_AUDIT_PATH
    lock_path              = lock_path              or DEFAULT_LOCK_PATH
    lock_conflict_log_path = lock_conflict_log_path or DEFAULT_LOCK_CONFLICT_LOG
    today                  = today                  or datetime.now(timezone.utc).date()
    invoker                = invoker                or _subprocess_invoker
    cwd                    = cwd                    or Path.cwd()

    # ── Refuse on unknown cadence (closed registry) ─────────────────
    try:
        dag = get_dag(cadence_id)
    except KeyError as exc:
        sys.stderr.write(f"[scheduler] {exc}\n")
        return {"scheduler_status": "refused_unknown_cadence",
                "cadence_id": cadence_id}

    # ── Refuse on lock contention ────────────────────────────────────
    lock_fd = _acquire_scheduler_lock(lock_path)
    if lock_fd is None:
        record = _record_lock_conflict(lock_path, lock_conflict_log_path)
        sys.stderr.write(
            f"[scheduler] refusing to start: lock already held; "
            f"holding_pid_in_lock={record['holding_pid_in_lock']!r} "
            f"(conflict logged to {lock_conflict_log_path})\n"
        )
        return {"scheduler_status": "refused_lock",
                "cadence_id": cadence_id,
                "conflict_record": record}

    try:
        # ── Pre-flight integrity check ──────────────────────────────
        chain_state = _chain_state_for_pre_flight(audit_path)
        if chain_state in ("invalid", "unverifiable"):
            sys.stderr.write(
                f"[scheduler] refusing to start: audit chain state="
                f"{chain_state!r}; remediate the chain before scheduling\n"
            )
            return {"scheduler_status": "refused_chain_invalid",
                    "cadence_id": cadence_id,
                    "chain_state": chain_state}

        # ── Execute DAG sequentially ────────────────────────────────
        started_at_dt = datetime.now(timezone.utc).replace(microsecond=0)
        started_at = started_at_dt.isoformat()
        tick_start_monotonic = time.monotonic()

        outcomes_by_name: Dict[str, str] = {}
        sub_job_records: List[Dict[str, Any]] = []

        for job in dag:
            resolved_argv = resolve_argv_templates(job.argv, today)
            # Skip due to upstream failure?
            if job.skip_on_dependency_failure and _any_upstream_failed(
                job, outcomes_by_name,
            ):
                outcomes_by_name[job.name] = _Outcome.SKIPPED_DEPENDENCY_FAILED
                sub_job_records.append({
                    "sub_job_name":     job.name,
                    "argv":             list(resolved_argv),
                    "outcome":          _Outcome.SKIPPED_DEPENDENCY_FAILED,
                    "exit_code":        None,
                    "duration_ms":      0,
                    "stderr_tail":      "",
                    "emitted_run_ids":  [],
                })
                continue

            # Record chain length before, scan delta after, to recover
            # any run_ids the sub-job appended.
            before_run_ids = _all_run_ids(audit_path)
            exit_code, stderr_tail, duration_ms = invoker(
                resolved_argv, job.timeout_sec, cwd,
            )
            after_run_ids = _all_run_ids(audit_path)
            emitted = sorted(set(after_run_ids) - set(before_run_ids))

            outcome = _outcome_from_exit(exit_code, job.refusal_exit_codes)
            outcomes_by_name[job.name] = outcome
            sub_job_records.append({
                "sub_job_name":     job.name,
                "argv":             list(resolved_argv),
                "outcome":          outcome,
                "exit_code":        exit_code,
                "duration_ms":      duration_ms,
                "stderr_tail":      stderr_tail if outcome == _Outcome.FAILED else "",
                "emitted_run_ids":  emitted,
            })

        completed_at_dt = datetime.now(timezone.utc).replace(microsecond=0)
        completed_at = completed_at_dt.isoformat()
        duration_ms = int((time.monotonic() - tick_start_monotonic) * 1000)

        # ── Aggregate overall outcome ───────────────────────────────
        outcomes_set = set(outcomes_by_name.values())
        if _Outcome.FAILED in outcomes_set or _Outcome.SKIPPED_DEPENDENCY_FAILED in outcomes_set:
            overall = "any_failed" if _Outcome.FAILED in outcomes_set else "partial"
        elif _Outcome.REFUSED in outcomes_set:
            overall = "partial"
        else:
            overall = "all_ok"

        # ── Post-cadence integrity probe (informational) ────────────
        post_chain_state = _chain_state_for_pre_flight(audit_path)
        chain_valid_after = post_chain_state == "valid"

        # ── Build payload + emit audit-only scheduled_run row ────────
        prior_run_id = _find_latest_scheduled_run_id(audit_path)
        audit_event: Dict[str, Any] = {
            "event_type":             "scheduled_run",
            "cadence_id":             cadence_id,
            "started_at":             started_at,
            "completed_at":           completed_at,
            "duration_ms":            duration_ms,
            "host_machine":           _hostname(),
            "today":                  today.isoformat(),
            "sub_jobs":               sub_job_records,
            "overall_outcome":        overall,
            "audit_chain_valid":      chain_valid_after,
            "chain_state_post":       post_chain_state,
            "schema_version":         SCHEDULED_RUN_SCHEMA_VERSION,
        }

        if emit_audit:
            append_audit_record(
                audit_path,
                audit_event,
                evidence_kind=None,            # audit-only
                parent_run_id=prior_run_id,    # chain orchestration ticks
            )

        return {
            "scheduler_status":  "completed",
            **audit_event,
        }
    finally:
        _release_scheduler_lock(lock_fd)


# ── Helpers ──────────────────────────────────────────────────────────


def _any_upstream_failed(
    job: SubJob, outcomes_by_name: Dict[str, str],
) -> bool:
    """True if any DIRECT or TRANSITIVE dependency has outcome
    ``failed`` or ``skipped_dependency_failed``. ``refused`` is
    honest evidence-bearing behavior and does NOT propagate as
    failure."""
    failure_set = {_Outcome.FAILED, _Outcome.SKIPPED_DEPENDENCY_FAILED}
    return any(
        outcomes_by_name.get(dep) in failure_set for dep in job.depends_on
    )


def _outcome_from_exit(
    exit_code: Optional[int], refusal_exit_codes,
) -> str:
    if exit_code is None:
        return _Outcome.FAILED  # timeout / subprocess crash
    if exit_code == 0:
        return _Outcome.OK
    if exit_code in refusal_exit_codes:
        return _Outcome.REFUSED
    return _Outcome.FAILED


def _all_run_ids(audit_path: Path) -> List[str]:
    """Scan the chain and return every event's run_id. Used to
    compute the delta of new run_ids appended by a sub-job. Cheap
    on bounded single-machine chains; replaceable with an offset-
    based read if needed."""
    if not audit_path.exists():
        return []
    out: List[str] = []
    with audit_path.open("r", encoding="utf-8") as h:
        for line in h:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            run_id = record.get("event", {}).get("run_id")
            if run_id:
                out.append(run_id)
    return out


def _find_latest_scheduled_run_id(audit_path: Path) -> Optional[str]:
    """Walk the chain bottom-up to find the most recent scheduled_run
    event's run_id. Used as parent_run_id for the new emit, threading
    cadence ticks together."""
    if not audit_path.exists():
        return None
    with audit_path.open("r", encoding="utf-8") as h:
        lines = h.readlines()
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        event = record.get("event", {})
        if event.get("event_type") == "scheduled_run":
            run_id = event.get("run_id")
            if run_id:
                return run_id
    return None


def _hostname() -> str:
    try:
        import socket
        return socket.gethostname()
    except Exception:  # noqa: BLE001
        return "unknown"


def find_scheduled_runs(
    audit_path: Optional[Path] = None,
    *,
    cadence_id: Optional[str] = None,
    overall_outcome: Optional[str] = None,
) -> List[dict]:
    """Linear chain scan for scheduled_run audit-event rows. Returns
    inner event dicts (not full audit records)."""
    audit_path = audit_path or DEFAULT_AUDIT_PATH
    if not audit_path.exists():
        return []
    rows: List[dict] = []
    with audit_path.open("r", encoding="utf-8") as h:
        for line in h:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = record.get("event", {})
            if event.get("event_type") != "scheduled_run":
                continue
            if cadence_id and event.get("cadence_id") != cadence_id:
                continue
            if overall_outcome and event.get("overall_outcome") != overall_outcome:
                continue
            rows.append(event)
    return rows
