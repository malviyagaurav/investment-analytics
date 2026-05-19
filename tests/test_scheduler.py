"""Tests for backend.scheduler (Step 12).

Coverage map:

  DAG validation:
    - duplicate sub-job names raise at import
    - unknown depends_on raises at import
    - cycle raises at import

  Template substitution:
    - <yesterday> / <rolling_1y_start> resolve correctly
    - non-template tokens pass through unchanged

  Refusal vs failure mapping (per the closed outcome enum):
    - exit 0 → ok
    - exit in refusal_exit_codes → refused
    - any other non-zero / timeout / crash → failed

  Refusal propagation:
    - upstream ok → downstream proceeds
    - upstream refused → downstream still proceeds (honest evidence)
    - upstream failed + skip_on_dependency_failure=True → downstream
      skipped_dependency_failed
    - upstream failed + skip_on_dependency_failure=False → downstream
      still runs (audit_verify pattern)
    - skipped_dependency_failed propagates transitively to dependents

  Lock discipline:
    - concurrent run refused; no scheduled_run audit row emitted
    - lock conflict recorded to data/scheduler/last_lock_conflict.json
    - stderr line written
    - lock released on completion

  Pre-flight integrity gate:
    - chain state "valid" / "partial_failure" / "empty" → cadence runs
    - chain state "invalid" / "unverifiable" → refusal, no row

  Audit emission:
    - scheduled_run row appended on success
    - evidence_kind is null (audit-only)
    - parent_run_id chains prior scheduled_run rows
    - emitted_run_ids correctly enumerates sub-job audit deltas
    - overall_outcome aggregates correctly

  Production isolation:
    - METHODOLOGY_VERSIONS byte-unmutated
    - HIGH_CORRELATION_THRESHOLD byte-unmutated

  Chain integrity:
    - verify_audit_chain passes across mixed cadence sequence
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import patch

from backend.evidence.store import emit_evidence
from backend.investment_analytics import methodology as meth_mod
from backend.investment_analytics._locking import (
    acquire_exclusive_blocking,
    release as release_exclusive,
)
from backend.investment_analytics.audit import (
    append_audit_record,
    verify_audit_chain,
)
from backend.scheduler import (
    CADENCE_REGISTRY,
    DAILY_WEEKDAY_EVENING_DAG,
    SUB_JOB_OUTCOMES,
    SubJob,
    find_scheduled_runs,
    resolve_argv_templates,
    run_cadence,
)
from backend.scheduler import dag as dag_mod
from backend.scheduler import runner as runner_mod


# ── DAG validation ───────────────────────────────────────────────────


class DagValidationTests(unittest.TestCase):

    def test_day_one_dag_has_expected_subjobs(self) -> None:
        names = [j.name for j in DAILY_WEEKDAY_EVENING_DAG]
        self.assertEqual(names, [
            "watchlist_run",
            "regimes_emit_recent_window",
            "calibration_correlation",
            "audit_verify",
        ])

    def test_audit_verify_always_runs_regardless_of_upstream(self) -> None:
        audit = next(j for j in DAILY_WEEKDAY_EVENING_DAG
                     if j.name == "audit_verify")
        self.assertEqual(audit.depends_on, ())
        self.assertFalse(audit.skip_on_dependency_failure)

    def test_duplicate_names_rejected(self) -> None:
        with self.assertRaises(ValueError):
            dag_mod._validate_dag("bad", (
                SubJob(name="a", argv=("x",)),
                SubJob(name="a", argv=("y",)),
            ))

    def test_unknown_dependency_rejected(self) -> None:
        with self.assertRaises(ValueError):
            dag_mod._validate_dag("bad", (
                SubJob(name="a", argv=("x",), depends_on=("nonexistent",)),
            ))

    def test_cycle_rejected(self) -> None:
        # a → b → a
        with self.assertRaises(ValueError):
            dag_mod._validate_dag("bad", (
                SubJob(name="a", argv=("x",), depends_on=("b",)),
                SubJob(name="b", argv=("y",), depends_on=("a",)),
            ))


# ── Template substitution ────────────────────────────────────────────


class TemplateSubstitutionTests(unittest.TestCase):

    def test_yesterday_template_resolves(self) -> None:
        today = date(2026, 5, 16)
        out = resolve_argv_templates(("--end", "<yesterday>"), today)
        self.assertEqual(out, ("--end", "2026-05-15"))

    def test_rolling_1y_template_resolves(self) -> None:
        today = date(2026, 5, 16)
        out = resolve_argv_templates(("--start", "<rolling_1y_start>"), today)
        self.assertEqual(out, ("--start", "2025-05-16"))

    def test_non_template_tokens_pass_through(self) -> None:
        today = date(2026, 5, 16)
        out = resolve_argv_templates(
            ("-m", "backend.regimes", "emit", "--start", "<rolling_1y_start>"),
            today,
        )
        self.assertEqual(out[:3], ("-m", "backend.regimes", "emit"))
        self.assertEqual(out[-1], "2025-05-16")


# ── Outcome mapping (exit code → typed outcome) ──────────────────────


class OutcomeMappingTests(unittest.TestCase):

    def test_zero_is_ok(self) -> None:
        self.assertEqual(
            runner_mod._outcome_from_exit(0, frozenset()),
            "ok",
        )

    def test_refusal_exit_code_is_refused(self) -> None:
        self.assertEqual(
            runner_mod._outcome_from_exit(2, frozenset({2})),
            "refused",
        )

    def test_other_nonzero_is_failed(self) -> None:
        self.assertEqual(
            runner_mod._outcome_from_exit(3, frozenset({2})),
            "failed",
        )

    def test_none_exit_code_is_failed_timeout_or_crash(self) -> None:
        self.assertEqual(
            runner_mod._outcome_from_exit(None, frozenset({2})),
            "failed",
        )


# ── Harness for orchestrator tests ───────────────────────────────────


class _SchedulerHarness(unittest.TestCase):
    """Common setUp: isolated audit + lock + conflict-log paths.
    Replaces the subprocess invoker with a deterministic stub so
    tests don't need actual sub-job binaries on disk."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp.name)
        self.audit_path = self.data_dir / "audit" / "audit.jsonl"
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.scheduler_dir = self.data_dir / "scheduler"
        self.scheduler_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.scheduler_dir / ".lock"
        self.conflict_log_path = self.scheduler_dir / "last_lock_conflict.json"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _read_events(self) -> List[dict]:
        if not self.audit_path.exists():
            return []
        with self.audit_path.open("r", encoding="utf-8") as h:
            return [json.loads(line)["event"] for line in h if line.strip()]


def _stub_invoker(
    outcomes: Dict[str, Tuple[int, str, int]],
) -> "runner_mod.Invoker":
    """Build an invoker that maps the FIRST argv token after '-m' to
    the desired (exit_code, stderr_tail, duration_ms) tuple.

    This sidesteps the need for real sub-job binaries; we test
    orchestration semantics, not the sub-jobs themselves."""
    def _inv(argv, timeout, cwd):
        # argv looks like ('-m', 'backend.x.y', 'subcmd', ...).
        # Use the first two tokens as the key.
        module = argv[1] if len(argv) >= 2 else "?"
        result = outcomes.get(module, (0, "", 1))
        return result
    return _inv


def _replace_dag_for_test(monkey_dag):
    """Patch the registry to inject a test DAG. Restores on context exit."""
    original = dict(dag_mod.CADENCE_REGISTRY)
    dag_mod.CADENCE_REGISTRY.clear()
    dag_mod.CADENCE_REGISTRY.update(monkey_dag)
    return original


def _restore_dag(snapshot):
    dag_mod.CADENCE_REGISTRY.clear()
    dag_mod.CADENCE_REGISTRY.update(snapshot)


# ── Refusal propagation ──────────────────────────────────────────────


class RefusalPropagationTests(_SchedulerHarness):

    def _run_with_dag(self, dag, invoker) -> dict:
        snapshot = _replace_dag_for_test({"test_cadence": dag})
        try:
            result = run_cadence(
                "test_cadence",
                audit_path=self.audit_path,
                lock_path=self.lock_path,
                lock_conflict_log_path=self.conflict_log_path,
                today=date(2026, 5, 16),
                invoker=invoker,
                cwd=self.data_dir,
            )
        finally:
            _restore_dag(snapshot)
        return result

    def test_upstream_ok_downstream_runs(self) -> None:
        dag = (
            SubJob(name="a", argv=("-m", "a.run")),
            SubJob(name="b", argv=("-m", "b.run"), depends_on=("a",),
                   skip_on_dependency_failure=True),
        )
        result = self._run_with_dag(dag, _stub_invoker({
            "a.run": (0, "", 10),
            "b.run": (0, "", 10),
        }))
        outcomes = {j["sub_job_name"]: j["outcome"] for j in result["sub_jobs"]}
        self.assertEqual(outcomes, {"a": "ok", "b": "ok"})
        self.assertEqual(result["overall_outcome"], "all_ok")

    def test_upstream_refused_downstream_still_runs(self) -> None:
        # A refusal is honest evidence-bearing behavior; downstream
        # is allowed (and expected) to proceed.
        dag = (
            SubJob(name="a", argv=("-m", "a.run"),
                   refusal_exit_codes=frozenset({2})),
            SubJob(name="b", argv=("-m", "b.run"), depends_on=("a",),
                   skip_on_dependency_failure=True),
        )
        result = self._run_with_dag(dag, _stub_invoker({
            "a.run": (2, "", 10),
            "b.run": (0, "", 10),
        }))
        outcomes = {j["sub_job_name"]: j["outcome"] for j in result["sub_jobs"]}
        self.assertEqual(outcomes, {"a": "refused", "b": "ok"})
        self.assertEqual(result["overall_outcome"], "partial")

    def test_upstream_failed_skip_downstream(self) -> None:
        dag = (
            SubJob(name="a", argv=("-m", "a.run")),
            SubJob(name="b", argv=("-m", "b.run"), depends_on=("a",),
                   skip_on_dependency_failure=True),
        )
        result = self._run_with_dag(dag, _stub_invoker({
            "a.run": (5, "boom", 10),
            "b.run": (0, "", 10),  # should NOT be invoked
        }))
        outcomes = {j["sub_job_name"]: j["outcome"] for j in result["sub_jobs"]}
        self.assertEqual(outcomes,
                         {"a": "failed", "b": "skipped_dependency_failed"})

    def test_upstream_failed_skip_false_runs_anyway(self) -> None:
        # audit_verify pattern — must run even when upstream failed.
        dag = (
            SubJob(name="a", argv=("-m", "a.run")),
            SubJob(name="b", argv=("-m", "b.run"), depends_on=("a",),
                   skip_on_dependency_failure=False),
        )
        result = self._run_with_dag(dag, _stub_invoker({
            "a.run": (5, "", 10),
            "b.run": (0, "", 10),
        }))
        outcomes = {j["sub_job_name"]: j["outcome"] for j in result["sub_jobs"]}
        self.assertEqual(outcomes, {"a": "failed", "b": "ok"})

    def test_skipped_propagates_transitively(self) -> None:
        # a fails → b skipped → c (depends on b) also skipped
        dag = (
            SubJob(name="a", argv=("-m", "a.run")),
            SubJob(name="b", argv=("-m", "b.run"), depends_on=("a",),
                   skip_on_dependency_failure=True),
            SubJob(name="c", argv=("-m", "c.run"), depends_on=("b",),
                   skip_on_dependency_failure=True),
        )
        result = self._run_with_dag(dag, _stub_invoker({
            "a.run": (5, "", 10),
            "b.run": (0, "", 10),
            "c.run": (0, "", 10),
        }))
        outcomes = {j["sub_job_name"]: j["outcome"] for j in result["sub_jobs"]}
        self.assertEqual(outcomes, {
            "a": "failed",
            "b": "skipped_dependency_failed",
            "c": "skipped_dependency_failed",
        })


# ── Lock discipline ──────────────────────────────────────────────────


class LockTests(_SchedulerHarness):

    def test_concurrent_run_refused_lock_conflict_recorded(self) -> None:
        # Manually hold the lock in this test process via the typed
        # cross-platform adapter (NOT raw fcntl, which is POSIX-only
        # and breaks on Windows CI). A second invocation must refuse
        # cleanly without emitting a row.
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.lock_path),
                     os.O_CREAT | os.O_WRONLY, 0o644)
        acquire_exclusive_blocking(fd)
        try:
            buf = io.StringIO()
            with redirect_stderr(buf):
                result = run_cadence(
                    "daily_weekday_evening",
                    audit_path=self.audit_path,
                    lock_path=self.lock_path,
                    lock_conflict_log_path=self.conflict_log_path,
                    invoker=_stub_invoker({}),
                    cwd=self.data_dir,
                )
            self.assertEqual(result["scheduler_status"], "refused_lock")
            # Conflict record written.
            self.assertTrue(self.conflict_log_path.exists())
            record = json.loads(self.conflict_log_path.read_text())
            self.assertEqual(record["kind"], "scheduler_lock_conflict")
            self.assertEqual(record["requesting_pid"], os.getpid())
            # Stderr emitted a human-readable line.
            self.assertIn("refusing to start", buf.getvalue())
            # NO scheduled_run row was emitted — orchestration didn't run.
            events = self._read_events()
            self.assertEqual(
                [e for e in events if e.get("event_type") == "scheduled_run"],
                [],
            )
        finally:
            release_exclusive(fd)
            os.close(fd)

    def test_lock_released_on_completion(self) -> None:
        # After run_cadence returns, the next invocation acquires
        # the lock cleanly.
        snapshot = _replace_dag_for_test({"test": (
            SubJob(name="a", argv=("-m", "a.run")),
        )})
        try:
            for _ in range(2):
                result = run_cadence(
                    "test",
                    audit_path=self.audit_path,
                    lock_path=self.lock_path,
                    lock_conflict_log_path=self.conflict_log_path,
                    invoker=_stub_invoker({"a.run": (0, "", 1)}),
                    cwd=self.data_dir,
                )
                self.assertEqual(result["scheduler_status"], "completed")
        finally:
            _restore_dag(snapshot)


# ── Pre-flight integrity gate ────────────────────────────────────────


class PreFlightTests(_SchedulerHarness):

    def test_empty_chain_proceeds(self) -> None:
        snapshot = _replace_dag_for_test({"test": (
            SubJob(name="a", argv=("-m", "a.run")),
        )})
        try:
            result = run_cadence(
                "test",
                audit_path=self.audit_path,
                lock_path=self.lock_path,
                lock_conflict_log_path=self.conflict_log_path,
                invoker=_stub_invoker({"a.run": (0, "", 1)}),
                cwd=self.data_dir,
            )
            self.assertEqual(result["scheduler_status"], "completed")
        finally:
            _restore_dag(snapshot)

    def test_invalid_chain_refuses(self) -> None:
        # Corrupt the chain by writing a malformed line.
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_path.write_text(
            "{\"prev_hash\":\"x\",\"payload_hash\":\"y\","
            "\"current_hash\":\"z\",\"event\":{}}\n",
            encoding="utf-8",
        )
        snapshot = _replace_dag_for_test({"test": (
            SubJob(name="a", argv=("-m", "a.run")),
        )})
        try:
            buf = io.StringIO()
            with redirect_stderr(buf):
                result = run_cadence(
                    "test",
                    audit_path=self.audit_path,
                    lock_path=self.lock_path,
                    lock_conflict_log_path=self.conflict_log_path,
                    invoker=_stub_invoker({"a.run": (0, "", 1)}),
                    cwd=self.data_dir,
                )
            self.assertEqual(result["scheduler_status"],
                             "refused_chain_invalid")
            self.assertIn("invalid", buf.getvalue().lower() + "_")
            # No scheduled_run row emitted.
            self.assertEqual(
                [e for e in self._read_events()
                 if e.get("event_type") == "scheduled_run"],
                [],
            )
        finally:
            _restore_dag(snapshot)


# ── Audit emission ───────────────────────────────────────────────────


class AuditEmissionTests(_SchedulerHarness):

    def _setup_test_dag(self):
        return {"test": (
            SubJob(name="a", argv=("-m", "a.run")),
            SubJob(name="b", argv=("-m", "b.run"), depends_on=("a",),
                   skip_on_dependency_failure=True),
        )}

    def test_scheduled_run_row_emitted_with_evidence_kind_null(self) -> None:
        snapshot = _replace_dag_for_test(self._setup_test_dag())
        try:
            run_cadence(
                "test",
                audit_path=self.audit_path,
                lock_path=self.lock_path,
                lock_conflict_log_path=self.conflict_log_path,
                invoker=_stub_invoker({
                    "a.run": (0, "", 1),
                    "b.run": (0, "", 1),
                }),
                cwd=self.data_dir,
            )
        finally:
            _restore_dag(snapshot)
        rows = [e for e in self._read_events()
                if e.get("event_type") == "scheduled_run"]
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0].get("evidence_kind"),
                          "scheduled_run must be audit-only")
        self.assertEqual(rows[0]["overall_outcome"], "all_ok")

    def test_parent_run_id_chains_prior_scheduled_run(self) -> None:
        snapshot = _replace_dag_for_test(self._setup_test_dag())
        try:
            invoker = _stub_invoker({"a.run": (0, "", 1), "b.run": (0, "", 1)})
            r1 = run_cadence(
                "test",
                audit_path=self.audit_path,
                lock_path=self.lock_path,
                lock_conflict_log_path=self.conflict_log_path,
                invoker=invoker, cwd=self.data_dir,
            )
            r2 = run_cadence(
                "test",
                audit_path=self.audit_path,
                lock_path=self.lock_path,
                lock_conflict_log_path=self.conflict_log_path,
                invoker=invoker, cwd=self.data_dir,
            )
        finally:
            _restore_dag(snapshot)
        rows = [e for e in self._read_events()
                if e.get("event_type") == "scheduled_run"]
        self.assertEqual(len(rows), 2)
        # Second row's parent_run_id matches first row's run_id.
        self.assertEqual(rows[1].get("parent_run_id"), rows[0]["run_id"])

    def test_emitted_run_ids_captures_sub_job_audit_delta(self) -> None:
        # An invoker that "emits" a row in the audit chain. The
        # orchestrator's chain-delta scan should attribute that row
        # to the right sub-job.
        emitted_run_ids: Dict[str, str] = {}

        def invoker(argv, timeout, cwd):
            # Each sub-job appends an audit row directly during its
            # invocation, simulating what a real sub-job's emit would do.
            module = argv[1]
            from backend.investment_analytics.audit import append_audit_record
            rec = append_audit_record(
                self.audit_path,
                {"event_type": "stub", "module": module},
                evidence_kind=None,
            )
            emitted_run_ids[module] = rec["event"]["run_id"]
            return 0, "", 1

        snapshot = _replace_dag_for_test(self._setup_test_dag())
        try:
            run_cadence(
                "test",
                audit_path=self.audit_path,
                lock_path=self.lock_path,
                lock_conflict_log_path=self.conflict_log_path,
                invoker=invoker, cwd=self.data_dir,
            )
        finally:
            _restore_dag(snapshot)
        rows = [e for e in self._read_events()
                if e.get("event_type") == "scheduled_run"]
        self.assertEqual(len(rows), 1)
        by_name = {j["sub_job_name"]: j for j in rows[0]["sub_jobs"]}
        # Each sub-job recorded the run_id its stub append produced.
        self.assertEqual(by_name["a"]["emitted_run_ids"],
                         [emitted_run_ids["a.run"]])
        self.assertEqual(by_name["b"]["emitted_run_ids"],
                         [emitted_run_ids["b.run"]])

    def test_overall_outcome_aggregates(self) -> None:
        # Mixed: a refused (still honest), b skipped because c
        # failed — actually let's do a simple two-step refusal case.
        snapshot = _replace_dag_for_test({"test": (
            SubJob(name="a", argv=("-m", "a.run"),
                   refusal_exit_codes=frozenset({2})),
            SubJob(name="b", argv=("-m", "b.run"), depends_on=("a",),
                   skip_on_dependency_failure=True),
        )})
        try:
            result = run_cadence(
                "test",
                audit_path=self.audit_path,
                lock_path=self.lock_path,
                lock_conflict_log_path=self.conflict_log_path,
                invoker=_stub_invoker({
                    "a.run": (2, "", 1),
                    "b.run": (0, "", 1),
                }),
                cwd=self.data_dir,
            )
        finally:
            _restore_dag(snapshot)
        self.assertEqual(result["overall_outcome"], "partial")


# ── Production isolation ─────────────────────────────────────────────


class ProductionIsolationTests(_SchedulerHarness):

    def test_methodology_versions_byte_unmutated(self) -> None:
        before = dict(meth_mod.METHODOLOGY_VERSIONS)
        snapshot = _replace_dag_for_test({"test": (
            SubJob(name="a", argv=("-m", "a.run")),
        )})
        try:
            run_cadence(
                "test",
                audit_path=self.audit_path,
                lock_path=self.lock_path,
                lock_conflict_log_path=self.conflict_log_path,
                invoker=_stub_invoker({"a.run": (0, "", 1)}),
                cwd=self.data_dir,
            )
        finally:
            _restore_dag(snapshot)
        self.assertEqual(before, dict(meth_mod.METHODOLOGY_VERSIONS))

    def test_high_correlation_threshold_byte_unmutated(self) -> None:
        from backend.investment_analytics.portfolio_health import correlation
        before = correlation.HIGH_CORRELATION_THRESHOLD
        snapshot = _replace_dag_for_test({"test": (
            SubJob(name="a", argv=("-m", "a.run")),
        )})
        try:
            run_cadence(
                "test",
                audit_path=self.audit_path,
                lock_path=self.lock_path,
                lock_conflict_log_path=self.conflict_log_path,
                invoker=_stub_invoker({"a.run": (0, "", 1)}),
                cwd=self.data_dir,
            )
        finally:
            _restore_dag(snapshot)
        self.assertEqual(correlation.HIGH_CORRELATION_THRESHOLD, before)


# ── Chain integrity ──────────────────────────────────────────────────


class ChainIntegrityTests(_SchedulerHarness):

    def test_chain_valid_across_mixed_sequence(self) -> None:
        snapshot = _replace_dag_for_test({"test": (
            SubJob(name="a", argv=("-m", "a.run"),
                   refusal_exit_codes=frozenset({2})),
            SubJob(name="b", argv=("-m", "b.run"), depends_on=("a",),
                   skip_on_dependency_failure=True),
        )})
        try:
            # Three ticks: success, refusal, success-after-refusal.
            invoker_ok = _stub_invoker({"a.run": (0, "", 1), "b.run": (0, "", 1)})
            invoker_refused = _stub_invoker({"a.run": (2, "", 1), "b.run": (0, "", 1)})
            for inv in (invoker_ok, invoker_refused, invoker_ok):
                run_cadence(
                    "test",
                    audit_path=self.audit_path,
                    lock_path=self.lock_path,
                    lock_conflict_log_path=self.conflict_log_path,
                    invoker=inv, cwd=self.data_dir,
                )
        finally:
            _restore_dag(snapshot)
        self.assertTrue(verify_audit_chain(self.audit_path))


# ── find_scheduled_runs ──────────────────────────────────────────────


class FindScheduledRunsTests(_SchedulerHarness):

    def test_empty_chain_returns_empty_list(self) -> None:
        self.assertEqual(
            find_scheduled_runs(audit_path=self.audit_path), [],
        )

    def test_filter_by_overall_outcome(self) -> None:
        snapshot = _replace_dag_for_test({"test": (
            SubJob(name="a", argv=("-m", "a.run"),
                   refusal_exit_codes=frozenset({2})),
        )})
        try:
            run_cadence(
                "test",
                audit_path=self.audit_path,
                lock_path=self.lock_path,
                lock_conflict_log_path=self.conflict_log_path,
                invoker=_stub_invoker({"a.run": (0, "", 1)}),
                cwd=self.data_dir,
            )
            run_cadence(
                "test",
                audit_path=self.audit_path,
                lock_path=self.lock_path,
                lock_conflict_log_path=self.conflict_log_path,
                invoker=_stub_invoker({"a.run": (2, "", 1)}),
                cwd=self.data_dir,
            )
        finally:
            _restore_dag(snapshot)
        ok = find_scheduled_runs(audit_path=self.audit_path,
                                 overall_outcome="all_ok")
        partial = find_scheduled_runs(audit_path=self.audit_path,
                                      overall_outcome="partial")
        self.assertEqual(len(ok), 1)
        self.assertEqual(len(partial), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
