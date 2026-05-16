"""POSIX-specific regression coverage for the locking adapter.

These tests prove the post-portability adapter on POSIX exercises
the SAME kernel primitive that pre-portability code did. They are
intentionally POSIX-only because the underlying assertion is about
``fcntl.LOCK_EX`` semantics, which does not exist on Windows.

## Why these tests live in a separate file

The cross-platform parity tests in ``tests/test_cross_platform.py``
must run uniformly on every platform — that file is guarded by the
CI workflow against any ``@unittest.skip*`` annotations. Platform-
specific regression coverage with typed skip annotations is the
correct discipline (typed, justified, documented) and lives here
where the CI guard does not apply.

## What these tests defend against

The portability work introduced a typed adapter ``_locking.py`` to
replace direct ``fcntl.flock`` calls. The first implementation of
that adapter defaulted to a bounded-60s retry loop on POSIX, which
silently shifted the audit-append concurrency contract from
"integrity over availability (truly-unbounded block)" to
"bounded availability with raise-on-timeout."

The regression-integrity audit caught the drift before commit;
the fix restored truly-unbounded blocking semantics on POSIX as
the default. These tests pin that behavior so a future change
that re-introduces the bounded default would surface immediately.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.investment_analytics import _locking


@unittest.skipIf(
    sys.platform.startswith("win"),
    reason=(
        "POSIX-specific behavioral contract — these tests assert the "
        "default blocking acquire invokes ``fcntl.flock(fd, LOCK_EX)`` "
        "with truly-unbounded kernel-blocking semantics. msvcrt has "
        "no truly-unbounded primitive, so the assertion does not "
        "apply to the Windows branch of the adapter. The Windows "
        "branch's bounded-retry behavior is verified by the cross-"
        "platform tests in test_cross_platform.py, which DO run on "
        "every platform without skips."
    ),
)
class PosixBlockingSemanticsRegressionTests(unittest.TestCase):
    """Pins the POSIX default to truly-unbounded LOCK_EX."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "lockfile"
        self.path.touch()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_default_blocking_acquire_calls_lock_ex_directly(self) -> None:
        """``acquire_exclusive_blocking(fd)`` with no timeout kwarg
        MUST invoke the kernel primitive directly — exactly one
        call to ``fcntl.flock(fd, LOCK_EX)`` with no retry loop,
        no polling, no LOCK_NB. This is the pre-portability sealed
        semantics: integrity over availability, kernel-blocking
        with zero polling overhead."""
        import fcntl as real_fcntl
        fd = os.open(str(self.path), os.O_RDWR)
        try:
            with patch.object(_locking, "fcntl") as mock_fcntl:
                # Preserve the integer constants the function body
                # references so the call shape is observable.
                mock_fcntl.LOCK_EX = real_fcntl.LOCK_EX
                mock_fcntl.LOCK_NB = real_fcntl.LOCK_NB
                mock_fcntl.LOCK_UN = real_fcntl.LOCK_UN

                _locking.acquire_exclusive_blocking(fd)

                # The truly-unbounded path makes ONE call:
                #   fcntl.flock(fd, LOCK_EX)
                # The bounded retry path would have called
                #   fcntl.flock(fd, LOCK_EX | LOCK_NB)
                # zero or more times. The assertion fails loudly
                # if the bounded path is re-introduced.
                mock_fcntl.flock.assert_called_once_with(
                    fd, real_fcntl.LOCK_EX,
                )
                # Defensive: also verify LOCK_NB was NOT used.
                for call in mock_fcntl.flock.call_args_list:
                    args, _kwargs = call
                    flag = args[1]
                    self.assertEqual(
                        flag, real_fcntl.LOCK_EX,
                        msg=(
                            f"acquire_exclusive_blocking with default "
                            f"timeout invoked fcntl.flock with flag "
                            f"{flag} != LOCK_EX={real_fcntl.LOCK_EX}; "
                            f"the bounded-retry path leaked into the "
                            f"POSIX default — pre-portability sealed "
                            f"concurrency contract is broken"
                        ),
                    )
        finally:
            # The mock didn't actually take a real lock, so just
            # close the fd. No release needed.
            os.close(fd)

    def test_explicit_finite_timeout_activates_bounded_retry(self) -> None:
        """When a caller EXPLICITLY passes ``timeout_seconds`` as a
        finite number, the bounded retry path activates: at least
        one call to ``fcntl.flock(fd, LOCK_EX | LOCK_NB)``, never
        a call to ``fcntl.flock(fd, LOCK_EX)`` directly. This is
        the opt-in path for callers that legitimately need bounded
        waits (used by tests + by any future caller that needs
        bounded semantics)."""
        import fcntl as real_fcntl
        fd = os.open(str(self.path), os.O_RDWR)
        try:
            with patch.object(_locking, "fcntl") as mock_fcntl:
                mock_fcntl.LOCK_EX = real_fcntl.LOCK_EX
                mock_fcntl.LOCK_NB = real_fcntl.LOCK_NB
                mock_fcntl.LOCK_UN = real_fcntl.LOCK_UN

                _locking.acquire_exclusive_blocking(fd, timeout_seconds=1.0)

                # The bounded path uses LOCK_EX|LOCK_NB on each
                # attempt. With no contention the first attempt
                # succeeds — exactly one call expected, with the
                # LOCK_NB flag set.
                expected_flag = real_fcntl.LOCK_EX | real_fcntl.LOCK_NB
                mock_fcntl.flock.assert_called_once_with(fd, expected_flag)
        finally:
            os.close(fd)

    def test_audit_append_uses_default_unbounded_path(self) -> None:
        """End-to-end pin: the audit append path calls the adapter
        WITHOUT passing a timeout_seconds, so it must execute the
        truly-unbounded LOCK_EX branch on POSIX. This guards
        against a future refactor accidentally passing a finite
        timeout to the adapter from the audit append call site."""
        import fcntl as real_fcntl
        import json
        import inspect

        from backend.investment_analytics import audit as audit_mod
        # Read the source of append_audit_record and assert it
        # calls acquire_exclusive_blocking with NO timeout kwarg.
        # If a future change adds a timeout kwarg there, this test
        # fires and forces a deliberate decision.
        src = inspect.getsource(audit_mod.append_audit_record)
        self.assertIn("acquire_exclusive_blocking(", src)
        # The current call shape is: acquire_exclusive_blocking(handle.fileno())
        # No timeout_seconds kwarg should appear in the call.
        # (This is a string-level guard; if the call site is
        # restructured legitimately, update this test deliberately.)
        self.assertNotIn(
            "acquire_exclusive_blocking(handle.fileno(), timeout_seconds",
            src,
            msg=(
                "append_audit_record now passes timeout_seconds to "
                "acquire_exclusive_blocking; this changes the POSIX "
                "concurrency contract from truly-unbounded blocking "
                "to bounded retry. If this change is intentional, "
                "the architectural seal documentation must be updated "
                "to reflect the new contract; otherwise revert."
            ),
        )

    def test_audit_migrate_uses_default_unbounded_path(self) -> None:
        """Same end-to-end pin for audit_migrate's lock call site."""
        import inspect
        from backend.investment_analytics import audit_migrate as audit_migrate_mod

        src = inspect.getsource(audit_migrate_mod.migrate_audit_to_epochs)
        self.assertIn("acquire_exclusive_blocking(", src)
        self.assertNotIn(
            "acquire_exclusive_blocking(handle.fileno(), timeout_seconds",
            src,
            msg=(
                "audit_migrate now passes timeout_seconds to "
                "acquire_exclusive_blocking; pre-portability sealed "
                "blocking contract is at risk. Revert or document."
            ),
        )


@unittest.skipIf(
    sys.platform.startswith("win"),
    reason=(
        "POSIX-specific cleanup discipline — the regression here is "
        "about ``os.close(fd)`` being invoked before a propagating "
        "OSError. On Windows the same code path exists but the test "
        "uses POSIX-specific os.open() flag combinations to construct "
        "the failure scenario; the corresponding Windows assertion "
        "would need a different fixture and is not required because "
        "the cleanup discipline lives in the call site (scheduler/"
        "runner.py), which is platform-agnostic."
    ),
)
class PosixSchedulerLockCleanupRegressionTests(unittest.TestCase):
    """Fix B regression coverage: scheduler lock acquisition MUST
    close the fd if the adapter raises a non-EAGAIN OSError."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.lock_path = Path(self.tmp.name) / "scheduler.lock"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_fd_closed_when_adapter_raises_propagating_oserror(self) -> None:
        """If ``acquire_exclusive_nonblocking`` raises (rare
        non-EAGAIN OSError path), ``_acquire_scheduler_lock`` MUST
        close the fd before the exception propagates — same
        cleanup discipline as the pre-portability inline fcntl
        code. Without this, the fd leaks on every such error."""
        from backend.scheduler import runner as scheduler_runner

        closed_fds: list = []
        original_close = os.close

        def tracking_close(fd: int) -> None:
            closed_fds.append(fd)
            original_close(fd)

        # Replace the adapter call with one that raises a non-
        # EAGAIN OSError — the contention path is EAGAIN/
        # EWOULDBLOCK; everything else propagates per the adapter's
        # documented contract.
        sentinel = OSError("simulated non-EAGAIN error")

        def raise_oserror(_fd: int) -> bool:
            raise sentinel

        with patch.object(
            scheduler_runner, "acquire_exclusive_nonblocking",
            side_effect=raise_oserror,
        ):
            with patch.object(os, "close", side_effect=tracking_close):
                with self.assertRaises(OSError) as ctx:
                    scheduler_runner._acquire_scheduler_lock(self.lock_path)
                self.assertIs(ctx.exception, sentinel)

        # The fd opened inside _acquire_scheduler_lock must have
        # been closed before the exception propagated. Exactly one
        # close call expected for that fd.
        self.assertEqual(
            len(closed_fds), 1,
            msg=(
                f"expected exactly one os.close call (the fd opened "
                f"inside _acquire_scheduler_lock before the adapter "
                f"raised); got {len(closed_fds)}: {closed_fds!r}. "
                f"Fix B regression: fd is leaking on propagating "
                f"non-EAGAIN OSError."
            ),
        )

    def test_fd_closed_on_normal_contention_path(self) -> None:
        """Sanity: the existing contention path (adapter returns
        False) still closes the fd correctly. This is byte-
        equivalent to pre-portability behavior; included to pin
        the discipline alongside the propagation path."""
        from backend.scheduler import runner as scheduler_runner

        closed_fds: list = []
        original_close = os.close

        def tracking_close(fd: int) -> None:
            closed_fds.append(fd)
            original_close(fd)

        with patch.object(
            scheduler_runner, "acquire_exclusive_nonblocking",
            return_value=False,
        ):
            with patch.object(os, "close", side_effect=tracking_close):
                result = scheduler_runner._acquire_scheduler_lock(
                    self.lock_path,
                )

        self.assertIsNone(result)
        self.assertEqual(len(closed_fds), 1)


if __name__ == "__main__":
    unittest.main()
