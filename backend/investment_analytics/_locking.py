"""Cross-platform exclusive file-lock adapter.

Replaces direct ``fcntl.flock`` usage so the audit chain, audit
migration, and scheduler cadence lock all work identically on POSIX
(Linux/macOS) and Windows. Below the explanatory surface — locking
is an operational mechanism the chain itself never reasons about,
so this module introduces NO new evidence_kind, methodology
component, or replay driver.

## API

  acquire_exclusive_blocking(fd, *, timeout_seconds=None) -> None
      Block until an exclusive cross-process lock on ``fd`` is held.

      POSIX (default, ``timeout_seconds=None``):
        ``fcntl.flock(fd, LOCK_EX)`` — TRULY UNBOUNDED kernel
        block. Byte-equivalent to the pre-portability sealed
        behavior. No polling overhead; the kernel wakes the
        waiter the instant the holder releases. Integrity over
        availability: the writer waits forever if needed rather
        than ever proceeding without the lock.

      POSIX (explicit finite ``timeout_seconds``):
        Non-blocking retry loop bounded by the caller-supplied
        timeout. Raises ``LockAcquisitionTimeout`` on timeout.
        This path is opt-in — callers that want bounded waits
        must ask for them explicitly. Used primarily by tests
        that need deterministic timeout behavior.

      Windows (any ``timeout_seconds``):
        ``msvcrt.locking(fd, LK_NBLCK, 1)`` in a retry loop. The
        Windows branch is necessarily bounded because msvcrt has
        no truly-unbounded blocking primitive (``LK_LOCK`` has a
        fixed 10-attempt retry budget that is too small for
        real-world contention). When ``timeout_seconds`` is None,
        the Windows branch uses a generous 60s default budget.
        Raises ``LockAcquisitionTimeout`` on timeout.

      The asymmetry between POSIX (unbounded default) and Windows
      (bounded default) is intentional and load-bearing:

        - POSIX preserves the pre-portability sealed concurrency
          contract exactly — same kernel primitive, same byte
          path. No silent shift from integrity-over-availability
          to bounded-availability.

        - Windows accepts a bounded default because the platform
          primitive does not offer an unbounded equivalent. The
          fail-closed timeout is the closest honest approximation;
          documented here so the difference is typed, not hidden.

      Both branches are fail-closed: the function either acquires
      the lock and returns, or raises. Neither branch ever proceeds
      without the lock.

  acquire_exclusive_nonblocking(fd) -> bool
      Try once to acquire; return True on success, False on
      contention. NEVER blocks. Maps to ``fcntl.LOCK_EX | LOCK_NB``
      on POSIX and ``msvcrt.LK_NBLCK`` on Windows.

  release(fd) -> None
      Release the lock. Assumes the caller holds it.

## Invariants preserved (same on both platforms)

  - Exclusive cross-process serialization on a local filesystem.
  - Kernel-managed release on process death (no stuck lock if a
    holder crashes).
  - Fail-closed: blocking acquire either succeeds or raises;
    non-blocking acquire either succeeds or returns False.
    Neither path silently proceeds without the lock.

## Platform divergence (documented, not hidden)

  POSIX blocking acquire is TRULY UNBOUNDED by default — the OS
  blocks the caller until the lock is available or the process
  is killed. This is byte-equivalent to the pre-portability
  sealed behavior: ``fcntl.flock(fd, LOCK_EX)`` is invoked
  directly with no retry loop and no polling.

  Windows blocking acquire has a finite default budget (60s)
  because ``msvcrt`` lacks an unbounded-block primitive. In
  normal operation audit-append contention resolves in
  milliseconds; the 60s budget triggers only if the holding
  process is hung or crashed in a way the OS hasn't yet cleaned
  up. Both platforms are fail-closed — POSIX waits; Windows
  raises — and neither platform proceeds without the lock.

  When a caller explicitly passes a finite ``timeout_seconds``,
  both platforms use a non-blocking retry loop bounded by that
  timeout. This explicit path exists for tests + for callers
  that legitimately need bounded waits; it is NEVER the default
  on POSIX.

  ``msvcrt.locking`` is byte-range mandatory locking. We lock 1
  byte at a sparse offset far beyond any plausible audit-file
  size as the cross-platform convention: all processes contend
  on the same well-known byte AND the lock never overlaps the
  file's real content. Locking byte 0 would block same-process
  readers (e.g. ``_last_record_hash``) that open a second handle
  and read from the start of the file — see ``_LOCK_OFFSET``.

## What this module is NOT

  - NOT a distributed lock (same scope as ``fcntl.flock``:
    single-machine local-filesystem only).
  - NOT a re-entrant lock (the same process re-locking the same
    fd has platform-dependent behavior; callers MUST use the
    existing ``_APPEND_LOCK = threading.Lock()`` for intra-process
    serialization, which is what audit.py already does).
  - NOT a substitute for the chain's hash-based integrity model;
    it serializes appenders so the hash chain has a single writer
    at a time. The hash chain itself is the integrity guarantee.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Final, Optional


class LockAcquisitionTimeout(RuntimeError):
    """Raised by ``acquire_exclusive_blocking`` when the lock cannot
    be acquired within the platform-specific timeout. Fail-closed:
    the caller MUST treat this as a refusal to proceed. Never as a
    soft fallback that would let an appender bypass serialization."""


_IS_WINDOWS: Final[bool] = sys.platform.startswith("win")

# Byte-range locking convention (Windows). All processes lock 1
# byte at a sparse offset far beyond any plausible audit-file size
# so contention is on one well-known byte AND the locked region
# never overlaps real file content. Windows byte-range locks are
# MANDATORY: a lock at byte 0 caused ``_last_record_hash`` (which
# opens a second handle and reads from byte 0) to fail with
# PermissionError on every audit append. The sparse offset
# preserves cross-process serialization (all writers seek+lock
# the same byte) without blocking same-process readers. POSIX
# ``fcntl.flock`` ignores byte offsets, so this constant is
# Windows-only in effect; the unbounded-blocking POSIX semantic
# documented above is unchanged.
_LOCK_OFFSET: Final[int] = 0x4000_0000_0000_0000
_LOCK_NBYTES: Final[int] = 1

# Retry tuning. The blocking-acquire retry interval is short so
# contention resolves promptly when callers opt into the bounded
# path; the Windows default budget is long so transient contention
# from a busy peer doesn't fail spuriously. NOTE: ``_DEFAULT_*``
# applies only to the Windows branch (and to POSIX callers that
# explicitly opt in); the POSIX default is truly-unbounded and
# does NOT consult this constant.
_RETRY_INTERVAL_SEC: Final[float] = 0.05
_DEFAULT_WINDOWS_BLOCKING_BUDGET_SEC: Final[float] = 60.0


if _IS_WINDOWS:
    import msvcrt  # type: ignore[import-not-found]

    def _seek_to_lock_byte(fd: int) -> None:
        os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)

    def acquire_exclusive_blocking(
        fd: int,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        # Windows has no truly-unbounded blocking primitive, so
        # ``timeout_seconds=None`` falls back to the platform default
        # budget (60s). Callers can still pass an explicit timeout
        # for shorter / longer bounded waits.
        budget = (
            timeout_seconds if timeout_seconds is not None
            else _DEFAULT_WINDOWS_BLOCKING_BUDGET_SEC
        )
        deadline = time.monotonic() + budget
        while True:
            try:
                _seek_to_lock_byte(fd)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, _LOCK_NBYTES)
                return
            except OSError:
                if time.monotonic() >= deadline:
                    raise LockAcquisitionTimeout(
                        f"could not acquire exclusive lock on fd={fd} "
                        f"within {budget}s — holder may be "
                        f"hung or crashed"
                    )
                time.sleep(_RETRY_INTERVAL_SEC)

    def acquire_exclusive_nonblocking(fd: int) -> bool:
        try:
            _seek_to_lock_byte(fd)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, _LOCK_NBYTES)
            return True
        except OSError:
            return False

    def release(fd: int) -> None:
        _seek_to_lock_byte(fd)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, _LOCK_NBYTES)

else:  # POSIX (Linux, macOS, BSD)
    import errno
    import fcntl

    def acquire_exclusive_blocking(
        fd: int,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        # LOAD-BEARING (post-portability seal restoration):
        # ``timeout_seconds=None`` (the default) invokes the kernel
        # primitive directly with truly-unbounded blocking semantics,
        # byte-equivalent to the pre-portability sealed call
        # ``fcntl.flock(fd, fcntl.LOCK_EX)``. No retry loop, no
        # polling, no timeout — the writer waits in the kernel
        # until the lock is released, exactly as it did before
        # the portability work.
        #
        # The bounded retry path activates ONLY when the caller
        # explicitly passes a finite ``timeout_seconds``. This is
        # opt-in semantics for tests + for callers that legitimately
        # need bounded waits; it is NEVER the default on POSIX, so
        # audit-append and audit-migrate (which call without a
        # timeout argument) execute the same kernel-blocking path
        # they always did.
        if timeout_seconds is None:
            fcntl.flock(fd, fcntl.LOCK_EX)
            return
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except (BlockingIOError, OSError) as exc:
                if getattr(exc, "errno", None) not in (
                    errno.EAGAIN, errno.EWOULDBLOCK,
                ):
                    raise
                if time.monotonic() >= deadline:
                    raise LockAcquisitionTimeout(
                        f"could not acquire exclusive lock on fd={fd} "
                        f"within {timeout_seconds}s — holder may be "
                        f"hung or crashed"
                    )
                time.sleep(_RETRY_INTERVAL_SEC)

    def acquire_exclusive_nonblocking(fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (BlockingIOError, OSError) as exc:
            if getattr(exc, "errno", None) in (
                errno.EAGAIN, errno.EWOULDBLOCK,
            ):
                return False
            raise

    def release(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


__all__ = [
    "LockAcquisitionTimeout",
    "acquire_exclusive_blocking",
    "acquire_exclusive_nonblocking",
    "release",
]
