"""Top-level operator CLI for audit-chain verification.

`python -m backend.audit verify` is the operator entry point for
checking the integrity of the hash-chained audit log without booting
the HTTP API. Read-only by construction: it consumes
``verify_audit_chain_multi`` from the audit primitive module and
formats the result; it does NOT touch the chain or any other state.

## Why a separate top-level module

The chain primitives live in ``backend.investment_analytics.audit`` —
that's the canonical home and that's where any future code that needs
to verify a chain should import from. This module is a thin
operator-facing shell:

  * exposes a stable ``python -m backend.audit verify`` invocation
    (kept shallow so ops scripting doesn't have to type the deep
    namespace);
  * formats the typed verification result in two shapes (human-
    readable summary by default, ``--json`` for scripts);
  * maps the typed ``overall_status`` to exit codes that mirror the
    operational convention established by Step 7's replay CLI
    (0 = clean, 2 = operationally-noteworthy-but-not-corrupt,
    3 = corrupt/unverifiable).

No verification logic lives here. No mutation of any kind. Polling
this CLI is functionally indistinguishable from polling the
``verify_audit_chain_multi`` function directly.

## Exit codes

  0 — overall_status == "valid"
  2 — overall_status in {"partial_failure", "empty"}
  3 — overall_status in {"invalid", "unverifiable"}
  2 — argparse error (Python default)

The "empty" → 2 mapping is deliberate: structurally a brand-new chain
is fine, but on a running system "empty" usually means something
upstream stopped emitting. Surfaces it operationally without
collapsing it into a fake-green state.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.investment_analytics.audit import verify_audit_chain_multi

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIT_DIR = ROOT / "data" / "audit"


EXIT_CODES: Dict[str, int] = {
    "valid":           0,
    "partial_failure": 2,
    "empty":           2,
    "invalid":         3,
    "unverifiable":    3,
}


def _format_human(result: Dict[str, Any]) -> str:
    """Operator-facing summary. One line per epoch, one line per orphan.

    Designed to be scan-readable at a glance; structured enough that a
    grep for "invalid" or "partial_failure" reliably finds problems.
    """
    overall = result.get("overall_status", "unknown")
    lines: List[str] = [f"audit chain: {overall}"]

    per_epoch = result.get("per_epoch", []) or []
    epochs_valid = result.get("epochs_valid", 0)
    epochs_failed = result.get("epochs_failed", 0)
    if per_epoch:
        lines.append(
            f"  epochs        : {len(per_epoch)} "
            f"(valid: {epochs_valid}, failed: {epochs_failed})"
        )
        for entry in per_epoch:
            status = entry.get("status", "unknown")
            lineno_lines = entry.get("lines_scanned", 0)
            bits = [f"{entry.get('file')}", f"{lineno_lines} lines"]
            if entry.get("first_bad_line") is not None:
                bits.append(f"first bad line {entry['first_bad_line']}")
            if entry.get("reason"):
                bits.append(f"reason: {entry['reason']}")
            lines.append(
                f"    epoch {entry.get('epoch')}     : "
                f"{status} ({', '.join(bits)})"
            )
    else:
        lines.append("  epochs        : 0")

    per_orphan = result.get("per_orphan", []) or []
    lines.append(f"  orphan chains : {len(per_orphan)}")
    for entry in per_orphan:
        bits = [
            f"classification={entry.get('classification')}",
            f"chain_root_type={entry.get('chain_root_type')}",
            f"total_lines={entry.get('total_lines')}",
        ]
        lines.append(f"    {entry.get('file'):<25} {' '.join(bits)}")

    total_lines = sum(e.get("lines_scanned", 0) or 0 for e in per_epoch)
    lines.append(f"  total lines   : {total_lines}")

    if result.get("reason") and overall == "unverifiable":
        lines.append(f"  reason        : {result['reason']}")

    return "\n".join(lines)


def _verify_subcommand(args: argparse.Namespace) -> int:
    audit_dir = Path(args.audit_dir)
    result = verify_audit_chain_multi(audit_dir)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(_format_human(result))
    return EXIT_CODES.get(result.get("overall_status", ""), 1)


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.audit",
        description="Operator CLI for audit-chain verification.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    verify_p = sub.add_parser(
        "verify",
        help="Verify the audit chain across all epochs and report status.",
    )
    verify_p.add_argument(
        "--audit-dir",
        default=str(DEFAULT_AUDIT_DIR),
        help=f"Audit directory (default: {DEFAULT_AUDIT_DIR}).",
    )
    verify_p.add_argument(
        "--json",
        action="store_true",
        help="Emit the typed verification result as JSON (for scripts).",
    )

    args = parser.parse_args(argv)

    if args.cmd == "verify":
        return _verify_subcommand(args)

    return 2  # unknown subcommand — argparse normally rejects first


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
