"""`python -m backend.calibration` — operator CLI for the
calibration engine.

Subcommands:
  calibrate  — run calibration for a target; emit one report row
  list       — list calibration_report rows from the audit chain
  replay     — convenience wrapper over `python -m backend.evidence.replay`
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from backend.calibration.config import CALIBRATION_TARGETS
from backend.calibration.runner import (
    DEFAULT_AUDIT_PATH,
    find_calibration_reports,
    run_calibration,
)
from backend.calibration.targets import DEFAULT_CACHE_DIR


def _calibrate_subcommand(args: argparse.Namespace) -> int:
    record = run_calibration(
        target=args.target,
        audit_path=Path(args.audit_path),
        cache_dir=Path(args.cache_dir),
        percentile=args.percentile,
    )
    out = {
        "run_id":              record["event"]["run_id"],
        "target":              record["event"]["target"],
        "recommendation":      record["event"]["recommendation"],
        "refusal_reason":      record["event"]["refusal_reason"],
        "valid_within_regimes_count": record["event"]["valid_within_regimes_count"],
        "excluded_regimes_count":     record["event"]["excluded_regimes_count"],
        "evidence_ref":        record["event"]["evidence_ref"],
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    # Exit code: 0 if recommendation present, 2 if refusal (operational
    # signal that ops can chain in scripts).
    return 0 if record["event"]["recommendation"] is not None else 2


def _list_subcommand(args: argparse.Namespace) -> int:
    rows = find_calibration_reports(
        audit_path=Path(args.audit_path),
        target=args.target,
        only_recommendations=args.only_recommendations,
        only_refusals=args.only_refusals,
    )
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    if not rows:
        print("(no calibration_report rows match the filters)")
        return 0
    print(f"calibration_report rows: {len(rows)}")
    for r in rows:
        rec = r.get("recommendation")
        if rec is not None:
            tag = f"recommendation={rec:.6f}"
        else:
            tag = f"refusal={r.get('refusal_reason')}"
        print(
            f"  run_id={r['run_id']}  target={r.get('target')}  {tag}  "
            f"engine={r.get('calibration_engine_version')}"
        )
    return 0


def _replay_subcommand(args: argparse.Namespace) -> int:
    from backend.evidence.replay import _main as replay_main
    replay_argv = [args.run_id, "--audit-path", args.audit_path]
    if args.no_verify_chain:
        replay_argv.append("--no-verify-chain")
    if args.dry_run:
        replay_argv.append("--dry-run")
    return replay_main(replay_argv)


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.calibration",
        description="Calibration engine operator CLI.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    c_p = sub.add_parser("calibrate", help="Run calibration for a target.")
    c_p.add_argument("--target", required=True,
                     help=f"One of {sorted(CALIBRATION_TARGETS)}")
    c_p.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH))
    c_p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    c_p.add_argument("--percentile", type=float, default=None,
                     help="Override the target's default percentile.")

    l_p = sub.add_parser("list", help="List calibration_report rows.")
    l_p.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH))
    l_p.add_argument("--target", default=None)
    l_p.add_argument("--only-recommendations", action="store_true",
                     dest="only_recommendations")
    l_p.add_argument("--only-refusals", action="store_true",
                     dest="only_refusals")
    l_p.add_argument("--json", action="store_true")

    r_p = sub.add_parser("replay",
                         help="Replay a calibration_report row.")
    r_p.add_argument("run_id")
    r_p.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH))
    r_p.add_argument("--no-verify-chain", action="store_true")
    r_p.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "calibrate":
        return _calibrate_subcommand(args)
    if args.cmd == "list":
        return _list_subcommand(args)
    if args.cmd == "replay":
        return _replay_subcommand(args)
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
