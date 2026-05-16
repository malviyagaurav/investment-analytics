"""`python -m backend.research_artifacts` — operator CLI for Step 13.

Subcommands:
  recommend <calibration_report_run_id>
            [--supersedes-run-id ID] [--rationale "..."]
            [--audit-path P]
            — emit a threshold_recommendation projecting from the
              cited calibration_report. Refuses if the cited row is
              a calibration refusal.

  list      [--target T] [--adoption-status S]
            [--include-superseded] [--audit-path P] [--json]
            — list threshold_recommendation rows.

  replay    <threshold_recommendation_run_id>
            [--audit-path P] [--no-verify-chain] [--dry-run]
            — convenience wrapper over backend.evidence.replay.

Exit codes:
  0 — success
  2 — typed refusal (e.g., cited calibration was a refusal)
  4 — bad argument / unknown ID / unknown subcommand
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from backend.research_artifacts.runner import (
    DEFAULT_AUDIT_PATH,
    emit_threshold_recommendation,
    find_threshold_recommendations,
)


def _recommend_subcommand(args: argparse.Namespace) -> int:
    try:
        record = emit_threshold_recommendation(
            args.calibration_report_run_id,
            audit_path=Path(args.audit_path),
            supersedes_run_id=args.supersedes_run_id,
            rationale=args.rationale,
        )
    except (KeyError, ValueError) as exc:
        sys.stderr.write(f"[research_artifacts] refused: {exc}\n")
        return 2
    out = {
        "run_id":                                  record["event"]["run_id"],
        "target_canonical_id":                     record["event"]["target_canonical_id"],
        "recommended_value":                       record["event"]["recommended_value"],
        "adoption_status":                         record["event"]["adoption_status"],
        "derived_from_calibration_report_run_id":  record["event"][
            "derived_from_calibration_report_run_id"],
        "supersedes_run_id":                       record["event"]["supersedes_run_id"],
        "evidence_ref":                            record["event"]["evidence_ref"],
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def _list_subcommand(args: argparse.Namespace) -> int:
    rows = find_threshold_recommendations(
        audit_path=Path(args.audit_path),
        target_canonical_id=args.target,
        adoption_status=args.adoption_status,
        include_superseded=args.include_superseded,
    )
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    if not rows:
        print("(no threshold_recommendation rows match the filters)")
        return 0
    print(f"threshold_recommendation rows: {len(rows)}")
    for r in rows:
        flag = " (superseded)" if r.get("supersedes_run_id") else ""
        print(
            f"  run_id={r['run_id']}  target={r.get('target_canonical_id')}  "
            f"value={r.get('recommended_value')}  "
            f"status={r.get('adoption_status')}{flag}"
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
        prog="python -m backend.research_artifacts",
        description="Research-artifact operator CLI (threshold_recommendation).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    rec_p = sub.add_parser("recommend",
                           help="Emit a threshold_recommendation projecting "
                                "from a calibration_report.")
    rec_p.add_argument("calibration_report_run_id")
    rec_p.add_argument("--supersedes-run-id", default=None,
                       dest="supersedes_run_id")
    rec_p.add_argument("--rationale", default=None)
    rec_p.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH))

    l_p = sub.add_parser("list",
                         help="List threshold_recommendation rows.")
    l_p.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH))
    l_p.add_argument("--target", default=None)
    l_p.add_argument("--adoption-status", default=None,
                     dest="adoption_status")
    l_p.add_argument("--include-superseded", action="store_true",
                     dest="include_superseded")
    l_p.add_argument("--json", action="store_true")

    rep_p = sub.add_parser("replay",
                           help="Replay a threshold_recommendation row.")
    rep_p.add_argument("run_id")
    rep_p.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH))
    rep_p.add_argument("--no-verify-chain", action="store_true")
    rep_p.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "recommend":
        return _recommend_subcommand(args)
    if args.cmd == "list":
        return _list_subcommand(args)
    if args.cmd == "replay":
        return _replay_subcommand(args)
    return 4


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
