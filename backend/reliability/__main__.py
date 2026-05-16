"""`python -m backend.reliability` — operator CLI for Step 14.

Subcommands:
  score   --target <target_canonical_id>
          --target-run-id <run_id>
          [--window-days N] [--supersedes-run-id ID]
          [--audit-path P]
          — compute reliability_score for a target artifact; emit
            one row.

  list    [--target T] [--only-recommendations | --only-refusals]
          [--audit-path P] [--json]
          — list reliability_score rows.

  replay  <reliability_score_run_id> [...]
          — convenience wrapper over backend.evidence.replay.

Exit codes:
  0 — recommendation emitted (overall_score non-null)
  2 — typed refusal emitted
  4 — bad argument / unknown ID / unknown subcommand
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from backend.reliability.config import DEFAULT_SCORING_WINDOW_DAYS
from backend.reliability.runner import (
    DEFAULT_AUDIT_PATH,
    find_reliability_scores,
    score_target,
)


def _score_subcommand(args: argparse.Namespace) -> int:
    try:
        record = score_target(
            target_canonical_id=args.target,
            target_run_id=args.target_run_id,
            audit_path=Path(args.audit_path),
            scoring_window_days=args.window_days,
            supersedes_run_id=args.supersedes_run_id,
        )
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[reliability] error: {exc}\n")
        return 4
    out = {
        "run_id":                   record["event"]["run_id"],
        "target_canonical_id":      record["event"]["target_canonical_id"],
        "target_run_id":            record["event"]["target_run_id"],
        "overall_score":            record["event"]["overall_score"],
        "overall_refusal_reason":   record["event"]["overall_refusal_reason"],
        "refused_dimension_count":  record["event"]["refused_dimension_count"],
        "weighting_table_version":  record["event"]["weighting_table_version"],
        "evidence_ref":             record["event"]["evidence_ref"],
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if record["event"]["overall_score"] is not None else 2


def _list_subcommand(args: argparse.Namespace) -> int:
    rows = find_reliability_scores(
        audit_path=Path(args.audit_path),
        target_canonical_id=args.target,
        only_recommendations=args.only_recommendations,
        only_refusals=args.only_refusals,
    )
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    if not rows:
        print("(no reliability_score rows match the filters)")
        return 0
    print(f"reliability_score rows: {len(rows)}")
    for r in rows:
        score = r.get("overall_score")
        refusal = r.get("overall_refusal_reason")
        tag = (f"score={score:.4f}" if score is not None
               else f"refusal={refusal}")
        print(
            f"  run_id={r['run_id']}  "
            f"target={r.get('target_canonical_id')}  "
            f"{tag}  refused_dims={r.get('refused_dimension_count', 0)}/"
            f"{r.get('dimension_count', 0)}"
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
        prog="python -m backend.reliability",
        description="Reliability scoring operator CLI.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    s_p = sub.add_parser("score",
                         help="Score reliability of a target artifact.")
    s_p.add_argument("--target", required=True)
    s_p.add_argument("--target-run-id", required=True,
                     dest="target_run_id")
    s_p.add_argument("--window-days", type=int,
                     default=DEFAULT_SCORING_WINDOW_DAYS,
                     dest="window_days")
    s_p.add_argument("--supersedes-run-id", default=None,
                     dest="supersedes_run_id")
    s_p.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH))

    l_p = sub.add_parser("list", help="List reliability_score rows.")
    l_p.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH))
    l_p.add_argument("--target", default=None)
    l_p.add_argument("--only-recommendations", action="store_true",
                     dest="only_recommendations")
    l_p.add_argument("--only-refusals", action="store_true",
                     dest="only_refusals")
    l_p.add_argument("--json", action="store_true")

    r_p = sub.add_parser("replay",
                         help="Replay a reliability_score row.")
    r_p.add_argument("run_id")
    r_p.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH))
    r_p.add_argument("--no-verify-chain", action="store_true")
    r_p.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "score":
        return _score_subcommand(args)
    if args.cmd == "list":
        return _list_subcommand(args)
    if args.cmd == "replay":
        return _replay_subcommand(args)
    return 4


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
