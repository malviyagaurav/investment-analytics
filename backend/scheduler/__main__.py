"""`python -m backend.scheduler` — operator CLI.

Subcommands:
  run-cadence <cadence_id>  — execute the cadence's DAG; emit one
                              scheduled_run row on success
  list                      — list scheduled_run rows from the
                              audit chain
  show-dag                  — print the registered DAG for a cadence

Exit codes:
  0   — cadence completed cleanly (overall_outcome=all_ok)
  2   — cadence completed but partial (refused/skipped sub-jobs)
  3   — cadence completed with at least one failed sub-job
  4   — scheduler refused to start (lock conflict / unknown cadence
        / chain pre-flight failed)

The dedicated exit code surface lets the cron wrapper distinguish
"scheduler ran but produced partial results" from "scheduler refused
to start" without parsing the audit chain.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from backend.scheduler.config import DEFAULT_AUDIT_PATH
from backend.scheduler.dag import CADENCE_REGISTRY, get_dag
from backend.scheduler.runner import (
    find_scheduled_runs,
    run_cadence,
)


def _run_cadence_subcommand(args: argparse.Namespace) -> int:
    result = run_cadence(
        args.cadence_id,
        audit_path=Path(args.audit_path),
    )
    print(json.dumps(_summarize_for_cli(result), indent=2, sort_keys=True))
    status = result.get("scheduler_status")
    if status != "completed":
        return 4
    overall = result.get("overall_outcome")
    if overall == "all_ok":
        return 0
    if overall == "partial":
        return 2
    return 3  # any_failed


def _summarize_for_cli(result: dict) -> dict:
    """Trim the full result dict to the most operator-actionable
    fields for stdout. stderr_tail strings stay on the audit row;
    operators chase them via the chain."""
    if result.get("scheduler_status") != "completed":
        return result
    return {
        "scheduler_status":    result["scheduler_status"],
        "cadence_id":          result["cadence_id"],
        "overall_outcome":     result["overall_outcome"],
        "duration_ms":         result["duration_ms"],
        "audit_chain_valid":   result["audit_chain_valid"],
        "chain_state_post":    result["chain_state_post"],
        "sub_jobs": [
            {
                "sub_job_name":      j["sub_job_name"],
                "outcome":           j["outcome"],
                "exit_code":         j["exit_code"],
                "duration_ms":       j["duration_ms"],
                "emitted_run_ids":   j["emitted_run_ids"],
            }
            for j in result["sub_jobs"]
        ],
    }


def _list_subcommand(args: argparse.Namespace) -> int:
    rows = find_scheduled_runs(
        audit_path=Path(args.audit_path),
        cadence_id=args.cadence_id,
        overall_outcome=args.overall_outcome,
    )
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    if not rows:
        print("(no scheduled_run rows match the filters)")
        return 0
    print(f"scheduled_run rows: {len(rows)}")
    for r in rows:
        sub_summary = ",".join(
            f"{j['sub_job_name']}={j['outcome']}" for j in r.get("sub_jobs", [])
        )
        print(
            f"  run_id={r['run_id']}  cadence={r.get('cadence_id')}  "
            f"overall={r.get('overall_outcome')}  "
            f"valid={r.get('audit_chain_valid')}  [{sub_summary}]"
        )
    return 0


def _show_dag_subcommand(args: argparse.Namespace) -> int:
    try:
        dag = get_dag(args.cadence_id)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 4
    out = [
        {
            "name":                       job.name,
            "argv":                       list(job.argv),
            "depends_on":                 list(job.depends_on),
            "skip_on_dependency_failure": job.skip_on_dependency_failure,
            "timeout_sec":                job.timeout_sec,
            "refusal_exit_codes":         sorted(job.refusal_exit_codes),
        }
        for job in dag
    ]
    print(json.dumps({"cadence_id": args.cadence_id, "dag": out},
                     indent=2, sort_keys=True))
    return 0


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.scheduler",
        description="Cadence orchestrator for evidence-producing sub-jobs.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    r_p = sub.add_parser("run-cadence",
                         help="Execute a registered cadence's DAG.")
    r_p.add_argument("cadence_id", help=f"One of {sorted(CADENCE_REGISTRY)}")
    r_p.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH))

    l_p = sub.add_parser("list", help="List scheduled_run rows.")
    l_p.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH))
    l_p.add_argument("--cadence-id", default=None, dest="cadence_id")
    l_p.add_argument("--overall-outcome", default=None,
                     dest="overall_outcome",
                     help="Filter by all_ok / partial / any_failed.")
    l_p.add_argument("--json", action="store_true")

    s_p = sub.add_parser("show-dag",
                         help="Print the registered DAG for a cadence.")
    s_p.add_argument("cadence_id")

    args = parser.parse_args(argv)
    if args.cmd == "run-cadence":
        return _run_cadence_subcommand(args)
    if args.cmd == "list":
        return _list_subcommand(args)
    if args.cmd == "show-dag":
        return _show_dag_subcommand(args)
    return 4


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
