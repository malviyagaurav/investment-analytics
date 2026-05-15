"""`python -m backend.experiments` — operator CLI for the experiment framework.

Subcommands:
  run     — execute a single experiment from a config JSON file
  list    — list experiment_run rows from the audit chain
  replay  — convenience wrapper over `python -m backend.evidence.replay`

The CLI is a thin shell over the runner module — no business logic
here. Same convention as Steps 7 (replay) and 8 (audit verify).

Exit codes:
  0  — success
  1  — argparse error or unexpected failure
  2  — validation error (missing target, bad config, missing parent)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from backend.experiments.config import ExperimentConfig
from backend.experiments.runner import (
    DEFAULT_AUDIT_PATH,
    DEFAULT_REGISTRY_PATH,
    find_experiment_runs,
    run_experiment,
)


def _load_config(path: Path) -> tuple[ExperimentConfig, str]:
    """Load a JSON config file and extract baseline_run_id.

    Config file shape:
      {
        "baseline_run_id": "...",
        "target": "rank_category",
        "target_inputs": {"category": "..."},
        "param_overrides": {"MIN_ALIGNED_POINTS": 500},
        "methodology_kind": "engineered_variant",
        "experiment_status": "exploratory",
        "derived_from_run_ids": [],
        "non_semantic_metadata": {"rationale": "..."}
      }
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    baseline = raw.pop("baseline_run_id", None)
    if not baseline:
        raise ValueError(
            f"config file {path} missing required 'baseline_run_id'"
        )
    return ExperimentConfig(**raw), baseline


def _run_subcommand(args: argparse.Namespace) -> int:
    config, baseline = _load_config(Path(args.config))
    record = run_experiment(
        config,
        baseline_run_id=baseline,
        audit_path=Path(args.audit_path),
        registry_path=Path(args.registry_path),
    )
    out = {
        "run_id":                   record["event"]["run_id"],
        "parent_run_id":            record["event"]["parent_run_id"],
        "config_fingerprint":       record["event"]["config_fingerprint"],
        "experiment_status":        record["event"]["experiment_status"],
        "methodology_kind":         record["event"]["methodology_kind"],
        "derivation_depth":         record["event"]["derivation_depth"],
        "evidence_ref":             record["event"]["evidence_ref"],
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def _list_subcommand(args: argparse.Namespace) -> int:
    rows = find_experiment_runs(
        audit_path=Path(args.audit_path),
        target=args.target,
        experiment_status=args.experiment_status,
        methodology_kind=args.methodology_kind,
    )
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        if not rows:
            print("(no experiment_run rows match the filters)")
            return 0
        print(f"experiment_run rows: {len(rows)}")
        for r in rows:
            print(
                f"  run_id={r['run_id']}  target={r.get('target')}  "
                f"status={r.get('experiment_status')}  "
                f"kind={r.get('methodology_kind')}  "
                f"depth={r.get('derivation_depth')}  "
                f"fingerprint={r.get('config_fingerprint','')[:12]}"
            )
    return 0


def _replay_subcommand(args: argparse.Namespace) -> int:
    # Thin wrapper — delegates straight to the replay CLI.
    from backend.evidence.replay import _main as replay_main
    replay_argv = [
        args.run_id,
        "--audit-path", args.audit_path,
        "--registry-path", args.registry_path,
    ]
    if args.no_verify_chain:
        replay_argv.append("--no-verify-chain")
    if args.dry_run:
        replay_argv.append("--dry-run")
    return replay_main(replay_argv)


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.experiments",
        description="Experiment framework operator CLI.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run a single experiment from a config file.")
    run_p.add_argument("--config", required=True,
                       help="Path to experiment config JSON.")
    run_p.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH),
                       help=f"Path to audit.jsonl (default: {DEFAULT_AUDIT_PATH}).")
    run_p.add_argument("--registry-path", default=str(DEFAULT_REGISTRY_PATH),
                       help=f"Path to registry (default: {DEFAULT_REGISTRY_PATH}).")

    list_p = sub.add_parser("list", help="List experiment_run rows in the audit chain.")
    list_p.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH))
    list_p.add_argument("--target", default=None,
                        help="Filter by target name (e.g. rank_category).")
    list_p.add_argument("--experiment-status", default=None,
                        dest="experiment_status",
                        help="Filter by experiment_status.")
    list_p.add_argument("--methodology-kind", default=None,
                        dest="methodology_kind",
                        help="Filter by methodology_kind.")
    list_p.add_argument("--json", action="store_true",
                        help="Emit raw events as JSON.")

    rep_p = sub.add_parser("replay",
                           help="Replay an experiment_run by run_id "
                                "(delegates to backend.evidence.replay).")
    rep_p.add_argument("run_id", help="Experiment_run row's run_id.")
    rep_p.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH))
    rep_p.add_argument("--registry-path", default=str(DEFAULT_REGISTRY_PATH))
    rep_p.add_argument("--no-verify-chain", action="store_true")
    rep_p.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        return _run_subcommand(args)
    if args.cmd == "list":
        return _list_subcommand(args)
    if args.cmd == "replay":
        return _replay_subcommand(args)
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
