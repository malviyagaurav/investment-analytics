"""`python -m backend.regimes` — operator CLI.

Subcommands:
  classify  — classify a single window (no emit) and print result
  emit      — classify + emit a regime_summary row
  list      — list regime_summary rows from the audit chain
  drift     — emit a drift_analysis between two windows
  replay    — convenience wrapper over `python -m backend.evidence.replay`
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from backend.regimes.classifier import (
    DEFAULT_CACHE_DIR,
    classification_to_payload,
    classify_window,
)
from backend.regimes.runner import (
    DEFAULT_AUDIT_PATH,
    emit_drift_analysis,
    emit_regime_summary,
    find_regime_summaries,
)


def _classify_subcommand(args: argparse.Namespace) -> int:
    c = classify_window(args.start, args.end, cache_dir=Path(args.cache_dir))
    payload = classification_to_payload(c)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if c.regime_class != "indeterminate" else 2


def _emit_subcommand(args: argparse.Namespace) -> int:
    record = emit_regime_summary(
        args.start, args.end,
        cache_dir=Path(args.cache_dir),
        audit_path=Path(args.audit_path),
        supersedes_run_id=args.supersedes_run_id,
    )
    out = {
        "run_id":                    record["event"]["run_id"],
        "regime_class":              record["event"]["regime_class"],
        "classification_confidence": record["event"]["classification_confidence"],
        "coverage_quality":          record["event"]["coverage_quality"],
        "supersedes_run_id":         record["event"]["supersedes_run_id"],
        "evidence_ref":              record["event"]["evidence_ref"],
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def _list_subcommand(args: argparse.Namespace) -> int:
    rows = find_regime_summaries(
        audit_path=Path(args.audit_path),
        regime_class=args.regime_class,
        include_superseded=args.include_superseded,
    )
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    if not rows:
        print("(no regime_summary rows match the filters)")
        return 0
    print(f"regime_summary rows: {len(rows)}")
    for r in rows:
        flag = " (superseded)" if r.get("supersedes_run_id") else ""
        print(
            f"  run_id={r['run_id']}  "
            f"[{r.get('window_start_date')} → {r.get('window_end_date')}]  "
            f"{r.get('regime_class')}  conf={r.get('classification_confidence')}  "
            f"coverage={r.get('coverage_quality')}{flag}"
        )
    return 0


def _drift_subcommand(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir)
    a = classify_window(args.a_start, args.a_end, cache_dir=cache_dir)
    b = classify_window(args.b_start, args.b_end, cache_dir=cache_dir)
    record = emit_drift_analysis(
        a, b,
        audit_path=Path(args.audit_path),
        window_a_run_id=args.window_a_run_id,
        window_b_run_id=args.window_b_run_id,
    )
    out = {
        "run_id":                  record["event"]["run_id"],
        "drift_kind":              record["event"]["drift_kind"],
        "vol_delta_pct":           record["event"]["vol_delta_pct"],
        "regime_transition":       record["event"]["regime_transition"],
        "transition_confidence":   record["event"]["transition_confidence"],
        "magnitude_band":          record["event"]["magnitude_band"],
        "evidence_ref":            record["event"]["evidence_ref"],
    }
    print(json.dumps(out, indent=2, sort_keys=True))
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
        prog="python -m backend.regimes",
        description="Regime detection & drift analysis CLI.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    c_p = sub.add_parser("classify",
                         help="Classify a window without emitting evidence.")
    c_p.add_argument("--start", required=True, help="YYYY-MM-DD")
    c_p.add_argument("--end", required=True, help="YYYY-MM-DD")
    c_p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))

    e_p = sub.add_parser("emit",
                         help="Classify and emit a regime_summary row.")
    e_p.add_argument("--start", required=True)
    e_p.add_argument("--end", required=True)
    e_p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    e_p.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH))
    e_p.add_argument("--supersedes-run-id", default=None, dest="supersedes_run_id",
                     help="run_id of the prior regime_summary this one supersedes.")

    l_p = sub.add_parser("list", help="List regime_summary rows.")
    l_p.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH))
    l_p.add_argument("--regime-class", default=None, dest="regime_class")
    l_p.add_argument("--include-superseded", action="store_true",
                     default=False, dest="include_superseded",
                     help="Show superseded rows in the listing.")
    l_p.add_argument("--json", action="store_true")

    d_p = sub.add_parser("drift",
                         help="Emit a drift_analysis between two windows.")
    d_p.add_argument("--a-start", required=True, dest="a_start")
    d_p.add_argument("--a-end", required=True, dest="a_end")
    d_p.add_argument("--b-start", required=True, dest="b_start")
    d_p.add_argument("--b-end", required=True, dest="b_end")
    d_p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    d_p.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH))
    d_p.add_argument("--window-a-run-id", default=None, dest="window_a_run_id")
    d_p.add_argument("--window-b-run-id", default=None, dest="window_b_run_id")

    r_p = sub.add_parser("replay",
                         help="Replay a regime_summary or drift_analysis row.")
    r_p.add_argument("run_id")
    r_p.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH))
    r_p.add_argument("--no-verify-chain", action="store_true")
    r_p.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "classify":
        return _classify_subcommand(args)
    if args.cmd == "emit":
        return _emit_subcommand(args)
    if args.cmd == "list":
        return _list_subcommand(args)
    if args.cmd == "drift":
        return _drift_subcommand(args)
    if args.cmd == "replay":
        return _replay_subcommand(args)
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
