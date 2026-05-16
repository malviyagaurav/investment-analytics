"""Governance-decision CLI.

Subcommands:

  approve         emit decision_type=approve
  reject          emit decision_type=reject
  request-review  emit decision_type=request_review
  rollback        emit decision_type=rollback
  show-eligibility  print eligibility result without emitting
  list            list governance_decision rows
  replay          replay a prior governance_decision

All emitting subcommands require:
  --subject-run-id <threshold_recommendation run_id>
  --operator-id <typed declaration of WHO>

Eligibility-bypass requires both flags:
  --force-ineligible --override-reason <enum>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from backend.governance.config import (
    DECISION_TYPES,
    OPERATOR_ROLES,
    OVERRIDE_REASONS,
)
from backend.governance.eligibility import check_eligibility
from backend.governance.runner import (
    DEFAULT_AUDIT_PATH,
    EligibilityRefused,
    emit_governance_decision,
    find_governance_decisions,
)


def _emit_via_cli(decision_type: str, args: argparse.Namespace) -> int:
    try:
        result = emit_governance_decision(
            decision_type=decision_type,
            subject_run_id=args.subject_run_id,
            operator_id=args.operator_id,
            operator_role=args.operator_role,
            attestation_method=args.attestation_method,
            supporting_evidence=args.supporting_evidence or None,
            rationale=args.rationale or "",
            force_ineligible=args.force_ineligible,
            override_reason=args.override_reason,
            supersedes_run_id=args.supersedes_run_id,
            rollback_target_run_id=args.rollback_target_run_id,
            audit_path=Path(args.audit_path),
        )
    except EligibilityRefused as exc:
        print(json.dumps({
            "status":             "eligibility_refused",
            "decision_type":      decision_type,
            "subject_run_id":     args.subject_run_id,
            "eligibility_result": exc.eligibility_result,
            "hint": (
                "to override, pass --force-ineligible --override-reason "
                f"<{'|'.join(sorted(OVERRIDE_REASONS))}>"
            ),
        }, indent=2, sort_keys=True))
        return 2
    except (ValueError, RuntimeError) as exc:
        print(json.dumps({
            "status":         "error",
            "decision_type":  decision_type,
            "subject_run_id": args.subject_run_id,
            "error":          str(exc),
        }, indent=2, sort_keys=True))
        return 3
    print(json.dumps({
        "status":         "emitted",
        "decision_type":  decision_type,
        "run_id":         result.get("event", {}).get("run_id"),
        "subject_run_id": args.subject_run_id,
    }, indent=2, sort_keys=True))
    return 0


def _show_eligibility(args: argparse.Namespace) -> int:
    result = check_eligibility(args.subject_run_id, Path(args.audit_path))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["eligibility_passed"] else 2


def _list(args: argparse.Namespace) -> int:
    decisions = find_governance_decisions(
        audit_path=Path(args.audit_path),
        subject_run_id=args.subject_run_id,
        subject_target_canonical_id=args.target_canonical_id,
        decision_type=args.decision_type,
    )
    print(json.dumps(decisions, indent=2, sort_keys=True))
    return 0


def _replay(args: argparse.Namespace) -> int:
    # Defer to the existing replay machinery — governance is a
    # registered handler in REPLAY_HANDLERS.
    from backend.evidence.replay import replay_run
    result = replay_run(
        audit_path=Path(args.audit_path),
        run_id=args.run_id,
        verify_chain=not args.no_verify_chain,
        emit_audit=not args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return {
        "exact_match": 0,
        "semantically_equivalent": 0,
        "expected_divergence": 0,
        "unreproducible": 2,
        "invalid_replay": 3,
    }.get(result.get("state", ""), 1)


def _add_emit_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--subject-run-id", required=True)
    p.add_argument("--operator-id", required=True)
    p.add_argument(
        "--operator-role", default="owner",
        choices=sorted(OPERATOR_ROLES),
    )
    p.add_argument(
        "--attestation-method", default="local_terminal",
    )
    p.add_argument(
        "--supporting-evidence", action="append", default=None,
        help="Pass multiple times for multiple breadcrumbs.",
    )
    p.add_argument("--rationale", default="")
    p.add_argument(
        "--force-ineligible", action="store_true",
        help="Proceed even if eligibility fails. Requires --override-reason.",
    )
    p.add_argument(
        "--override-reason", default=None,
        choices=sorted(OVERRIDE_REASONS),
        help="Typed reason for the override (required with --force-ineligible).",
    )
    p.add_argument(
        "--supersedes-run-id", default=None,
        help="Prior governance_decision this one supersedes.",
    )
    p.add_argument(
        "--rollback-target-run-id", default=None,
        help="Prior approve being rolled back (rollback only).",
    )
    p.add_argument(
        "--audit-path", default=str(DEFAULT_AUDIT_PATH),
    )


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.governance",
        description="Governance + promotion evidence CLI.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    for dt in sorted(DECISION_TYPES):
        sp = sub.add_parser(dt.replace("_", "-"), help=f"Emit decision_type={dt}")
        _add_emit_args(sp)
        sp.set_defaults(_decision_type=dt)

    show = sub.add_parser("show-eligibility")
    show.add_argument("--subject-run-id", required=True)
    show.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH))

    lst = sub.add_parser("list")
    lst.add_argument("--subject-run-id", default=None)
    lst.add_argument("--target-canonical-id", default=None)
    lst.add_argument(
        "--decision-type", default=None, choices=sorted(DECISION_TYPES),
    )
    lst.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH))

    rp = sub.add_parser("replay")
    rp.add_argument("run_id")
    rp.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH))
    rp.add_argument("--no-verify-chain", action="store_true")
    rp.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "show-eligibility":
        return _show_eligibility(args)
    if args.cmd == "list":
        return _list(args)
    if args.cmd == "replay":
        return _replay(args)

    # Map dashed CLI form back to the underscore enum value.
    return _emit_via_cli(args._decision_type, args)


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
