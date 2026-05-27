"""Operator-invoked CLI for the LLM research assistant.

Usage:

    python -m backend.research_artifacts.llm_assist \\
        --evidence data/evidence/portfolio_health_snapshot/<run_id>.json \\
        --question "What does the recorded coverage_integrity flag mean?"

Multiple ``--evidence`` flags may be passed; each file is sent to
the model under its own labelled section in the prompt.

## Exit codes

  0 — note written successfully
  2 — operationally-noteworthy refusal (model returned empty,
      missing API key, dependency missing). Mirrors the
      ``empty / partial_failure`` convention used by
      ``python -m backend.audit verify``.
  3 — invocation error (bad arguments, evidence file missing).

## What this CLI does NOT do

  * Does NOT add LLM output to any analyzer response.
  * Does NOT mutate any existing evidence file.
  * Does NOT call back into the chain except for ONE lightweight
    pointer event per run (``evidence_kind=None``), which records
    that an LLM run happened and where its non-evidentiary output
    lives. The chain remains the source of truth for what was
    done; the note remains the source of operator-facing
    commentary.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.research_artifacts.llm_assist.gemini_client import (
    DEFAULT_MODEL,
    LlmAssistError,
    LlmConfigError,
    LlmDependencyMissing,
    LlmRefusedToAnswer,
    call_gemini,
)
from backend.research_artifacts.llm_assist.notes_writer import (
    LLM_NOTES_DIR,
    reference_evidence_file,
    write_research_note,
)
from backend.research_artifacts.llm_assist.prompt_templates import (
    PROMPT_TEMPLATE_VERSION,
    build_evidence_review_prompt,
)


ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_AUDIT_PATH = ROOT / "data" / "audit" / "audit.jsonl"


EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_BAD_INVOCATION = 3


def _load_evidence(path: Path) -> Dict[str, Any]:
    """Read an evidence envelope JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"evidence file not found: {path}")
    with path.open("r", encoding="utf-8") as h:
        return json.load(h)


def _append_pointer_event(
    *,
    audit_path: Path,
    note_path: Path,
    note_run_id: str,
    model: str,
    prompt_template_version: str,
    evidence_refs: List[Any],
) -> Optional[Dict[str, Any]]:
    """Append a ``llm_research_note_emitted`` event to the audit chain.

    The event is ``evidence_kind=None`` (intentionally unclassified)
    so it cannot be mistaken for a typed evidence emission. Its only
    purpose is to make the chain aware that an LLM run happened, by
    pointing to the non-evidentiary note file by hash.

    Lazy import of ``append_audit_record`` keeps the LLM sandbox
    runnable in test environments that mock out the audit primitive.
    Returns None if the chain is unavailable (e.g. tests using a
    transient notes dir without an audit chain alongside).
    """
    if not audit_path.exists():
        return None
    # Lazy import: keeps `import backend.research_artifacts.llm_assist`
    # cheap and avoids pulling the audit primitive into mock-heavy
    # test paths that do not exercise the chain integration.
    from backend.investment_analytics.audit import append_audit_record

    sha = _file_sha256(note_path)
    repo_rel = note_path.resolve().relative_to(ROOT.resolve()).as_posix()
    event = {
        "event_type":            "llm_research_note_emitted",
        "non_evidentiary":       True,
        "non_replayable":        True,
        "llm_note_run_id":       note_run_id,
        "llm_note_path":         repo_rel,
        "llm_note_sha256":       sha,
        "model":                 model,
        "prompt_template_version": prompt_template_version,
        "input_evidence_refs":   [
            {
                "evidence_kind":   r.evidence_kind,
                "evidence_path":   r.evidence_path,
                "evidence_sha256": r.evidence_sha256,
                "run_id":          r.run_id,
            }
            for r in evidence_refs
        ],
        "operator_warning": (
            "This audit row points at a NON-EVIDENTIARY LLM note. "
            "Do NOT replay, do NOT cite, do NOT treat the note as "
            "a typed claim. The pointer exists only so the chain "
            "knows an LLM run happened; the note's content is "
            "operator-only commentary."
        ),
    }
    return append_audit_record(
        audit_path,
        event,
        evidence_kind=None,
        run_id=str(uuid.uuid4()),
    )


def _file_sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m backend.research_artifacts.llm_assist",
        description=(
            "Off-chain LLM research-assistant CLI. Produces "
            "non-evidentiary commentary on existing evidence files. "
            "Operator-invoked only; never runs on a schedule."
        ),
    )
    p.add_argument(
        "--evidence",
        action="append",
        required=True,
        metavar="PATH",
        help=(
            "Path to one evidence file (data/evidence/<kind>/<run_id>.json). "
            "May be passed multiple times to send several files in one prompt."
        ),
    )
    p.add_argument(
        "--question",
        required=True,
        help="Operator question for the model. Passed through verbatim.",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model id (default: {DEFAULT_MODEL}).",
    )
    p.add_argument(
        "--notes-dir",
        default=str(LLM_NOTES_DIR),
        help=(
            "Override the notes output directory (default: data/llm_notes). "
            "Useful for routing notes into a subdirectory per research run."
        ),
    )
    p.add_argument(
        "--no-audit-pointer",
        action="store_true",
        help=(
            "Skip the chain pointer event. Default is to append one "
            "lightweight evidence_kind=None pointer per run so the chain "
            "knows an LLM run happened. Pass this flag for ad-hoc "
            "experimentation when you do not want chain pollution."
        ),
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    evidence_paths = [Path(p) for p in args.evidence]
    for p in evidence_paths:
        if not p.exists():
            print(f"error: evidence file not found: {p}", file=sys.stderr)
            return EXIT_BAD_INVOCATION

    try:
        evidence_envelopes = {p.name: _load_evidence(p) for p in evidence_paths}
        evidence_refs = [
            reference_evidence_file(
                evidence_kind=str(
                    evidence_envelopes[p.name].get("evidence_kind")
                    or "unknown"
                ),
                evidence_path=p,
                repo_root=ROOT,
            )
            for p in evidence_paths
        ]
        prompt = build_evidence_review_prompt(
            evidence_files=evidence_envelopes,
            operator_question=args.question,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_BAD_INVOCATION

    try:
        raw_output = call_gemini(prompt, model=args.model)
    except LlmDependencyMissing as exc:
        print(f"llm dependency missing: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except LlmConfigError as exc:
        print(f"llm config error: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except LlmRefusedToAnswer as exc:
        print(f"llm refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except LlmAssistError as exc:
        print(f"llm error: {exc}", file=sys.stderr)
        return EXIT_REFUSED

    note_run_id = str(uuid.uuid4())
    note_path = write_research_note(
        operator_question=args.question,
        raw_output=raw_output,
        model=args.model,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        input_evidence_refs=evidence_refs,
        notes_dir=Path(args.notes_dir),
        run_id=note_run_id,
    )

    pointer_record: Optional[Dict[str, Any]] = None
    if not args.no_audit_pointer:
        pointer_record = _append_pointer_event(
            audit_path=DEFAULT_AUDIT_PATH,
            note_path=note_path,
            note_run_id=note_run_id,
            model=args.model,
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
            evidence_refs=evidence_refs,
        )

    print(f"note written: {note_path}")
    if pointer_record is not None:
        print(f"chain pointer run_id: {pointer_record['event']['run_id']}")
    else:
        print("chain pointer: skipped (no audit chain present or --no-audit-pointer)")
    print(
        "REMINDER: this artifact is non-evidentiary and non-replayable. "
        "Verify every numeric claim against the cited evidence files "
        "before acting on it."
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
