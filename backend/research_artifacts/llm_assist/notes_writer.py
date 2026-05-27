"""Writes off-chain LLM research notes to data/llm_notes/.

Every artifact this module produces is marked, on every field that
matters, as non-evidentiary and non-replayable. The marking exists
in the JSON envelope, in the filename prefix, and in an operator-
warning string. The redundancy is deliberate: a future reader (or
a sloppy automation that scans the directory) should fail at
every checkpoint before they could mistake an LLM note for evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


ROOT = Path(__file__).resolve().parent.parent.parent.parent
LLM_NOTES_DIR = ROOT / "data" / "llm_notes"


# ── Refusal vocabulary the writer enforces ───────────────────────────


@dataclass(frozen=True)
class _EvidenceReference:
    """Reference to one evidence file the LLM was given as input."""
    evidence_kind: str
    evidence_path: str          # repo-relative, forward-slash
    evidence_sha256: str
    run_id: Optional[str]       # the evidence file's run_id (if any)


@dataclass
class LlmResearchNote:
    """One research note written to ``data/llm_notes/``.

    Every field carrying a ``non_*`` or ``operator_*`` marker is
    load-bearing: the file MUST not be mistaken for evidence by any
    downstream reader. The serialised JSON is what gets persisted;
    the dataclass is a typed builder.
    """
    run_id:                          str
    generated_at:                    str
    model:                           str
    prompt_template_version:         str
    operator_question:               str
    raw_output:                      str
    input_evidence_refs:             List[_EvidenceReference]
    artifact_kind:                   str = "llm_research_note"
    non_evidentiary:                 bool = True
    non_replayable:                  bool = True
    non_attributable:                bool = True
    operator_decides:                bool = True
    operator_warning:                str = (
        "NOT EVIDENCE. NOT REPLAYABLE. NOT ATTRIBUTABLE. Do not cite this "
        "file in any consequential decision without independently "
        "verifying every numeric claim against the input_evidence_refs "
        "listed above. The model is stochastic, may hallucinate, and may "
        "have been trained on data inconsistent with this engine's "
        "deterministic outputs."
    )
    # Free-form audit metadata captured at write time. Not part of the
    # semantic identity of the note — never feeds back into anything.
    non_semantic_metadata:           Dict[str, Any] = field(default_factory=dict)


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _serialise_note(note: LlmResearchNote) -> str:
    """Canonical-ish JSON for the note. Sorted keys + indent for human
    readability; the artifact is operator-facing, not byte-hashed by
    any downstream consumer (it's explicitly non-evidentiary)."""
    data = asdict(note)
    return json.dumps(data, indent=2, sort_keys=True)


def write_research_note(
    *,
    operator_question: str,
    raw_output: str,
    model: str,
    prompt_template_version: str,
    input_evidence_refs: List[_EvidenceReference],
    notes_dir: Optional[Path] = None,
    non_semantic_metadata: Optional[Mapping[str, Any]] = None,
    run_id: Optional[str] = None,
) -> Path:
    """Write one LLM research note and return its filesystem path.

    The filename prefix is ``LLM_NOTE_`` so a directory listing makes
    the non-evidentiary status visually obvious. The run_id suffix
    is unique enough to avoid collisions even if two notes are
    written in the same second.

    Args:
      operator_question:        the verbatim question the operator
                                asked. Recorded for replay context
                                even though the artifact itself is
                                non-replayable.
      raw_output:               the model's literal response. NOT
                                parsed, NOT modified, NOT cleaned.
                                The operator reads it as-is.
      model:                    model identifier (e.g. "gemini-2.5-pro").
      prompt_template_version:  version of the prompt template used.
      input_evidence_refs:      every evidence file handed to the
                                model. Required: empty refs means
                                the LLM had no grounding, which the
                                operator must be able to spot
                                immediately.
      notes_dir:                override target dir (used by tests).
      non_semantic_metadata:    operator-supplied breadcrumbs. Never
                                feeds back into any typed claim.
      run_id:                   override the generated UUID (used by
                                tests for byte-stable assertions).
    """
    if not input_evidence_refs:
        raise ValueError(
            "input_evidence_refs must not be empty: an LLM note with no "
            "grounding evidence is exactly the failure mode the sandbox "
            "is designed to prevent. Refuse rather than write."
        )
    if not operator_question or not operator_question.strip():
        raise ValueError(
            "operator_question must be a non-empty string"
        )
    if not raw_output or not raw_output.strip():
        raise ValueError(
            "raw_output must be non-empty; the LLM returned nothing "
            "useful, which is itself a refusal — record that upstream "
            "rather than writing an empty note"
        )

    target_dir = notes_dir or LLM_NOTES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    note = LlmResearchNote(
        run_id=run_id or str(uuid.uuid4()),
        generated_at=_now_iso(),
        model=model,
        prompt_template_version=prompt_template_version,
        operator_question=operator_question,
        raw_output=raw_output,
        input_evidence_refs=list(input_evidence_refs),
        non_semantic_metadata=dict(non_semantic_metadata or {}),
    )

    filename = f"LLM_NOTE_{note.run_id}.json"
    out_path = target_dir / filename
    serialised = _serialise_note(note)
    # Write + fsync so a crashed terminal doesn't leave a partial file
    # that looks valid to a casual reader.
    with out_path.open("w", encoding="utf-8", newline="\n") as h:
        h.write(serialised)
        h.flush()
        os.fsync(h.fileno())
    return out_path


def reference_evidence_file(
    *,
    evidence_kind: str,
    evidence_path: Path,
    repo_root: Path,
) -> _EvidenceReference:
    """Build an _EvidenceReference by hashing the file at ``evidence_path``.

    The reference captures (path, sha256, run_id-if-present) so the
    research note can be cross-checked against the exact evidence
    file the LLM saw — even if the file is later moved or modified
    (which would change the sha256 mismatch and flag the note as
    stale).
    """
    if not evidence_path.exists():
        raise FileNotFoundError(
            f"evidence file does not exist: {evidence_path}"
        )
    raw = evidence_path.read_bytes()
    sha = _sha256_bytes(raw)
    # Try to extract run_id from the envelope; tolerate any shape
    # since this is a reference, not a typed read.
    run_id: Optional[str] = None
    try:
        envelope = json.loads(raw.decode("utf-8"))
        run_id = envelope.get("run_id")
    except (UnicodeDecodeError, json.JSONDecodeError):
        run_id = None

    # Canonical forward-slash form — matches the byte-stability
    # convention used by watchlist category_snapshot_refs. Fall back
    # to absolute path (forward-slash form) when the evidence file
    # is outside the repo root, which happens in tests using
    # TemporaryDirectory and in any ad-hoc operator invocation
    # where evidence has been staged outside the tree. The path is
    # metadata; the sha256 is the integrity primitive.
    resolved = evidence_path.resolve()
    try:
        rel = resolved.relative_to(repo_root.resolve())
        rel_str = rel.as_posix()
    except ValueError:
        rel_str = resolved.as_posix()

    return _EvidenceReference(
        evidence_kind=evidence_kind,
        evidence_path=rel_str,
        evidence_sha256=sha,
        run_id=run_id,
    )
