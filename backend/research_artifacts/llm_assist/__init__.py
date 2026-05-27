"""Off-chain LLM research-assistant sandbox.

This subpackage is the ENTIRE LLM surface in the project. It exists
under an explicit standing-instruction override (governance event
recorded 2026-05-27, run_id 344e84af-a03b-49f6-85c2-419d1ce143f0)
and operates under hard constraints documented in that audit row.

## What this package does

Read-only commentary layer. Takes already-emitted evidence files as
input, sends them to Gemini with a paranoid prompt that forbids
invented numbers, writes the model's output to a clearly-marked
non-evidentiary research note under ``data/llm_notes/``. The operator
reads the note and decides whether to act on it. Nothing more.

## What this package MUST NOT do (load-bearing)

  * Write to ``data/audit/`` or ``data/evidence/``. Only the lightweight
    pointer event (``llm_research_note_emitted``) goes on-chain; it is
    explicitly ``evidence_kind=None`` (intentionally unclassified) so
    it cannot be replayed as a typed claim.
  * Add a new ``evidence_kind`` or methodology component. The LLM is
    not a methodology — it is operator-facing commentary that the
    operator alone is responsible for.
  * Be imported by ``api/``, ``backend/evidence/``,
    ``backend/investment_analytics/``, ``backend/jobs/``, or any
    analyzer. An import-firewall test enforces this and fails CI
    if violated.
  * Mix LLM output back into any typed claim, insight template,
    analyzer response, or ranking. LLM output is non-evidentiary
    by construction and stays that way.
  * Run on a schedule or expose an HTTP endpoint. Operator-invoked
    CLI only.

## Why the firewall is enforced in tests, not just convention

The dormant-artifact memory (2026-05-16) named the exact failure
mode: authority migration through advisory channels. The shape
"we're just adding insights" → "operators rely on the insights" →
"governance defers to the insights" is how an architecture loses
its operator-attested governance property without explicitly
changing it. A CI-enforced import firewall makes accidental
authority migration a build failure, not a gradual drift.
"""
from __future__ import annotations

from backend.research_artifacts.llm_assist.notes_writer import (
    LLM_NOTES_DIR,
    LlmResearchNote,
    write_research_note,
)
from backend.research_artifacts.llm_assist.prompt_templates import (
    PROMPT_TEMPLATE_VERSION,
    build_evidence_review_prompt,
)

__all__ = [
    "LLM_NOTES_DIR",
    "LlmResearchNote",
    "PROMPT_TEMPLATE_VERSION",
    "build_evidence_review_prompt",
    "write_research_note",
]
