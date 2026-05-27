"""Versioned prompt templates for the LLM research assistant.

Bump ``PROMPT_TEMPLATE_VERSION`` whenever the prompt body changes
in a way that affects model behavior. The version is recorded in
every research note so two notes produced by different prompts
cannot be confused.

## Why the prompt is paranoid

The model has no access to the live audit chain or evidence layer.
It only sees what we hand it as text in the prompt. If the prompt
permits invention, the model will invent. If the prompt permits
hedging, the model will hedge. The template below explicitly:

  * forbids inventing any numeric value not present in the input;
  * forbids speculating about future prices or returns;
  * requires the model to refuse rather than guess when uncertain;
  * frames the output as commentary for an operator who has the
    actual numbers, not as standalone advice;
  * forbids the use of advisory language ("you should", "buy",
    "sell", "I recommend") so the output cannot be mistaken for
    a personalized investment action.

These are belt-and-suspenders alongside the architectural firewall.
The architectural firewall guarantees no LLM text reaches a user
via a typed claim; the prompt guards reduce the chance the operator
themselves is misled when reading the note.
"""
from __future__ import annotations

import json
from typing import Any, Mapping


PROMPT_TEMPLATE_VERSION = "v1"


_SYSTEM_INSTRUCTION = """\
You are a read-only research assistant reviewing one or more
deterministic evidence files from an investment analytics engine.
You are NOT an advisor, NOT a portfolio manager, and NOT a decision-
maker. Your output is non-evidentiary commentary that a human
operator will read alongside the actual numbers.

Hard rules — violating any of these is a refusal failure, and the
operator is instructed to discard your output if you violate them:

  1. Do not invent, estimate, or extrapolate any numeric value that
     is not literally present in the evidence the user gives you.
     If a number would be useful and is missing, say "not present
     in provided evidence" — do not fill it in.
  2. Do not speculate about future prices, future returns, or
     future market conditions. Past observations only.
  3. Do not use advisory language. Forbidden phrases include
     "you should", "I recommend", "buy", "sell", "hold", "invest",
     "trade", "target price". Use neutral, observational language:
     "the data shows", "the recorded metric is", "one possible
     reading of this is".
  4. When uncertain, refuse the specific sub-question. The exact
     refusal text "REFUSED: <reason>" must appear on its own line.
     Refusal is a valid and preferred response over a guess.
  5. Do not claim access to anything beyond the evidence text the
     user provides. You have no live market data, no news feed,
     no other files.
  6. If the evidence appears internally inconsistent, flag the
     inconsistency rather than averaging or guessing around it.

Your output is appended to a file marked NON-EVIDENTIARY and
NON-REPLAYABLE. The operator will verify every numeric claim you
make against the underlying evidence file before acting on it.
"""


def build_evidence_review_prompt(
    evidence_files: Mapping[str, Any],
    operator_question: str,
) -> str:
    """Render the full prompt for a research-note request.

    Args:
      evidence_files: mapping of ``{label: parsed_envelope_dict}``
        where each value is the full evidence envelope (the JSON
        loaded from ``data/evidence/<kind>/<run_id>.json``). The
        full envelope is sent so the model can see methodology
        versions, run_id, generated_at — provenance the operator
        will want commentary on.
      operator_question: free-text question the operator wants
        answered. Passed through verbatim; the system instruction
        above constrains what the model can say in response.

    Returns:
      The complete prompt string to send to Gemini.
    """
    if not evidence_files:
        raise ValueError(
            "evidence_files must contain at least one file; the LLM "
            "assistant refuses to operate without grounding evidence."
        )
    if not operator_question or not operator_question.strip():
        raise ValueError(
            "operator_question must be a non-empty string; the LLM "
            "assistant refuses to operate without a typed question."
        )

    sections = [_SYSTEM_INSTRUCTION, "", "=== EVIDENCE FILES ==="]
    for label, envelope in evidence_files.items():
        sections.append(f"\n--- {label} ---")
        sections.append(json.dumps(envelope, indent=2, sort_keys=True))
    sections.append("\n=== OPERATOR QUESTION ===")
    sections.append(operator_question.strip())
    sections.append("\n=== YOUR RESPONSE ===")
    sections.append(
        "Respond with: (a) a short observational answer grounded in "
        "the evidence above, (b) any 'REFUSED: <reason>' lines for "
        "sub-questions you cannot answer, and (c) a list of every "
        "numeric value you cited and which evidence file it came "
        "from, so the operator can verify."
    )
    return "\n".join(sections)
