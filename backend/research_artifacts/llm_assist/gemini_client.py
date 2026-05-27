"""Thin Gemini SDK wrapper with refusal-by-default error handling.

Encapsulates the entire google-generativeai surface area used by the
sandbox so the rest of the package never imports the SDK directly.
That keeps the import-firewall test simple (any import of
``google.generativeai`` outside this file is a violation) and makes
it trivial to swap or mock the underlying client in tests.

## Failure modes are typed refusals, not silent shrugs

  * SDK not installed → ``LlmDependencyMissing`` with install
    instructions.
  * API key missing → ``LlmConfigError`` naming the env var.
  * Network / API error → ``LlmCallFailed`` with the underlying
    exception preserved.
  * Empty / blocked model response → ``LlmRefusedToAnswer`` — the
    sandbox treats this as a valid refusal, NOT as an excuse to
    write an empty note. Upstream callers MUST surface the
    refusal to the operator rather than retry until a non-empty
    response appears.

All four are subclasses of ``LlmAssistError`` so a CLI can catch
the whole family and exit non-zero with a clean message.
"""
from __future__ import annotations

import os
from typing import Optional


DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_TEMPERATURE = 0.0
API_KEY_ENV_VAR = "GEMINI_API_KEY"


class LlmAssistError(Exception):
    """Base class for any failure in the LLM sandbox."""


class LlmDependencyMissing(LlmAssistError):
    """The google-generativeai SDK is not installed."""


class LlmConfigError(LlmAssistError):
    """Required configuration is missing (e.g. API key env var)."""


class LlmCallFailed(LlmAssistError):
    """The model call itself errored (network, quota, API error)."""


class LlmRefusedToAnswer(LlmAssistError):
    """The model returned empty or a safety-block — treated as refusal."""


def _import_sdk():
    """Lazy import so the package is usable in environments where
    google-generativeai is not installed (the rest of the analytics
    engine remains fully functional). The import failure is converted
    into a typed refusal the CLI can present cleanly.
    """
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise LlmDependencyMissing(
            "google-generativeai is not installed. Install the optional "
            "LLM dependencies with: python -m pip install -r "
            "requirements-llm.txt"
        ) from exc
    return genai


def call_gemini(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """Send ``prompt`` to Gemini and return the model's text response.

    Args:
      prompt:       full prompt string (system instruction + evidence
                    + question; built by prompt_templates).
      model:        model id. Defaults to gemini-2.5-pro.
      api_key:      optional explicit key; otherwise read from the
                    GEMINI_API_KEY env var. Never written to any file.
      temperature:  defaults to 0 for as-deterministic-as-possible
                    output. NOTE: temperature=0 does NOT make Gemini
                    replayable — model weights are opaque and may be
                    updated by the vendor at any time. The research
                    note envelope marks every artifact non_replayable
                    regardless of this setting.

    Returns the raw text of the model's response.

    Raises:
      LlmDependencyMissing: SDK not installed.
      LlmConfigError:       API key not available.
      LlmCallFailed:        network / API error.
      LlmRefusedToAnswer:   model returned empty / safety-blocked.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    resolved_key = api_key or os.environ.get(API_KEY_ENV_VAR)
    if not resolved_key:
        raise LlmConfigError(
            f"API key not provided. Set the {API_KEY_ENV_VAR} environment "
            f"variable or pass api_key explicitly. The key is never "
            f"written to any file by this sandbox."
        )

    genai = _import_sdk()
    try:
        genai.configure(api_key=resolved_key)
        model_obj = genai.GenerativeModel(model)
        response = model_obj.generate_content(
            prompt,
            generation_config={"temperature": temperature},
        )
    except LlmAssistError:
        raise
    except Exception as exc:
        raise LlmCallFailed(
            f"Gemini call failed: {type(exc).__name__}: {exc}"
        ) from exc

    # Defensive: the SDK has multiple ways to express "no output"
    # (empty .text, safety-blocked candidates, partial responses).
    # Treat all of them as a refusal — empty text is meaningful
    # signal, not an excuse to write an empty note.
    text = getattr(response, "text", None)
    if not text or not text.strip():
        raise LlmRefusedToAnswer(
            "Model returned no usable text (empty response, safety "
            "block, or malformed candidates). Recording as a refusal "
            "upstream; do not retry blindly."
        )
    return text
