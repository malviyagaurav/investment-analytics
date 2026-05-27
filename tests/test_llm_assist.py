"""Tests for backend.research_artifacts.llm_assist.

The most load-bearing test in this module is
``TestImportFirewall.test_no_production_module_imports_llm_assist``:
that single test enforces the architectural promise that LLM output
cannot reach a typed claim. Every other test here is belt-and-
suspenders around the firewall.
"""
from __future__ import annotations

import io
import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from backend.research_artifacts.llm_assist import gemini_client
from backend.research_artifacts.llm_assist.cli import EXIT_OK, EXIT_REFUSED, main as cli_main
from backend.research_artifacts.llm_assist.notes_writer import (
    LlmResearchNote,
    _EvidenceReference,
    reference_evidence_file,
    write_research_note,
)
from backend.research_artifacts.llm_assist.prompt_templates import (
    PROMPT_TEMPLATE_VERSION,
    build_evidence_review_prompt,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_ref(label: str = "ev") -> _EvidenceReference:
    return _EvidenceReference(
        evidence_kind="portfolio_health_snapshot",
        evidence_path=f"data/evidence/portfolio_health_snapshot/{label}.json",
        evidence_sha256="0" * 64,
        run_id=label,
    )


# ────────────────────────────────────────────────────────────────────
# Import firewall — the most important test in this module.
# ────────────────────────────────────────────────────────────────────


class TestImportFirewall(unittest.TestCase):
    """LLM sandbox must not be imported by any production module.

    If this test fails, an LLM dependency has leaked into a path
    that produces typed claims — which is exactly the architectural
    failure mode the sandbox exists to prevent.
    """

    FORBIDDEN_ROOTS = [
        REPO_ROOT / "api",
        REPO_ROOT / "backend" / "evidence",
        REPO_ROOT / "backend" / "investment_analytics",
        REPO_ROOT / "backend" / "jobs",
        REPO_ROOT / "backend" / "calibration",
        REPO_ROOT / "backend" / "governance",
        REPO_ROOT / "backend" / "regimes",
        REPO_ROOT / "backend" / "reliability",
        REPO_ROOT / "backend" / "data_discovery",
        REPO_ROOT / "backend" / "data_ingestion",
        REPO_ROOT / "backend" / "experiments",
        REPO_ROOT / "backend" / "scheduler",
    ]

    def _scan(self, needle: str):
        """Return list of (path, lineno, line) for any occurrence."""
        hits = []
        for root in self.FORBIDDEN_ROOTS:
            if not root.exists():
                continue
            for path in root.rglob("*.py"):
                # Skip __pycache__ etc.
                if "__pycache__" in path.parts:
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                for i, line in enumerate(text.splitlines(), start=1):
                    if needle in line:
                        hits.append((path, i, line.strip()))
        return hits

    def test_no_production_module_imports_llm_assist(self):
        hits = self._scan("llm_assist")
        self.assertEqual(
            hits, [],
            "llm_assist must not be referenced by any production "
            "module. Found:\n" + "\n".join(
                f"  {p}:{ln}: {ls}" for p, ln, ls in hits
            ),
        )

    def test_no_production_module_imports_google_generativeai(self):
        hits = self._scan("google.generativeai")
        # Allow the firewall test file itself (this file) and the
        # gemini_client module, but nothing else.
        offending = [
            (p, ln, ls) for p, ln, ls in hits
            if p.resolve() != Path(__file__).resolve()
        ]
        self.assertEqual(
            offending, [],
            "google.generativeai must only be imported from the LLM "
            "sandbox's gemini_client.py. Leaks:\n" + "\n".join(
                f"  {p}:{ln}: {ls}" for p, ln, ls in offending
            ),
        )

    def test_only_gemini_client_imports_sdk(self):
        """Within the sandbox, only gemini_client.py imports the SDK."""
        sandbox = REPO_ROOT / "backend" / "research_artifacts" / "llm_assist"
        offending = []
        for path in sandbox.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            if path.name == "gemini_client.py":
                continue
            text = path.read_text(encoding="utf-8")
            for i, line in enumerate(text.splitlines(), start=1):
                if "google.generativeai" in line:
                    offending.append((path, i, line.strip()))
        self.assertEqual(
            offending, [],
            "Only gemini_client.py may import google.generativeai. "
            "Leaks within the sandbox:\n" + "\n".join(
                f"  {p}:{ln}: {ls}" for p, ln, ls in offending
            ),
        )


# ────────────────────────────────────────────────────────────────────
# Note envelope shape — every non_* marker is load-bearing.
# ────────────────────────────────────────────────────────────────────


class TestNoteEnvelope(unittest.TestCase):
    def test_envelope_has_every_non_evidentiary_marker(self):
        with TemporaryDirectory() as tmp:
            out = write_research_note(
                operator_question="what does this say?",
                raw_output="observation: the recorded value is X",
                model="gemini-2.5-pro",
                prompt_template_version=PROMPT_TEMPLATE_VERSION,
                input_evidence_refs=[_make_ref()],
                notes_dir=Path(tmp),
            )
            data = json.loads(out.read_text(encoding="utf-8"))
        # Every marker must be present and true.
        self.assertTrue(data["non_evidentiary"])
        self.assertTrue(data["non_replayable"])
        self.assertTrue(data["non_attributable"])
        self.assertTrue(data["operator_decides"])
        self.assertEqual(data["artifact_kind"], "llm_research_note")
        # Operator warning text must contain the canonical phrase.
        self.assertIn("NOT EVIDENCE", data["operator_warning"])
        self.assertIn("NOT REPLAYABLE", data["operator_warning"])
        # Filename prefix makes the non-evidentiary status visible
        # in any directory listing.
        self.assertTrue(out.name.startswith("LLM_NOTE_"))

    def test_envelope_records_input_evidence_refs(self):
        with TemporaryDirectory() as tmp:
            out = write_research_note(
                operator_question="q",
                raw_output="a",
                model="gemini-2.5-pro",
                prompt_template_version=PROMPT_TEMPLATE_VERSION,
                input_evidence_refs=[_make_ref("a"), _make_ref("b")],
                notes_dir=Path(tmp),
            )
            data = json.loads(out.read_text(encoding="utf-8"))
        refs = data["input_evidence_refs"]
        self.assertEqual(len(refs), 2)
        self.assertEqual(refs[0]["run_id"], "a")
        self.assertEqual(refs[1]["run_id"], "b")

    def test_envelope_records_template_version(self):
        with TemporaryDirectory() as tmp:
            out = write_research_note(
                operator_question="q",
                raw_output="a",
                model="m",
                prompt_template_version="v99",
                input_evidence_refs=[_make_ref()],
                notes_dir=Path(tmp),
            )
            data = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(data["prompt_template_version"], "v99")


# ────────────────────────────────────────────────────────────────────
# Refusal-by-default — the writer refuses to produce malformed notes.
# ────────────────────────────────────────────────────────────────────


class TestNoteWriterRefusals(unittest.TestCase):
    def test_refuses_empty_evidence_refs(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as cm:
                write_research_note(
                    operator_question="q",
                    raw_output="a",
                    model="m",
                    prompt_template_version="v1",
                    input_evidence_refs=[],
                    notes_dir=Path(tmp),
                )
        self.assertIn("grounding", str(cm.exception))

    def test_refuses_empty_question(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_research_note(
                    operator_question="   ",
                    raw_output="a",
                    model="m",
                    prompt_template_version="v1",
                    input_evidence_refs=[_make_ref()],
                    notes_dir=Path(tmp),
                )

    def test_refuses_empty_output(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_research_note(
                    operator_question="q",
                    raw_output="",
                    model="m",
                    prompt_template_version="v1",
                    input_evidence_refs=[_make_ref()],
                    notes_dir=Path(tmp),
                )


# ────────────────────────────────────────────────────────────────────
# Evidence reference hashing.
# ────────────────────────────────────────────────────────────────────


class TestEvidenceReference(unittest.TestCase):
    def test_reference_captures_sha256_and_run_id(self):
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "ev.json"
            payload = {"run_id": "abc-123", "payload": {"x": 1}}
            f.write_text(json.dumps(payload), encoding="utf-8")
            ref = reference_evidence_file(
                evidence_kind="portfolio_health_snapshot",
                evidence_path=f,
                repo_root=Path(tmp),
            )
        self.assertEqual(ref.run_id, "abc-123")
        self.assertEqual(len(ref.evidence_sha256), 64)
        # Forward-slash form regardless of OS — matches the byte-
        # stability convention used elsewhere.
        self.assertNotIn("\\", ref.evidence_path)

    def test_reference_to_missing_file_refuses(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                reference_evidence_file(
                    evidence_kind="x",
                    evidence_path=Path(tmp) / "nope.json",
                    repo_root=Path(tmp),
                )


# ────────────────────────────────────────────────────────────────────
# Prompt template — refusals and version.
# ────────────────────────────────────────────────────────────────────


class TestPromptTemplate(unittest.TestCase):
    def test_template_version_is_pinned(self):
        # Bumping the version is intentional; this test exists to make
        # a silent change to the prompt body visible in code review.
        self.assertEqual(PROMPT_TEMPLATE_VERSION, "v1")

    def test_builder_refuses_empty_evidence(self):
        with self.assertRaises(ValueError):
            build_evidence_review_prompt({}, "what is this?")

    def test_builder_refuses_empty_question(self):
        with self.assertRaises(ValueError):
            build_evidence_review_prompt({"a": {"k": 1}}, "   ")

    def test_prompt_includes_no_hallucination_rule(self):
        out = build_evidence_review_prompt(
            {"a": {"run_id": "x"}}, "summarise this",
        )
        # Rule #1 must be present verbatim — the prompt's whole point.
        self.assertIn("not invent", out.lower())
        self.assertIn("REFUSED:", out)

    def test_prompt_forbids_advisory_language(self):
        out = build_evidence_review_prompt(
            {"a": {"run_id": "x"}}, "summarise",
        )
        # The advisory-language ban echoes the project's existing
        # advisory-language linter constraint.
        for forbidden in ["you should", "I recommend", "buy", "sell"]:
            self.assertIn(forbidden, out)


# ────────────────────────────────────────────────────────────────────
# Gemini client — typed refusals.
# ────────────────────────────────────────────────────────────────────


class TestGeminiClientRefusals(unittest.TestCase):
    def test_missing_api_key_raises_config_error(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(gemini_client.LlmConfigError):
                gemini_client.call_gemini("hello", api_key=None)

    def test_empty_prompt_raises_value_error(self):
        with self.assertRaises(ValueError):
            gemini_client.call_gemini("", api_key="x")

    def test_empty_response_is_refusal(self):
        fake_response = mock.Mock()
        fake_response.text = "   "
        fake_model = mock.Mock()
        fake_model.generate_content.return_value = fake_response
        fake_genai = mock.Mock()
        fake_genai.GenerativeModel.return_value = fake_model

        with mock.patch.object(
            gemini_client, "_import_sdk", return_value=fake_genai,
        ):
            with self.assertRaises(gemini_client.LlmRefusedToAnswer):
                gemini_client.call_gemini("hi", api_key="x")

    def test_sdk_error_becomes_call_failed(self):
        fake_model = mock.Mock()
        fake_model.generate_content.side_effect = RuntimeError("net down")
        fake_genai = mock.Mock()
        fake_genai.GenerativeModel.return_value = fake_model

        with mock.patch.object(
            gemini_client, "_import_sdk", return_value=fake_genai,
        ):
            with self.assertRaises(gemini_client.LlmCallFailed):
                gemini_client.call_gemini("hi", api_key="x")


# ────────────────────────────────────────────────────────────────────
# CLI — exit codes and flow.
# ────────────────────────────────────────────────────────────────────


class TestCli(unittest.TestCase):
    def _write_evidence(self, dir: Path, run_id: str) -> Path:
        path = dir / f"{run_id}.json"
        envelope = {
            "evidence_kind": "portfolio_health_snapshot",
            "run_id": run_id,
            "payload": {"coverage_pct": 0.85},
        }
        path.write_text(json.dumps(envelope), encoding="utf-8")
        return path

    def test_cli_success_writes_note(self):
        with TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            ev = self._write_evidence(tmpp, "abc")
            notes_dir = tmpp / "notes"
            with mock.patch(
                "backend.research_artifacts.llm_assist.cli.call_gemini",
                return_value="observation: coverage_pct=0.85",
            ):
                exit_code = cli_main([
                    "--evidence", str(ev),
                    "--question", "what is the coverage?",
                    "--notes-dir", str(notes_dir),
                    "--no-audit-pointer",
                ])
            self.assertEqual(exit_code, EXIT_OK)
            notes = list(notes_dir.glob("LLM_NOTE_*.json"))
            self.assertEqual(len(notes), 1)
            data = json.loads(notes[0].read_text(encoding="utf-8"))
            self.assertTrue(data["non_evidentiary"])

    def test_cli_missing_evidence_exits_bad_invocation(self):
        with TemporaryDirectory() as tmp:
            exit_code = cli_main([
                "--evidence", str(Path(tmp) / "nope.json"),
                "--question", "q",
                "--notes-dir", tmp,
                "--no-audit-pointer",
            ])
        self.assertEqual(exit_code, 3)

    def test_cli_llm_refusal_exits_refused(self):
        with TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            ev = self._write_evidence(tmpp, "abc")
            with mock.patch(
                "backend.research_artifacts.llm_assist.cli.call_gemini",
                side_effect=gemini_client.LlmRefusedToAnswer("empty"),
            ):
                exit_code = cli_main([
                    "--evidence", str(ev),
                    "--question", "q",
                    "--notes-dir", str(tmpp / "notes"),
                    "--no-audit-pointer",
                ])
        self.assertEqual(exit_code, EXIT_REFUSED)


if __name__ == "__main__":
    unittest.main()
