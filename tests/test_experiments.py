"""Tests for backend.experiments (Step 9).

Coverage map:
  Config governance:
    - methodology_kind closed enum validation
    - experiment_status closed enum validation
    - data_driven_variant requires non-empty derived_from_run_ids
    - non_semantic_metadata excluded from config_fingerprint
    - frozen: post-construction mutation of nested dicts is prevented
    - identical configs → identical fingerprints (content-addressable)

  Runner — validation:
    - unknown target → KeyError
    - param_overrides outside allowed set → ValueError
    - missing baseline_run_id → ValueError
    - missing derived_from_run_ids → ValueError

  Runner — production isolation:
    - METHODOLOGY_VERSIONS byte-identical before and after run_experiment
    - production_methodology_versions present in payload, separate from
      experiment_overrides
    - production rank_category emit path still works unchanged

  Runner — derivation depth:
    - 0 for engineered root (empty derived_from_run_ids)
    - max(parent depths) + 1 for derived experiments

  Runner — emission shape:
    - exactly one experiment_run row appended
    - parent_run_id chains to baseline
    - registry_contract recorded in payload + fingerprint in audit_event

  Replay handler:
    - present in REPLAY_HANDLERS
    - replay of an experiment_run reproduces the same content (semantic
      equivalence or exact match)
    - replay refuses when registry_contract drifts

  CLI surface:
    - list returns rows; filters work
"""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from backend.evidence import replay as rp
from backend.evidence.store import emit_evidence
from backend.experiments import (
    EXPERIMENT_STATUSES,
    METHODOLOGY_KINDS,
    ExperimentConfig,
    REGISTERED_PARAMETERIZED_FUNCS,
    find_experiment_runs,
    registry_contract,
    run_experiment,
)
from backend.investment_analytics import methodology as meth_mod
from backend.investment_analytics.audit import verify_audit_chain
from backend.investment_analytics.ranking import (
    CategoryRanking,
    FundMetrics,
    RankedFund,
)


# ── Fixtures ─────────────────────────────────────────────────────────


def _f(code: int, **kw) -> FundMetrics:
    base = dict(
        scheme_code=code,
        fund_name=f"Fund {code} Direct Plan - Growth",
        fund_house=f"AMC {code // 100}",
        excess_return_pct=2.0, max_drawdown_pct=-15.0, consistency_pct=55.0,
        volatility_pct=12.0, downside_capture_ratio=1.0,
        fund_cagr_pct=10.0, benchmark_cagr_pct=8.0,
        aligned_points=1500, history_years=8.0, drawdown_trough_date=None,
    )
    base.update(kw)
    return FundMetrics(**base)


def _ranking(category: str, codes: list,
             computed_at: str = "2026-05-15T00:00:00+00:00") -> CategoryRanking:
    funds = [_f(c, excess_return_pct=4.0 - i) for i, c in enumerate(codes)]
    ranked = [
        RankedFund(rank=i + 1, fund=f, dominance_count=len(funds) - 1 - i,
                   total_peers=len(funds), confidence_level="High",
                   strengths=[], weaknesses=[])
        for i, f in enumerate(funds)
    ]
    return CategoryRanking(
        category=category, benchmark_name="Nifty 100", benchmark_code=999,
        benchmark_fallback=False, ranked=ranked, excluded=[],
        computed_at=computed_at, total_funds_in_category=len(funds),
    )


class _ExperimentHarness(unittest.TestCase):
    """Common setUp: isolated audit + evidence + registry under tmpdir."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp.name)
        self.audit_path = self.data_dir / "audit" / "audit.jsonl"
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.data_dir / "registry" / "schemes.json"
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text("[]", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _read_events(self) -> list:
        if not self.audit_path.exists():
            return []
        with self.audit_path.open("r", encoding="utf-8") as h:
            return [json.loads(line)["event"] for line in h if line.strip()]

    def _emit_baseline_ranking(self, category="Equity Scheme - Large Cap Fund",
                                codes=(101, 102, 103)) -> str:
        from backend.investment_analytics.ranking import ranking_to_dict
        result = _ranking(category, list(codes))
        payload = ranking_to_dict(result)
        record = emit_evidence(
            audit_log_path=self.audit_path,
            evidence_kind="ranking_snapshot",
            audit_event={
                "event_type": "rank_category",
                "subject_token": "test",
                "category": category,
                "ranked_count": len(result.ranked),
                "excluded_count": 0,
                "benchmark_code": result.benchmark_code,
                "benchmark_fallback": False,
                "schema_version": "v1",
            },
            payload=payload,
        )
        return record["event"]["run_id"]


# ── Config governance ────────────────────────────────────────────────


class ExperimentConfigGovernanceTests(unittest.TestCase):

    def _valid_config(self, **overrides) -> ExperimentConfig:
        defaults = dict(
            target="rank_category",
            target_inputs={"category": "Equity Scheme - Large Cap Fund"},
            param_overrides={"MIN_ALIGNED_POINTS": 500},
            methodology_kind="engineered_variant",
            experiment_status="exploratory",
        )
        defaults.update(overrides)
        return ExperimentConfig(**defaults)

    def test_invalid_methodology_kind_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._valid_config(methodology_kind="bogus")

    def test_invalid_experiment_status_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._valid_config(experiment_status="random")

    def test_data_driven_without_derived_from_raises(self) -> None:
        # Anti-evidence-laundering invariant: a data_driven_variant
        # without a citation chain is structurally inadmissible.
        with self.assertRaises(ValueError):
            self._valid_config(
                methodology_kind="data_driven_variant",
                derived_from_run_ids=(),
            )

    def test_data_driven_with_derived_from_is_valid(self) -> None:
        cfg = self._valid_config(
            methodology_kind="data_driven_variant",
            derived_from_run_ids=("some-prior-run-id",),
        )
        self.assertEqual(cfg.methodology_kind, "data_driven_variant")
        self.assertEqual(cfg.derived_from_run_ids, ("some-prior-run-id",))

    def test_identical_configs_have_identical_fingerprints(self) -> None:
        a = self._valid_config()
        b = self._valid_config()
        self.assertEqual(a.config_fingerprint, b.config_fingerprint)

    def test_param_override_change_changes_fingerprint(self) -> None:
        a = self._valid_config(param_overrides={"MIN_ALIGNED_POINTS": 500})
        b = self._valid_config(param_overrides={"MIN_ALIGNED_POINTS": 600})
        self.assertNotEqual(a.config_fingerprint, b.config_fingerprint)

    def test_non_semantic_metadata_excluded_from_fingerprint(self) -> None:
        # rationale changes must NEVER change the experiment identity.
        # Otherwise prose silently encodes assumptions the system would
        # treat as semantic content.
        a = self._valid_config(
            non_semantic_metadata={"rationale": "first version"})
        b = self._valid_config(
            non_semantic_metadata={"rationale": "totally different text"})
        self.assertEqual(a.config_fingerprint, b.config_fingerprint)

    def test_status_change_changes_fingerprint(self) -> None:
        # experiment_status IS semantic — different status = different
        # experiment for promotion-gating purposes.
        a = self._valid_config(experiment_status="exploratory")
        b = self._valid_config(experiment_status="shadow_candidate")
        self.assertNotEqual(a.config_fingerprint, b.config_fingerprint)

    def test_nested_dict_mutation_after_construction_is_prevented(self) -> None:
        cfg = self._valid_config()
        with self.assertRaises(TypeError):
            cfg.target_inputs["category"] = "different"  # type: ignore
        with self.assertRaises(TypeError):
            cfg.param_overrides["MIN_ALIGNED_POINTS"] = 999  # type: ignore

    def test_to_payload_includes_fingerprint_and_metadata(self) -> None:
        cfg = self._valid_config(
            non_semantic_metadata={"rationale": "test"})
        payload = cfg.to_payload()
        self.assertEqual(payload["config_fingerprint"], cfg.config_fingerprint)
        self.assertEqual(payload["non_semantic_metadata"]["rationale"], "test")


# ── Runner: input validation ─────────────────────────────────────────


class RunnerValidationTests(_ExperimentHarness):

    def _valid_config(self, **overrides) -> ExperimentConfig:
        defaults = dict(
            target="rank_category",
            target_inputs={"category": "Equity Scheme - Large Cap Fund"},
            param_overrides={"MIN_ALIGNED_POINTS": 500},
            methodology_kind="engineered_variant",
            experiment_status="exploratory",
        )
        defaults.update(overrides)
        return ExperimentConfig(**defaults)

    def test_unknown_target_raises_keyerror(self) -> None:
        # Build a config with a valid target then mutate to bypass the
        # frozen-dataclass guard. Runner must refuse before any compute.
        cfg = self._valid_config()
        object.__setattr__(cfg, "target", "no_such_target")
        baseline = self._emit_baseline_ranking()
        with patch("backend.investment_analytics.ranking.rank_category") as rc:
            with self.assertRaises(KeyError):
                run_experiment(cfg, baseline_run_id=baseline,
                               audit_path=self.audit_path,
                               registry_path=self.registry_path)
            rc.assert_not_called()

    def test_disallowed_param_key_raises_before_compute(self) -> None:
        baseline = self._emit_baseline_ranking()
        cfg = self._valid_config(
            param_overrides={"NOT_ALLOWED_KEY": 999})
        with patch("backend.investment_analytics.ranking.rank_category") as rc:
            with self.assertRaises(ValueError) as ctx:
                run_experiment(cfg, baseline_run_id=baseline,
                               audit_path=self.audit_path,
                               registry_path=self.registry_path)
            rc.assert_not_called()  # MUST fail before any compute
        self.assertIn("NOT_ALLOWED_KEY", str(ctx.exception))

    def test_missing_baseline_run_id_raises(self) -> None:
        cfg = self._valid_config()
        with self.assertRaises(ValueError) as ctx:
            run_experiment(cfg,
                           baseline_run_id="00000000-0000-0000-0000-000000000000",
                           audit_path=self.audit_path,
                           registry_path=self.registry_path)
        self.assertIn("baseline_run_id", str(ctx.exception))

    def test_missing_derived_from_run_id_raises(self) -> None:
        baseline = self._emit_baseline_ranking()
        cfg = self._valid_config(
            methodology_kind="data_driven_variant",
            derived_from_run_ids=("nonexistent-run-id",),
        )
        with self.assertRaises(ValueError) as ctx:
            run_experiment(cfg, baseline_run_id=baseline,
                           audit_path=self.audit_path,
                           registry_path=self.registry_path)
        self.assertIn("nonexistent-run-id", str(ctx.exception))


# ── Runner: production isolation ─────────────────────────────────────


class RunnerProductionIsolationTests(_ExperimentHarness):
    """The most important invariant of Step 9: running an experiment
    cannot mutate METHODOLOGY_VERSIONS, must record what production
    believed separately from what the experiment varied, and must not
    contaminate production code paths."""

    def test_methodology_versions_byte_identical_before_and_after(self) -> None:
        before = json.dumps(meth_mod.METHODOLOGY_VERSIONS, sort_keys=True)
        baseline = self._emit_baseline_ranking()
        cfg = ExperimentConfig(
            target="rank_category",
            target_inputs={"category": "Equity Scheme - Large Cap Fund"},
            param_overrides={"MIN_ALIGNED_POINTS": 500},
            methodology_kind="engineered_variant",
            experiment_status="exploratory",
        )
        with patch("backend.investment_analytics.ranking.rank_category",
                   side_effect=lambda c, p: _ranking(c, [1, 2, 3])):
            run_experiment(cfg, baseline_run_id=baseline,
                           audit_path=self.audit_path,
                           registry_path=self.registry_path)
        after = json.dumps(meth_mod.METHODOLOGY_VERSIONS, sort_keys=True)
        self.assertEqual(before, after,
                         "running an experiment must NEVER mutate "
                         "METHODOLOGY_VERSIONS")

    def test_payload_splits_production_versions_from_experiment_overrides(self) -> None:
        baseline = self._emit_baseline_ranking()
        cfg = ExperimentConfig(
            target="rank_category",
            target_inputs={"category": "Equity Scheme - Large Cap Fund"},
            param_overrides={"MIN_ALIGNED_POINTS": 500},
            methodology_kind="engineered_variant",
            experiment_status="exploratory",
        )
        with patch("backend.investment_analytics.ranking.rank_category",
                   side_effect=lambda c, p: _ranking(c, [1, 2, 3])):
            record = run_experiment(cfg, baseline_run_id=baseline,
                                    audit_path=self.audit_path,
                                    registry_path=self.registry_path)
        # Load the evidence file (the payload lives there, not on the
        # audit row).
        evidence_path = self.data_dir / record["event"]["evidence_ref"]["path"]
        envelope = json.loads(evidence_path.read_text())
        payload = envelope["payload"]
        self.assertIn("production_methodology_versions", payload)
        self.assertIn("experiment_overrides", payload)
        self.assertEqual(payload["experiment_overrides"],
                         {"MIN_ALIGNED_POINTS": 500})
        # The two fields must NOT be the same object — they record
        # different concepts and must not collapse.
        self.assertNotEqual(payload["production_methodology_versions"],
                            payload["experiment_overrides"])

    def test_production_rank_category_path_unaffected(self) -> None:
        # An experiment swap is supposed to be temporary. After it
        # completes, the production rank_category should see the
        # original module constants on its very next call.
        from backend.investment_analytics.ranking import equity
        original_min = equity.MIN_ALIGNED_POINTS

        baseline = self._emit_baseline_ranking()
        cfg = ExperimentConfig(
            target="rank_category",
            target_inputs={"category": "Equity Scheme - Large Cap Fund"},
            param_overrides={"MIN_ALIGNED_POINTS": 42},
            methodology_kind="engineered_variant",
            experiment_status="exploratory",
        )
        with patch("backend.investment_analytics.ranking.rank_category",
                   side_effect=lambda c, p: _ranking(c, [1, 2, 3])):
            run_experiment(cfg, baseline_run_id=baseline,
                           audit_path=self.audit_path,
                           registry_path=self.registry_path)
        # The temporary swap must have been restored.
        self.assertEqual(equity.MIN_ALIGNED_POINTS, original_min)


# ── Runner: derivation depth ─────────────────────────────────────────


class DerivationDepthTests(_ExperimentHarness):

    def _run_basic(self, *, derived_from=(),
                   methodology_kind="engineered_variant") -> dict:
        baseline = self._emit_baseline_ranking()
        cfg = ExperimentConfig(
            target="rank_category",
            target_inputs={"category": "Equity Scheme - Large Cap Fund"},
            param_overrides={"MIN_ALIGNED_POINTS": 500},
            methodology_kind=methodology_kind,
            experiment_status="exploratory",
            derived_from_run_ids=tuple(derived_from),
        )
        with patch("backend.investment_analytics.ranking.rank_category",
                   side_effect=lambda c, p: _ranking(c, [1, 2, 3])):
            return run_experiment(cfg, baseline_run_id=baseline,
                                  audit_path=self.audit_path,
                                  registry_path=self.registry_path)

    def test_engineered_root_has_depth_zero(self) -> None:
        record = self._run_basic()
        self.assertEqual(record["event"]["derivation_depth"], 0)

    def test_derived_experiment_has_depth_one(self) -> None:
        # First experiment: depth 0 (engineered root).
        root_record = self._run_basic()
        root_run_id = root_record["event"]["run_id"]
        # Second experiment derives from the first.
        derived = self._run_basic(
            derived_from=(root_run_id,),
            methodology_kind="data_driven_variant",
        )
        self.assertEqual(derived["event"]["derivation_depth"], 1)

    def test_recursive_depth_is_max_plus_one(self) -> None:
        # depth0 root, depth1 derived, depth2 derived-from-derived.
        root = self._run_basic()
        depth1 = self._run_basic(
            derived_from=(root["event"]["run_id"],),
            methodology_kind="data_driven_variant",
        )
        self.assertEqual(depth1["event"]["derivation_depth"], 1)
        depth2 = self._run_basic(
            derived_from=(depth1["event"]["run_id"],),
            methodology_kind="data_driven_variant",
        )
        self.assertEqual(depth2["event"]["derivation_depth"], 2)


# ── Runner: emission shape ───────────────────────────────────────────


class EmissionShapeTests(_ExperimentHarness):

    def test_exactly_one_experiment_run_row_appended(self) -> None:
        baseline = self._emit_baseline_ranking()
        before = sum(1 for e in self._read_events()
                     if e.get("evidence_kind") == "experiment_run")
        cfg = ExperimentConfig(
            target="rank_category",
            target_inputs={"category": "Equity Scheme - Large Cap Fund"},
            param_overrides={"MIN_ALIGNED_POINTS": 500},
            methodology_kind="engineered_variant",
            experiment_status="exploratory",
        )
        with patch("backend.investment_analytics.ranking.rank_category",
                   side_effect=lambda c, p: _ranking(c, [1, 2, 3])):
            run_experiment(cfg, baseline_run_id=baseline,
                           audit_path=self.audit_path,
                           registry_path=self.registry_path)
        after = sum(1 for e in self._read_events()
                    if e.get("evidence_kind") == "experiment_run")
        self.assertEqual(after - before, 1)

    def test_parent_run_id_chains_to_baseline(self) -> None:
        baseline = self._emit_baseline_ranking()
        cfg = ExperimentConfig(
            target="rank_category",
            target_inputs={"category": "Equity Scheme - Large Cap Fund"},
            param_overrides={"MIN_ALIGNED_POINTS": 500},
            methodology_kind="engineered_variant",
            experiment_status="exploratory",
        )
        with patch("backend.investment_analytics.ranking.rank_category",
                   side_effect=lambda c, p: _ranking(c, [1, 2, 3])):
            record = run_experiment(cfg, baseline_run_id=baseline,
                                    audit_path=self.audit_path,
                                    registry_path=self.registry_path)
        self.assertEqual(record["event"]["parent_run_id"], baseline)

    def test_audit_event_carries_registry_contract_fingerprint(self) -> None:
        baseline = self._emit_baseline_ranking()
        cfg = ExperimentConfig(
            target="rank_category",
            target_inputs={"category": "Equity Scheme - Large Cap Fund"},
            param_overrides={"MIN_ALIGNED_POINTS": 500},
            methodology_kind="engineered_variant",
            experiment_status="exploratory",
        )
        with patch("backend.investment_analytics.ranking.rank_category",
                   side_effect=lambda c, p: _ranking(c, [1, 2, 3])):
            record = run_experiment(cfg, baseline_run_id=baseline,
                                    audit_path=self.audit_path,
                                    registry_path=self.registry_path)
        expected = registry_contract("rank_category")["fingerprint"]
        self.assertEqual(record["event"]["registry_contract_fingerprint"],
                         expected)


# ── Replay handler ───────────────────────────────────────────────────


class ExperimentReplayHandlerTests(_ExperimentHarness):

    def test_experiment_run_present_in_replay_handlers(self) -> None:
        self.assertIn("experiment_run", rp.REPLAY_HANDLERS)

    def test_replay_reproduces_recorded_content(self) -> None:
        baseline = self._emit_baseline_ranking()
        cfg = ExperimentConfig(
            target="rank_category",
            target_inputs={"category": "Equity Scheme - Large Cap Fund"},
            param_overrides={"MIN_ALIGNED_POINTS": 500},
            methodology_kind="engineered_variant",
            experiment_status="exploratory",
        )
        with patch("backend.investment_analytics.ranking.rank_category",
                   side_effect=lambda c, p: _ranking(c, [1, 2, 3])):
            record = run_experiment(cfg, baseline_run_id=baseline,
                                    audit_path=self.audit_path,
                                    registry_path=self.registry_path)
            run_id = record["event"]["run_id"]
            result = rp.replay_run(
                self.audit_path, run_id,
                verify_chain=False, emit_audit=False,
                registry_path=self.registry_path,
            )
        # Identical computed_at means we hit exact_match; otherwise
        # the volatile-strip layer would land us on semantic equivalent.
        self.assertIn(result["state"], ("exact_match", "semantically_equivalent"),
                      f"got state={result['state']}, reason={result.get('reason')}")

    def test_replay_refuses_when_registry_contract_drifts(self) -> None:
        baseline = self._emit_baseline_ranking()
        cfg = ExperimentConfig(
            target="rank_category",
            target_inputs={"category": "Equity Scheme - Large Cap Fund"},
            param_overrides={"MIN_ALIGNED_POINTS": 500},
            methodology_kind="engineered_variant",
            experiment_status="exploratory",
        )
        with patch("backend.investment_analytics.ranking.rank_category",
                   side_effect=lambda c, p: _ranking(c, [1, 2, 3])):
            record = run_experiment(cfg, baseline_run_id=baseline,
                                    audit_path=self.audit_path,
                                    registry_path=self.registry_path)
            run_id = record["event"]["run_id"]

            # Simulate contract drift: shrink allowed_param_keys for
            # the target. Replay must refuse.
            from backend.experiments import registry as reg_mod
            original_entry = reg_mod.REGISTERED_PARAMETERIZED_FUNCS["rank_category"]
            from dataclasses import replace
            tightened = replace(original_entry,
                                allowed_param_keys=frozenset({"MIN_ALIGNED_POINTS"}))
            reg_mod.REGISTERED_PARAMETERIZED_FUNCS["rank_category"] = tightened
            try:
                result = rp.replay_run(
                    self.audit_path, run_id,
                    verify_chain=False, emit_audit=False,
                    registry_path=self.registry_path,
                )
            finally:
                reg_mod.REGISTERED_PARAMETERIZED_FUNCS["rank_category"] = original_entry
        self.assertEqual(result["state"], "invalid_replay")
        self.assertIn("registry_contract drift", result["reason"])


# ── find_experiment_runs ─────────────────────────────────────────────


class FindExperimentRunsTests(_ExperimentHarness):

    def test_empty_chain_returns_empty_list(self) -> None:
        # File doesn't even exist.
        self.assertEqual(find_experiment_runs(audit_path=self.audit_path), [])

    def test_target_filter_works(self) -> None:
        baseline = self._emit_baseline_ranking()
        cfg = ExperimentConfig(
            target="rank_category",
            target_inputs={"category": "Equity Scheme - Large Cap Fund"},
            param_overrides={"MIN_ALIGNED_POINTS": 500},
            methodology_kind="engineered_variant",
            experiment_status="exploratory",
        )
        with patch("backend.investment_analytics.ranking.rank_category",
                   side_effect=lambda c, p: _ranking(c, [1, 2, 3])):
            run_experiment(cfg, baseline_run_id=baseline,
                           audit_path=self.audit_path,
                           registry_path=self.registry_path)
        found = find_experiment_runs(audit_path=self.audit_path,
                                      target="rank_category")
        self.assertEqual(len(found), 1)
        none = find_experiment_runs(audit_path=self.audit_path,
                                    target="nonexistent")
        self.assertEqual(none, [])

    def test_status_filter_works(self) -> None:
        baseline = self._emit_baseline_ranking()
        for status in ("exploratory", "validation"):
            cfg = ExperimentConfig(
                target="rank_category",
                target_inputs={"category": "Equity Scheme - Large Cap Fund"},
                param_overrides={"MIN_ALIGNED_POINTS": 500},
                methodology_kind="engineered_variant",
                experiment_status=status,
            )
            with patch("backend.investment_analytics.ranking.rank_category",
                       side_effect=lambda c, p: _ranking(c, [1, 2, 3])):
                run_experiment(cfg, baseline_run_id=baseline,
                               audit_path=self.audit_path,
                               registry_path=self.registry_path)
        exploratory = find_experiment_runs(audit_path=self.audit_path,
                                           experiment_status="exploratory")
        validation = find_experiment_runs(audit_path=self.audit_path,
                                          experiment_status="validation")
        self.assertEqual(len(exploratory), 1)
        self.assertEqual(len(validation), 1)


# ── Chain integrity after experiments ────────────────────────────────


class ChainIntegrityTests(_ExperimentHarness):

    def test_audit_chain_valid_after_experiment_sequence(self) -> None:
        baseline = self._emit_baseline_ranking()
        for i in range(3):
            cfg = ExperimentConfig(
                target="rank_category",
                target_inputs={"category": "Equity Scheme - Large Cap Fund"},
                param_overrides={"MIN_ALIGNED_POINTS": 500 + i},
                methodology_kind="engineered_variant",
                experiment_status="exploratory",
            )
            with patch("backend.investment_analytics.ranking.rank_category",
                       side_effect=lambda c, p: _ranking(c, [1, 2, 3])):
                run_experiment(cfg, baseline_run_id=baseline,
                               audit_path=self.audit_path,
                               registry_path=self.registry_path)
        self.assertTrue(verify_audit_chain(self.audit_path))


# ── CLI ──────────────────────────────────────────────────────────────


class CliTests(_ExperimentHarness):

    def _run_cli(self, argv: list) -> tuple[int, str]:
        from backend.experiments.__main__ import _main
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _main(argv)
        return rc, buf.getvalue()

    def test_list_subcommand_prints_human_summary(self) -> None:
        baseline = self._emit_baseline_ranking()
        cfg = ExperimentConfig(
            target="rank_category",
            target_inputs={"category": "Equity Scheme - Large Cap Fund"},
            param_overrides={"MIN_ALIGNED_POINTS": 500},
            methodology_kind="engineered_variant",
            experiment_status="exploratory",
        )
        with patch("backend.investment_analytics.ranking.rank_category",
                   side_effect=lambda c, p: _ranking(c, [1, 2, 3])):
            run_experiment(cfg, baseline_run_id=baseline,
                           audit_path=self.audit_path,
                           registry_path=self.registry_path)
        rc, out = self._run_cli(["list",
                                 "--audit-path", str(self.audit_path)])
        self.assertEqual(rc, 0, out)
        self.assertIn("experiment_run rows: 1", out)
        self.assertIn("rank_category", out)

    def test_list_filters_pass_through(self) -> None:
        rc, out = self._run_cli([
            "list",
            "--audit-path", str(self.audit_path),
            "--target", "nonexistent",
        ])
        self.assertEqual(rc, 0)
        self.assertIn("(no experiment_run rows match", out)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
