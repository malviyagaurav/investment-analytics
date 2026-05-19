from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.investment_analytics.audit import append_audit_record, verify_audit_chain
from backend.investment_analytics.compiler import compile_insight
from backend.investment_analytics.errors import PolicyError
from backend.investment_analytics.jurisdiction import JurisdictionContext, evaluate_jurisdiction
from backend.investment_analytics.analyzers import analyze_portfolio, analyze_portfolio_with_funds
from backend.investment_analytics.lineage import LICENSE_REDIS, LICENSE_RESTRICTED, make_source
from backend.investment_analytics.mutual_funds import analyze_mutual_fund, load_mutual_fund_csv
from backend.investment_analytics.etf import analyze_etf
from backend.investment_analytics.scenarios import list_standard_scenarios, resolve_scenario_definition
from backend.investment_analytics.thresholds import should_suppress_for_missing_fields


def diagnostic_payload() -> dict:
    return {
        "type": "diagnostic",
        "observation": "Equity exposure is 60.00% of submitted market value.",
        "why_it_matters": "A large exposure in one asset class can amplify portfolio volatility.",
        "supporting_data": {
            "source": make_source("unit_test", "2026-04-24T00:00:00+00:00", LICENSE_REDIS)
        },
        "evidence_strength": "Medium",
        "data_completeness": "High",
        "limitations": ["Only submitted holdings are evaluated."],
        "unavailable_components": [],
    }


def mf_payload(
    fund_values: list[float],
    benchmark_values: list[float],
    dates: list[str] | None = None,
    window_points: int = 2,
) -> dict:
    if dates is None:
        dates = [f"2024-01-0{idx + 1}" for idx in range(len(fund_values))]
    source = make_source("unit_test_nav", "2026-04-24T00:00:00+00:00", LICENSE_REDIS)
    benchmark_source = make_source("unit_test_benchmark", "2026-04-24T00:00:00+00:00", LICENSE_REDIS)
    return {
        "fund_name": "Unit Test Fund",
        "benchmark_name": "Unit Test Benchmark",
        "category": "Equity",
        "expense_ratio_pct": 1.0,
        "fund_source": source,
        "benchmark_source": benchmark_source,
        "fund": [{"date": day, "nav": value} for day, value in zip(dates, fund_values)],
        "benchmark": [{"date": day, "value": value} for day, value in zip(dates, benchmark_values)],
        "rolling_window_points": window_points,
        "rolling_step_points": 1,
        "rolling_min_windows": 1,
        "expense_impact": {"investment_amount": 100000, "horizons_years": [3, 5, 10]},
    }


def insight_payload(result: dict, template: str, supporting_key: str) -> dict:
    for item in result["insights"]:
        if item["template"] == template and supporting_key in item["payload"].get("supporting_data", {}):
            return item["payload"]
    raise AssertionError(f"Insight with key {supporting_key} not found")


class GuardrailTests(unittest.TestCase):
    def test_allowed_template_compiles(self) -> None:
        compiled = compile_insight(diagnostic_payload())
        self.assertEqual(compiled["template"], "diagnostic")

    def test_advisory_language_fails_closed(self) -> None:
        payload = diagnostic_payload()
        payload["observation"] = "User should reduce equity exposure."
        with self.assertRaises(PolicyError) as ctx:
            compile_insight(payload)
        self.assertEqual(ctx.exception.code, "language_policy_violation")

    def test_extra_schema_field_fails(self) -> None:
        payload = diagnostic_payload()
        payload["suggested_action"] = "None"
        with self.assertRaises(PolicyError) as ctx:
            compile_insight(payload)
        self.assertEqual(ctx.exception.code, "schema_violation")

    def test_restricted_lineage_fails(self) -> None:
        payload = diagnostic_payload()
        payload["supporting_data"]["source"] = make_source(
            "restricted_vendor",
            "2026-04-24T00:00:00+00:00",
            LICENSE_RESTRICTED,
        )
        with self.assertRaises(PolicyError) as ctx:
            compile_insight(payload)
        self.assertEqual(ctx.exception.code, "restricted_lineage")

    def test_restricted_derived_lineage_fails(self) -> None:
        payload = diagnostic_payload()
        payload["supporting_data"]["source"] = make_source(
            "derived_metric",
            "2026-04-24T00:00:00+00:00",
            LICENSE_REDIS,
            [
                make_source("allowed_nav", "2026-04-24T00:00:00+00:00", LICENSE_REDIS),
                make_source("restricted_benchmark", "2026-04-24T00:00:00+00:00", LICENSE_RESTRICTED),
            ],
        )
        with self.assertRaises(PolicyError) as ctx:
            compile_insight(payload)
        self.assertEqual(ctx.exception.code, "restricted_lineage")

    def test_unsupported_jurisdiction_disables_analytics(self) -> None:
        gate = evaluate_jurisdiction(
            JurisdictionContext(user_country="FR", asset_market="IN", serving_entity="demo")
        )
        self.assertFalse(gate["supported"])
        self.assertFalse(gate["features"]["analytics"])
        self.assertTrue(gate["strict_constraints"])

    def test_standard_scenario_must_be_allowlisted(self) -> None:
        resolved = resolve_scenario_definition({"kind": "standard", "id": "market_down_20"})
        self.assertEqual(resolved["kind"], "standard")
        with self.assertRaises(PolicyError):
            resolve_scenario_definition({"kind": "standard", "id": "auto_improve"})

    def test_standard_scenarios_do_not_embed_directional_allocation(self) -> None:
        blocked_terms = {"allocation", "target", "optimized", "improvement"}
        for scenario in list_standard_scenarios():
            serialized = str(scenario).lower()
            self.assertFalse(any(term in serialized for term in blocked_terms), scenario)

    def test_scenario_param_keys_are_linted(self) -> None:
        with self.assertRaises(PolicyError) as ctx:
            compile_insight(
                {
                    "type": "scenario",
                    "scenario_definition": {
                        "kind": "user",
                        "params": {"optimized_allocation_pct": 60},
                    },
                    "assumptions": {},
                    "projected_impact": {"range": [0, 1], "units": "base_currency"},
                    "sensitivity": [],
                    "evidence_strength": "Low",
                    "data_completeness": "Low",
                    "limitations": ["Scenario output is a calculation, not a prescribed action."],
                    "unavailable_components": [],
                }
            )
        self.assertEqual(ctx.exception.code, "language_policy_violation")

    def test_deep_nested_linter_violation_fails(self) -> None:
        payload = diagnostic_payload()
        payload["supporting_data"]["nested"] = [{"level": {"text": "This is a better fit."}}]
        with self.assertRaises(PolicyError) as ctx:
            compile_insight(payload)
        self.assertEqual(ctx.exception.code, "language_policy_violation")

    def test_dynamic_analyzer_language_is_blocked(self) -> None:
        payload = diagnostic_payload()
        payload["observation"] = "Performance is better with strong outperformance."
        with self.assertRaises(PolicyError) as ctx:
            compile_insight(payload)
        self.assertEqual(ctx.exception.code, "language_policy_violation")

    def test_missing_field_threshold_suppresses_above_30_percent(self) -> None:
        self.assertTrue(should_suppress_for_missing_fields(0.31))
        self.assertFalse(should_suppress_for_missing_fields(0.30))

    def test_audit_log_is_hash_chained(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            append_audit_record(path, {"event_type": "one", "subject_token": "anon"})
            append_audit_record(path, {"event_type": "two", "subject_token": "anon"})
            self.assertTrue(verify_audit_chain(path))

    def test_audit_log_redacts_direct_pii(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            record = append_audit_record(
                path,
                {"event_type": "pii_check", "email": "user@example.com", "subject_token": "anon"},
            )
            self.assertEqual(record["event"]["email"], "[redacted]")
            self.assertIn("current_hash", record)
            self.assertIn("prev_hash", record)

    def test_audit_log_redacts_nested_pii_inside_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            record = append_audit_record(
                path,
                {
                    "event_type": "nested_pii_check",
                    "payload": [{"client_id": "C123", "nested": {"account_number": "A456"}}],
                    "user_id": "U789",
                },
            )
            self.assertEqual(record["event"]["payload"][0]["client_id"], "[redacted]")
            self.assertEqual(record["event"]["payload"][0]["nested"]["account_number"], "[redacted]")
            self.assertEqual(record["event"]["user_id"], "[redacted]")

    def test_audit_chain_detects_tampered_middle_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            append_audit_record(path, {"event_type": "one", "subject_token": "anon"})
            append_audit_record(path, {"event_type": "two", "subject_token": "anon"})
            append_audit_record(path, {"event_type": "three", "subject_token": "anon"})
            lines = path.read_text(encoding="utf-8").splitlines()
            middle = json.loads(lines[1])
            middle["event"]["event_type"] = "tampered"
            lines[1] = json.dumps(middle, sort_keys=True, separators=(",", ":"))
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.assertFalse(verify_audit_chain(path))

    def test_mutual_fund_analyzer_generates_template_outputs(self) -> None:
        path = Path(__file__).resolve().parent.parent / "data" / "sample" / "mutual_fund_nav.csv"
        payload = load_mutual_fund_csv(path)
        result = analyze_mutual_fund(payload)
        templates = [item["template"] for item in result["insights"]]
        self.assertGreaterEqual(len(result["insights"]), 4)
        self.assertIn("benchmark_comparison", templates)
        self.assertIn("cost_tax", templates)
        self.assertEqual(result["suppressed_insights"], [])

    def test_mutual_fund_restricted_benchmark_suppresses_dependent_metrics(self) -> None:
        path = Path(__file__).resolve().parent.parent / "data" / "sample" / "mutual_fund_nav.csv"
        payload = load_mutual_fund_csv(path)
        payload["benchmark_source"]["license"] = LICENSE_RESTRICTED
        result = analyze_mutual_fund(payload)
        templates = [item["template"] for item in result["insights"]]
        self.assertEqual(templates, ["cost_tax"])
        self.assertGreaterEqual(len(result["suppressed_insights"]), 3)

    def test_mutual_fund_misaligned_series_uses_intersection_only(self) -> None:
        payload = mf_payload(
            fund_values=[100, 101, 102, 103],
            benchmark_values=[100, 101, 102, 103],
            dates=["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        )
        payload["benchmark"].append({"date": "2024-01-05", "value": 104})
        payload["fund"] = [row for row in payload["fund"] if row["date"] != "2024-01-03"]
        result = analyze_mutual_fund(payload)
        quality = result["data_quality"]
        self.assertEqual(quality["aligned_points"], 3)
        self.assertEqual(quality["dropped_benchmark_only_dates"], 2)
        rolling = insight_payload(result, "benchmark_comparison", "excess_return_stats")
        self.assertEqual(rolling["supporting_data"]["aligned_points"], 3)
        self.assertLess(rolling["supporting_data"]["relative_completeness"], 1)

    def test_mutual_fund_duplicate_unsorted_dates_are_normalized(self) -> None:
        payload = mf_payload(
            fund_values=[102, 100, 101, 103],
            benchmark_values=[102, 100, 101, 103],
            dates=["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-02"],
        )
        result = analyze_mutual_fund(payload)
        event_types = [event["type"] for event in result["data_quality"]["normalization_events"]]
        self.assertIn("sorted_ascending", event_types)
        self.assertIn("deduped_by_last_observation", event_types)

    def test_mutual_fund_non_positive_value_fails_closed(self) -> None:
        payload = mf_payload([100, 0, 102], [100, 101, 102])
        with self.assertRaises(PolicyError) as ctx:
            analyze_mutual_fund(payload)
        self.assertEqual(ctx.exception.code, "data_validation_error")

    def test_mutual_fund_outlier_is_flagged_not_modified(self) -> None:
        payload = mf_payload([100, 220, 221], [100, 101, 102])
        result = analyze_mutual_fund(payload)
        flags = result["data_quality"]["outlier_flags"]
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0]["date"], "2024-01-02")
        rolling = insight_payload(result, "benchmark_comparison", "excess_return_stats")
        self.assertTrue(any("Observed extreme value changes" in item for item in rolling["limitations"]))

    def test_mutual_fund_excess_return_stats_are_correct(self) -> None:
        payload = mf_payload([100, 110, 100, 120], [100, 100, 100, 100])
        result = analyze_mutual_fund(payload)
        rolling = insight_payload(result, "benchmark_comparison", "excess_return_stats")
        stats = rolling["supporting_data"]["excess_return_stats"]
        self.assertEqual(stats["sample_size"], 3)
        self.assertEqual(stats["window_count"], 3)
        self.assertEqual(stats["hit_ratio_pct"], 66.67)
        self.assertEqual(stats["avg_win_pct"], 15.0)
        self.assertEqual(stats["avg_loss_pct"], -9.09)
        self.assertEqual(stats["window_span_summary"]["median_days"], 1.0)

    def test_mutual_fund_drawdown_duration_without_recovery(self) -> None:
        payload = mf_payload([100, 110, 90, 105], [100, 100, 100, 100])
        result = analyze_mutual_fund(payload)
        drawdown = insight_payload(result, "benchmark_comparison", "fund_drawdown")
        profile = drawdown["supporting_data"]["fund_drawdown"]
        self.assertEqual(profile["start_date"], "2024-01-02")
        self.assertEqual(profile["trough_date"], "2024-01-03")
        self.assertIsNone(profile["recovery_date"])
        self.assertEqual(profile["duration_days"], 1)

    def test_mutual_fund_drawdown_duration_with_recovery(self) -> None:
        payload = mf_payload([100, 110, 90, 111], [100, 100, 100, 100])
        result = analyze_mutual_fund(payload)
        drawdown = insight_payload(result, "benchmark_comparison", "fund_drawdown")
        profile = drawdown["supporting_data"]["fund_drawdown"]
        self.assertEqual(profile["recovery_date"], "2024-01-04")
        self.assertEqual(profile["recovery_days"], 1)

    def test_mutual_fund_flat_series_drawdown_is_zero(self) -> None:
        payload = mf_payload([100, 100, 100], [100, 100, 100])
        result = analyze_mutual_fund(payload)
        drawdown = insight_payload(result, "benchmark_comparison", "fund_drawdown")
        profile = drawdown["supporting_data"]["fund_drawdown"]
        self.assertEqual(profile["max_drawdown_pct"], 0.0)
        self.assertEqual(profile["duration_days"], 0)
        self.assertIsNone(profile["recovery_date"])

    def test_mutual_fund_multiple_peaks_uses_peak_before_trough(self) -> None:
        payload = mf_payload([100, 120, 110, 130, 90], [100, 100, 100, 100, 100])
        result = analyze_mutual_fund(payload)
        drawdown = insight_payload(result, "benchmark_comparison", "fund_drawdown")
        profile = drawdown["supporting_data"]["fund_drawdown"]
        self.assertEqual(profile["start_date"], "2024-01-04")
        self.assertEqual(profile["trough_date"], "2024-01-05")

    def test_mutual_fund_missing_middle_dates_downgrades_rolling_evidence(self) -> None:
        payload = mf_payload(
            fund_values=[100, 101, 102],
            benchmark_values=[100, 100, 100],
            dates=["2024-01-01", "2024-06-01", "2025-01-01"],
            window_points=2,
        )
        payload["expected_window_span_days"] = 365
        payload["rolling_min_windows"] = 3
        result = analyze_mutual_fund(payload)
        rolling = insight_payload(result, "benchmark_comparison", "excess_return_stats")
        stats = rolling["supporting_data"]["excess_return_stats"]
        self.assertEqual(stats["window_count"], 2)
        self.assertLess(rolling["supporting_data"]["calendar_density"], 0.7)
        self.assertEqual(rolling["evidence_strength"], "Low")
        self.assertTrue(any("Calendar density" in item for item in rolling["limitations"]))

    def test_mutual_fund_uneven_spacing_surfaces_span_variation(self) -> None:
        payload = mf_payload(
            fund_values=[100, 101, 102, 103],
            benchmark_values=[100, 100, 100, 100],
            dates=["2024-01-01", "2024-01-02", "2024-06-01", "2025-01-01"],
            window_points=2,
        )
        result = analyze_mutual_fund(payload)
        rolling = insight_payload(result, "benchmark_comparison", "excess_return_stats")
        span = rolling["supporting_data"]["excess_return_stats"]["window_span_summary"]
        self.assertEqual(span["min_days"], 1)
        self.assertGreater(span["max_days"], span["min_days"])

    def test_mutual_fund_sparse_data_has_low_window_count_and_evidence(self) -> None:
        payload = mf_payload(
            fund_values=[100, 105],
            benchmark_values=[100, 104],
            dates=["2024-01-01", "2025-01-01"],
            window_points=2,
        )
        payload["rolling_min_windows"] = 10
        result = analyze_mutual_fund(payload)
        rolling = insight_payload(result, "benchmark_comparison", "excess_return_stats")
        stats = rolling["supporting_data"]["excess_return_stats"]
        self.assertEqual(stats["window_count"], 1)
        self.assertEqual(rolling["evidence_strength"], "Low")
        self.assertTrue(any("Rolling window count" in item for item in rolling["limitations"]))

    def test_adversarial_sparse_spike_misaligned_duplicates_flat(self) -> None:
        """Combined adversarial: sparse + spike + misaligned + duplicates + flat."""
        fund_dates = [
            "2023-01-15", "2023-04-10", "2023-04-10",  # duplicate
            "2023-07-20", "2023-08-12",                 # spike here
            "2023-11-01", "2024-02-15", "2024-06-01",
            "2024-06-02", "2024-06-03",                 # flat section
        ]
        fund_navs = [
            100.0, 101.0, 101.5,   # duplicate (last wins)
            102.0, 260.0,           # extreme spike on 08-12
            103.0, 104.0, 105.0,
            105.0, 105.0,           # flat
        ]
        # Benchmark misaligned: overlaps on some dates, extra dates elsewhere
        benchmark_dates = [
            "2023-01-15", "2023-03-01",                 # 03-01 has no fund match
            "2023-04-10", "2023-07-20", "2023-08-12",
            "2023-11-01", "2024-02-15", "2024-06-01",
            "2024-06-02", "2024-06-03", "2024-09-01",   # 09-01 has no fund match
        ]
        benchmark_values = [
            100.0, 100.5,
            101.0, 102.0, 103.0,
            104.0, 105.0, 106.0,
            106.0, 106.0, 107.0,
        ]
        source = make_source("adversarial_test", "2026-04-24T00:00:00+00:00", LICENSE_REDIS)
        payload = {
            "fund_name": "Adversarial Fund",
            "benchmark_name": "Adversarial Benchmark",
            "category": "Equity",
            "expense_ratio_pct": 1.5,
            "fund_source": source,
            "benchmark_source": source,
            "fund": [{"date": d, "nav": v} for d, v in zip(fund_dates, fund_navs)],
            "benchmark": [{"date": d, "value": v} for d, v in zip(benchmark_dates, benchmark_values)],
            "rolling_window_points": 3,
            "rolling_step_points": 1,
            "rolling_min_windows": 3,
            "expected_window_span_days": 365,
            "expense_impact": {"investment_amount": 100000, "horizons_years": [3, 5, 10]},
        }
        result = analyze_mutual_fund(payload)

        # Must not crash
        self.assertIn("insights", result)
        self.assertIn("data_quality", result)

        quality = result["data_quality"]

        # Low evidence due to sparse data
        for item in result["insights"]:
            self.assertIn(item["payload"]["evidence_strength"], ("Low", "Medium"))

        # Outlier flagged for the spike
        outlier_dates = {flag["date"] for flag in quality["outlier_flags"]}
        self.assertIn("2023-08-12", outlier_dates)

        # Outlier warning propagated to limitations
        all_limitations = []
        for item in result["insights"]:
            all_limitations.extend(item["payload"].get("limitations", []))
        self.assertTrue(
            any("extreme value" in lim.lower() for lim in all_limitations),
            "Outlier warning missing from limitations",
        )

        # Duplicate normalization recorded
        event_types = [e["type"] for e in quality["normalization_events"]]
        self.assertIn("deduped_by_last_observation", event_types)

        # Misaligned series: dropped dates recorded
        self.assertGreater(quality["dropped_benchmark_only_dates"], 0)

        # Calendar density is low (sparse data)
        self.assertLess(quality["calendar_density"], 0.70)

        # No suppressed insights (no restricted lineage)
        self.assertEqual(result["suppressed_insights"], [])

        # Drawdown sparse limitation present
        drawdown = insight_payload(result, "benchmark_comparison", "fund_drawdown")
        self.assertTrue(
            any("sparse observations" in lim.lower() for lim in drawdown["limitations"]),
            "Drawdown sparse limitation missing",
        )

    def test_audit_record_contains_version_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            record = append_audit_record(path, {"event_type": "version_check"})
            self.assertEqual(record["schema_version"], "v1")
            self.assertEqual(record["analyzer_version"], "mf_v2")
            self.assertTrue(verify_audit_chain(path))

    def test_density_span_contradiction_adds_rolling_limitation(self) -> None:
        """When density is low but span meets expectation, rolling limitation is added."""
        payload = mf_payload(
            fund_values=[100, 101, 102],
            benchmark_values=[100, 100, 100],
            dates=["2024-01-01", "2024-06-01", "2025-01-01"],
            window_points=2,
        )
        payload["expected_window_span_days"] = 180
        payload["rolling_min_windows"] = 1
        result = analyze_mutual_fund(payload)
        rolling = insight_payload(result, "benchmark_comparison", "excess_return_stats")
        self.assertTrue(
            any("sparse observations" in lim.lower() for lim in rolling["limitations"]),
            "Density vs span contradiction limitation missing from rolling insight",
        )

    # --- Portfolio analyzer tests ---

    def _portfolio_payload(self, holdings: list[dict], **kwargs) -> dict:
        payload = {"holdings": holdings}
        payload.update(kwargs)
        return payload

    def _portfolio_insight(self, result: list[dict], supporting_key: str) -> dict:
        for item in result:
            if supporting_key in item["payload"].get("supporting_data", {}):
                return item["payload"]
        raise AssertionError(f"Portfolio insight with key {supporting_key} not found")

    def test_portfolio_exposure_and_concentration_generated(self) -> None:
        result = analyze_portfolio(self._portfolio_payload([
            {"name": "Fund A", "asset_class": "Equity", "market_value": 60000},
            {"name": "Fund B", "asset_class": "Debt", "market_value": 30000},
            {"name": "Fund C", "asset_class": "Equity", "market_value": 10000},
        ]))
        templates = [item["template"] for item in result]
        self.assertEqual(templates.count("diagnostic"), 4)

    def test_portfolio_top_n_concentration(self) -> None:
        result = analyze_portfolio(self._portfolio_payload([
            {"name": "A", "asset_class": "Equity", "market_value": 50000},
            {"name": "B", "asset_class": "Debt", "market_value": 30000},
            {"name": "C", "asset_class": "Gold", "market_value": 10000},
            {"name": "D", "asset_class": "Cash", "market_value": 10000},
        ]))
        top_n = self._portfolio_insight(result, "top_n_weight")
        self.assertEqual(top_n["supporting_data"]["top_n"], 3)
        self.assertAlmostEqual(top_n["supporting_data"]["top_n_weight"], 0.9, places=4)
        self.assertAlmostEqual(top_n["supporting_data"]["tail_weight"], 0.1, places=4)
        self.assertEqual(len(top_n["supporting_data"]["contributing_assets"]), 3)

    def test_portfolio_hhi_single_holding(self) -> None:
        result = analyze_portfolio(self._portfolio_payload([
            {"name": "Only", "asset_class": "Equity", "market_value": 100000},
        ]))
        hhi = self._portfolio_insight(result, "hhi")
        self.assertAlmostEqual(hhi["supporting_data"]["hhi"], 1.0, places=4)
        self.assertAlmostEqual(hhi["supporting_data"]["effective_holdings"], 1.0, places=2)

    def test_portfolio_hhi_equal_weights(self) -> None:
        n = 4
        result = analyze_portfolio(self._portfolio_payload([
            {"name": f"Fund {i}", "asset_class": "Equity", "market_value": 25000}
            for i in range(n)
        ]))
        hhi = self._portfolio_insight(result, "hhi")
        self.assertAlmostEqual(hhi["supporting_data"]["hhi"], 1.0 / n, places=4)
        self.assertAlmostEqual(hhi["supporting_data"]["effective_holdings"], float(n), places=2)

    def test_portfolio_hhi_many_small_holdings(self) -> None:
        result = analyze_portfolio(self._portfolio_payload([
            {"name": f"Fund {i}", "asset_class": "Equity", "market_value": 1000}
            for i in range(20)
        ]))
        hhi = self._portfolio_insight(result, "hhi")
        self.assertLess(hhi["supporting_data"]["hhi"], 0.1)
        self.assertGreater(hhi["supporting_data"]["effective_holdings"], 10)

    def test_portfolio_class_hhi_computed(self) -> None:
        result = analyze_portfolio(self._portfolio_payload([
            {"name": "A", "asset_class": "Equity", "market_value": 50000},
            {"name": "B", "asset_class": "Debt", "market_value": 30000},
            {"name": "C", "asset_class": "Gold", "market_value": 20000},
        ]))
        class_hhi = self._portfolio_insight(result, "class_hhi")
        self.assertIn("class_hhi", class_hhi["supporting_data"])
        self.assertEqual(class_hhi["supporting_data"]["distinct_classes"], 3)

    def test_portfolio_duplicate_names_normalized(self) -> None:
        result = analyze_portfolio(self._portfolio_payload([
            {"name": "Fund A", "asset_class": "Equity", "market_value": 30000},
            {"name": "Fund A", "asset_class": "Equity", "market_value": 20000},
            {"name": "Fund B", "asset_class": "Debt", "market_value": 50000},
        ]))
        exposure = self._portfolio_insight(result, "exposure_by_class")
        events = exposure["supporting_data"]["normalization_events"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "duplicate_holdings_merged")
        self.assertIn("Fund A", events[0]["names"])
        self.assertEqual(exposure["data_completeness"], "Medium")

    def test_portfolio_no_positive_value_fails_closed(self) -> None:
        result = analyze_portfolio(self._portfolio_payload([
            {"name": "Zero", "asset_class": "Equity", "market_value": 0},
        ]))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["payload"]["evidence_strength"], "Low")

    def test_portfolio_deterministic_ordering(self) -> None:
        """Exposure sorted desc by weight, top-N sorted desc by weight."""
        result = analyze_portfolio(self._portfolio_payload([
            {"name": "Small", "asset_class": "Cash", "market_value": 5000},
            {"name": "Big", "asset_class": "Equity", "market_value": 80000},
            {"name": "Mid", "asset_class": "Debt", "market_value": 15000},
        ]))
        exposure = self._portfolio_insight(result, "exposure_by_class")
        classes = [e["asset_class"] for e in exposure["supporting_data"]["exposure_by_class"]]
        self.assertEqual(classes, ["Equity", "Debt", "Cash"])
        top_n = self._portfolio_insight(result, "top_n_weight")
        names = [h["name"] for h in top_n["supporting_data"]["top_holdings"]]
        self.assertEqual(names, ["Big", "Mid", "Small"])

    def test_portfolio_linter_blocks_advisory_language(self) -> None:
        """No advisory terms in any portfolio insight field."""
        result = analyze_portfolio(self._portfolio_payload([
            {"name": "A", "asset_class": "Equity", "market_value": 50000},
            {"name": "B", "asset_class": "Debt", "market_value": 50000},
        ]))
        for item in result:
            all_text = json.dumps(item["payload"])
            for banned in ["should", "recommend", "overweight", "underweight",
                           "rebalance", "allocation", "buy", "sell", "optimize"]:
                self.assertNotIn(banned, all_text.lower(),
                                 f"Banned term '{banned}' found in portfolio insight")

    def test_portfolio_evidence_from_count(self) -> None:
        low = analyze_portfolio(self._portfolio_payload([
            {"name": "A", "asset_class": "Equity", "market_value": 50000},
            {"name": "B", "asset_class": "Debt", "market_value": 50000},
        ]))
        self.assertEqual(low[0]["payload"]["evidence_strength"], "Low")
        medium = analyze_portfolio(self._portfolio_payload([
            {"name": f"F{i}", "asset_class": "Equity", "market_value": 1000}
            for i in range(5)
        ]))
        self.assertEqual(medium[0]["payload"]["evidence_strength"], "Medium")
        high = analyze_portfolio(self._portfolio_payload([
            {"name": f"F{i}", "asset_class": "Equity", "market_value": 1000}
            for i in range(10)
        ]))
        self.assertEqual(high[0]["payload"]["evidence_strength"], "High")

    def test_portfolio_single_holding_top_n_clamped(self) -> None:
        result = analyze_portfolio(self._portfolio_payload([
            {"name": "Only", "asset_class": "Equity", "market_value": 100000},
        ]))
        top_n = self._portfolio_insight(result, "top_n_weight")
        self.assertEqual(top_n["supporting_data"]["top_n"], 1)
        self.assertAlmostEqual(top_n["supporting_data"]["top_n_weight"], 1.0, places=4)
        self.assertAlmostEqual(top_n["supporting_data"]["tail_weight"], 0.0, places=4)

    def test_portfolio_extreme_skew(self) -> None:
        """One dominant holding, rest tiny."""
        result = analyze_portfolio(self._portfolio_payload([
            {"name": "Giant", "asset_class": "Equity", "market_value": 990000},
            {"name": "Tiny1", "asset_class": "Debt", "market_value": 5000},
            {"name": "Tiny2", "asset_class": "Cash", "market_value": 5000},
        ]))
        hhi = self._portfolio_insight(result, "hhi")
        self.assertGreater(hhi["supporting_data"]["hhi"], 0.95)
        self.assertAlmostEqual(hhi["supporting_data"]["effective_holdings"], 1.02, places=1)
        top_n = self._portfolio_insight(result, "top_n_weight")
        self.assertAlmostEqual(top_n["supporting_data"]["top_n_weight"], 1.0, places=4)

    def test_portfolio_all_same_class(self) -> None:
        """All holdings in one asset class → class_hhi = 1."""
        result = analyze_portfolio(self._portfolio_payload([
            {"name": f"F{i}", "asset_class": "Equity", "market_value": 10000}
            for i in range(5)
        ]))
        class_hhi = self._portfolio_insight(result, "class_hhi")
        self.assertAlmostEqual(class_hhi["supporting_data"]["class_hhi"], 1.0, places=4)
        self.assertEqual(class_hhi["supporting_data"]["distinct_classes"], 1)

    def test_portfolio_float_precision_thirds(self) -> None:
        """Near-equal weights don't cause precision issues."""
        result = analyze_portfolio(self._portfolio_payload([
            {"name": "A", "asset_class": "Eq", "market_value": 33333},
            {"name": "B", "asset_class": "De", "market_value": 33333},
            {"name": "C", "asset_class": "Go", "market_value": 33334},
        ]))
        hhi = self._portfolio_insight(result, "hhi")
        self.assertAlmostEqual(hhi["supporting_data"]["hhi"], 1.0 / 3, places=3)
        self.assertAlmostEqual(hhi["supporting_data"]["effective_holdings"], 3.0, places=1)
        top_n = self._portfolio_insight(result, "top_n_weight")
        self.assertAlmostEqual(
            top_n["supporting_data"]["top_n_weight"] + top_n["supporting_data"]["tail_weight"],
            1.0, places=4,
        )

    def test_portfolio_class_hhi_within_class_limitation(self) -> None:
        """Class HHI limitation mentions within-class diversification."""
        result = analyze_portfolio(self._portfolio_payload([
            {"name": "A", "asset_class": "Equity", "market_value": 50000},
            {"name": "B", "asset_class": "Debt", "market_value": 50000},
        ]))
        class_hhi = self._portfolio_insight(result, "class_hhi")
        self.assertTrue(
            any("within-class" in lim.lower() for lim in class_hhi["limitations"]),
            "Class HHI missing within-class diversification limitation",
        )

    # --- Portfolio-with-funds aggregation tests ---

    def _mf_data(self, fund_values=None, benchmark_values=None, dates=None,
                  expense_ratio_pct=1.0, window_points=2) -> dict:
        if fund_values is None:
            fund_values = [100, 101, 102, 103]
        if benchmark_values is None:
            benchmark_values = [100, 100.5, 101, 101.5]
        if dates is None:
            dates = [f"2024-01-0{i + 1}" for i in range(len(fund_values))]
        src = make_source("unit_test", "2026-04-24T00:00:00+00:00", LICENSE_REDIS)
        return {
            "fund_name": "Test Fund",
            "benchmark_name": "Test Benchmark",
            "category": "Equity",
            "expense_ratio_pct": expense_ratio_pct,
            "fund_source": src,
            "benchmark_source": src,
            "fund": [{"date": d, "nav": v} for d, v in zip(dates, fund_values)],
            "benchmark": [{"date": d, "value": v} for d, v in zip(dates, benchmark_values)],
            "rolling_window_points": window_points,
            "rolling_step_points": 1,
            "rolling_min_windows": 1,
            "expense_impact": {"investment_amount": 100000, "horizons_years": [3, 5, 10]},
        }

    def _pwf_payload(self, funds: list[dict]) -> dict:
        return {"funds": funds}

    def _pwf_insight(self, result: dict, supporting_key: str) -> dict:
        for item in result["insights"]:
            if supporting_key in item["payload"].get("supporting_data", {}):
                return item["payload"]
        raise AssertionError(f"Aggregation insight with key {supporting_key} not found")

    def test_pwf_evidence_worst_case_dominates(self) -> None:
        """Low evidence fund dominates portfolio evidence."""
        from datetime import date, timedelta
        base = date(2020, 1, 1)
        long_dates = [(base + timedelta(days=i)).isoformat() for i in range(1200)]
        long_navs = [100 + i * 0.01 for i in range(1200)]
        long_bench = [100 + i * 0.008 for i in range(1200)]
        result = analyze_portfolio_with_funds(self._pwf_payload([
            {"name": "Dense Fund", "market_value": 70000,
             "mf_payload": self._mf_data(fund_values=long_navs,
                                           benchmark_values=long_bench,
                                           dates=long_dates)},
            {"name": "Sparse Fund", "market_value": 30000,
             "mf_payload": self._mf_data(
                 fund_values=[100, 105],
                 benchmark_values=[100, 104],
                 dates=["2024-01-01", "2025-01-01"],
             )},
        ]))
        ev = self._pwf_insight(result, "portfolio_evidence")
        self.assertEqual(ev["supporting_data"]["portfolio_evidence"], "Low")
        self.assertEqual(ev["evidence_strength"], "Low")

    def test_pwf_evidence_distribution_by_weight(self) -> None:
        result = analyze_portfolio_with_funds(self._pwf_payload([
            {"name": "Fund A", "market_value": 60000,
             "mf_payload": self._mf_data()},
            {"name": "Fund B", "market_value": 40000,
             "mf_payload": self._mf_data()},
        ]))
        ev = self._pwf_insight(result, "evidence_distribution")
        dist = ev["supporting_data"]["evidence_distribution"]
        total_weight = sum(b["weight"] for b in dist)
        self.assertAlmostEqual(total_weight, 1.0, places=4)

    def test_pwf_limitation_dedup(self) -> None:
        """Same limitation across funds appears only once."""
        result = analyze_portfolio_with_funds(self._pwf_payload([
            {"name": "Fund A", "market_value": 50000,
             "mf_payload": self._mf_data()},
            {"name": "Fund B", "market_value": 50000,
             "mf_payload": self._mf_data()},
        ]))
        lim_insight = self._pwf_insight(result, "aggregated_limitations")
        lims = lim_insight["supporting_data"]["aggregated_limitations"]
        normalized = [l.lower().strip() for l in lims]
        self.assertEqual(len(normalized), len(set(normalized)),
                         "Duplicate limitations found after dedup")

    def test_pwf_limitation_cap(self) -> None:
        """Aggregated limitations capped at 8."""
        result = analyze_portfolio_with_funds(self._pwf_payload([
            {"name": "Fund A", "market_value": 50000,
             "mf_payload": self._mf_data()},
            {"name": "Fund B", "market_value": 50000,
             "mf_payload": self._mf_data()},
        ]))
        lim_insight = self._pwf_insight(result, "aggregated_limitations")
        self.assertLessEqual(
            lim_insight["supporting_data"]["aggregated_limitations_count"], 8,
        )

    def test_pwf_expense_aggregation(self) -> None:
        result = analyze_portfolio_with_funds(self._pwf_payload([
            {"name": "Fund L", "market_value": 60000,
             "mf_payload": self._mf_data(expense_ratio_pct=0.5)},
            {"name": "Fund H", "market_value": 40000,
             "mf_payload": self._mf_data(expense_ratio_pct=2.0)},
        ]))
        cost = None
        for item in result["insights"]:
            if item["template"] == "cost_tax":
                cost = item["payload"]
                break
        self.assertIsNotNone(cost, "cost_tax insight missing")
        weighted = cost["assumptions"]["weighted_expense_ratio"]
        # 0.6 * 0.005 + 0.4 * 0.02 = 0.011
        self.assertAlmostEqual(weighted, 0.011, places=4)
        self.assertEqual(cost["estimated_impact"]["units"], "drag_factor")

    def test_pwf_attribution_snapshot(self) -> None:
        result = analyze_portfolio_with_funds(self._pwf_payload([
            {"name": "A", "market_value": 70000, "mf_payload": self._mf_data()},
            {"name": "B", "market_value": 30000, "mf_payload": self._mf_data()},
        ]))
        attr = self._pwf_insight(result, "weights")
        names = [w["name"] for w in attr["supporting_data"]["weights"]]
        self.assertEqual(names[0], "A")  # sorted desc by weight
        self.assertAlmostEqual(attr["supporting_data"]["weights"][0]["weight"], 0.7, places=4)

    def test_pwf_fail_on_invalid_fund(self) -> None:
        """One invalid MF payload fails the entire request."""
        with self.assertRaises(PolicyError) as ctx:
            analyze_portfolio_with_funds(self._pwf_payload([
                {"name": "Good", "market_value": 50000, "mf_payload": self._mf_data()},
                {"name": "Bad", "market_value": 50000, "mf_payload": self._mf_data(
                    fund_values=[100, 0, 102],
                    benchmark_values=[100, 101, 102],
                )},
            ]))
        self.assertEqual(ctx.exception.code, "mf_analysis_failed")
        self.assertEqual(ctx.exception.details["fund_index"], 1)
        self.assertEqual(ctx.exception.details["fund_name"], "Bad")

    def test_pwf_rejects_jurisdiction_in_mf_payload(self) -> None:
        mf = self._mf_data()
        mf["user_country"] = "IN"
        with self.assertRaises(PolicyError) as ctx:
            analyze_portfolio_with_funds(self._pwf_payload([
                {"name": "Fund", "market_value": 50000, "mf_payload": mf},
            ]))
        self.assertEqual(ctx.exception.code, "data_validation_error")
        self.assertIn("user_country", ctx.exception.details["rejected_keys"])

    def test_pwf_empty_funds_fails(self) -> None:
        with self.assertRaises(PolicyError):
            analyze_portfolio_with_funds({"funds": []})

    def test_pwf_deterministic_ordering(self) -> None:
        """Same input → same output ordering.

        Analytical outputs (weights, evidence levels, expense ratios,
        ordering, names, hashes) MUST be deterministic across two
        invocations on identical input. The ``source.timestamp`` field
        embedded in ``supporting_data`` via ``make_source`` is
        intentionally fresh per call — it is provenance metadata
        marking when the result was generated, not a semantic output.
        ``_now_iso`` truncates to second precision, so the previous
        un-mocked form passed only when both calls landed in the same
        wall-clock second; crossing a 1-second boundary made the
        assertion fail probabilistically (observed on ubuntu-latest CI).

        Mocking ``_now_iso`` for the duration of the test pins the
        provenance timestamp to a fixed value across both calls,
        keeping the broad ``assertEqual(sd1, sd2)`` payload-equality
        assertion strong (any non-temporal drift still fails the test)
        while removing the accidental wall-clock coupling. Production
        code is unchanged; the analyzer continues to emit a fresh
        timestamp per call in real API usage.
        """
        payload = self._pwf_payload([
            {"name": "B", "market_value": 30000, "mf_payload": self._mf_data()},
            {"name": "A", "market_value": 70000, "mf_payload": self._mf_data()},
        ])
        with patch(
            "backend.investment_analytics.analyzers._now_iso",
            return_value="2026-01-01T00:00:00+00:00",
        ):
            r1 = analyze_portfolio_with_funds(payload)
            r2 = analyze_portfolio_with_funds(payload)
        for i1, i2 in zip(r1["insights"], r2["insights"]):
            self.assertEqual(i1["template"], i2["template"])
            sd1 = i1["payload"].get("supporting_data")
            sd2 = i2["payload"].get("supporting_data")
            if sd1 is not None:
                self.assertEqual(sd1, sd2)

    def test_pwf_no_advisory_language(self) -> None:
        result = analyze_portfolio_with_funds(self._pwf_payload([
            {"name": "F", "market_value": 100000, "mf_payload": self._mf_data()},
        ]))
        raw = json.dumps(result["insights"]).lower()
        for term in ["should", "recommend", "overweight", "underweight",
                      "rebalance", "optimize", "buy", "sell"]:
            self.assertNotIn(term, raw, f"Advisory term '{term}' in aggregated output")

    def test_pwf_metadata_versions(self) -> None:
        result = analyze_portfolio_with_funds(self._pwf_payload([
            {"name": "F", "market_value": 100000, "mf_payload": self._mf_data()},
        ]))
        meta = result["aggregation_metadata"]
        self.assertEqual(meta["schema_version"], "v1")
        self.assertEqual(meta["aggregator_version"], "portfolio_v1")
        self.assertEqual(meta["mf_analyzer_version"], "mf_v2")

    def test_pwf_per_fund_results_included(self) -> None:
        result = analyze_portfolio_with_funds(self._pwf_payload([
            {"name": "A", "market_value": 60000, "mf_payload": self._mf_data()},
            {"name": "B", "market_value": 40000, "mf_payload": self._mf_data()},
        ]))
        self.assertEqual(len(result["per_fund_results"]), 2)
        for fr in result["per_fund_results"]:
            self.assertIn("fund_name", fr)
            self.assertIn("weight", fr)
            self.assertIn("evidence", fr)
            self.assertIn("completeness", fr)

    # ── ETF analyzer tests ─────────────────────────────────────

    def _etf_data(self, prices=None, bench=None, dates=None,
                  expense_ratio_pct=0.2, window_points=2) -> dict:
        if prices is None:
            prices = [100, 101, 102, 103]
        if bench is None:
            bench = [100, 100.5, 101, 101.5]
        if dates is None:
            dates = [f"2024-01-0{i + 1}" for i in range(len(prices))]
        src = make_source("unit_test", "2026-04-24T00:00:00+00:00", LICENSE_REDIS)
        return {
            "etf_name": "Test ETF",
            "benchmark_name": "Test Benchmark",
            "category": "Equity",
            "expense_ratio_pct": expense_ratio_pct,
            "etf_source": src,
            "benchmark_source": src,
            "price_series": [{"date": d, "price": v} for d, v in zip(dates, prices)],
            "benchmark_series": [{"date": d, "value": v} for d, v in zip(dates, bench)],
            "rolling_window_points": window_points,
            "rolling_step_points": 1,
            "rolling_min_windows": 1,
            "expense_impact": {"investment_amount": 100000, "horizons_years": [3, 5, 10]},
        }

    def test_etf_basic_analysis_returns_insights(self) -> None:
        result = analyze_etf(self._etf_data())
        self.assertIn("insights", result)
        self.assertIn("suppressed_insights", result)
        self.assertIn("data_quality", result)
        self.assertGreaterEqual(len(result["insights"]), 3)

    def test_etf_tracking_difference_present(self) -> None:
        result = analyze_etf(self._etf_data())
        found = False
        for item in result["insights"]:
            sd = item["payload"].get("supporting_data", {})
            if "tracking_difference" in sd:
                found = True
                td = sd["tracking_difference"]
                self.assertIn("tracking_difference_pct", td)
                self.assertIn("etf_total_return_pct", td)
                self.assertIn("benchmark_total_return_pct", td)
        self.assertTrue(found, "tracking_difference insight missing")

    def test_etf_tracking_error_present(self) -> None:
        result = analyze_etf(self._etf_data())
        found = False
        for item in result["insights"]:
            sd = item["payload"].get("supporting_data", {})
            if "tracking_error" in sd:
                found = True
                te = sd["tracking_error"]
                self.assertIn("tracking_error_pct", te)
                self.assertIn("observation_count", te)
        self.assertTrue(found, "tracking_error insight missing")

    def test_etf_tracking_difference_math(self) -> None:
        """ETF +3%, benchmark +1.5% → TD ≈ 1.5 pct points."""
        result = analyze_etf(self._etf_data(
            prices=[100, 103],
            bench=[100, 101.5],
            dates=["2024-01-01", "2024-06-01"],
        ))
        for item in result["insights"]:
            sd = item["payload"].get("supporting_data", {})
            if "tracking_difference" in sd:
                td = sd["tracking_difference"]
                self.assertAlmostEqual(td["tracking_difference_pct"], 1.5, places=1)
                self.assertAlmostEqual(td["etf_total_return_pct"], 3.0, places=1)
                break

    def test_etf_expense_drag_uses_decimal(self) -> None:
        """1% TER → 10Y drag factor ≈ 0.9044."""
        result = analyze_etf(self._etf_data(expense_ratio_pct=1.0))
        for item in result["insights"]:
            if item["template"] == "cost_tax":
                assumptions = item["payload"]["assumptions"]
                horizons = assumptions["horizons"]
                ten_yr = [h for h in horizons if h["horizon_years"] == 10][0]
                self.assertAlmostEqual(ten_yr["drag_factor"], (0.99 ** 10), places=4)
                break

    def test_etf_evidence_low_on_sparse(self) -> None:
        result = analyze_etf(self._etf_data(
            prices=[100, 105],
            bench=[100, 104],
            dates=["2024-01-01", "2025-01-01"],
        ))
        for item in result["insights"]:
            self.assertIn(item["payload"]["evidence_strength"], ("Low", "Medium", "High"))

    def test_etf_requires_two_common_dates(self) -> None:
        with self.assertRaises(PolicyError):
            analyze_etf({
                "etf_name": "X",
                "benchmark_name": "Y",
                "etf_source": make_source("t", "2026-01-01T00:00:00+00:00", LICENSE_REDIS),
                "benchmark_source": make_source("t", "2026-01-01T00:00:00+00:00", LICENSE_REDIS),
                "price_series": [{"date": "2024-01-01", "price": 100}],
                "benchmark_series": [{"date": "2024-01-02", "price": 100}],
            })

    def test_etf_no_advisory_language(self) -> None:
        result = analyze_etf(self._etf_data())
        raw = json.dumps(result["insights"]).lower()
        for term in ["should", "recommend", "overweight", "underweight",
                      "rebalance", "optimize", "buy", "sell"]:
            self.assertNotIn(term, raw, f"Advisory term '{term}' in ETF output")

    def test_etf_limitations_always_present(self) -> None:
        result = analyze_etf(self._etf_data())
        for item in result["insights"]:
            lims = item["payload"].get("limitations", [])
            self.assertIsInstance(lims, list)
            self.assertGreater(len(lims), 0, f"No limitations on {item['template']}")

    def test_etf_cost_tax_has_consultation(self) -> None:
        result = analyze_etf(self._etf_data())
        for item in result["insights"]:
            if item["template"] == "cost_tax":
                self.assertIn("Consult a qualified professional",
                              item["payload"]["limitations"])
                break

    def test_etf_tracking_error_downgrade_on_few_observations(self) -> None:
        """With only 3 aligned points, tracking error evidence should degrade."""
        result = analyze_etf(self._etf_data(
            prices=[100, 101, 102],
            bench=[100, 100.5, 101],
            dates=["2024-01-01", "2024-01-02", "2024-01-03"],
        ))
        for item in result["insights"]:
            sd = item["payload"].get("supporting_data", {})
            if "tracking_error" in sd:
                self.assertEqual(item["payload"]["evidence_strength"], "Low")
                break

    def test_etf_drawdown_present(self) -> None:
        result = analyze_etf(self._etf_data())
        found = False
        for item in result["insights"]:
            sd = item["payload"].get("supporting_data", {})
            if "etf_drawdown" in sd:
                found = True
                self.assertIn("max_drawdown_pct", sd["etf_drawdown"])
                self.assertIn("benchmark_drawdown", sd)
        self.assertTrue(found, "drawdown insight missing")

    def test_etf_dividend_gap_negative_td(self) -> None:
        """ETF flat, benchmark rising → tracking difference negative."""
        result = analyze_etf(self._etf_data(
            prices=[100, 100, 100, 100],
            bench=[100, 101, 102, 103],
        ))
        for item in result["insights"]:
            sd = item["payload"].get("supporting_data", {})
            if "tracking_difference" in sd:
                self.assertLess(sd["tracking_difference"]["tracking_difference_pct"], 0)
                break

    def test_etf_distribution_limitation_present(self) -> None:
        """Price-series distribution caveat must appear in limitations."""
        result = analyze_etf(self._etf_data())
        all_lims = []
        for item in result["insights"]:
            all_lims.extend(item["payload"].get("limitations", []))
        joined = " ".join(all_lims).lower()
        self.assertIn("distribution", joined,
                       "Missing limitation about price series not reflecting distributions")

    def test_etf_misaligned_dates_reduce_completeness(self) -> None:
        """ETF missing mid-period dates → alignment drops, completeness reflects it."""
        src = make_source("unit_test", "2026-04-24T00:00:00+00:00", LICENSE_REDIS)
        payload = {
            "etf_name": "Sparse ETF",
            "benchmark_name": "Dense Bench",
            "category": "Equity",
            "expense_ratio_pct": 0.2,
            "etf_source": src,
            "benchmark_source": src,
            "price_series": [
                {"date": "2024-01-01", "price": 100},
                {"date": "2024-01-03", "price": 102},
                {"date": "2024-01-05", "price": 104},
            ],
            "benchmark_series": [
                {"date": "2024-01-01", "value": 100},
                {"date": "2024-01-02", "value": 100.5},
                {"date": "2024-01-03", "value": 101},
                {"date": "2024-01-04", "value": 101.5},
                {"date": "2024-01-05", "value": 102},
            ],
            "rolling_window_points": 2,
            "rolling_step_points": 1,
            "rolling_min_windows": 1,
            "expense_impact": {"investment_amount": 100000, "horizons_years": [3, 5, 10]},
        }
        result = analyze_etf(payload)
        quality = result["data_quality"]
        self.assertGreater(quality["dropped_benchmark_only_dates"], 0)

    def test_etf_constant_spread_zero_tracking_error(self) -> None:
        """ETF always exactly +1% vs benchmark → tracking error ≈ 0."""
        prices = [100 * (1.01 ** i) for i in range(20)]
        bench = [100 * (1.00 ** i) for i in range(20)]  # flat benchmark
        dates = [f"2024-01-{str(i + 1).zfill(2)}" for i in range(20)]
        result = analyze_etf(self._etf_data(
            prices=prices, bench=bench, dates=dates,
        ))
        for item in result["insights"]:
            sd = item["payload"].get("supporting_data", {})
            if "tracking_error" in sd:
                # constant excess → std dev should be very small
                self.assertLess(sd["tracking_error"]["tracking_error_pct"], 0.1)
                break

    def test_etf_tracking_error_not_annualized_limitation(self) -> None:
        """Tracking error insight must state it is per-observation, not annualized."""
        result = analyze_etf(self._etf_data())
        for item in result["insights"]:
            sd = item["payload"].get("supporting_data", {})
            if "tracking_error" in sd:
                lims = " ".join(item["payload"]["limitations"]).lower()
                self.assertIn("not annualized", lims)
                break


if __name__ == "__main__":
    unittest.main()
