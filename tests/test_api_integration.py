from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from starlette.testclient import TestClient

from api.main import app, AUDIT_PATH


def _source_dict(name: str = "test_source") -> dict:
    return {
        "source": name,
        "timestamp": "2026-04-24T00:00:00+00:00",
        "license": "redistributable",
        "lineage": [],
    }


def _mf_api_payload(**overrides) -> dict:
    base = {
        "subject_token": "integration_test",
        "user_country": "IN",
        "asset_market": "IN",
        "serving_entity": "test_entity",
        "fund_name": "Test Fund",
        "benchmark_name": "Test Benchmark",
        "category": "Equity",
        "expense_ratio_pct": 1.0,
        "fund_source": _source_dict("test_fund"),
        "benchmark_source": _source_dict("test_benchmark"),
        "fund": [
            {"date": "2024-01-01", "nav": 100.0},
            {"date": "2024-01-02", "nav": 101.0},
            {"date": "2024-01-03", "nav": 102.0},
            {"date": "2024-01-04", "nav": 103.0},
        ],
        "benchmark": [
            {"date": "2024-01-01", "value": 100.0},
            {"date": "2024-01-02", "value": 100.5},
            {"date": "2024-01-03", "value": 101.0},
            {"date": "2024-01-04", "value": 101.5},
        ],
        "rolling_window_points": 2,
        "rolling_step_points": 1,
        "rolling_min_windows": 1,
    }
    base.update(overrides)
    return base


def _portfolio_api_payload(**overrides) -> dict:
    base = {
        "subject_token": "integration_test",
        "user_country": "IN",
        "asset_market": "IN",
        "serving_entity": "test_entity",
        "holdings": [
            {"name": "Fund A", "asset_class": "Equity", "market_value": 60000},
            {"name": "Fund B", "asset_class": "Debt", "market_value": 30000},
            {"name": "Fund C", "asset_class": "Gold", "market_value": 10000},
        ],
    }
    base.update(overrides)
    return base


class APIIntegrationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    # --- Health ---

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("audit_chain_valid", data)

    # --- Jurisdiction ---

    def test_jurisdiction_supported(self) -> None:
        response = self.client.post("/policy/jurisdiction", json={
            "subject_token": "test",
            "user_country": "IN",
            "asset_market": "IN",
            "serving_entity": "test",
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["supported"])
        self.assertTrue(data["features"]["analytics"])

    def test_jurisdiction_unsupported(self) -> None:
        response = self.client.post("/policy/jurisdiction", json={
            "subject_token": "test",
            "user_country": "FR",
            "asset_market": "FR",
            "serving_entity": "test",
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["supported"])
        self.assertFalse(data["features"]["analytics"])

    # --- Portfolio API ---

    def test_portfolio_api_full_round_trip(self) -> None:
        response = self.client.post("/analytics/portfolio", json=_portfolio_api_payload())
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("gate", data)
        self.assertTrue(data["gate"]["supported"])
        self.assertIn("insights", data)
        self.assertGreaterEqual(len(data["insights"]), 4)
        templates = [item["template"] for item in data["insights"]]
        self.assertEqual(templates.count("diagnostic"), 4)

    def test_portfolio_api_returns_hhi_and_top_n(self) -> None:
        response = self.client.post("/analytics/portfolio", json=_portfolio_api_payload())
        data = response.json()
        all_keys = set()
        for item in data["insights"]:
            all_keys.update(item["payload"]["supporting_data"].keys())
        self.assertIn("hhi", all_keys)
        self.assertIn("top_n_weight", all_keys)
        self.assertIn("class_hhi", all_keys)
        self.assertIn("exposure_by_class", all_keys)

    def test_portfolio_api_unsupported_jurisdiction_blocks(self) -> None:
        response = self.client.post("/analytics/portfolio", json=_portfolio_api_payload(
            user_country="FR", asset_market="FR",
        ))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["gate"]["features"]["analytics"])
        self.assertEqual(data["insights"], [])

    def test_portfolio_api_validation_rejects_empty_name(self) -> None:
        response = self.client.post("/analytics/portfolio", json=_portfolio_api_payload(
            holdings=[{"name": "", "asset_class": "Equity", "market_value": 1000}],
        ))
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertEqual(data["reason"], "REQUEST_VALIDATION_ERROR")

    def test_portfolio_api_empty_holdings_returns_low_evidence(self) -> None:
        response = self.client.post("/analytics/portfolio", json=_portfolio_api_payload(
            holdings=[],
        ))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["insights"]), 1)
        self.assertEqual(data["insights"][0]["payload"]["evidence_strength"], "Low")

    def test_portfolio_api_compiler_enforces_schema(self) -> None:
        """Every insight through API has template + labels + payload."""
        response = self.client.post("/analytics/portfolio", json=_portfolio_api_payload())
        data = response.json()
        for item in data["insights"]:
            self.assertIn("template", item)
            self.assertIn("labels", item)
            self.assertIn("payload", item)
            self.assertIn("evidence_strength", item["payload"])
            self.assertIn("limitations", item["payload"])
            self.assertIn("data_completeness", item["payload"])

    def test_portfolio_api_with_horizon(self) -> None:
        response = self.client.post("/analytics/portfolio", json=_portfolio_api_payload(
            profile={"horizon_years": 10},
        ))
        data = response.json()
        templates = [item["template"] for item in data["insights"]]
        self.assertIn("benchmark_comparison", templates)

    # --- Mutual Fund API ---

    def test_mf_api_full_round_trip(self) -> None:
        response = self.client.post("/analytics/mutual-fund", json=_mf_api_payload())
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("gate", data)
        self.assertTrue(data["gate"]["supported"])
        self.assertIn("insights", data)
        self.assertIn("suppressed_insights", data)
        self.assertIn("data_quality", data)
        templates = [item["template"] for item in data["insights"]]
        self.assertIn("benchmark_comparison", templates)
        self.assertIn("cost_tax", templates)

    def test_mf_api_data_quality_in_response(self) -> None:
        response = self.client.post("/analytics/mutual-fund", json=_mf_api_payload())
        data = response.json()
        quality = data["data_quality"]
        self.assertIn("aligned_points", quality)
        self.assertIn("calendar_density", quality)
        self.assertIn("data_completeness", quality)
        self.assertIn("outlier_flags", quality)

    def test_mf_api_unsupported_jurisdiction_blocks(self) -> None:
        response = self.client.post("/analytics/mutual-fund", json=_mf_api_payload(
            user_country="FR", asset_market="FR",
        ))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["gate"]["features"]["analytics"])
        self.assertEqual(data["insights"], [])

    def test_mf_api_validation_rejects_single_point(self) -> None:
        response = self.client.post("/analytics/mutual-fund", json=_mf_api_payload(
            fund=[{"date": "2024-01-01", "nav": 100.0}],
            benchmark=[{"date": "2024-01-01", "value": 100.0}],
        ))
        self.assertEqual(response.status_code, 422)

    def test_mf_api_compiler_enforces_structure(self) -> None:
        """Every MF insight through API has template + labels + payload."""
        response = self.client.post("/analytics/mutual-fund", json=_mf_api_payload())
        data = response.json()
        for item in data["insights"]:
            self.assertIn("template", item)
            self.assertIn("labels", item)
            self.assertIn("payload", item)
            payload = item["payload"]
            self.assertIn("evidence_strength", payload)
            self.assertIn("limitations", payload)
            self.assertIn("data_completeness", payload)
            self.assertIn(payload["evidence_strength"], ("Low", "Medium", "High"))

    def test_mf_demo_endpoint(self) -> None:
        response = self.client.get("/analytics/mutual-fund/demo")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("insights", data)
        self.assertGreaterEqual(len(data["insights"]), 4)
        self.assertIn("data_quality", data)

    # --- Scenario API ---

    def test_scenario_api_standard(self) -> None:
        response = self.client.post("/scenarios/run", json={
            "subject_token": "test",
            "portfolio_value": 100000,
            "scenario_definition": {"kind": "standard", "id": "market_down_20"},
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("insight", data)
        self.assertEqual(data["insight"]["template"], "scenario")

    def test_scenario_api_invalid_id_rejected(self) -> None:
        response = self.client.post("/scenarios/run", json={
            "subject_token": "test",
            "portfolio_value": 100000,
            "scenario_definition": {"kind": "standard", "id": "nonexistent"},
        })
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertEqual(data["reason"], "SCENARIO_VIOLATION")

    def test_scenarios_list(self) -> None:
        response = self.client.get("/scenarios/standard")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 1)
        ids = [item["id"] for item in data]
        self.assertIn("market_down_20", ids)

    # --- End-to-end: linter enforcement through API ---

    def test_mf_api_non_positive_nav_rejected(self) -> None:
        response = self.client.post("/analytics/mutual-fund", json=_mf_api_payload(
            fund=[
                {"date": "2024-01-01", "nav": 100.0},
                {"date": "2024-01-02", "nav": 0.0},
                {"date": "2024-01-03", "nav": 102.0},
            ],
            benchmark=[
                {"date": "2024-01-01", "value": 100.0},
                {"date": "2024-01-02", "value": 101.0},
                {"date": "2024-01-03", "value": 102.0},
            ],
        ))
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertEqual(data["reason"], "DATA_VALIDATION_ERROR")

    # --- End-to-end: no advisory language in any API response ---

    def test_no_advisory_language_in_portfolio_response(self) -> None:
        response = self.client.post("/analytics/portfolio", json=_portfolio_api_payload())
        raw = response.text.lower()
        for term in ["should", "recommend", "overweight", "underweight",
                      "rebalance", "optimize", "buy", "sell"]:
            self.assertNotIn(term, raw, f"Advisory term '{term}' found in portfolio API response")

    def test_no_advisory_language_in_mf_response(self) -> None:
        response = self.client.post("/analytics/mutual-fund", json=_mf_api_payload())
        raw = response.text.lower()
        for term in ["should", "recommend", "overweight", "underweight",
                      "rebalance", "optimize", "buy", "sell"]:
            self.assertNotIn(term, raw, f"Advisory term '{term}' found in MF API response")

    # --- Portfolio-with-funds API ---

    def _pwf_api_payload(self, **overrides) -> dict:
        src = {
            "source": "integration_test",
            "timestamp": "2026-04-24T00:00:00+00:00",
            "license": "redistributable",
            "lineage": [],
        }
        mf = {
            "fund_name": "API Test Fund",
            "benchmark_name": "API Test Benchmark",
            "category": "Equity",
            "expense_ratio_pct": 1.0,
            "fund_source": src,
            "benchmark_source": src,
            "fund": [
                {"date": "2024-01-01", "nav": 100.0},
                {"date": "2024-01-02", "nav": 101.0},
                {"date": "2024-01-03", "nav": 102.0},
                {"date": "2024-01-04", "nav": 103.0},
            ],
            "benchmark": [
                {"date": "2024-01-01", "value": 100.0},
                {"date": "2024-01-02", "value": 100.5},
                {"date": "2024-01-03", "value": 101.0},
                {"date": "2024-01-04", "value": 101.5},
            ],
            "rolling_window_points": 2,
            "rolling_step_points": 1,
            "rolling_min_windows": 1,
            "expense_impact": {"investment_amount": 100000, "horizons_years": [3, 5, 10]},
        }
        base = {
            "subject_token": "integration_test",
            "user_country": "IN",
            "asset_market": "IN",
            "serving_entity": "test_entity",
            "funds": [
                {"name": "Fund A", "market_value": 60000, "mf_payload": mf},
                {"name": "Fund B", "market_value": 40000, "mf_payload": dict(mf, expense_ratio_pct=2.0)},
            ],
        }
        base.update(overrides)
        return base

    def test_pwf_api_full_round_trip(self) -> None:
        response = self.client.post(
            "/analytics/portfolio-with-funds", json=self._pwf_api_payload(),
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("gate", data)
        self.assertTrue(data["gate"]["supported"])
        self.assertIn("insights", data)
        self.assertGreaterEqual(len(data["insights"]), 4)
        self.assertIn("per_fund_results", data)
        self.assertEqual(len(data["per_fund_results"]), 2)
        self.assertIn("aggregation_metadata", data)

    def test_pwf_api_compiler_enforces_structure(self) -> None:
        response = self.client.post(
            "/analytics/portfolio-with-funds", json=self._pwf_api_payload(),
        )
        data = response.json()
        for item in data["insights"]:
            self.assertIn("template", item)
            self.assertIn("labels", item)
            self.assertIn("payload", item)
            self.assertIn("evidence_strength", item["payload"])
            self.assertIn("limitations", item["payload"])

    def test_pwf_api_unsupported_jurisdiction_blocks(self) -> None:
        response = self.client.post(
            "/analytics/portfolio-with-funds",
            json=self._pwf_api_payload(user_country="FR", asset_market="FR"),
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["gate"]["features"]["analytics"])
        self.assertEqual(data["insights"], [])

    def test_pwf_api_invalid_fund_returns_422(self) -> None:
        payload = self._pwf_api_payload()
        payload["funds"][1]["mf_payload"]["fund"] = [
            {"date": "2024-01-01", "nav": 100.0},
            {"date": "2024-01-02", "nav": 0.0},
        ]
        response = self.client.post("/analytics/portfolio-with-funds", json=payload)
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertEqual(data["reason"], "MF_ANALYSIS_FAILED")
        self.assertEqual(data["details"]["fund_index"], 1)

    def test_pwf_api_no_advisory_language(self) -> None:
        response = self.client.post(
            "/analytics/portfolio-with-funds", json=self._pwf_api_payload(),
        )
        raw = response.text.lower()
        for term in ["should", "recommend", "overweight", "underweight",
                      "rebalance", "optimize", "buy", "sell"]:
            self.assertNotIn(term, raw, f"Advisory term '{term}' in pwf API response")

    def test_pwf_api_metadata_versions(self) -> None:
        response = self.client.post(
            "/analytics/portfolio-with-funds", json=self._pwf_api_payload(),
        )
        data = response.json()
        meta = data["aggregation_metadata"]
        self.assertEqual(meta["schema_version"], "v1")
        self.assertEqual(meta["aggregator_version"], "portfolio_v1")
        self.assertEqual(meta["mf_analyzer_version"], "mf_v2")

    def test_pwf_api_validation_rejects_empty_funds(self) -> None:
        response = self.client.post("/analytics/portfolio-with-funds", json={
            "subject_token": "test",
            "user_country": "IN",
            "asset_market": "IN",
            "serving_entity": "test",
            "funds": [],
        })
        self.assertEqual(response.status_code, 422)

    # ── ETF API integration tests ──────────────────────────────

    def _etf_api_payload(self, **overrides) -> dict:
        base = {
            "subject_token": "integration_test",
            "user_country": "IN",
            "asset_market": "IN",
            "serving_entity": "test_entity",
            "etf_name": "Test ETF",
            "benchmark_name": "Test Benchmark",
            "category": "Equity",
            "expense_ratio_pct": 0.5,
            "etf_source": _source_dict("test_etf"),
            "benchmark_source": _source_dict("test_benchmark"),
            "price_series": [
                {"date": "2024-01-01", "price": 100.0},
                {"date": "2024-01-02", "price": 101.0},
                {"date": "2024-01-03", "price": 102.0},
                {"date": "2024-01-04", "price": 103.0},
            ],
            "benchmark_series": [
                {"date": "2024-01-01", "value": 100.0},
                {"date": "2024-01-02", "value": 100.5},
                {"date": "2024-01-03", "value": 101.0},
                {"date": "2024-01-04", "value": 101.5},
            ],
            "rolling_window_points": 2,
            "rolling_step_points": 1,
            "rolling_min_windows": 1,
        }
        base.update(overrides)
        return base

    def test_etf_api_full_round_trip(self) -> None:
        response = self.client.post(
            "/analytics/etf", json=self._etf_api_payload(),
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("gate", data)
        self.assertIn("insights", data)
        self.assertIn("data_quality", data)
        self.assertGreaterEqual(len(data["insights"]), 3)
        templates = [i["template"] for i in data["insights"]]
        self.assertIn("cost_tax", templates)

    def test_etf_api_tracking_difference_in_output(self) -> None:
        response = self.client.post(
            "/analytics/etf", json=self._etf_api_payload(),
        )
        data = response.json()
        found = False
        for item in data["insights"]:
            sd = item["payload"].get("supporting_data", {})
            if "tracking_difference" in sd:
                found = True
                self.assertIn("tracking_difference_pct",
                              sd["tracking_difference"])
        self.assertTrue(found, "tracking_difference missing from API response")

    def test_etf_api_tracking_error_in_output(self) -> None:
        response = self.client.post(
            "/analytics/etf", json=self._etf_api_payload(),
        )
        data = response.json()
        found = False
        for item in data["insights"]:
            sd = item["payload"].get("supporting_data", {})
            if "tracking_error" in sd:
                found = True
                self.assertIn("tracking_error_pct", sd["tracking_error"])
        self.assertTrue(found, "tracking_error missing from API response")

    def test_etf_api_unsupported_jurisdiction_blocks(self) -> None:
        response = self.client.post(
            "/analytics/etf",
            json=self._etf_api_payload(user_country="ZZ", asset_market="ZZ"),
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["insights"], [])

    def test_etf_api_no_advisory_language(self) -> None:
        response = self.client.post(
            "/analytics/etf", json=self._etf_api_payload(),
        )
        data = response.json()
        raw = json.dumps(data["insights"]).lower()
        for term in ["should", "recommend", "buy", "sell",
                      "overweight", "rebalance"]:
            self.assertNotIn(term, raw,
                             f"Advisory term '{term}' in ETF API output")

    def test_etf_api_compiler_enforces_structure(self) -> None:
        response = self.client.post(
            "/analytics/etf", json=self._etf_api_payload(),
        )
        data = response.json()
        for item in data["insights"]:
            self.assertIn("template", item)
            self.assertIn("labels", item)
            self.assertIn("payload", item)
            self.assertIn("evidence_strength", item["payload"])
            self.assertIsInstance(item["payload"].get("limitations"), list)

    def test_etf_api_validation_rejects_empty_series(self) -> None:
        response = self.client.post("/analytics/etf", json={
            "subject_token": "test",
            "user_country": "IN",
            "asset_market": "IN",
            "serving_entity": "test",
            "etf_name": "X",
            "benchmark_name": "Y",
            "etf_source": _source_dict(),
            "benchmark_source": _source_dict(),
            "price_series": [],
            "benchmark_series": [],
        })
        self.assertEqual(response.status_code, 422)

    # ── From-source API integration tests ──────────────────────

    def _from_source_payload(self, **overrides) -> dict:
        base = {
            "subject_token": "integration_test",
            "user_country": "IN",
            "asset_market": "IN",
            "serving_entity": "test_entity",
            "source": "csv_sample",
            "symbol": "messy_mf",
            "fund_name": "Messy Fund",
            "benchmark_name": "Messy Bench",
            "category": "Equity",
            "expense_ratio_pct": 1.5,
            "rolling_window_points": 4,
            "rolling_step_points": 1,
            "rolling_min_windows": 1,
        }
        base.update(overrides)
        return base

    def test_from_source_mf_round_trip(self) -> None:
        response = self.client.post(
            "/analytics/mutual-fund/from-source",
            json=self._from_source_payload(),
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("gate", data)
        self.assertIn("insights", data)
        self.assertIn("ingestion_report", data)
        self.assertGreaterEqual(len(data["insights"]), 1)
        report = data["ingestion_report"]
        self.assertGreater(report["fund_series"]["rejected_count"], 0)
        self.assertGreater(len(report["ingestion_limitations"]), 0)

    def test_from_source_unknown_symbol_fails(self) -> None:
        response = self.client.post(
            "/analytics/mutual-fund/from-source",
            json=self._from_source_payload(symbol="nonexistent"),
        )
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertEqual(data["reason"], "INGESTION_SOURCE_ERROR")

    def test_from_source_jurisdiction_gate(self) -> None:
        response = self.client.post(
            "/analytics/mutual-fund/from-source",
            json=self._from_source_payload(
                user_country="ZZ", asset_market="ZZ",
            ),
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["insights"], [])

    def test_from_source_evidence_reflects_messy_data(self) -> None:
        response = self.client.post(
            "/analytics/mutual-fund/from-source",
            json=self._from_source_payload(),
        )
        data = response.json()
        evidence_levels = {
            i["payload"]["evidence_strength"] for i in data["insights"]
        }
        self.assertIn("Low", evidence_levels)

    def test_from_source_no_advisory_language(self) -> None:
        response = self.client.post(
            "/analytics/mutual-fund/from-source",
            json=self._from_source_payload(),
        )
        data = response.json()
        raw = json.dumps(data["insights"]).lower()
        for term in ["should", "recommend", "buy", "sell",
                      "overweight", "rebalance"]:
            self.assertNotIn(term, raw,
                             f"Advisory term '{term}' in from-source output")

    def test_from_source_clean_csv_works(self) -> None:
        response = self.client.post(
            "/analytics/mutual-fund/from-source",
            json=self._from_source_payload(symbol="clean_mf"),
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(len(data["insights"]), 1)
        report = data["ingestion_report"]
        self.assertEqual(report["fund_series"]["rejected_count"], 0)

    def test_from_source_lineage_in_response(self) -> None:
        response = self.client.post(
            "/analytics/mutual-fund/from-source",
            json=self._from_source_payload(),
        )
        data = response.json()
        report = data["ingestion_report"]
        self.assertEqual(report["source"], "csv")
        self.assertIn("ingestion_timestamp", report)
        self.assertIn("license", report)
        self.assertIn("source_path", report)

    def test_from_source_limitation_caveats_present(self) -> None:
        """Messy data limitations must include threshold + calendar caveats."""
        response = self.client.post(
            "/analytics/mutual-fund/from-source",
            json=self._from_source_payload(),
        )
        data = response.json()
        lims_joined = " ".join(
            data["ingestion_report"]["ingestion_limitations"]
        ).lower()
        self.assertIn("fixed threshold", lims_joined)
        self.assertIn("calendar days", lims_joined)

    # ── Schema mapping API tests ───────────────────────────────

    def test_from_source_amfi_with_mapping(self) -> None:
        response = self.client.post(
            "/analytics/mutual-fund/from-source",
            json=self._from_source_payload(
                symbol="amfi_mf",
                schema_mapping="amfi_nav",
                fund_name="AMFI Fund",
                benchmark_name="Nifty 50",
            ),
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(len(data["insights"]), 1)
        self.assertEqual(
            data["ingestion_report"]["mapping_label"], "AMFI NAV",
        )

    def test_from_source_unknown_mapping_fails(self) -> None:
        response = self.client.post(
            "/analytics/mutual-fund/from-source",
            json=self._from_source_payload(schema_mapping="nonexistent"),
        )
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertEqual(data["reason"], "INGESTION_SCHEMA_ERROR")

    def test_from_source_wrong_mapping_for_csv_fails(self) -> None:
        """AMFI mapping applied to default-format CSV should fail."""
        response = self.client.post(
            "/analytics/mutual-fund/from-source",
            json=self._from_source_payload(
                symbol="clean_mf",
                schema_mapping="amfi_nav",
            ),
        )
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertEqual(data["reason"], "INGESTION_SCHEMA_ERROR")

    def test_from_source_etf_csv_with_mapping(self) -> None:
        response = self.client.post(
            "/analytics/mutual-fund/from-source",
            json=self._from_source_payload(
                symbol="etf_price",
                schema_mapping="etf_price",
                fund_name="ETFX",
                benchmark_name="Broad Market",
            ),
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(len(data["insights"]), 1)
        self.assertEqual(
            data["ingestion_report"]["mapping_label"], "ETF price",
        )

    def test_from_source_alt_mf_with_mapping(self) -> None:
        response = self.client.post(
            "/analytics/mutual-fund/from-source",
            json=self._from_source_payload(
                symbol="alt_mf",
                schema_mapping="alt_mf",
                fund_name="Alternate Fund",
                benchmark_name="Alt Bench",
            ),
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(len(data["insights"]), 1)
        self.assertEqual(
            data["ingestion_report"]["mapping_label"], "alternate MF",
        )

    def test_from_source_error_message_actionable(self) -> None:
        """Schema mismatch error should name columns and mapping."""
        response = self.client.post(
            "/analytics/mutual-fund/from-source",
            json=self._from_source_payload(
                symbol="clean_mf",
                schema_mapping="etf_price",
            ),
        )
        self.assertEqual(response.status_code, 422)
        msg = response.json()["message"]
        self.assertIn("Close Price", msg)
        self.assertIn("Check schema_mapping=", msg)

    # ── Impact links API tests ─────────────────────────────────

    def test_from_source_messy_has_impact_links(self) -> None:
        response = self.client.post(
            "/analytics/mutual-fund/from-source",
            json=self._from_source_payload(),
        )
        data = response.json()
        links = data["ingestion_report"]["impact_links"]
        self.assertIsInstance(links, list)
        self.assertGreater(len(links), 0)
        # Every link has required fields
        for link in links:
            self.assertIn("issue", link)
            self.assertIn("affected_metrics", link)
            self.assertIn("explanation", link)
            self.assertIn("series", link)

    def test_from_source_clean_no_outlier_or_rejected_links(self) -> None:
        response = self.client.post(
            "/analytics/mutual-fund/from-source",
            json=self._from_source_payload(symbol="clean_mf"),
        )
        data = response.json()
        links = data["ingestion_report"]["impact_links"]
        issue_types = {l["issue"] for l in links}
        # Quarterly data has natural gaps, but no outliers or rejected
        self.assertNotIn("extreme_values", issue_types)
        self.assertNotIn("records_rejected", issue_types)

    # ── Infrastructure tests ───────────────────────────────────

    def test_request_id_in_response_header(self) -> None:
        """Every response must include X-Request-ID header."""
        response = self.client.get("/health")
        self.assertIn("x-request-id", response.headers)
        req_id = response.headers["x-request-id"]
        self.assertEqual(len(req_id), 12)

    def test_health_endpoint_fields(self) -> None:
        """Health endpoint returns operational checks."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("audit_chain_valid", data)
        self.assertIn("data_directory_accessible", data)
        self.assertIn("registered_sources", data)
        self.assertTrue(data["data_directory_accessible"])
        self.assertGreater(data["registered_sources"], 0)

    def test_health_unhealthy_when_data_dir_missing(self) -> None:
        """Missing data directory → unhealthy (not degraded)."""
        from unittest.mock import patch, PropertyMock
        with patch("api.main.ROOT", Path("/nonexistent/path")):
            response = self.client.get("/health")
            data = response.json()
            self.assertEqual(data["status"], "unhealthy")
            self.assertFalse(data["data_directory_accessible"])

    def test_gzip_compression_accepted(self) -> None:
        """Server supports gzip for eligible responses."""
        response = self.client.get("/health", headers={"Accept-Encoding": "gzip"})
        self.assertEqual(response.status_code, 200)

    def test_csv_row_limit_via_api(self) -> None:
        """Oversized CSV must be rejected through the API."""
        from unittest.mock import patch
        with patch("backend.data_ingestion.sources.MAX_CSV_ROWS", 3):
            response = self.client.post(
                "/analytics/mutual-fund/from-source",
                json=self._from_source_payload(),
            )
            self.assertEqual(response.status_code, 422)
            data = response.json()
            self.assertEqual(data["reason"], "INGESTION_SIZE_ERROR")

    def test_error_response_includes_request_id(self) -> None:
        """Error JSON body must include request_id for traceability."""
        response = self.client.post(
            "/analytics/mutual-fund/from-source",
            json=self._from_source_payload(symbol="nonexistent"),
        )
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("request_id", data)
        self.assertEqual(len(data["request_id"]), 12)


# ─────────────────────────────────────────────────────────────────────
# Evidence layer integration tests (step 4 of evidence-layer roadmap)
#
# For the 4 migrated endpoints (rank_category, rank_all_categories,
# rank_all_assets, portfolio_health_check) the heavy payload is now
# written to data/evidence/<kind>/<run_id>.json and the audit row
# carries evidence_ref instead. These tests verify the end-to-end
# pipeline for each endpoint:
#   1. The response JSON shape is unchanged.
#   2. The audit row has evidence_ref with the expected shape.
#   3. The evidence file exists at the referenced path.
#   4. The sha256 in evidence_ref matches the file bytes.
# ─────────────────────────────────────────────────────────────────────

class EvidenceLayerEndpointTests(unittest.TestCase):
    """Endpoints with by-reference evidence: rank_category,
    rank_all_categories, rank_all_assets, portfolio_health_check."""

    def setUp(self) -> None:
        import hashlib as _hashlib
        from datetime import datetime, timezone

        self._hashlib = _hashlib
        self._now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)
        self.audit_path = self.tmpdir / "data" / "audit" / "audit.jsonl"
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @staticmethod
    def _gate_payload() -> dict:
        return {
            "subject_token": "evidence_test",
            "user_country": "IN",
            "asset_market": "IN",
            "serving_entity": "test_entity",
        }

    def _read_audit_records(self) -> list:
        if not self.audit_path.exists():
            return []
        return [json.loads(line) for line in self.audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _verify_evidence_ref(self, audit_record: dict, expected_kind: str) -> None:
        """Common assertions for any by-reference audit record."""
        event = audit_record["event"]
        self.assertEqual(event.get("evidence_kind"), expected_kind)
        self.assertIn("evidence_ref", event)
        ref = event["evidence_ref"]
        self.assertEqual(set(ref.keys()), {"path", "sha256", "size_bytes"})
        # Evidence file exists at the referenced path (relative to data/).
        evidence_file = self.tmpdir / "data" / ref["path"]
        self.assertTrue(evidence_file.exists(), f"missing evidence file {evidence_file}")
        # SHA256 matches the file bytes.
        actual_sha = self._hashlib.sha256(evidence_file.read_bytes()).hexdigest()
        self.assertEqual(actual_sha, ref["sha256"])
        self.assertEqual(evidence_file.stat().st_size, ref["size_bytes"])
        # The audit row's run_id matches the evidence file's basename.
        self.assertTrue(ref["path"].endswith(f"{event['run_id']}.json"))
        # The audit row carries the lightweight fields BUT NOT the heavy payload.
        self.assertNotIn("payload", event)

    def test_rank_category_writes_evidence_ref(self) -> None:
        from backend.investment_analytics.ranking.equity import CategoryRanking

        synthetic = CategoryRanking(
            category="Equity Scheme - Large Cap Fund",
            benchmark_name="Nifty 50",
            benchmark_code=120716,
            benchmark_fallback=False,
            ranked=[],
            excluded=[],
            computed_at=self._now_iso,
            total_funds_in_category=0,
        )
        with patch("api.main.AUDIT_PATH", self.audit_path), \
             patch("api.main.rank_category", return_value=synthetic):
            response = self.client.post("/analytics/rank-category", json={
                **self._gate_payload(),
                "category": "Equity Scheme - Large Cap Fund",
            })
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        # Response shape unchanged (still has gate + ranking).
        self.assertIn("gate", body)
        self.assertIn("ranking", body)
        # Audit + evidence verified.
        records = self._read_audit_records()
        self.assertEqual(len(records), 1)
        self._verify_evidence_ref(records[0], "ranking_snapshot")
        self.assertEqual(records[0]["event"]["event_type"], "rank_category")

    def test_rank_all_categories_writes_evidence_ref(self) -> None:
        from backend.investment_analytics.ranking.multi import MultiCategoryRanking

        synthetic = MultiCategoryRanking(
            categories={},
            errors={},
            computed_at=self._now_iso,
            top_n=5,
        )
        with patch("api.main.AUDIT_PATH", self.audit_path), \
             patch("api.main.rank_all_categories", return_value=synthetic):
            response = self.client.post("/analytics/rank-all-categories", json={
                **self._gate_payload(),
                "top_n": 5,
            })
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertIn("gate", body)
        self.assertIn("multi_ranking", body)
        records = self._read_audit_records()
        self.assertEqual(len(records), 1)
        self._verify_evidence_ref(records[0], "ranking_snapshot")
        self.assertEqual(records[0]["event"]["event_type"], "rank_all_categories")

    def test_rank_all_assets_writes_evidence_ref(self) -> None:
        # Multi-category endpoints get ONE evidence file per call (the
        # full multi-result), not one per category — per architecture.
        from backend.investment_analytics.ranking.all_assets import AllAssetsRanking

        synthetic = AllAssetsRanking(
            equity={},
            debt={},
            gold=None,
            equity_errors={},
            debt_errors={},
            gold_error=None,
            computed_at=self._now_iso,
            top_n=5,
        )
        with patch("api.main.AUDIT_PATH", self.audit_path), \
             patch("api.main.rank_all_assets", return_value=synthetic):
            response = self.client.post("/analytics/rank-all-assets", json={
                **self._gate_payload(),
                "top_n": 5,
            })
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertIn("gate", body)
        self.assertIn("all_assets", body)
        records = self._read_audit_records()
        # Exactly one audit row + one evidence file for the whole call.
        self.assertEqual(len(records), 1)
        self._verify_evidence_ref(records[0], "ranking_snapshot")
        self.assertEqual(records[0]["event"]["event_type"], "rank_all_assets")
        # And exactly one evidence file in the kind subdir.
        ranking_dir = self.tmpdir / "data" / "evidence" / "ranking_snapshot"
        self.assertTrue(ranking_dir.is_dir())
        self.assertEqual(len(list(ranking_dir.glob("*.json"))), 1)

    def test_portfolio_health_writes_evidence_ref(self) -> None:
        from backend.investment_analytics.portfolio_health.models import (
            PortfolioHealthResult,
        )

        synthetic = PortfolioHealthResult(
            holdings=[],
            not_found=[],
            concentration=[],
            mistakes=[],
            redundancies=[],
            exposure_gaps=[],
            risk_summary={},
            portfolio_status="Well diversified",
            computed_at=self._now_iso,
        )
        with patch("api.main.AUDIT_PATH", self.audit_path), \
             patch("api.main.check_portfolio_health", return_value=synthetic):
            response = self.client.post("/analytics/portfolio-health", json={
                **self._gate_payload(),
                "scheme_codes": [101, 102, 103],
                "weights": {"101": 0.5, "102": 0.3, "103": 0.2},
            })
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertIn("gate", body)
        self.assertIn("health", body)
        records = self._read_audit_records()
        self.assertEqual(len(records), 1)
        self._verify_evidence_ref(records[0], "portfolio_health_snapshot")
        self.assertEqual(records[0]["event"]["event_type"], "portfolio_health_check")

    def test_evidence_envelope_matches_audit_envelope(self) -> None:
        # run_id, evidence_kind, inputs MUST match between the audit
        # row and the evidence file so replay can correlate them.
        from backend.investment_analytics.ranking.equity import CategoryRanking

        synthetic = CategoryRanking(
            category="Equity Scheme - Large Cap Fund",
            benchmark_name="Nifty 50", benchmark_code=120716,
            benchmark_fallback=False, ranked=[], excluded=[],
            computed_at=self._now_iso, total_funds_in_category=0,
        )
        with patch("api.main.AUDIT_PATH", self.audit_path), \
             patch("api.main.rank_category", return_value=synthetic):
            self.client.post("/analytics/rank-category", json={
                **self._gate_payload(),
                "category": "Equity Scheme - Large Cap Fund",
            })
        records = self._read_audit_records()
        audit_event = records[0]["event"]
        evidence_file = self.tmpdir / "data" / audit_event["evidence_ref"]["path"]
        evidence = json.loads(evidence_file.read_text(encoding="utf-8"))
        # Same run_id, evidence_kind, and inputs across both records.
        self.assertEqual(audit_event["run_id"], evidence["run_id"])
        self.assertEqual(audit_event["evidence_kind"], evidence["evidence_kind"])
        self.assertEqual(audit_event["inputs"], evidence["inputs"])


if __name__ == "__main__":
    unittest.main()
