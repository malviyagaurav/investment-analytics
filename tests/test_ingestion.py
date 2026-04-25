from __future__ import annotations

import json
import unittest
from pathlib import Path

from backend.data_ingestion.normalize import (
    normalize_series,
    NormalizedPoint,
    SeriesNormalization,
)
from backend.data_ingestion.validate import (
    build_ingestion_limitations,
    validate_series,
)
from backend.data_ingestion.schema_map import (
    ALT_MF_MAPPING,
    AMFI_NAV_MAPPING,
    CANONICAL_BENCHMARK_VALUE,
    CANONICAL_DATE,
    CANONICAL_FUND_VALUE,
    ColumnMapping,
    DEFAULT_MAPPING,
    ETF_PRICE_MAPPING,
    KNOWN_MAPPINGS,
    apply_mapping,
    validate_mapping,
)
from backend.data_ingestion.sources import load_csv
from backend.data_ingestion.ingest import ingest_mf_from_csv
from backend.data_ingestion.impact_links import build_impact_links
from backend.investment_analytics.errors import PolicyError
from backend.investment_analytics.mutual_funds import analyze_mutual_fund

ROOT = Path(__file__).resolve().parent.parent


class NormalizationTests(unittest.TestCase):

    def _records(self, pairs: list[tuple[str, str]], key: str = "v") -> list[dict]:
        return [{"date": d, key: v} for d, v in pairs]

    def test_sorts_unsorted_input(self) -> None:
        records = self._records([
            ("2024-01-03", "103"),
            ("2024-01-01", "100"),
            ("2024-01-02", "101"),
        ])
        result = normalize_series(records, "date", "v", "test")
        dates = [p.as_of.isoformat() for p in result.points]
        self.assertEqual(dates, ["2024-01-01", "2024-01-02", "2024-01-03"])
        sort_events = [e for e in result.events if e["type"] == "sorted_ascending"]
        self.assertEqual(len(sort_events), 1)

    def test_deduplicates_dates_keeps_last(self) -> None:
        records = self._records([
            ("2024-01-01", "100"),
            ("2024-01-01", "105"),
            ("2024-01-02", "110"),
        ])
        result = normalize_series(records, "date", "v", "test")
        self.assertEqual(len(result.points), 2)
        self.assertAlmostEqual(result.points[0].value, 105.0)
        self.assertEqual(result.metadata["duplicate_dates_merged"], 1)

    def test_rejects_empty_values(self) -> None:
        records = self._records([
            ("2024-01-01", "100"),
            ("2024-01-02", ""),
            ("2024-01-03", "102"),
        ])
        result = normalize_series(records, "date", "v", "test")
        self.assertEqual(len(result.points), 2)
        self.assertEqual(result.metadata["rejected_count"], 1)
        self.assertEqual(result.rejected[0]["reason"], "missing_or_non_positive_value")

    def test_rejects_non_positive_values(self) -> None:
        records = self._records([
            ("2024-01-01", "100"),
            ("2024-01-02", "0"),
            ("2024-01-03", "-5"),
        ])
        result = normalize_series(records, "date", "v", "test")
        self.assertEqual(len(result.points), 1)
        self.assertEqual(result.metadata["rejected_count"], 2)

    def test_rejects_non_numeric_values(self) -> None:
        records = self._records([
            ("2024-01-01", "100"),
            ("2024-01-02", "N.A."),
            ("2024-01-03", "102"),
        ])
        result = normalize_series(records, "date", "v", "test")
        self.assertEqual(len(result.points), 2)
        self.assertEqual(result.rejected[0]["reason"], "missing_or_non_positive_value")

    def test_rejects_unparseable_dates(self) -> None:
        records = self._records([
            ("2024-01-01", "100"),
            ("not-a-date", "101"),
            ("2024-01-03", "102"),
        ])
        result = normalize_series(records, "date", "v", "test")
        self.assertEqual(len(result.points), 2)
        self.assertEqual(result.rejected[0]["reason"], "unparseable_date")

    def test_detects_extreme_moves(self) -> None:
        records = self._records([
            ("2024-01-01", "100"),
            ("2024-01-02", "160"),
            ("2024-01-03", "105"),
        ])
        result = normalize_series(records, "date", "v", "test")
        # 100→160 = +60% (above 50%), 160→105 = -34% (below 50%)
        self.assertEqual(result.metadata["anomalies_flagged"], 1)
        self.assertEqual(result.anomalies[0]["type"], "extreme_move")

    def test_detects_gaps(self) -> None:
        records = self._records([
            ("2024-01-01", "100"),
            ("2024-01-20", "105"),
        ])
        result = normalize_series(records, "date", "v", "test")
        self.assertEqual(result.metadata["gaps_detected"], 1)
        gap_events = [e for e in result.events if e["type"] == "gap_detected"]
        self.assertEqual(gap_events[0]["gap_days"], 19)

    def test_no_false_gap_for_normal_spacing(self) -> None:
        records = self._records([
            ("2024-01-01", "100"),
            ("2024-01-08", "105"),
        ])
        result = normalize_series(records, "date", "v", "test")
        self.assertEqual(result.metadata["gaps_detected"], 0)

    def test_metadata_tracks_all_counts(self) -> None:
        records = self._records([
            ("2024-01-03", "100"),
            ("2024-01-01", "98"),
            ("2024-01-01", "99"),
            ("2024-01-02", ""),
            ("2024-01-20", "200"),
        ])
        result = normalize_series(records, "date", "v", "test")
        m = result.metadata
        self.assertEqual(m["input_records"], 5)
        self.assertEqual(m["rejected_count"], 1)
        self.assertEqual(m["duplicate_dates_merged"], 1)
        self.assertGreaterEqual(m["anomalies_flagged"], 1)
        self.assertGreaterEqual(m["gaps_detected"], 1)


class ValidationTests(unittest.TestCase):

    def _make_result(self, n_points: int) -> SeriesNormalization:
        points = [
            NormalizedPoint(as_of=__import__("datetime").date(2024, 1, i + 1), value=100 + i)
            for i in range(n_points)
        ]
        return SeriesNormalization(
            points=points,
            events=[],
            rejected=[],
            anomalies=[],
            metadata={
                "input_records": n_points,
                "output_points": n_points,
                "rejected_count": 0,
                "duplicate_dates_merged": 0,
                "anomalies_flagged": 0,
                "gaps_detected": 0,
                "date_range": {"start": None, "end": None},
            },
        )

    def test_rejects_too_few_points(self) -> None:
        with self.assertRaises(PolicyError) as ctx:
            validate_series(self._make_result(1), "fund", min_points=2)
        self.assertEqual(ctx.exception.code, "ingestion_validation_error")

    def test_passes_sufficient_points(self) -> None:
        validate_series(self._make_result(5), "fund", min_points=2)

    def test_build_limitations_captures_all_issues(self) -> None:
        result = SeriesNormalization(
            points=[],
            events=[],
            rejected=[{"reason": "test"}],
            anomalies=[{"type": "extreme_move"}],
            metadata={
                "input_records": 10,
                "output_points": 8,
                "rejected_count": 2,
                "duplicate_dates_merged": 1,
                "anomalies_flagged": 1,
                "gaps_detected": 1,
                "date_range": {"start": None, "end": None},
            },
        )
        lims = build_ingestion_limitations(result)
        joined = " ".join(lims).lower()
        self.assertIn("excluded", joined)
        self.assertIn("duplicate", joined)
        self.assertIn("fixed threshold", joined)
        self.assertIn("valid market events", joined)
        self.assertIn("gap", joined)
        self.assertIn("calendar days", joined)


class SourceTests(unittest.TestCase):

    def test_csv_loads_successfully(self) -> None:
        fund, bench, meta = load_csv(
            ROOT, "data/sample/messy_mf_nav.csv",
        )
        self.assertGreater(len(fund), 0)
        self.assertGreater(len(bench), 0)
        self.assertIn("date", fund[0])
        self.assertIn("fund_nav", fund[0])

    def test_csv_rejects_path_traversal(self) -> None:
        with self.assertRaises(PolicyError) as ctx:
            load_csv(ROOT, "data/../../../etc/passwd")
        self.assertEqual(ctx.exception.code, "ingestion_security_error")

    def test_csv_rejects_outside_data_dir(self) -> None:
        with self.assertRaises(PolicyError) as ctx:
            load_csv(ROOT, "backend/investment_analytics/errors.py")
        self.assertEqual(ctx.exception.code, "ingestion_security_error")

    def test_csv_rejects_nonexistent_file(self) -> None:
        with self.assertRaises(PolicyError) as ctx:
            load_csv(ROOT, "data/sample/does_not_exist.csv")
        self.assertEqual(ctx.exception.code, "ingestion_source_error")

    def test_csv_rejects_oversized_file(self) -> None:
        """CSV exceeding MAX_CSV_ROWS must be rejected."""
        from backend.data_ingestion.sources import MAX_CSV_ROWS
        from unittest.mock import patch
        with patch("backend.data_ingestion.sources.MAX_CSV_ROWS", 5):
            with self.assertRaises(PolicyError) as ctx:
                load_csv(ROOT, "data/sample/messy_mf_nav.csv")
            self.assertEqual(ctx.exception.code, "ingestion_size_error")


class IngestionPipelineTests(unittest.TestCase):

    def test_messy_csv_full_pipeline(self) -> None:
        """Messy CSV → normalize → validate → analyze → insights."""
        payload, report = ingest_mf_from_csv(
            ROOT,
            "data/sample/messy_mf_nav.csv",
            fund_name="Messy Test Fund",
            benchmark_name="Test Benchmark",
            expense_ratio_pct=1.5,
            rolling_window_points=4,
            rolling_step_points=1,
            rolling_min_windows=1,
        )
        result = analyze_mutual_fund(payload)
        self.assertIn("insights", result)
        self.assertGreaterEqual(len(result["insights"]), 1)

    def test_messy_csv_ingestion_report_complete(self) -> None:
        _, report = ingest_mf_from_csv(
            ROOT,
            "data/sample/messy_mf_nav.csv",
            fund_name="Test",
            benchmark_name="Bench",
            rolling_window_points=4,
            rolling_step_points=1,
            rolling_min_windows=1,
        )
        self.assertGreater(report["fund_series"]["rejected_count"], 0)
        self.assertGreater(report["fund_series"]["duplicate_dates_merged"], 0)
        self.assertGreater(report["fund_series"]["anomalies_flagged"], 0)
        self.assertGreater(report["fund_series"]["gaps_detected"], 0)
        self.assertGreater(len(report["ingestion_limitations"]), 0)

    def test_messy_csv_lineage_in_report(self) -> None:
        """Ingestion report must include source lineage for traceability."""
        _, report = ingest_mf_from_csv(
            ROOT,
            "data/sample/messy_mf_nav.csv",
            fund_name="Test",
            benchmark_name="Bench",
            rolling_window_points=4,
            rolling_step_points=1,
            rolling_min_windows=1,
        )
        self.assertEqual(report["source"], "csv")
        self.assertIn("data/sample/messy_mf_nav.csv", report["source_path"])
        self.assertIn("ingestion_timestamp", report)
        self.assertIsInstance(report["ingestion_timestamp"], str)
        self.assertGreater(len(report["ingestion_timestamp"]), 0)
        self.assertIn("license", report)

    def test_messy_csv_evidence_degrades(self) -> None:
        """Short messy history → evidence should be Low."""
        payload, _ = ingest_mf_from_csv(
            ROOT,
            "data/sample/messy_mf_nav.csv",
            fund_name="Test",
            benchmark_name="Bench",
            rolling_window_points=4,
            rolling_step_points=1,
            rolling_min_windows=1,
        )
        result = analyze_mutual_fund(payload)
        evidence_levels = {
            item["payload"]["evidence_strength"] for item in result["insights"]
        }
        self.assertIn("Low", evidence_levels)

    def test_messy_csv_outlier_flagged_by_analyzer(self) -> None:
        """Spike in messy data should produce outlier flags in analyzer."""
        payload, _ = ingest_mf_from_csv(
            ROOT,
            "data/sample/messy_mf_nav.csv",
            fund_name="Test",
            benchmark_name="Bench",
            rolling_window_points=4,
            rolling_step_points=1,
            rolling_min_windows=1,
        )
        result = analyze_mutual_fund(payload)
        self.assertGreater(len(result["data_quality"]["outlier_flags"]), 0)

    def test_messy_csv_limitations_surface(self) -> None:
        """Ingestion limitations should be human-readable and non-empty."""
        _, report = ingest_mf_from_csv(
            ROOT,
            "data/sample/messy_mf_nav.csv",
            fund_name="Test",
            benchmark_name="Bench",
            rolling_window_points=4,
            rolling_step_points=1,
            rolling_min_windows=1,
        )
        lims = report["ingestion_limitations"]
        self.assertIsInstance(lims, list)
        self.assertGreater(len(lims), 0)
        for lim in lims:
            self.assertIsInstance(lim, str)
            self.assertGreater(len(lim), 10)

    def test_messy_csv_no_advisory_language(self) -> None:
        payload, _ = ingest_mf_from_csv(
            ROOT,
            "data/sample/messy_mf_nav.csv",
            fund_name="Test",
            benchmark_name="Bench",
            rolling_window_points=4,
            rolling_step_points=1,
            rolling_min_windows=1,
        )
        result = analyze_mutual_fund(payload)
        raw = json.dumps(result["insights"]).lower()
        for term in ["should", "recommend", "buy", "sell",
                      "overweight", "rebalance", "optimize"]:
            self.assertNotIn(term, raw, f"Advisory term '{term}' in output")

    def test_clean_csv_full_pipeline(self) -> None:
        """Clean sample CSV should also work through ingestion pipeline."""
        payload, report = ingest_mf_from_csv(
            ROOT,
            "data/sample/mutual_fund_nav.csv",
            fund_name="Clean Fund",
            benchmark_name="Clean Bench",
            expense_ratio_pct=1.2,
            rolling_window_points=4,
            rolling_step_points=1,
            rolling_min_windows=1,
        )
        result = analyze_mutual_fund(payload)
        self.assertIn("insights", result)
        self.assertEqual(report["fund_series"]["rejected_count"], 0)


class SchemaMappingTests(unittest.TestCase):
    """Tests for the explicit schema mapping layer."""

    def test_validate_mapping_accepts_valid_columns(self) -> None:
        cols = ["Date", "NAV", "Index Value", "Category"]
        validate_mapping(AMFI_NAV_MAPPING, cols)  # should not raise

    def test_validate_mapping_rejects_missing_columns(self) -> None:
        cols = ["Date", "NAV"]  # missing Index Value
        with self.assertRaises(PolicyError) as ctx:
            validate_mapping(AMFI_NAV_MAPPING, cols)
        self.assertEqual(ctx.exception.code, "ingestion_schema_error")
        self.assertIn("Index Value", ctx.exception.message)

    def test_apply_mapping_remaps_to_canonical_keys(self) -> None:
        rows = [
            {"Date": "31-03-2020", "NAV": "100.00", "Index Value": "105.00"},
            {"Date": "30-06-2020", "NAV": "110.00", "Index Value": "115.00"},
        ]
        fund, bench = apply_mapping(rows, AMFI_NAV_MAPPING)
        self.assertEqual(len(fund), 2)
        self.assertEqual(len(bench), 2)
        # Canonical keys, not source keys
        self.assertIn(CANONICAL_DATE, fund[0])
        self.assertIn(CANONICAL_FUND_VALUE, fund[0])
        self.assertIn(CANONICAL_BENCHMARK_VALUE, bench[0])
        self.assertEqual(fund[0][CANONICAL_DATE], "31-03-2020")
        self.assertEqual(fund[0][CANONICAL_FUND_VALUE], "100.00")

    def test_default_mapping_matches_existing_csvs(self) -> None:
        self.assertEqual(DEFAULT_MAPPING.date_column, "date")
        self.assertEqual(DEFAULT_MAPPING.fund_column, "fund_nav")
        self.assertEqual(DEFAULT_MAPPING.benchmark_column, "benchmark_nav")

    def test_known_mappings_registry_contains_expected(self) -> None:
        self.assertIn("default", KNOWN_MAPPINGS)
        self.assertIn("amfi_nav", KNOWN_MAPPINGS)
        self.assertIn("etf_price", KNOWN_MAPPINGS)
        self.assertIn("alt_mf", KNOWN_MAPPINGS)

    def test_custom_mapping_works(self) -> None:
        custom = ColumnMapping(
            date_column="trade_date",
            fund_column="price",
            benchmark_column="index",
            label="custom_test",
        )
        rows = [{"trade_date": "2024-01-01", "price": "50", "index": "100"}]
        fund, bench = apply_mapping(rows, custom)
        self.assertEqual(fund[0][CANONICAL_FUND_VALUE], "50")
        self.assertEqual(bench[0][CANONICAL_BENCHMARK_VALUE], "100")

    def test_amfi_csv_full_pipeline(self) -> None:
        """AMFI-format CSV with non-default columns ingests correctly."""
        payload, report = ingest_mf_from_csv(
            ROOT,
            "data/sample/amfi_nav.csv",
            fund_name="AMFI Fund",
            benchmark_name="Nifty 50",
            mapping=AMFI_NAV_MAPPING,
            rolling_window_points=4,
            rolling_step_points=1,
            rolling_min_windows=1,
        )
        result = analyze_mutual_fund(payload)
        self.assertIn("insights", result)
        self.assertGreaterEqual(len(result["insights"]), 1)
        self.assertEqual(report["mapping_label"], "AMFI NAV")
        self.assertEqual(report["fund_series"]["rejected_count"], 0)

    def test_amfi_csv_without_mapping_fails(self) -> None:
        """AMFI CSV without mapping produces empty series (no 'fund_nav' column)."""
        # Without mapping, load_csv uses default column names which don't exist
        # in the AMFI CSV — normalization will reject all rows.
        with self.assertRaises(PolicyError):
            ingest_mf_from_csv(
                ROOT,
                "data/sample/amfi_nav.csv",
                fund_name="AMFI Fund",
                benchmark_name="Nifty 50",
                rolling_window_points=4,
                rolling_step_points=1,
                rolling_min_windows=1,
            )

    def test_wrong_mapping_for_csv_fails(self) -> None:
        """Applying AMFI mapping to default-format CSV fails on missing columns."""
        with self.assertRaises(PolicyError) as ctx:
            ingest_mf_from_csv(
                ROOT,
                "data/sample/mutual_fund_nav.csv",
                fund_name="Test",
                benchmark_name="Bench",
                mapping=AMFI_NAV_MAPPING,
                rolling_window_points=4,
                rolling_step_points=1,
                rolling_min_windows=1,
            )
        self.assertEqual(ctx.exception.code, "ingestion_schema_error")

    def test_mapping_label_in_report(self) -> None:
        """Direct (no mapping) ingestion records 'direct' as mapping label."""
        _, report = ingest_mf_from_csv(
            ROOT,
            "data/sample/mutual_fund_nav.csv",
            fund_name="Test",
            benchmark_name="Bench",
            rolling_window_points=4,
            rolling_step_points=1,
            rolling_min_windows=1,
        )
        self.assertEqual(report["mapping_label"], "direct")

    def test_etf_csv_full_pipeline(self) -> None:
        """ETF-format CSV with price columns ingests correctly via mapping."""
        payload, report = ingest_mf_from_csv(
            ROOT,
            "data/sample/etf_price.csv",
            fund_name="ETFX",
            benchmark_name="Broad Market Index",
            mapping=ETF_PRICE_MAPPING,
            rolling_window_points=4,
            rolling_step_points=1,
            rolling_min_windows=1,
        )
        result = analyze_mutual_fund(payload)
        self.assertIn("insights", result)
        self.assertGreaterEqual(len(result["insights"]), 1)
        self.assertEqual(report["mapping_label"], "ETF price")
        self.assertEqual(report["fund_series"]["rejected_count"], 0)
        # ETF CSV has 50 daily rows
        self.assertEqual(report["fund_series"]["output_points"], 50)

    def test_etf_csv_without_mapping_fails(self) -> None:
        """ETF CSV without mapping has no 'fund_nav' column → fails."""
        with self.assertRaises(PolicyError):
            ingest_mf_from_csv(
                ROOT,
                "data/sample/etf_price.csv",
                fund_name="ETFX",
                benchmark_name="Bench",
                rolling_window_points=4,
                rolling_step_points=1,
                rolling_min_windows=1,
            )

    def test_alt_mf_csv_full_pipeline(self) -> None:
        """Alternate MF format with 'Benchmark NAV' column ingests correctly."""
        payload, report = ingest_mf_from_csv(
            ROOT,
            "data/sample/alt_mf_nav.csv",
            fund_name="Alternate Fund",
            benchmark_name="Alt Bench",
            mapping=ALT_MF_MAPPING,
            rolling_window_points=4,
            rolling_step_points=1,
            rolling_min_windows=1,
        )
        result = analyze_mutual_fund(payload)
        self.assertIn("insights", result)
        self.assertGreaterEqual(len(result["insights"]), 1)
        self.assertEqual(report["mapping_label"], "alternate MF")
        self.assertEqual(report["fund_series"]["rejected_count"], 0)

    def test_error_message_includes_column_name_and_mapping(self) -> None:
        """Schema error message must name the missing column and mapping."""
        with self.assertRaises(PolicyError) as ctx:
            ingest_mf_from_csv(
                ROOT,
                "data/sample/mutual_fund_nav.csv",
                fund_name="Test",
                benchmark_name="Bench",
                mapping=ETF_PRICE_MAPPING,
                rolling_window_points=4,
                rolling_step_points=1,
                rolling_min_windows=1,
            )
        msg = ctx.exception.message
        self.assertIn("'Close Price'", msg)
        self.assertIn("ETF price", msg)
        self.assertIn("Check schema_mapping=", msg)

    def test_all_mappings_produce_insights_on_matching_csv(self) -> None:
        """Each known mapping that has a matching CSV should produce insights."""
        mapping_to_csv = {
            "default": "data/sample/mutual_fund_nav.csv",
            "amfi_nav": "data/sample/amfi_nav.csv",
            "etf_price": "data/sample/etf_price.csv",
            "alt_mf": "data/sample/alt_mf_nav.csv",
        }
        for name, csv_path in mapping_to_csv.items():
            with self.subTest(mapping=name):
                mapping = KNOWN_MAPPINGS[name]
                payload, report = ingest_mf_from_csv(
                    ROOT,
                    csv_path,
                    fund_name=f"Fund ({name})",
                    benchmark_name=f"Bench ({name})",
                    mapping=mapping if name != "default" else None,
                    rolling_window_points=4,
                    rolling_step_points=1,
                    rolling_min_windows=1,
                )
                result = analyze_mutual_fund(payload)
                self.assertGreaterEqual(
                    len(result["insights"]), 1,
                    f"Mapping '{name}' produced no insights",
                )
                self.assertEqual(report["fund_series"]["rejected_count"], 0)


class ImpactLinkTests(unittest.TestCase):
    """Tests for deterministic issue → metric linking."""

    @staticmethod
    def _empty_meta(**overrides: int) -> dict:
        base = {
            "input_records": 100,
            "output_points": 100,
            "rejected_count": 0,
            "duplicate_dates_merged": 0,
            "anomalies_flagged": 0,
            "gaps_detected": 0,
            "date_range": {"start": "2020-01-01", "end": "2024-12-31"},
        }
        base.update(overrides)
        return base

    def test_no_issues_produces_no_links(self) -> None:
        links = build_impact_links(
            self._empty_meta(), self._empty_meta(), [], [], [], [],
        )
        self.assertEqual(links, [])

    def test_gaps_link_to_rolling_drawdown_trailing(self) -> None:
        meta = self._empty_meta(gaps_detected=1)
        events = [{"type": "gap_detected", "start_date": "2024-02-12",
                   "end_date": "2024-02-26", "gap_days": 14, "series": "fund"}]
        links = build_impact_links(meta, self._empty_meta(), [], [], events, [])
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]["issue"], "gaps_detected")
        self.assertIn("rolling excess returns", links[0]["affected_metrics"])
        self.assertIn("drawdown profile", links[0]["affected_metrics"])
        self.assertIn("trailing returns", links[0]["affected_metrics"])
        self.assertEqual(links[0]["series"], "fund")
        # Dates surfaced in explanation
        self.assertIn("2024-02-12", links[0]["explanation"])

    def test_anomalies_link_to_rolling_drawdown_trailing(self) -> None:
        meta = self._empty_meta(anomalies_flagged=2)
        anomalies = [
            {"type": "extreme_move", "date": "2024-03-12", "series": "fund"},
            {"type": "extreme_move", "date": "2024-05-20", "series": "fund"},
        ]
        links = build_impact_links(
            meta, self._empty_meta(), anomalies, [], [], [],
        )
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]["issue"], "extreme_values")
        self.assertIn("rolling excess returns", links[0]["affected_metrics"])
        self.assertIn("drawdown profile", links[0]["affected_metrics"])
        self.assertIn("2024-03-12", links[0]["explanation"])

    def test_rejected_records_link_to_evidence(self) -> None:
        meta = self._empty_meta(rejected_count=3)
        links = build_impact_links(
            meta, self._empty_meta(), [], [], [], [],
        )
        rejected_link = [l for l in links if l["issue"] == "records_rejected"]
        self.assertEqual(len(rejected_link), 1)
        self.assertIn("evidence strength", rejected_link[0]["affected_metrics"])
        self.assertIn("3 record(s)", rejected_link[0]["explanation"])

    def test_short_history_links_to_all_timeseries(self) -> None:
        meta = self._empty_meta(output_points=10)
        links = build_impact_links(
            meta, self._empty_meta(output_points=10), [], [], [], [],
        )
        short_link = [l for l in links if l["issue"] == "short_history"]
        self.assertEqual(len(short_link), 1)
        self.assertIn("trailing returns", short_link[0]["affected_metrics"])
        self.assertIn("rolling excess returns", short_link[0]["affected_metrics"])
        self.assertIn("drawdown profile", short_link[0]["affected_metrics"])
        self.assertIn("evidence strength", short_link[0]["affected_metrics"])

    def test_sufficient_history_no_short_link(self) -> None:
        meta = self._empty_meta(output_points=100)
        links = build_impact_links(
            meta, self._empty_meta(output_points=100), [], [], [], [],
        )
        short_links = [l for l in links if l["issue"] == "short_history"]
        self.assertEqual(len(short_links), 0)

    def test_duplicates_link_to_evidence(self) -> None:
        meta = self._empty_meta(duplicate_dates_merged=2)
        links = build_impact_links(
            meta, self._empty_meta(), [], [], [], [],
        )
        dupe_link = [l for l in links if l["issue"] == "duplicates_merged"]
        self.assertEqual(len(dupe_link), 1)
        self.assertIn("evidence strength", dupe_link[0]["affected_metrics"])

    def test_both_series_reported(self) -> None:
        fund_meta = self._empty_meta(gaps_detected=1)
        bench_meta = self._empty_meta(gaps_detected=1)
        fund_events = [{"type": "gap_detected", "start_date": "2024-02-12",
                        "end_date": "2024-02-26", "gap_days": 14, "series": "fund"}]
        bench_events = [{"type": "gap_detected", "start_date": "2024-03-01",
                         "end_date": "2024-03-15", "gap_days": 14, "series": "benchmark"}]
        links = build_impact_links(
            fund_meta, bench_meta, [], [], fund_events, bench_events,
        )
        gap_link = [l for l in links if l["issue"] == "gaps_detected"]
        self.assertEqual(gap_link[0]["series"], "both")

    def test_multiple_issues_produce_multiple_links(self) -> None:
        """Messy data should produce multiple distinct links."""
        meta = self._empty_meta(
            gaps_detected=1, anomalies_flagged=1, rejected_count=2,
            output_points=10,
        )
        events = [{"type": "gap_detected", "start_date": "2024-02-12",
                   "end_date": "2024-02-26", "gap_days": 14, "series": "fund"}]
        anomalies = [{"type": "extreme_move", "date": "2024-03-12",
                      "series": "fund"}]
        links = build_impact_links(
            meta, self._empty_meta(output_points=10),
            anomalies, [], events, [],
        )
        issue_types = {l["issue"] for l in links}
        self.assertIn("gaps_detected", issue_types)
        self.assertIn("extreme_values", issue_types)
        self.assertIn("records_rejected", issue_types)
        self.assertIn("short_history", issue_types)
        self.assertGreaterEqual(len(links), 4)

    def test_link_ordering_is_deterministic(self) -> None:
        """Links must appear in fixed broadest-impact-first order."""
        meta = self._empty_meta(
            gaps_detected=1, anomalies_flagged=1, rejected_count=1,
            output_points=10, duplicate_dates_merged=1,
        )
        events = [{"type": "gap_detected", "start_date": "2024-02-12",
                   "end_date": "2024-02-26", "gap_days": 14, "series": "fund"}]
        anomalies = [{"type": "extreme_move", "date": "2024-03-12",
                      "series": "fund"}]
        links = build_impact_links(
            meta, self._empty_meta(output_points=10),
            anomalies, [], events, [],
        )
        order = [l["issue"] for l in links]
        expected = [
            "short_history",
            "records_rejected",
            "duplicates_merged",
            "gaps_detected",
            "extreme_values",
        ]
        self.assertEqual(order, expected)

    def test_no_advisory_language_in_links(self) -> None:
        """Impact link text must be informational, not advisory."""
        meta = self._empty_meta(
            gaps_detected=1, anomalies_flagged=1, rejected_count=1,
            output_points=10, duplicate_dates_merged=1,
        )
        events = [{"type": "gap_detected", "start_date": "2024-02-12",
                   "end_date": "2024-02-26", "gap_days": 14, "series": "fund"}]
        anomalies = [{"type": "extreme_move", "date": "2024-03-12",
                      "series": "fund"}]
        links = build_impact_links(
            meta, self._empty_meta(output_points=10),
            anomalies, [], events, [],
        )
        raw = json.dumps(links).lower()
        for term in ["unreliable", "degraded", "poor", "bad",
                      "should", "recommend", "avoid"]:
            self.assertNotIn(term, raw,
                             f"Advisory/judgmental term '{term}' in links")

    def test_messy_csv_pipeline_has_impact_links(self) -> None:
        """End-to-end: messy CSV ingestion report includes impact links."""
        _, report = ingest_mf_from_csv(
            ROOT,
            "data/sample/messy_mf_nav.csv",
            fund_name="Test",
            benchmark_name="Bench",
            rolling_window_points=4,
            rolling_step_points=1,
            rolling_min_windows=1,
        )
        links = report["impact_links"]
        self.assertIsInstance(links, list)
        self.assertGreater(len(links), 0)
        # Messy CSV has gaps + outliers + rejected + short history
        issue_types = {l["issue"] for l in links}
        self.assertIn("gaps_detected", issue_types)
        self.assertIn("extreme_values", issue_types)

    def test_clean_csv_minimal_impact_links(self) -> None:
        """Clean CSV (quarterly) should have gap links but no outlier/rejected."""
        _, report = ingest_mf_from_csv(
            ROOT,
            "data/sample/mutual_fund_nav.csv",
            fund_name="Clean",
            benchmark_name="Bench",
            rolling_window_points=4,
            rolling_step_points=1,
            rolling_min_windows=1,
        )
        links = report["impact_links"]
        issue_types = {l["issue"] for l in links}
        # Quarterly data has natural ~90-day gaps → gap detection is correct
        # But no anomalies, no rejected records
        self.assertNotIn("extreme_values", issue_types)
        self.assertNotIn("records_rejected", issue_types)


if __name__ == "__main__":
    unittest.main()
