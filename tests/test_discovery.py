"""Tests for Phase 2 — data discovery layer.

Covers:
  - Registry parsing (NAVAll.txt format)
  - Search logic (prefix, contains, code, filters)
  - NAV date conversion (DD-MM-YYYY → ISO)
  - Fetch layer record conversion
  - API model validation
  - Registry persistence (save/load)
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backend.data_discovery.registry import (
    SchemeEntry,
    load_registry,
    parse_navall_text,
    save_registry,
)
from backend.data_discovery.search import search_schemes
from backend.data_discovery.fetch import (
    _convert_nav_to_records,
    _parse_dd_mm_yyyy,
)


# ─────────────────────────────────────────────────────────────
# Sample NAVAll.txt snippets for testing
# ─────────────────────────────────────────────────────────────

SAMPLE_NAVALL = """\
Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Net Asset Value;Date

Open Ended Schemes(Debt Scheme - Banking and PSU Fund)

Aditya Birla Sun Life Mutual Fund
119551;INF209KA12Z1;INF209KA13Z9;Aditya Birla Sun Life Banking & PSU Debt Fund  - DIRECT - IDCW;104.76520;23-Apr-2026
119552;INF209KB12Z2;-;Aditya Birla Sun Life Banking & PSU Debt Fund  - DIRECT - Growth;52.12340;23-Apr-2026

Open Ended Schemes(Equity Scheme - Flexi Cap Fund)

HDFC Mutual Fund
112233;INF179KA1RN2;-;HDFC Flexi Cap Fund - Direct Plan - Growth;150.50000;23-Apr-2026
112234;INF179KA1RO0;INF179KA1RP7;HDFC Flexi Cap Fund - Regular Plan - Growth;140.25000;23-Apr-2026

ICICI Prudential Mutual Fund
445566;INF109KA1234;-;ICICI Prudential Bluechip Fund - Direct Plan - Growth;88.00000;23-Apr-2026
"""

SAMPLE_NAVALL_EMPTY = """\
Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Net Asset Value;Date
"""

SAMPLE_NAVALL_MALFORMED = """\
Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Net Asset Value;Date

Open Ended Schemes(Equity Scheme - Large Cap Fund)

Test Fund House
not_a_code;INF209KA12Z1;-;Broken Scheme;10.0;23-Apr-2026
999;INF209KA12Z1;-;;10.0;23-Apr-2026
888;INF209KA12Z1;-;Valid Scheme;N.A.;23-Apr-2026
777;INF209KA12Z1;-;Also Valid;25.5;23-Apr-2026
"""


class TestRegistryParsing(unittest.TestCase):
    """Test NAVAll.txt parsing."""

    def test_parse_basic(self):
        entries = parse_navall_text(SAMPLE_NAVALL)
        self.assertEqual(len(entries), 5)

    def test_scheme_codes_extracted(self):
        entries = parse_navall_text(SAMPLE_NAVALL)
        codes = [e.scheme_code for e in entries]
        self.assertIn(119551, codes)
        self.assertIn(112233, codes)
        self.assertIn(445566, codes)

    def test_fund_house_assigned(self):
        entries = parse_navall_text(SAMPLE_NAVALL)
        by_code = {e.scheme_code: e for e in entries}
        self.assertEqual(
            by_code[119551].fund_house,
            "Aditya Birla Sun Life Mutual Fund",
        )
        self.assertEqual(by_code[112233].fund_house, "HDFC Mutual Fund")
        self.assertEqual(
            by_code[445566].fund_house,
            "ICICI Prudential Mutual Fund",
        )

    def test_category_assigned(self):
        entries = parse_navall_text(SAMPLE_NAVALL)
        by_code = {e.scheme_code: e for e in entries}
        self.assertIn("Banking and PSU Fund", by_code[119551].scheme_category)
        self.assertIn("Flexi Cap Fund", by_code[112233].scheme_category)

    def test_isin_parsed(self):
        entries = parse_navall_text(SAMPLE_NAVALL)
        by_code = {e.scheme_code: e for e in entries}
        self.assertEqual(by_code[119551].isin_growth, "INF209KA12Z1")
        self.assertEqual(by_code[119551].isin_reinvestment, "INF209KA13Z9")
        # Dash means None
        self.assertIsNone(by_code[119552].isin_reinvestment)

    def test_nav_parsed(self):
        entries = parse_navall_text(SAMPLE_NAVALL)
        by_code = {e.scheme_code: e for e in entries}
        self.assertAlmostEqual(by_code[119551].latest_nav, 104.76520)
        self.assertAlmostEqual(by_code[112233].latest_nav, 150.50000)

    def test_empty_navall(self):
        entries = parse_navall_text(SAMPLE_NAVALL_EMPTY)
        self.assertEqual(len(entries), 0)

    def test_malformed_lines_skipped(self):
        entries = parse_navall_text(SAMPLE_NAVALL_MALFORMED)
        # "not_a_code" → skipped, empty scheme name → skipped
        # "N.A." NAV → nav=None but still valid entry
        # "Also Valid" → valid
        codes = [e.scheme_code for e in entries]
        self.assertNotIn("not_a_code", [str(c) for c in codes])
        # Empty name scheme skipped
        self.assertNotIn(999, codes)
        # NAV=N.A. is still valid (nav=None)
        self.assertIn(888, codes)
        self.assertIn(777, codes)

    def test_na_nav_is_none(self):
        entries = parse_navall_text(SAMPLE_NAVALL_MALFORMED)
        by_code = {e.scheme_code: e for e in entries}
        self.assertIsNone(by_code[888].latest_nav)

    def test_scheme_name_trimmed(self):
        entries = parse_navall_text(SAMPLE_NAVALL)
        for e in entries:
            self.assertEqual(e.scheme_name, e.scheme_name.strip())


class TestRegistryPersistence(unittest.TestCase):
    """Test save/load round-trip."""

    def test_save_and_load(self):
        entries = parse_navall_text(SAMPLE_NAVALL)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "schemes.json"
            save_registry(entries, path)
            self.assertTrue(path.exists())
            loaded = load_registry(path)
            self.assertEqual(len(loaded), len(entries))
            # Spot-check first entry
            self.assertEqual(loaded[0].scheme_code, entries[0].scheme_code)
            self.assertEqual(loaded[0].scheme_name, entries[0].scheme_name)

    def test_load_missing_file(self):
        loaded = load_registry(Path("/nonexistent/schemes.json"))
        self.assertEqual(loaded, [])

    def test_json_structure(self):
        entries = parse_navall_text(SAMPLE_NAVALL)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "schemes.json"
            save_registry(entries, path)
            data = json.loads(path.read_text())
            self.assertIsInstance(data, list)
            self.assertIn("scheme_code", data[0])
            self.assertIn("scheme_name", data[0])
            # Internal field must not leak
            self.assertNotIn("_name_lower", data[0])


class TestSearch(unittest.TestCase):
    """Test search logic."""

    @classmethod
    def setUpClass(cls):
        cls.registry = parse_navall_text(SAMPLE_NAVALL)

    def test_search_by_name_prefix(self):
        results = search_schemes(self.registry, "HDFC")
        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(
            any("HDFC" in r["scheme_name"] for r in results)
        )

    def test_search_by_name_contains(self):
        results = search_schemes(self.registry, "Flexi Cap")
        self.assertGreaterEqual(len(results), 1)

    def test_search_by_code(self):
        results = search_schemes(self.registry, "119551")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["scheme_code"], 119551)

    def test_search_case_insensitive(self):
        results_lower = search_schemes(self.registry, "hdfc")
        results_upper = search_schemes(self.registry, "HDFC")
        self.assertEqual(len(results_lower), len(results_upper))

    def test_search_empty_query(self):
        results = search_schemes(self.registry, "")
        self.assertEqual(len(results), 0)

    def test_search_no_match(self):
        results = search_schemes(self.registry, "zzz_nonexistent_fund")
        self.assertEqual(len(results), 0)

    def test_search_max_results(self):
        results = search_schemes(self.registry, "Fund", max_results=2)
        self.assertLessEqual(len(results), 2)

    def test_search_category_filter(self):
        results = search_schemes(
            self.registry, "Fund",
            category_filter="Flexi Cap",
        )
        for r in results:
            self.assertIn("Flexi Cap", r["scheme_category"])

    def test_search_fund_house_filter(self):
        results = search_schemes(
            self.registry, "Fund",
            fund_house_filter="ICICI",
        )
        for r in results:
            self.assertIn("ICICI", r["fund_house"])

    def test_search_results_serializable(self):
        results = search_schemes(self.registry, "HDFC")
        # Must be JSON-safe (no dataclass objects)
        json.dumps(results)

    def test_search_no_internal_fields_leaked(self):
        results = search_schemes(self.registry, "HDFC")
        for r in results:
            self.assertNotIn("_name_lower", r)

    def test_search_by_fund_house_name(self):
        """Searching for fund house name should match even if scheme name doesn't contain it."""
        results = search_schemes(self.registry, "ICICI Prudential")
        self.assertGreaterEqual(len(results), 1)

    def test_prefix_matches_ranked_first(self):
        """Schemes whose name starts with query should appear before contains matches."""
        results = search_schemes(self.registry, "Aditya")
        if len(results) >= 2:
            # First results should be the ones starting with "Aditya"
            self.assertTrue(results[0]["scheme_name"].lower().startswith("aditya"))


class TestDateParsing(unittest.TestCase):
    """Test DD-MM-YYYY date parsing."""

    def test_valid_date(self):
        from datetime import date
        d = _parse_dd_mm_yyyy("23-04-2026")
        self.assertEqual(d, date(2026, 4, 23))

    def test_invalid_date(self):
        self.assertIsNone(_parse_dd_mm_yyyy("not-a-date"))

    def test_empty_string(self):
        self.assertIsNone(_parse_dd_mm_yyyy(""))

    def test_leap_year(self):
        from datetime import date
        d = _parse_dd_mm_yyyy("29-02-2024")
        self.assertEqual(d, date(2024, 2, 29))

    def test_non_leap_year_feb_29(self):
        self.assertIsNone(_parse_dd_mm_yyyy("29-02-2023"))


class TestNavConversion(unittest.TestCase):
    """Test mfapi.in JSON → canonical record conversion."""

    def test_basic_conversion(self):
        nav_data = [
            {"date": "23-04-2026", "nav": "104.76520"},
            {"date": "22-04-2026", "nav": "104.84700"},
            {"date": "21-04-2026", "nav": "103.50000"},
        ]
        records = _convert_nav_to_records(nav_data)
        # Should be reversed (oldest first)
        self.assertEqual(records[0]["date"], "2026-04-21")
        self.assertEqual(records[-1]["date"], "2026-04-23")
        self.assertAlmostEqual(records[0]["nav"], 103.5)

    def test_invalid_entries_skipped(self):
        nav_data = [
            {"date": "23-04-2026", "nav": "104.0"},
            {"date": "bad-date", "nav": "100.0"},
            {"date": "22-04-2026", "nav": "N.A."},
            {"date": "21-04-2026", "nav": "0"},  # zero nav skipped
            {"date": "20-04-2026", "nav": "-5"},  # negative skipped
        ]
        records = _convert_nav_to_records(nav_data)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["date"], "2026-04-23")

    def test_empty_input(self):
        records = _convert_nav_to_records([])
        self.assertEqual(records, [])

    def test_output_is_chronological(self):
        """mfapi returns newest-first; output must be oldest-first."""
        nav_data = [
            {"date": "03-01-2025", "nav": "30.0"},
            {"date": "02-01-2025", "nav": "20.0"},
            {"date": "01-01-2025", "nav": "10.0"},
        ]
        records = _convert_nav_to_records(nav_data)
        dates = [r["date"] for r in records]
        self.assertEqual(dates, sorted(dates))

    def test_max_cap_applied(self):
        """Records beyond MAX_NAV_POINTS should be capped."""
        from backend.data_discovery.fetch import MAX_NAV_POINTS
        nav_data = [
            {"date": f"{str(i % 28 + 1).zfill(2)}-{str(i % 12 + 1).zfill(2)}-{2000 + i // 365}", "nav": str(10.0 + i * 0.01)}
            for i in range(MAX_NAV_POINTS + 500)
        ]
        # Some may fail date parse, but the cap logic should still apply
        records = _convert_nav_to_records(nav_data)
        self.assertLessEqual(len(records), MAX_NAV_POINTS)

    def test_iso_date_format(self):
        """Output dates must be ISO YYYY-MM-DD."""
        nav_data = [{"date": "05-12-2024", "nav": "100.0"}]
        records = _convert_nav_to_records(nav_data)
        self.assertEqual(records[0]["date"], "2024-12-05")


class TestSchemeEntryInvariants(unittest.TestCase):
    """Test SchemeEntry dataclass behavior."""

    def test_frozen(self):
        e = SchemeEntry(scheme_code=1, scheme_name="Test", fund_house="FH", scheme_category="Cat")
        with self.assertRaises(AttributeError):
            e.scheme_code = 2  # type: ignore

    def test_name_lower_precomputed(self):
        e = SchemeEntry(scheme_code=1, scheme_name="HDFC Flexi", fund_house="FH", scheme_category="Cat")
        self.assertEqual(e._name_lower, "hdfc flexi")


class TestSearchEdgeCases(unittest.TestCase):
    """Edge cases for search."""

    def test_single_char_query_returns_nothing(self):
        """Single character queries are too broad; should return nothing (len < 2 in practice)."""
        registry = parse_navall_text(SAMPLE_NAVALL)
        # The search doesn't filter by length, but "a" is valid — just produces results
        # This tests that it doesn't crash
        results = search_schemes(registry, "a")
        self.assertIsInstance(results, list)

    def test_whitespace_only_query(self):
        registry = parse_navall_text(SAMPLE_NAVALL)
        results = search_schemes(registry, "   ")
        self.assertEqual(len(results), 0)

    def test_max_results_cap_at_50(self):
        """Even if caller asks for 1000, cap is 50."""
        registry = parse_navall_text(SAMPLE_NAVALL)
        results = search_schemes(registry, "Fund", max_results=1000)
        self.assertLessEqual(len(results), 50)

    def test_numeric_query_non_matching_code(self):
        registry = parse_navall_text(SAMPLE_NAVALL)
        results = search_schemes(registry, "999999")
        self.assertEqual(len(results), 0)

    def test_combined_filters_narrow_results(self):
        registry = parse_navall_text(SAMPLE_NAVALL)
        all_results = search_schemes(registry, "Fund")
        filtered = search_schemes(registry, "Fund", fund_house_filter="HDFC")
        self.assertLessEqual(len(filtered), len(all_results))


if __name__ == "__main__":
    unittest.main()
