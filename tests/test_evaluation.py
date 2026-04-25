"""Tests for the portfolio evaluation engine."""
import unittest
from datetime import date, timedelta

from backend.investment_analytics.evaluation import (
    DEFAULT_CONSTRAINTS,
    _check_constraint,
    _extract_metrics,
    detect_flags,
    evaluate_constraints,
    evaluate_portfolio,
)


# ── Helpers ───────────────────────────────────────────────────

def _make_insight(supporting_data: dict) -> dict:
    """Wrap supporting_data into a minimal insight dict."""
    return {"supporting_data": supporting_data}


def _make_compiled_insight(supporting_data: dict) -> dict:
    """Wrap supporting_data into a compiled insight dict with payload."""
    return {"template": "diagnostic", "payload": {"supporting_data": supporting_data}}


def _realistic_insights() -> list:
    """Return a list of realistic portfolio insight supporting_data blocks."""
    return [
        _make_insight({"trailing_cagr": 9.5, "fund_count": 2, "fund_weights": {}}),
        _make_insight({
            "distribution": {
                "window_count": 20,
                "median_cagr_pct": 8.5,
                "p25_cagr_pct": 6.0,
                "p75_cagr_pct": 11.0,
                "min_cagr_pct": 2.0,
                "max_cagr_pct": 14.0,
            },
        }),
        _make_insight({
            "drawdown": {
                "max_drawdown_pct": -22.5,
                "drawdowns_gt_threshold_pct": 1,
                "threshold_pct": -5.0,
                "avg_recovery_days": 180,
                "max_recovery_days": 450,
            },
            "fund_count": 2,
        }),
        _make_insight({
            "volatility": {
                "periodic_return_std_pct": 1.2,
                "observation_count": 500,
                "note": "daily",
            },
        }),
        _make_insight({
            "correlation_pairs": [
                {"fund_a": "FundA", "fund_b": "FundB", "code_a": 1, "code_b": 2,
                 "correlation": 0.65},
            ],
            "average_correlation": 0.65,
        }),
        _make_insight({
            "concentration": {
                "hhi": 0.42,
                "effective_fund_count": 2.38,
                "actual_fund_count": 2,
                "largest_weight": 0.55,
                "smallest_weight": 0.45,
            },
        }),
        _make_insight({
            "contribution": {
                "portfolio_return_pct": 9.5,
                "funds": [
                    {"scheme_code": 1, "name": "FundA", "weight": 0.55,
                     "fund_return_pct": 11.0, "weighted_return_pct": 6.05,
                     "fund_max_drawdown_pct": -25.0},
                    {"scheme_code": 2, "name": "FundB", "weight": 0.45,
                     "fund_return_pct": 7.5, "weighted_return_pct": 3.38,
                     "fund_max_drawdown_pct": -18.0},
                ],
            },
        }),
    ]


# ── _extract_metrics ──────────────────────────────────────────

class TestExtractMetrics(unittest.TestCase):
    """Test metric extraction from insight supporting_data."""

    def test_extracts_all_metric_types(self):
        m = _extract_metrics(_realistic_insights())
        self.assertEqual(m["max_drawdown_pct"], -22.5)
        self.assertEqual(m["max_recovery_days"], 450)
        self.assertEqual(m["median_rolling_cagr_pct"], 8.5)
        self.assertEqual(m["volatility_pct"], 1.2)
        self.assertEqual(m["max_pairwise_correlation"], 0.65)
        self.assertEqual(m["hhi"], 0.42)
        self.assertEqual(m["portfolio_return_pct"], 9.5)
        self.assertEqual(m["trailing_cagr"], 9.5)
        self.assertEqual(len(m["fund_contributions"]), 2)

    def test_empty_insights(self):
        m = _extract_metrics([])
        self.assertEqual(m, {})

    def test_insight_without_supporting_data(self):
        m = _extract_metrics([{"foo": "bar"}])
        self.assertEqual(m, {})

    def test_compiled_insight_with_payload(self):
        compiled = [_make_compiled_insight({
            "drawdown": {"max_drawdown_pct": -15.0, "max_recovery_days": 200},
        })]
        m = _extract_metrics(compiled)
        self.assertEqual(m["max_drawdown_pct"], -15.0)
        self.assertEqual(m["max_recovery_days"], 200)

    def test_mixed_raw_and_compiled_insights(self):
        mixed = [
            _make_insight({"volatility": {"periodic_return_std_pct": 1.8}}),
            _make_compiled_insight({"drawdown": {"max_drawdown_pct": -10.0}}),
        ]
        m = _extract_metrics(mixed)
        self.assertEqual(m["volatility_pct"], 1.8)
        self.assertEqual(m["max_drawdown_pct"], -10.0)

    def test_partial_data(self):
        """Only drawdown — other metrics should be absent."""
        m = _extract_metrics([
            _make_insight({"drawdown": {"max_drawdown_pct": -5.0}}),
        ])
        self.assertEqual(m["max_drawdown_pct"], -5.0)
        self.assertNotIn("volatility_pct", m)
        self.assertNotIn("hhi", m)

    def test_correlation_max_is_absolute_value(self):
        m = _extract_metrics([_make_insight({
            "correlation_pairs": [
                {"fund_a": "A", "fund_b": "B", "code_a": 1, "code_b": 2,
                 "correlation": -0.95},
            ],
            "average_correlation": -0.95,
        })])
        self.assertEqual(m["max_pairwise_correlation"], 0.95)

    def test_rolling_window_count_extracted(self):
        m = _extract_metrics([_make_insight({
            "distribution": {"window_count": 42, "median_cagr_pct": 7.0},
        })])
        self.assertEqual(m["rolling_window_count"], 42)

    def test_missing_drawdown_fields_are_none_not_zero(self):
        """Missing fields in drawdown dict should be None, not 0."""
        m = _extract_metrics([_make_insight({"drawdown": {}})])
        self.assertIsNone(m.get("max_drawdown_pct"))
        self.assertIsNone(m.get("max_recovery_days"))
        self.assertIsNone(m.get("drawdowns_gt_threshold"))

    def test_missing_distribution_fields_are_none(self):
        m = _extract_metrics([_make_insight({"distribution": {}})])
        self.assertIsNone(m.get("median_rolling_cagr_pct"))
        self.assertIsNone(m.get("rolling_window_count"))

    def test_missing_volatility_fields_are_none(self):
        m = _extract_metrics([_make_insight({"volatility": {}})])
        self.assertIsNone(m.get("volatility_pct"))
        self.assertIsNone(m.get("observation_count"))

    def test_observation_count_extracted(self):
        m = _extract_metrics([_make_insight({
            "volatility": {"periodic_return_std_pct": 1.5, "observation_count": 500},
        })])
        self.assertEqual(m["observation_count"], 500)


# ── _check_constraint ─────────────────────────────────────────

class TestCheckConstraint(unittest.TestCase):
    """Unit tests for single constraint checking."""

    def test_gte_pass(self):
        r = _check_constraint("test", 10.0, 5.0, "gte")
        self.assertEqual(r["status"], "PASS")

    def test_gte_fail(self):
        r = _check_constraint("test", 3.0, 5.0, "gte")
        self.assertEqual(r["status"], "FAIL")

    def test_gte_exact_boundary(self):
        r = _check_constraint("test", 5.0, 5.0, "gte")
        self.assertEqual(r["status"], "PASS")

    def test_lte_pass(self):
        r = _check_constraint("test", 3.0, 5.0, "lte")
        self.assertEqual(r["status"], "PASS")

    def test_lte_fail(self):
        r = _check_constraint("test", 7.0, 5.0, "lte")
        self.assertEqual(r["status"], "FAIL")

    def test_lte_exact_boundary(self):
        r = _check_constraint("test", 5.0, 5.0, "lte")
        self.assertEqual(r["status"], "PASS")

    def test_insufficient_data(self):
        r = _check_constraint("test", None, 5.0, "gte")
        self.assertEqual(r["status"], "INSUFFICIENT_DATA")
        self.assertIsNone(r["observed"])

    def test_preserves_name_and_threshold(self):
        r = _check_constraint("max_vol", 1.5, 2.0, "lte")
        self.assertEqual(r["name"], "max_vol")
        self.assertEqual(r["threshold"], 2.0)
        self.assertEqual(r["observed"], 1.5)

    def test_unknown_comparator_fails(self):
        r = _check_constraint("test", 5.0, 5.0, "eq")
        self.assertEqual(r["status"], "FAIL")

    def test_why_present_on_fail(self):
        r = _check_constraint("max_vol", 3.0, 2.0, "lte")
        self.assertEqual(r["status"], "FAIL")
        self.assertIn("why", r)
        self.assertIn("3.0", r["why"])
        self.assertIn("2.0", r["why"])

    def test_why_present_on_insufficient(self):
        r = _check_constraint("max_vol", None, 2.0, "lte")
        self.assertIn("why", r)
        self.assertIn("No data", r["why"])

    def test_why_absent_on_pass(self):
        r = _check_constraint("max_vol", 1.0, 2.0, "lte")
        self.assertEqual(r["status"], "PASS")
        self.assertNotIn("why", r)

    def test_drawdown_negative_proof_pass(self):
        """Explicit proof: threshold -30, observed -20 → PASS (less severe)."""
        r = _check_constraint("max_dd", -20.0, -30.0, "gte")
        self.assertEqual(r["status"], "PASS")

    def test_drawdown_negative_proof_fail(self):
        """Explicit proof: threshold -30, observed -40 → FAIL (more severe)."""
        r = _check_constraint("max_dd", -40.0, -30.0, "gte")
        self.assertEqual(r["status"], "FAIL")

    def test_drawdown_negative_proof_exact_boundary(self):
        """Explicit proof: threshold -30, observed -30 → PASS (exactly at limit)."""
        r = _check_constraint("max_dd", -30.0, -30.0, "gte")
        self.assertEqual(r["status"], "PASS")


# ── evaluate_constraints ──────────────────────────────────────

class TestEvaluateConstraints(unittest.TestCase):
    """Test constraint evaluation with various metric/constraint combos."""

    def test_all_pass_realistic(self):
        m = _extract_metrics(_realistic_insights())
        checks = evaluate_constraints(m, DEFAULT_CONSTRAINTS)
        # Count results
        statuses = {c["name"]: c["status"] for c in checks}
        self.assertEqual(statuses["max_drawdown"], "PASS")  # -22.5 >= -30
        self.assertEqual(statuses["max_recovery_days"], "PASS")  # 450 <= 730
        self.assertEqual(statuses["min_median_rolling_cagr"], "PASS")  # 8.5 >= 5
        self.assertEqual(statuses["max_volatility"], "PASS")  # 1.2 <= 2
        self.assertEqual(statuses["max_pairwise_correlation"], "PASS")  # 0.65 <= 0.85
        self.assertEqual(statuses["max_concentration_hhi"], "PASS")  # 0.42 <= 0.50
        self.assertEqual(statuses["fund_drawdown_1"], "PASS")  # -25 >= -40
        self.assertEqual(statuses["fund_drawdown_2"], "PASS")  # -18 >= -40

    def test_drawdown_fail(self):
        m = _extract_metrics([
            _make_insight({"drawdown": {"max_drawdown_pct": -35.0}}),
        ])
        checks = evaluate_constraints(m, {"max_drawdown_pct": -30.0})
        self.assertEqual(checks[0]["status"], "FAIL")

    def test_drawdown_pass_negative_comparison(self):
        """Drawdown -20 should pass threshold -30 (less severe)."""
        m = _extract_metrics([
            _make_insight({"drawdown": {"max_drawdown_pct": -20.0}}),
        ])
        checks = evaluate_constraints(m, {"max_drawdown_pct": -30.0})
        self.assertEqual(checks[0]["status"], "PASS")

    def test_recovery_days_fail(self):
        m = _extract_metrics([
            _make_insight({"drawdown": {"max_recovery_days": 800}}),
        ])
        checks = evaluate_constraints(m, {"max_recovery_days": 730})
        self.assertEqual(checks[0]["status"], "FAIL")

    def test_cagr_below_threshold(self):
        m = _extract_metrics([
            _make_insight({"distribution": {"median_cagr_pct": 3.0, "window_count": 10}}),
        ])
        checks = evaluate_constraints(m, {"min_median_rolling_cagr_pct": 5.0})
        self.assertEqual(checks[0]["status"], "FAIL")

    def test_volatility_high(self):
        m = _extract_metrics([
            _make_insight({"volatility": {"periodic_return_std_pct": 3.5}}),
        ])
        checks = evaluate_constraints(m, {"max_volatility_pct": 2.0})
        self.assertEqual(checks[0]["status"], "FAIL")

    def test_correlation_high(self):
        m = _extract_metrics([_make_insight({
            "correlation_pairs": [
                {"fund_a": "A", "fund_b": "B", "correlation": 0.92},
            ],
        })])
        checks = evaluate_constraints(m, {"max_correlation": 0.85})
        self.assertEqual(checks[0]["status"], "FAIL")

    def test_concentration_high(self):
        m = _extract_metrics([_make_insight({
            "concentration": {"hhi": 0.65},
        })])
        checks = evaluate_constraints(m, {"max_concentration_hhi": 0.50})
        self.assertEqual(checks[0]["status"], "FAIL")

    def test_per_fund_drawdown_one_fails(self):
        m = _extract_metrics([_make_insight({
            "contribution": {
                "portfolio_return_pct": 5.0,
                "funds": [
                    {"scheme_code": 1, "name": "A", "fund_max_drawdown_pct": -45.0},
                    {"scheme_code": 2, "name": "B", "fund_max_drawdown_pct": -20.0},
                ],
            },
        })])
        checks = evaluate_constraints(m, {"max_single_fund_drawdown_pct": -40.0})
        statuses = {c["name"]: c["status"] for c in checks}
        self.assertEqual(statuses["fund_drawdown_1"], "FAIL")  # -45 < -40
        self.assertEqual(statuses["fund_drawdown_2"], "PASS")  # -20 >= -40

    def test_empty_constraints_no_checks(self):
        m = _extract_metrics(_realistic_insights())
        checks = evaluate_constraints(m, {})
        self.assertEqual(len(checks), 0)

    def test_missing_metric_gives_insufficient(self):
        """No drawdown data → INSUFFICIENT_DATA."""
        m = _extract_metrics([
            _make_insight({"volatility": {"periodic_return_std_pct": 1.0}}),
        ])
        checks = evaluate_constraints(m, {"max_drawdown_pct": -30.0})
        self.assertEqual(checks[0]["status"], "INSUFFICIENT_DATA")


# ── detect_flags ──────────────────────────────────────────────

class TestDetectFlags(unittest.TestCase):
    """Test structural red flag detection."""

    def test_no_flags_on_healthy_data(self):
        m = _extract_metrics(_realistic_insights())
        flags = detect_flags(m)
        self.assertEqual(len(flags), 0)

    def test_high_correlation_flag(self):
        m = _extract_metrics([_make_insight({
            "correlation_pairs": [
                {"fund_a": "A", "fund_b": "B", "correlation": 0.95},
            ],
        })])
        flags = detect_flags(m)
        flag_names = [f["flag"] for f in flags]
        self.assertIn("high_correlation", flag_names)

    def test_negative_high_correlation_flag(self):
        """Negative correlation with abs >= 0.90 should trigger flag."""
        m = _extract_metrics([_make_insight({
            "correlation_pairs": [
                {"fund_a": "A", "fund_b": "B", "correlation": -0.92},
            ],
        })])
        flags = detect_flags(m)
        flag_names = [f["flag"] for f in flags]
        self.assertIn("high_correlation", flag_names)

    def test_dominant_contributor_flag(self):
        m = _extract_metrics([_make_insight({
            "contribution": {
                "portfolio_return_pct": 10.0,
                "funds": [
                    {"scheme_code": 1, "name": "BigFund", "weight": 0.5,
                     "fund_return_pct": 15.0, "weighted_return_pct": 7.5,
                     "fund_max_drawdown_pct": -10.0},
                    {"scheme_code": 2, "name": "TinyFund", "weight": 0.5,
                     "fund_return_pct": 5.0, "weighted_return_pct": 2.5,
                     "fund_max_drawdown_pct": -5.0},
                ],
            },
        })])
        flags = detect_flags(m)
        flag_names = [f["flag"] for f in flags]
        self.assertIn("dominant_contributor", flag_names)

    def test_no_dominant_when_balanced(self):
        m = _extract_metrics([_make_insight({
            "contribution": {
                "portfolio_return_pct": 10.0,
                "funds": [
                    {"scheme_code": 1, "name": "A", "weighted_return_pct": 5.1},
                    {"scheme_code": 2, "name": "B", "weighted_return_pct": 4.9},
                ],
            },
        })])
        flags = detect_flags(m)
        flag_names = [f["flag"] for f in flags]
        self.assertNotIn("dominant_contributor", flag_names)

    def test_high_concentration_flag(self):
        m = _extract_metrics([_make_insight({
            "concentration": {"hhi": 0.65},
        })])
        flags = detect_flags(m)
        flag_names = [f["flag"] for f in flags]
        self.assertIn("high_concentration", flag_names)

    def test_no_concentration_flag_below_threshold(self):
        m = _extract_metrics([_make_insight({
            "concentration": {"hhi": 0.55},
        })])
        flags = detect_flags(m)
        flag_names = [f["flag"] for f in flags]
        self.assertNotIn("high_concentration", flag_names)

    def test_no_rolling_data_flag(self):
        m = _extract_metrics([_make_insight({
            "distribution": {"window_count": 0, "median_cagr_pct": None},
        })])
        flags = detect_flags(m)
        flag_names = [f["flag"] for f in flags]
        self.assertIn("no_rolling_data", flag_names)

    def test_no_rolling_flag_with_positive_count(self):
        m = _extract_metrics([_make_insight({
            "distribution": {"window_count": 10, "median_cagr_pct": 8.0},
        })])
        flags = detect_flags(m)
        flag_names = [f["flag"] for f in flags]
        self.assertNotIn("no_rolling_data", flag_names)

    def test_multiple_flags_simultaneously(self):
        m = _extract_metrics([
            _make_insight({
                "correlation_pairs": [
                    {"fund_a": "A", "fund_b": "B", "correlation": 0.98},
                ],
            }),
            _make_insight({"concentration": {"hhi": 0.70}}),
            _make_insight({"distribution": {"window_count": 0}}),
        ])
        flags = detect_flags(m)
        flag_names = {f["flag"] for f in flags}
        self.assertIn("high_correlation", flag_names)
        self.assertIn("high_concentration", flag_names)
        self.assertIn("no_rolling_data", flag_names)

    def test_zero_portfolio_return_no_dominant(self):
        """Zero return should not trigger dominant_contributor."""
        m = _extract_metrics([_make_insight({
            "contribution": {
                "portfolio_return_pct": 0.0,
                "funds": [
                    {"scheme_code": 1, "name": "A", "weighted_return_pct": 0.005},
                    {"scheme_code": 2, "name": "B", "weighted_return_pct": -0.005},
                ],
            },
        })])
        flags = detect_flags(m)
        flag_names = [f["flag"] for f in flags]
        self.assertNotIn("dominant_contributor", flag_names)

    def test_tiny_portfolio_return_no_dominant(self):
        """Return below 1% should suppress dominant contributor flag (noise)."""
        m = _extract_metrics([_make_insight({
            "contribution": {
                "portfolio_return_pct": 0.5,
                "funds": [
                    {"scheme_code": 1, "name": "A", "weighted_return_pct": 0.45},
                    {"scheme_code": 2, "name": "B", "weighted_return_pct": 0.05},
                ],
            },
        })])
        flags = detect_flags(m)
        flag_names = [f["flag"] for f in flags]
        self.assertNotIn("dominant_contributor", flag_names)

    def test_correlation_flag_suppressed_with_few_observations(self):
        """High correlation should NOT be flagged when observation_count < 30."""
        m = {
            "correlation_pairs": [
                {"fund_a": "A", "fund_b": "B", "correlation": 0.98},
            ],
            "observation_count": 10,
        }
        flags = detect_flags(m)
        flag_names = [f["flag"] for f in flags]
        self.assertNotIn("high_correlation", flag_names)

    def test_correlation_flag_fires_with_enough_observations(self):
        """High correlation should flag when observation_count >= 30."""
        m = {
            "correlation_pairs": [
                {"fund_a": "A", "fund_b": "B", "correlation": 0.98},
            ],
            "observation_count": 100,
        }
        flags = detect_flags(m)
        flag_names = [f["flag"] for f in flags]
        self.assertIn("high_correlation", flag_names)

    def test_correlation_flag_fires_when_obs_count_unknown(self):
        """When obs count is missing (None), flag conservatively."""
        m = {
            "correlation_pairs": [
                {"fund_a": "A", "fund_b": "B", "correlation": 0.95},
            ],
        }
        flags = detect_flags(m)
        flag_names = [f["flag"] for f in flags]
        self.assertIn("high_correlation", flag_names)

    def test_no_rolling_flag_absent_without_distribution(self):
        """No distribution data at all → should NOT fire no_rolling_data."""
        m = _extract_metrics([
            _make_insight({"volatility": {"periodic_return_std_pct": 1.0}}),
        ])
        flags = detect_flags(m)
        flag_names = [f["flag"] for f in flags]
        self.assertNotIn("no_rolling_data", flag_names)


# ── evaluate_portfolio (end-to-end) ───────────────────────────

class TestEvaluatePortfolio(unittest.TestCase):
    """End-to-end tests for the main evaluation function."""

    def test_realistic_all_pass(self):
        report = evaluate_portfolio(_realistic_insights())
        s = report["summary"]
        self.assertEqual(s["verdict"], "ALL_PASS")
        self.assertEqual(s["failed"], 0)
        self.assertEqual(s["flag_count"], 0)
        self.assertGreater(s["total_checks"], 0)

    def test_custom_strict_constraints_fail(self):
        report = evaluate_portfolio(
            _realistic_insights(),
            {"max_drawdown_pct": -10.0},  # realistic drawdown is -22.5
        )
        s = report["summary"]
        self.assertEqual(s["verdict"], "FAIL")
        self.assertGreater(s["failed"], 0)

    def test_defaults_are_applied(self):
        report = evaluate_portfolio(_realistic_insights())
        for key in DEFAULT_CONSTRAINTS:
            self.assertIn(key, report["constraints_applied"])

    def test_user_overrides_defaults(self):
        report = evaluate_portfolio(
            _realistic_insights(),
            {"max_drawdown_pct": -50.0},
        )
        self.assertEqual(report["constraints_applied"]["max_drawdown_pct"], -50.0)
        # Other defaults preserved
        self.assertEqual(
            report["constraints_applied"]["max_volatility_pct"],
            DEFAULT_CONSTRAINTS["max_volatility_pct"],
        )

    def test_empty_insights_all_insufficient(self):
        report = evaluate_portfolio([])
        s = report["summary"]
        self.assertEqual(s["verdict"], "INCOMPLETE")
        self.assertEqual(s["passed"], 0)
        self.assertEqual(s["failed"], 0)
        self.assertGreater(s["insufficient_data"], 0)

    def test_no_constraints_override(self):
        """Passing None uses all defaults."""
        report = evaluate_portfolio(_realistic_insights(), None)
        self.assertEqual(report["constraints_applied"], DEFAULT_CONSTRAINTS)

    def test_extracted_metrics_included(self):
        report = evaluate_portfolio(_realistic_insights())
        self.assertIn("extracted_metrics", report)
        self.assertIn("max_drawdown_pct", report["extracted_metrics"])

    def test_checks_structure(self):
        report = evaluate_portfolio(_realistic_insights())
        for check in report["checks"]:
            self.assertIn("name", check)
            self.assertIn("status", check)
            self.assertIn("observed", check)
            self.assertIn("threshold", check)
            self.assertIn(check["status"], ("PASS", "FAIL", "INSUFFICIENT_DATA"))

    def test_flags_structure(self):
        # Use data that triggers flags
        insights = [_make_insight({
            "correlation_pairs": [
                {"fund_a": "A", "fund_b": "B", "correlation": 0.95},
            ],
        })]
        report = evaluate_portfolio(insights)
        self.assertGreater(report["summary"]["flag_count"], 0)
        for flag in report["flags"]:
            self.assertIn("flag", flag)
            self.assertIn("detail", flag)
            self.assertIn("value", flag)

    def test_verdict_incomplete_when_only_insufficient(self):
        """If no failures but some INSUFFICIENT_DATA → INCOMPLETE."""
        insights = [_make_insight({"trailing_cagr": 9.0})]
        report = evaluate_portfolio(insights)
        s = report["summary"]
        self.assertEqual(s["verdict"], "INCOMPLETE")
        self.assertEqual(s["failed"], 0)

    def test_verdict_fail_trumps_incomplete(self):
        """If any FAIL → verdict is FAIL even with INSUFFICIENT_DATA too."""
        insights = [
            _make_insight({"drawdown": {"max_drawdown_pct": -50.0}}),
        ]
        report = evaluate_portfolio(insights, {"max_drawdown_pct": -30.0})
        s = report["summary"]
        self.assertEqual(s["verdict"], "FAIL")
        # Must also have insufficient (no volatility/correlation/etc data)
        self.assertGreater(s["insufficient_data"], 0)
        # But verdict stays FAIL, not INCOMPLETE
        self.assertNotEqual(s["verdict"], "INCOMPLETE")

    def test_why_in_failed_checks(self):
        """Failed checks should include a 'why' explanation."""
        report = evaluate_portfolio(
            _realistic_insights(),
            {"max_drawdown_pct": -10.0},
        )
        failed = [c for c in report["checks"] if c["status"] == "FAIL"]
        self.assertGreater(len(failed), 0)
        for c in failed:
            self.assertIn("why", c)
            self.assertTrue(len(c["why"]) > 0)

    def test_why_in_insufficient_checks(self):
        """INSUFFICIENT_DATA checks should include a 'why' explanation."""
        report = evaluate_portfolio([])
        insufficient = [c for c in report["checks"] if c["status"] == "INSUFFICIENT_DATA"]
        self.assertGreater(len(insufficient), 0)
        for c in insufficient:
            self.assertIn("why", c)

    def test_why_absent_in_passing_checks(self):
        """Passing checks should NOT include a 'why' field."""
        report = evaluate_portfolio(_realistic_insights())
        passing = [c for c in report["checks"] if c["status"] == "PASS"]
        self.assertGreater(len(passing), 0)
        for c in passing:
            self.assertNotIn("why", c)


# ── Integration with real portfolio builder ───────────────────

class TestEvaluationWithRealPortfolio(unittest.TestCase):
    """Test evaluation engine with output from build_portfolio_insights."""

    @classmethod
    def setUpClass(cls):
        from backend.investment_analytics.portfolio import build_portfolio_insights
        from backend.investment_analytics.comparison import (
            align_multiple_series,
            alignment_quality,
        )
        from backend.investment_analytics.mutual_funds import SeriesPoint

        start = date(2019, 1, 1)
        n = 600
        s1 = [SeriesPoint(as_of=start + timedelta(days=i),
                           value=100 * (1.10 ** (i / 365)))
              for i in range(n)]
        s2 = [SeriesPoint(as_of=start + timedelta(days=i),
                           value=100 * (1.08 ** (i / 365)))
              for i in range(n)]
        dates, aligned = align_multiple_series({1: s1, 2: s2})
        q = alignment_quality(dates, {1: s1, 2: s2})
        cls.insights = build_portfolio_insights(
            dates, aligned, {1: "FundA", 2: "FundB"},
            {1: 0.6, 2: 0.4}, q,
        )

    def test_real_insights_evaluate(self):
        report = evaluate_portfolio(self.insights)
        s = report["summary"]
        self.assertIn(s["verdict"], ("ALL_PASS", "FAIL", "INCOMPLETE"))
        self.assertGreater(s["total_checks"], 0)

    def test_real_metrics_extracted(self):
        report = evaluate_portfolio(self.insights)
        m = report["extracted_metrics"]
        # Should have at least drawdown, volatility, correlation, concentration
        self.assertIn("max_drawdown_pct", m)
        self.assertIn("volatility_pct", m)
        self.assertIn("hhi", m)

    def test_strict_drawdown_fails(self):
        """Synthetic data has 0 drawdown, so -0.01 threshold should work."""
        report = evaluate_portfolio(self.insights, {"min_median_rolling_cagr_pct": 99.0})
        s = report["summary"]
        self.assertEqual(s["verdict"], "FAIL")

    def test_lenient_constraints_all_pass(self):
        report = evaluate_portfolio(self.insights, {
            "max_drawdown_pct": -99.0,
            "max_recovery_days": 9999,
            "min_median_rolling_cagr_pct": 0.01,
            "max_volatility_pct": 99.0,
            "max_correlation": 1.0,
            "max_concentration_hhi": 1.0,
            "max_single_fund_drawdown_pct": -99.0,
        })
        s = report["summary"]
        # Should be ALL_PASS or INCOMPLETE (if recovery days is None)
        self.assertEqual(s["failed"], 0)

    def test_per_fund_drawdown_checks_present(self):
        report = evaluate_portfolio(self.insights)
        check_names = [c["name"] for c in report["checks"]]
        fund_dd_checks = [n for n in check_names if n.startswith("fund_drawdown_")]
        self.assertGreaterEqual(len(fund_dd_checks), 2)


if __name__ == "__main__":
    unittest.main()
