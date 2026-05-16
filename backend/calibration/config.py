"""Calibration governance + day-one defaults.

Closed enums (REFUSAL_REASONS, CALIBRATION_TARGETS) and engineered
coverage floors. Step 11's day-one engine is deliberately narrow:
one target, one signal substrate, one weighting scheme, one
aggregation rule. Future targets / weighting schemes / aggregation
rules become typed extensions, not configuration sprawl.

## Why refusal_reason is a closed enum

Free-text refusal rationales eventually become inconsistent,
non-queryable, non-replayable, and impossible to aggregate
statistically. Typed refusal preserves replay semantics, reliability
scoring, calibration quality metrics, and future governance
analysis. Refusal is itself a claim — typed claims must come from a
bounded vocabulary.
"""
from __future__ import annotations


CALIBRATION_REPORT_SCHEMA_VERSION = "v1"
CALIBRATION_ENGINE_VERSION = "v1"


# Closed enum. Adding a target requires registering its substrate
# fetcher in backend/calibration/targets.py AND surfacing the
# canonical id here.
CALIBRATION_TARGETS: frozenset = frozenset({
    "portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD",
})


# Closed enum. Adding a reason requires understanding what new
# epistemic refusal category is being introduced.
#
#   insufficient_substrate        — no regime_summary rows in chain
#   insufficient_coverage         — observations below per-regime floor
#   confidence_floor_unmet        — effective weight below floor
#   regime_indeterminate          — requested regime is indeterminate
#   target_unsupported            — target not in CALIBRATION_TARGETS
#   regime_dependency_superseded  — cited regime_summary has been
#                                   superseded by a newer claim
REFUSAL_REASONS: frozenset = frozenset({
    "insufficient_substrate",
    "insufficient_coverage",
    "confidence_floor_unmet",
    "regime_indeterminate",
    "target_unsupported",
    "regime_dependency_superseded",
})


# Day-one engineered coverage floors. These are reasoned defaults,
# not calibrated. They will produce refusals for most regimes until
# substrate accumulates — exactly the intended Phase 2 discipline.
#
#   MIN_OBSERVATIONS_PER_REGIME     — raw sample count floor; below this,
#                                     percentile estimation is statistically
#                                     unstable regardless of weighting.
#   MIN_EFFECTIVE_WEIGHT_PER_REGIME — weighted sample mass floor; a regime
#                                     can have many samples but all at
#                                     near-zero confidence (e.g., all
#                                     near band boundaries) — the weighted
#                                     mass must independently clear a floor.
MIN_OBSERVATIONS_PER_REGIME     = 50
MIN_EFFECTIVE_WEIGHT_PER_REGIME = 25.0


# Coverage-quality floor. A regime whose samples are entirely from
# "low" coverage_quality regime_summary rows is refused — those
# samples are not authoritative enough to derive a recommendation.
#
# Acceptable per-regime mix: at least one sample from a "high" or
# "medium" coverage_quality regime_summary.
ACCEPTED_COVERAGE_QUALITIES: frozenset = frozenset({"high", "medium"})


# Day-one universe for correlation calibration: deep-history funds
# only. A fund qualifies if its NAV cache has at least this many
# observations. The audit (see Step 11 audit) found ~50 deep equity
# funds at this threshold — enough universe for stable per-regime
# correlation distributions.
MIN_FUND_HISTORY_POINTS_FOR_CALIBRATION = 2500


# Default percentile per target. p95 for correlation thresholds
# means "flag pairs at the top 5% of observed cross-asset
# correlations per regime" — operationally salient.
DEFAULT_PERCENTILE_BY_TARGET: dict = {
    "portfolio_health.correlation.HIGH_CORRELATION_THRESHOLD": 95.0,
}


# Categories used as the deep-history equity universe for correlation
# calibration. Restricted to large/flexi/multi-cap where active funds
# typically have ≥10y of history. Excludes small/mid-cap (younger
# segment funds; small-cap proxy index only goes back to 2021) and
# sectoral/thematic (excluded from ranking entirely).
CALIBRATION_EQUITY_UNIVERSE_CATEGORIES: frozenset = frozenset({
    "Equity Scheme - Large Cap Fund",
    "Equity Scheme - Large & Mid Cap Fund",
    "Equity Scheme - Flexi Cap Fund",
    "Equity Scheme - Multi Cap Fund",
    "Equity Scheme - Focused Fund",
    "Equity Scheme - Value Fund",
    "Equity Scheme - ELSS",
})
