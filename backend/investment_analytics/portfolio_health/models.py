"""Public dataclass models for the portfolio_health package.

Extracted to a leaf module so decision.py, serializer.py, and the
package __init__ can all depend on these types without any of them
forming a cycle through __init__.py. Tests and external callers
import these via `portfolio_health.<Name>` thanks to __init__.py's
re-export.

Discipline:
- Models are pure dataclasses. Behavior lives in decision/serializer.
- No model field is computed at construction time; computation
  happens upstream and the value is passed in.
- Adding a field is additive (existing code continues to work).
  Removing or renaming a field is a breaking change requiring a
  PORTFOLIO_HEALTH_SCHEMA_VERSION bump.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.investment_analytics.portfolio_health.coverage import (
    CoverageReport,
)
from backend.investment_analytics.portfolio_health.correlation import (
    HIGH_CORRELATION_THRESHOLD,
)


@dataclass
class FundHealthResult:
    """Health check result for a single holding."""
    scheme_code: int
    fund_name: str
    fund_house: str
    category: str
    asset_class: str  # "equity" or "debt"
    rank: int
    total_in_category: int
    status: str  # Strong / Neutral / Weak / Not Ranked
    confidence_level: str
    history_years: float
    horizon: str  # Short-term / Mid-term / Long-term / Unknown
    action: str  # Continue / Monitor / Review
    action_note: str  # optional factual observation
    strengths: List[str]
    weaknesses: List[str]
    metrics: Dict[str, Any]
    alternatives: List[Dict[str, Any]]  # top 3 from same category (if Weak/Neutral)
    data_quality_flags: List[Dict[str, str]] = field(default_factory=list)
    outlier_flags: List[str] = field(default_factory=list)
    your_fund_gaps: List[str] = field(default_factory=list)  # personal comparison items
    benchmark_name: str = ""  # benchmark used for ranking (equity only)


@dataclass
class ConcentrationWarning:
    """Warns about category/risk concentration."""
    category: str
    count: int
    weight_pct: float
    message: str


@dataclass
class PortfolioHealthResult:
    """Complete portfolio health check result."""
    holdings: List[FundHealthResult]
    not_found: List[Dict[str, Any]]  # scheme codes not in registry
    concentration: List[ConcentrationWarning]
    mistakes: List[Dict[str, Any]]
    redundancies: List[Dict[str, Any]]
    exposure_gaps: List[Dict[str, Any]]
    risk_summary: Dict[str, Any]
    portfolio_status: str  # Well diversified / Moderately concentrated / Highly concentrated
    computed_at: str
    correlations: List[Dict[str, Any]] = field(default_factory=list)
    correlation_threshold: float = HIGH_CORRELATION_THRESHOLD
    coverage: Optional[CoverageReport] = None
    action_priority: Optional[Dict[str, Any]] = None
    structural_priority: Optional[Dict[str, Any]] = None
    top_ranked_by_category: List[Dict[str, Any]] = field(default_factory=list)
    plan_efficiency_flags: List[Dict[str, Any]] = field(default_factory=list)
