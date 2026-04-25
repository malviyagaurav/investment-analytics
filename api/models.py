from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class JurisdictionRequest(BaseModel):
    subject_token: str = "anonymous"
    user_country: str = Field(min_length=2, max_length=64)
    asset_market: str = Field(min_length=2, max_length=64)
    serving_entity: str = Field(min_length=1, max_length=128)


class Holding(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    asset_class: str = Field(min_length=1, max_length=64)
    market_value: float = Field(ge=0)


class PortfolioProfile(BaseModel):
    horizon_years: float = Field(default=0, ge=0, le=100)


class PortfolioAnalyticsRequest(JurisdictionRequest):
    profile: PortfolioProfile = Field(default_factory=PortfolioProfile)
    holdings: list[Holding] = Field(default_factory=list, max_length=500)


class ScenarioDefinition(BaseModel):
    kind: Literal["standard", "user"]
    id: Optional[str] = Field(default=None, max_length=80)
    params: dict[str, Any] = Field(default_factory=dict)


class ScenarioRunRequest(BaseModel):
    subject_token: str = "anonymous"
    portfolio_value: float = Field(ge=0)
    scenario_definition: ScenarioDefinition


class DataSourceModel(BaseModel):
    source: str = Field(min_length=1, max_length=160)
    timestamp: str = Field(min_length=10, max_length=40)
    license: Literal["redistributable", "restricted", "user_supplied"]
    lineage: list[dict[str, Any]] = Field(default_factory=list)


class MutualFundPoint(BaseModel):
    date: str = Field(min_length=10, max_length=10)
    nav: float


class BenchmarkPoint(BaseModel):
    date: str = Field(min_length=10, max_length=10)
    value: float


class ExpenseImpactAssumptions(BaseModel):
    investment_amount: float = Field(default=100000.0, gt=0)
    horizons_years: list[float] = Field(default_factory=lambda: [3.0, 5.0, 10.0], min_length=1, max_length=10)


class MutualFundAnalysisRequest(BaseModel):
    subject_token: str = "anonymous"
    user_country: str = Field(default="IN", min_length=2, max_length=64)
    asset_market: str = Field(default="IN", min_length=2, max_length=64)
    serving_entity: str = Field(default="local_demo", min_length=1, max_length=128)
    fund_name: str = Field(min_length=1, max_length=160)
    benchmark_name: str = Field(min_length=1, max_length=160)
    category: str = Field(default="Unknown", max_length=120)
    expense_ratio_pct: float = Field(default=0.0, ge=0, le=10)
    expense_impact: ExpenseImpactAssumptions = Field(default_factory=ExpenseImpactAssumptions)
    fund_source: DataSourceModel
    benchmark_source: DataSourceModel
    fund: list[MutualFundPoint] = Field(min_length=2, max_length=5000)
    benchmark: list[BenchmarkPoint] = Field(min_length=2, max_length=5000)
    rolling_window_points: int = Field(default=252, ge=2, le=1260)
    rolling_step_points: int = Field(default=5, ge=1, le=252)
    rolling_min_windows: int = Field(default=126, ge=1, le=2000)


class MFPayload(BaseModel):
    """MF analysis payload without jurisdiction fields."""
    fund_name: str = Field(min_length=1, max_length=160)
    benchmark_name: str = Field(min_length=1, max_length=160)
    category: str = Field(default="Unknown", max_length=120)
    expense_ratio_pct: float = Field(default=0.0, ge=0, le=10)
    expense_impact: ExpenseImpactAssumptions = Field(default_factory=ExpenseImpactAssumptions)
    fund_source: DataSourceModel
    benchmark_source: DataSourceModel
    fund: list[MutualFundPoint] = Field(min_length=2, max_length=5000)
    benchmark: list[BenchmarkPoint] = Field(min_length=2, max_length=5000)
    rolling_window_points: int = Field(default=252, ge=2, le=1260)
    rolling_step_points: int = Field(default=5, ge=1, le=252)
    rolling_min_windows: int = Field(default=126, ge=1, le=2000)


class FundEntry(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    market_value: float = Field(gt=0)
    mf_payload: MFPayload


class PortfolioWithFundsRequest(JurisdictionRequest):
    funds: list[FundEntry] = Field(min_length=1, max_length=50)


class FromSourceRequest(BaseModel):
    subject_token: str = "anonymous"
    user_country: str = Field(default="IN", min_length=2, max_length=64)
    asset_market: str = Field(default="IN", min_length=2, max_length=64)
    serving_entity: str = Field(default="local_demo", min_length=1, max_length=128)
    source: str = Field(min_length=1, max_length=64)
    symbol: str = Field(min_length=1, max_length=160)
    fund_name: str = Field(default="Ingested Fund", max_length=160)
    benchmark_name: str = Field(default="Ingested Benchmark", max_length=160)
    category: str = Field(default="Unknown", max_length=120)
    expense_ratio_pct: float = Field(default=0.0, ge=0, le=10)
    rolling_window_points: int = Field(default=4, ge=2, le=1260)
    rolling_step_points: int = Field(default=1, ge=1, le=252)
    rolling_min_windows: int = Field(default=1, ge=1, le=2000)
    schema_mapping: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Name of a known schema mapping (e.g. 'amfi_nav', 'etf_price'). "
                    "If omitted, the CSV must use default column names.",
    )


class PricePoint(BaseModel):
    date: str = Field(min_length=10, max_length=10)
    price: float


class ETFAnalysisRequest(BaseModel):
    subject_token: str = "anonymous"
    user_country: str = Field(default="IN", min_length=2, max_length=64)
    asset_market: str = Field(default="IN", min_length=2, max_length=64)
    serving_entity: str = Field(default="local_demo", min_length=1, max_length=128)
    etf_name: str = Field(min_length=1, max_length=160)
    benchmark_name: str = Field(min_length=1, max_length=160)
    category: str = Field(default="Unknown", max_length=120)
    expense_ratio_pct: float = Field(default=0.0, ge=0, le=10)
    expense_impact: ExpenseImpactAssumptions = Field(default_factory=ExpenseImpactAssumptions)
    etf_source: DataSourceModel
    benchmark_source: DataSourceModel
    price_series: list[PricePoint] = Field(min_length=2, max_length=5000)
    benchmark_series: list[BenchmarkPoint] = Field(min_length=2, max_length=5000)
    rolling_window_points: int = Field(default=252, ge=2, le=1260)
    rolling_step_points: int = Field(default=5, ge=1, le=252)
    rolling_min_windows: int = Field(default=126, ge=1, le=2000)


class DiscoverFetchRequest(BaseModel):
    subject_token: str = "anonymous"
    user_country: str = Field(default="IN", min_length=2, max_length=64)
    asset_market: str = Field(default="IN", min_length=2, max_length=64)
    serving_entity: str = Field(default="local_demo", min_length=1, max_length=128)
    scheme_code: int = Field(gt=0)
    fund_name: str = Field(default="AMFI Fund", max_length=200)
    category: str = Field(default="Unknown", max_length=120)
    expense_ratio_pct: float = Field(default=0.0, ge=0, le=10)
    rolling_window_points: int = Field(default=4, ge=2, le=1260)
    rolling_step_points: int = Field(default=1, ge=1, le=252)
    rolling_min_windows: int = Field(default=1, ge=1, le=2000)


class FundCompareEntry(BaseModel):
    scheme_code: int = Field(gt=0)
    fund_name: str = Field(default="", max_length=200)
    category: str = Field(default="Unknown", max_length=120)
    expense_ratio_pct: float = Field(default=0.0, ge=0, le=10)


class CompareRequest(BaseModel):
    subject_token: str = "anonymous"
    user_country: str = Field(default="IN", min_length=2, max_length=64)
    asset_market: str = Field(default="IN", min_length=2, max_length=64)
    serving_entity: str = Field(default="local_demo", min_length=1, max_length=128)
    funds: list[FundCompareEntry] = Field(min_length=2, max_length=5)
    rolling_window_points: int = Field(default=60, ge=2, le=1260)
    rolling_step_points: int = Field(default=5, ge=1, le=252)


class SipFundEntry(BaseModel):
    scheme_code: int = Field(gt=0)
    fund_name: str = Field(default="", max_length=200)
    category: str = Field(default="Unknown", max_length=120)
    expense_ratio_pct: float = Field(default=0.0, ge=0, le=10)
    weight: float = Field(ge=0.0, le=1.0)


class SipRequest(BaseModel):
    subject_token: str = "anonymous"
    user_country: str = Field(default="IN", min_length=2, max_length=64)
    asset_market: str = Field(default="IN", min_length=2, max_length=64)
    serving_entity: str = Field(default="local_demo", min_length=1, max_length=128)
    funds: list[SipFundEntry] = Field(min_length=1, max_length=5)
    monthly_amount: float = Field(gt=0, le=10_000_000)
    rolling_window_months: int = Field(default=36, ge=6, le=240)
    step_months: int = Field(default=1, ge=1, le=12)


class PortfolioAggregateRequest(BaseModel):
    subject_token: str = "anonymous"
    user_country: str = Field(default="IN", min_length=2, max_length=64)
    asset_market: str = Field(default="IN", min_length=2, max_length=64)
    serving_entity: str = Field(default="local_demo", min_length=1, max_length=128)
    funds: list[SipFundEntry] = Field(min_length=2, max_length=5)
    rolling_window_points: int = Field(default=252, ge=20, le=2520)
    rolling_step_points: int = Field(default=21, ge=1, le=252)


class EvaluationConstraints(BaseModel):
    max_drawdown_pct: Optional[float] = Field(default=None, le=0)
    max_recovery_days: Optional[int] = Field(default=None, ge=1)
    min_median_rolling_cagr_pct: Optional[float] = Field(default=None)
    max_volatility_pct: Optional[float] = Field(default=None, ge=0)
    max_correlation: Optional[float] = Field(default=None, ge=-1, le=1)
    max_concentration_hhi: Optional[float] = Field(default=None, ge=0, le=1)
    max_single_fund_drawdown_pct: Optional[float] = Field(default=None, le=0)


class PortfolioEvaluateRequest(BaseModel):
    subject_token: str = "anonymous"
    user_country: str = Field(default="IN", min_length=2, max_length=64)
    asset_market: str = Field(default="IN", min_length=2, max_length=64)
    serving_entity: str = Field(default="local_demo", min_length=1, max_length=128)
    funds: list[SipFundEntry] = Field(min_length=2, max_length=5)
    rolling_window_points: int = Field(default=252, ge=20, le=2520)
    rolling_step_points: int = Field(default=21, ge=1, le=252)
    constraints: EvaluationConstraints = Field(default_factory=EvaluationConstraints)
