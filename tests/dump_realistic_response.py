"""Dump a realistic /analytics/portfolio-health response to JSON.

Builds a mixed portfolio that exercises EVERY code path the frontend
expects to render, then writes the resulting health dict to disk for
visual inspection. The intent: catch schema-contract drift between
backend and frontend without a browser.

Mix:
- Strong, Neutral, Weak holdings (action_priority + alternatives)
- An ETF (Not Ranked → coverage band drops)
- A Regular plan + its Direct sibling (plan_efficiency_flags +
  structural_priority)
- Two cross-category funds with NAV correlation = 1.0 (high-overlap
  signal — but suppressed by low coverage path D unless coverage
  recovers)
- Mixed weights so the priority headlines have distinct winners

Run:  venv/bin/python -m tests.dump_realistic_response
"""
from __future__ import annotations

import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

from backend.investment_analytics import portfolio_health as ph
from backend.investment_analytics.ranking import (
    CategoryRanking, FundMetrics, RankedFund,
)


def F(code, **kw):
    base = dict(
        scheme_code=code,
        fund_name=f"Synthetic Fund {code} Direct Plan - Growth",
        fund_house=f"AMC {code // 100}",
        excess_return_pct=0.0, max_drawdown_pct=-15.0, consistency_pct=50.0,
        volatility_pct=12.0, downside_capture_ratio=1.0,
        fund_cagr_pct=10.0, benchmark_cagr_pct=8.0,
        aligned_points=1500, history_years=8.0, drawdown_trough_date=None,
    )
    base.update(kw)
    return FundMetrics(**base)


def RF(rank, fund, conf="High"):
    return RankedFund(
        rank=rank, fund=fund, dominance_count=max(0, 5 - rank), total_peers=5,
        confidence_level=conf, strengths=[], weaknesses=[],
    )


class Scheme:
    def __init__(self, code, name, category, house="AMC X"):
        self.scheme_code = code
        self.scheme_name = name
        self.scheme_category = category
        self.fund_house = house


def _gen_nav(seed, days=600):
    rng = random.Random(seed)
    d = date(2021, 1, 1)
    out, nav = [], 100.0
    for _ in range(days):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        ret = 0.0003 + 0.01 * rng.gauss(0, 1)
        nav = max(0.01, nav * (1 + ret))
        out.append({"date": d.isoformat(), "nav": round(nav, 4)})
        d += timedelta(days=1)
    return out


def _scaled_nav(series, factor=1.0):
    out = [{"date": series[0]["date"], "nav": series[0]["nav"]}]
    for i in range(1, len(series)):
        ret = (series[i]["nav"] / series[i - 1]["nav"]) - 1.0
        out.append({"date": series[i]["date"],
                    "nav": max(0.01, out[-1]["nav"] * (1 + factor * ret))})
    return out


def main() -> int:
    # ── Categories + rankings ────────────────────────────────────
    lc = "Equity Scheme - Large Cap Fund"
    fc = "Equity Scheme - Flexi Cap Fund"
    debt = "Debt Scheme - Short Duration Fund"

    # Large Cap: 5 funds, held = 1101 (rank 5, Weak)
    lc_funds = [
        F(1101, excess_return_pct=4.5, consistency_pct=72, max_drawdown_pct=-11, volatility_pct=10),  # rank 1
        F(1102, excess_return_pct=3.5, consistency_pct=66, max_drawdown_pct=-13, volatility_pct=11),
        F(1103, excess_return_pct=2.5, consistency_pct=58, max_drawdown_pct=-15, volatility_pct=12),
        F(1104, excess_return_pct=1.5, consistency_pct=52, max_drawdown_pct=-17, volatility_pct=13),
        F(1105, excess_return_pct=-2.0, consistency_pct=20, max_drawdown_pct=-30, volatility_pct=22),  # held weak
    ]
    fc_funds = [
        F(1201, excess_return_pct=4.0, consistency_pct=70, max_drawdown_pct=-12, volatility_pct=11),
        F(1202, excess_return_pct=3.0, consistency_pct=64, max_drawdown_pct=-14, volatility_pct=12),
        F(1203, excess_return_pct=2.0, consistency_pct=55, max_drawdown_pct=-16, volatility_pct=13),
        F(1204, excess_return_pct=1.5, consistency_pct=50, max_drawdown_pct=-18, volatility_pct=14),
    ]
    debt_funds = [
        F(2001, fund_cagr_pct=7.5, volatility_pct=2.0, max_drawdown_pct=-1.0,
          consistency_pct=82, downside_capture_ratio=3.5),
        F(2002, fund_cagr_pct=7.0, volatility_pct=2.5, max_drawdown_pct=-1.5,
          consistency_pct=78, downside_capture_ratio=2.8),
    ]
    rankings = {
        lc: CategoryRanking(category=lc, benchmark_name="Nifty 100",
                            benchmark_code=999, benchmark_fallback=False,
                            ranked=[RF(i + 1, f) for i, f in enumerate(lc_funds)],
                            excluded=[], computed_at="2026-04-30T00:00:00+00:00",
                            total_funds_in_category=5),
        fc: CategoryRanking(category=fc, benchmark_name="Nifty 500",
                            benchmark_code=998, benchmark_fallback=False,
                            ranked=[RF(i + 1, f) for i, f in enumerate(fc_funds)],
                            excluded=[], computed_at="2026-04-30T00:00:00+00:00",
                            total_funds_in_category=4),
        debt: CategoryRanking(category=debt, benchmark_name="None (absolute)",
                              benchmark_code=0, benchmark_fallback=False,
                              ranked=[RF(i + 1, f) for i, f in enumerate(debt_funds)],
                              excluded=[], computed_at="2026-04-30T00:00:00+00:00",
                              total_funds_in_category=2),
    }

    # ── Registry ────────────────────────────────────────────────
    registry = [
        # Held: weak large-cap Direct
        Scheme(1105, "Synthetic Fund 1105 Direct Plan - Growth", lc, "AMC 11"),
        # Held: flexi-cap Direct (will land Neutral)
        Scheme(1203, "Synthetic Fund 1203 Direct Plan - Growth", fc, "AMC 12"),
        # Held: debt fund Direct
        Scheme(2001, "Synthetic Fund 2001 Direct Plan - Growth", debt, "AMC 20"),
        # Held: Regular plan with Direct sibling in registry
        Scheme(7001, "BlueChip Bond Regular Plan - Growth", debt, "Acme MF"),
        Scheme(7002, "BlueChip Bond Direct Plan - Growth", debt, "Acme MF"),
        # Held: ETF — Not Ranked
        Scheme(9001, "Nifty 50 ETF Direct - Growth",
               "Other Scheme - Index Funds", "Index AMC"),
    ]
    held_codes = [1105, 1203, 2001, 7001, 9001]
    weights = {
        1105: 0.10,  # Weak large-cap, small slice → Review headline severity
        1203: 0.30,  # Flexi-cap Neutral
        2001: 0.10,  # Debt
        7001: 0.30,  # Regular plan — heaviest single source of structural cost
        9001: 0.20,  # ETF → Not Ranked, contributes to not_ranked_pct
    }
    # Coverage: ranked share = 1105+1203+2001+7001 = 80% (7001 is in
    # registry under debt category but its name lacks "Direct" → it
    # falls into "found is None" path → Not Ranked. So actually only
    # 1105+1203+2001 = 50% rank-able. Coverage band will be PARTIAL.

    # ── NAV stubs for the correlation path ──────────────────────
    nav_a = _gen_nav(seed=11)
    nav_b = _scaled_nav(nav_a, factor=1.0)
    nav_by_code: Dict[int, Any] = {1105: nav_a, 1203: nav_b}

    # ── Orchestrate ─────────────────────────────────────────────
    def eq_lookup(category, _path):
        return rankings.get(category) if category in (lc, fc) else None

    def dt_lookup(category, _path):
        return rankings.get(category) if category == debt else None

    def fake_fetch(code):
        if code in nav_by_code:
            return {"data": [{"date": r["date"], "nav": r["nav"]}
                             for r in nav_by_code[code]]}
        raise RuntimeError(f"no NAV for {code}")

    def fake_convert(data):
        return [{"date": d["date"], "nav": d["nav"]} for d in data]

    with patch.object(ph, "load_registry", return_value=registry), \
         patch.object(ph, "_get_or_rank_equity", side_effect=eq_lookup), \
         patch.object(ph, "_get_or_rank_debt", side_effect=dt_lookup), \
         patch.object(ph, "fetch_scheme_nav", side_effect=fake_fetch), \
         patch.object(ph, "_convert_nav_to_records", side_effect=fake_convert):
        result = ph.check_portfolio_health(
            scheme_codes=held_codes,
            weights=weights,
            registry_path="ignored",
        )
    out = ph.portfolio_health_to_dict(result, weights=weights)

    target = Path("data/sample_realistic_response.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {target} ({target.stat().st_size} bytes)")

    # ── Spot-check: every section the frontend reads exists ────
    expected_keys = [
        "computed_at", "schema_version", "total_holdings", "not_found_count",
        "decision_summary", "decision_summary_weight_pct",
        "holdings", "not_found", "concentration", "mistakes",
        "redundancies", "exposure_gaps", "correlations",
        "correlation_threshold", "coverage", "action_priority",
        "structural_priority", "top_ranked_by_category",
        "plan_efficiency_flags", "risk_summary", "portfolio_status",
        "no_major_issues", "data_as_of", "limitations",
    ]
    missing = [k for k in expected_keys if k not in out]
    print(f"missing top-level keys: {missing or 'none'}")
    print(f"action_priority      : {out.get('action_priority')}")
    print(f"structural_priority  : {out.get('structural_priority')}")
    print(f"coverage band        : {(out.get('coverage') or {}).get('confidence_band')}")
    print(f"plan_flags           : {len(out.get('plan_efficiency_flags', []))}")
    print(f"top_ranked rows      : {len(out.get('top_ranked_by_category', []))}")
    print(f"correlations         : {len(out.get('correlations', []))}")
    print(f"portfolio_status     : {out.get('portfolio_status')}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
