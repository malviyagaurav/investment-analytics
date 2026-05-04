"""Decision-quality validation harness — runs synthetic portfolios through
the live `check_portfolio_health` + `portfolio_health_to_dict` pipeline and
asserts properties that DON'T show up in code-only unit tests.

These are the "looks correct but is wrong" failures that matter for a
capital-allocation tool: ordering, magnitude believability, gate
behavior, edge-case framing.

Run with:  venv/bin/python -m tests.validate_decision_quality
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

from backend.investment_analytics import portfolio_health as ph
from backend.investment_analytics.ranking import (
    CategoryRanking, FundMetrics, RankedFund,
)


# ─────────────────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────────────────

def F(code: int, **kw) -> FundMetrics:
    base = dict(
        scheme_code=code,
        fund_name=f"Scheme {code} Direct Plan - Growth",
        fund_house=f"AMC {code // 100}",
        excess_return_pct=0.0,
        max_drawdown_pct=-15.0,
        consistency_pct=50.0,
        volatility_pct=12.0,
        downside_capture_ratio=1.0,
        fund_cagr_pct=10.0,
        benchmark_cagr_pct=8.0,
        aligned_points=1500,
        history_years=8.0,
        drawdown_trough_date=None,
    )
    base.update(kw)
    return FundMetrics(**base)


def RF(rank: int, fund: FundMetrics, conf: str = "High") -> RankedFund:
    return RankedFund(
        rank=rank, fund=fund,
        dominance_count=max(0, 5 - rank), total_peers=5,
        confidence_level=conf, strengths=[], weaknesses=[],
    )


class Scheme:
    def __init__(self, code: int, name: str, category: str, house: str = "AMC X"):
        self.scheme_code = code
        self.scheme_name = name
        self.scheme_category = category
        self.fund_house = house


def category_ranking(funds_in_order: List[FundMetrics],
                     category: str, benchmark: str = "Nifty 100") -> CategoryRanking:
    ranked = [RF(i + 1, f, conf=("Low" if f.history_years < 5 else "High"))
              for i, f in enumerate(funds_in_order)]
    return CategoryRanking(
        category=category,
        benchmark_name=benchmark,
        benchmark_code=999,
        benchmark_fallback=False,
        ranked=ranked,
        excluded=[],
        computed_at="2026-04-30T00:00:00+00:00",
        total_funds_in_category=len(funds_in_order),
    )


# ─────────────────────────────────────────────────────────────────────────
# Reporter
# ─────────────────────────────────────────────────────────────────────────

REPORT: List[Dict[str, Any]] = []


def check(case: str, condition: bool, detail: str = "") -> bool:
    REPORT.append({"case": case, "pass": bool(condition), "detail": detail})
    return bool(condition)


def banner(title: str) -> None:
    REPORT.append({"banner": title})


# ─────────────────────────────────────────────────────────────────────────
# Scenarios
# ─────────────────────────────────────────────────────────────────────────

def run_with_rankings(scheme_codes: List[int],
                      registry: List[Scheme],
                      eq_rankings: Dict[str, CategoryRanking],
                      dt_rankings: Dict[str, CategoryRanking],
                      weights=None,
                      nav_by_code=None) -> Dict[str, Any]:
    def eq_lookup(category: str, _registry_path: str):
        return eq_rankings.get(category)

    def dt_lookup(category: str, _registry_path: str):
        return dt_rankings.get(category)

    # Always mock fetch_scheme_nav so the harness never hits the network.
    # If `nav_by_code` is provided, return synthetic NAV for those codes;
    # otherwise raise so the correlation wrapper records a fetch failure
    # and returns an empty pair list (the legitimate "no NAV" path).
    nav_by_code = nav_by_code or {}

    def fake_fetch(code: int):
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
            scheme_codes=scheme_codes,
            weights=weights,
            registry_path="ignored",
        )
    return ph.portfolio_health_to_dict(result, weights=weights)


def scenario_single_fund() -> None:
    banner("SCENARIO 1: Single fund portfolio (Weak)")
    cat = "Equity Scheme - Large Cap Fund"
    funds = [
        F(101, excess_return_pct=4.0, consistency_pct=70, max_drawdown_pct=-12, volatility_pct=10),
        F(102, excess_return_pct=3.0, consistency_pct=65, max_drawdown_pct=-13, volatility_pct=11),
        F(103, excess_return_pct=2.0, consistency_pct=58, max_drawdown_pct=-14, volatility_pct=12),
        F(104, excess_return_pct=1.0, consistency_pct=55, max_drawdown_pct=-16, volatility_pct=13),
        F(105, excess_return_pct=-2.0, consistency_pct=20, max_drawdown_pct=-30, volatility_pct=22),
    ]
    out = run_with_rankings(
        scheme_codes=[105],
        registry=[Scheme(105, funds[4].fund_name, cat)],
        eq_rankings={cat: category_ranking(funds, cat)},
        dt_rankings={},
    )
    h = out["holdings"][0]
    check("1.1 status is Weak", h["status"] == "Weak", h["status"])
    check("1.2 action is Review", h["action"] == "Review", h["action"])
    check("1.3 weight_pct is 100.0", h.get("scheme_code") == 105 and
          out["decision_summary"]["Review"][0]["weight_pct"] == 100.0,
          str(out["decision_summary"]["Review"][0].get("weight_pct")))
    check("1.4 portfolio_status flags concentration",
          out["portfolio_status"] == "Highly concentrated",
          out["portfolio_status"])
    check("1.5 alternatives are non-empty (peers materially better)",
          len(h["alternatives"]) > 0, f"alts={len(h['alternatives'])}")
    if h["alternatives"]:
        first_alt = h["alternatives"][0]
        check("1.6 first alt has metrics dict (UI-7 ready)",
              "metrics" in first_alt and isinstance(first_alt["metrics"], dict))
        check("1.7 justification carries magnitude",
              all("magnitude" in j and j["magnitude"] in {"small", "moderate", "large"}
                  for j in first_alt["justification"]))


def scenario_all_weak() -> None:
    banner("SCENARIO 2: All-Weak portfolio (every holding ranks bottom)")
    cats = [
        ("Equity Scheme - Large Cap Fund", "Nifty 100"),
        ("Equity Scheme - Mid Cap Fund", "Nifty Midcap 150"),
        ("Equity Scheme - Small Cap Fund", "Nifty Smallcap 250"),
    ]
    eq_rankings = {}
    registry = []
    held_codes = []
    for i, (cat, bench) in enumerate(cats):
        # 5 funds; held is rank 5 (Weak)
        funds = [
            F(100 * (i + 1) + j, excess_return_pct=4.0 - j,
              consistency_pct=70 - j * 8,
              max_drawdown_pct=-12 - j * 2,
              volatility_pct=10 + j)
            for j in range(5)
        ]
        # Make the bottom fund clearly weaker so peers materially dominate.
        funds[4] = F(100 * (i + 1) + 4,
                     excess_return_pct=-3.0,
                     consistency_pct=15,
                     max_drawdown_pct=-30,
                     volatility_pct=22)
        eq_rankings[cat] = category_ranking(funds, cat, bench)
        held_codes.append(funds[4].scheme_code)
        registry.append(Scheme(funds[4].scheme_code, funds[4].fund_name, cat))

    weights = {held_codes[0]: 0.5, held_codes[1]: 0.3, held_codes[2]: 0.2}
    out = run_with_rankings(held_codes, registry, eq_rankings, {}, weights=weights)

    review_items = out["decision_summary"]["Review"]
    check("2.1 all holdings in Review", len(review_items) == 3, str(len(review_items)))
    review_weights = [it["weight_pct"] for it in review_items]
    check("2.2 review entries sorted by weight desc",
          review_weights == sorted(review_weights, reverse=True), str(review_weights))
    check("2.3 review bucket total is 100%",
          out["decision_summary_weight_pct"]["Review"] == 100.0,
          str(out["decision_summary_weight_pct"]["Review"]))
    # Each holding should have alternatives because peers are materially better.
    for h in out["holdings"]:
        check(f"2.4 [{h['scheme_code']}] alts non-empty",
              len(h["alternatives"]) > 0, f"alts={len(h['alternatives'])}")
        # Make sure no Low-conf fund leaks into alternatives.
        check(f"2.5 [{h['scheme_code']}] no Low-conf alt",
              all(a["confidence_level"] != "Low" for a in h["alternatives"]))


def scenario_all_low_confidence() -> None:
    banner("SCENARIO 3: All-Low-confidence portfolio (every fund <5y history)")
    cat = "Equity Scheme - Mid Cap Fund"
    # Every fund: 3-year history → confidence_level becomes "Low"
    funds = [
        F(200 + i, excess_return_pct=3.0 - i, consistency_pct=60 - i * 5,
          max_drawdown_pct=-15 - i, volatility_pct=12 + i, history_years=3.0)
        for i in range(5)
    ]
    eq_rankings = {cat: category_ranking(funds, cat)}
    registry = [Scheme(funds[0].scheme_code, funds[0].fund_name, cat),
                Scheme(funds[1].scheme_code, funds[1].fund_name, cat)]

    out = run_with_rankings(
        scheme_codes=[funds[0].scheme_code, funds[1].scheme_code],
        registry=registry, eq_rankings=eq_rankings, dt_rankings={},
    )
    # Strong should be downgraded to Neutral when conf is Low (existing safety)
    statuses = [h["status"] for h in out["holdings"]]
    check("3.1 no Strong status when all peers Low-conf",
          "Strong" not in statuses, str(statuses))
    # Alternatives should NOT include Low-conf peers — so all alts blocks empty
    for h in out["holdings"]:
        check(f"3.2 [{h['scheme_code']}] alts empty (every peer is Low-conf)",
              len(h["alternatives"]) == 0,
              f"alts={[a['scheme_code'] for a in h['alternatives']]}")
        # OBS-2: action_note must say "data limitations" when every peer
        # was filtered for data reasons (no peer survived to the gate).
        check(f"3.3 [{h['scheme_code']}] action_note cites data limitations",
              "data limitations" in (h.get("action_note") or "").lower(),
              h.get("action_note"))


def scenario_etf() -> None:
    banner("SCENARIO 4: ETF holding (unsupported asset)")
    registry = [Scheme(300, "Some Index ETF Direct - Growth",
                       "Other Scheme - Index Funds")]
    out = run_with_rankings(
        scheme_codes=[300], registry=registry,
        eq_rankings={}, dt_rankings={},
    )
    h = out["holdings"][0]
    check("4.1 ETF status is Not Ranked", h["status"] == "Not Ranked", h["status"])
    check("4.2 ETF action is Monitor", h["action"] == "Monitor", h["action"])
    check("4.3 ETF action_note mentions ETF",
          "etf" in (h["action_note"] or "").lower(), h["action_note"])
    check("4.4 ETF has zero alternatives", len(h["alternatives"]) == 0,
          str(len(h["alternatives"])))


def scenario_marginal_no_alts() -> None:
    banner("SCENARIO 5: Held + 4 marginally-better peers → gate drops all")
    cat = "Equity Scheme - Flexi Cap Fund"
    held = F(401, excess_return_pct=2.0, consistency_pct=50,
             max_drawdown_pct=-20, volatility_pct=15)
    peers = [
        F(402, excess_return_pct=2.5, consistency_pct=52,
          max_drawdown_pct=-19, volatility_pct=14.5),
        F(403, excess_return_pct=2.3, consistency_pct=51,
          max_drawdown_pct=-19.5, volatility_pct=14.7),
        F(404, excess_return_pct=2.1, consistency_pct=50.5,
          max_drawdown_pct=-19.8, volatility_pct=14.9),
        F(405, excess_return_pct=2.05, consistency_pct=50.2,
          max_drawdown_pct=-19.9, volatility_pct=14.95),
    ]
    funds_in_order = peers + [held]  # held is last → status Weak
    eq_rankings = {cat: category_ranking(funds_in_order, cat)}
    out = run_with_rankings(
        scheme_codes=[401],
        registry=[Scheme(401, held.fund_name, cat)],
        eq_rankings=eq_rankings, dt_rankings={},
    )
    h = out["holdings"][0]
    check("5.1 status Weak", h["status"] == "Weak", h["status"])
    check("5.2 alternatives empty (gate drops marginal peers)",
          len(h["alternatives"]) == 0,
          f"alts={[a['scheme_code'] for a in h['alternatives']]}")
    # OBS-2: when peers were considered but none cleared the material gate,
    # the action_note now says "no materially better peer" instead of
    # the previous "data limitations" wording.
    check("5.3 action_note explains no materially better peer",
          "no materially better" in (h["action_note"] or "").lower(),
          h["action_note"])


def scenario_magnitude_calibration() -> None:
    banner("SCENARIO 6: Magnitude labels match metric deltas")
    cat = "Equity Scheme - Large Cap Fund"
    held = F(501, excess_return_pct=1.0, consistency_pct=50,
             max_drawdown_pct=-15, volatility_pct=12)
    # Large-magnitude alt: well above all "large" thresholds
    big_alt = F(502, excess_return_pct=5.0, consistency_pct=80,
                max_drawdown_pct=-5, volatility_pct=8)
    # Moderate-magnitude alt: just above "moderate", below "large"
    mod_alt = F(503, excess_return_pct=2.7, consistency_pct=62,
                max_drawdown_pct=-9, volatility_pct=10)
    # Small-magnitude alt: just above 0 but below "moderate"
    small_alt = F(504, excess_return_pct=1.5, consistency_pct=55,
                  max_drawdown_pct=-13, volatility_pct=11.5)
    funds_in_order = [big_alt, mod_alt, small_alt, held,
                      F(505, excess_return_pct=-2, consistency_pct=20,
                        max_drawdown_pct=-30, volatility_pct=22)]
    eq_rankings = {cat: category_ranking(funds_in_order, cat)}
    out = run_with_rankings(
        scheme_codes=[501],
        registry=[Scheme(501, held.fund_name, cat)],
        eq_rankings=eq_rankings, dt_rankings={},
    )
    h = out["holdings"][0]
    by_code = {a["scheme_code"]: a for a in h["alternatives"]}

    # Big alt: at least one "large" magnitude bullet
    if 502 in by_code:
        big_mags = {j["magnitude"] for j in by_code[502]["justification"]}
        check("6.1 big_alt produces 'large' magnitude bullets",
              "large" in big_mags, str(big_mags))
    # Moderate alt: should have at least one moderate, no "large" required
    if 503 in by_code:
        mod_mags = {j["magnitude"] for j in by_code[503]["justification"]}
        check("6.2 mod_alt produces 'moderate' magnitude bullets",
              "moderate" in mod_mags or "large" in mod_mags, str(mod_mags))
    # Small alt: gate may filter it; if surfaced, should not advertise "large"
    if 504 in by_code:
        sm_mags = {j["magnitude"] for j in by_code[504]["justification"]}
        check("6.3 small_alt does NOT advertise 'large'",
              "large" not in sm_mags, str(sm_mags))
    else:
        # Acceptable: gate dropped it because no moderate+ improvement
        check("6.3 small_alt filtered out by gate (acceptable)", True, "filtered")


def scenario_mixed_categories_and_concentration() -> None:
    banner("SCENARIO 7: Mixed equity+debt portfolio with concentration")
    eq_cat = "Equity Scheme - Large Cap Fund"
    dt_cat = "Debt Scheme - Short Duration Fund"
    eq_funds = [
        F(601 + i, excess_return_pct=4 - i, consistency_pct=70 - i * 8,
          max_drawdown_pct=-12 - i, volatility_pct=10 + i)
        for i in range(5)
    ]
    dt_funds = [
        F(701 + i, fund_cagr_pct=8 - i * 0.3, volatility_pct=2 + i * 0.5,
          max_drawdown_pct=-1 - i * 0.5, consistency_pct=80 - i * 5)
        for i in range(5)
    ]
    eq_rankings = {eq_cat: category_ranking(eq_funds, eq_cat)}
    dt_rankings = {dt_cat: category_ranking(dt_funds, dt_cat)}
    registry = [
        Scheme(eq_funds[0].scheme_code, eq_funds[0].fund_name, eq_cat),
        Scheme(eq_funds[1].scheme_code, eq_funds[1].fund_name, eq_cat),
        Scheme(dt_funds[0].scheme_code, dt_funds[0].fund_name, dt_cat),
    ]
    weights = {eq_funds[0].scheme_code: 0.5,
               eq_funds[1].scheme_code: 0.3,
               dt_funds[0].scheme_code: 0.2}
    out = run_with_rankings(
        scheme_codes=[eq_funds[0].scheme_code, eq_funds[1].scheme_code,
                      dt_funds[0].scheme_code],
        registry=registry, eq_rankings=eq_rankings, dt_rankings=dt_rankings,
        weights=weights,
    )
    rs = out["risk_summary"]
    check("7.1 equity_pct is 80.0", rs["equity_pct"] == 80.0, str(rs["equity_pct"]))
    check("7.2 debt_pct is 20.0", rs["debt_pct"] == 20.0, str(rs["debt_pct"]))
    # Risk-level table in _build_risk_summary: eq_w >= 0.80 → "High".
    # 80% equity at the boundary is correctly "High" (not "Medium-High").
    check("7.3 risk_level High at 80% equity",
          rs["risk_level"] == "High", rs["risk_level"])
    check("7.4 concentration warning for >25% Large Cap",
          any("Large Cap" in c["category_short"] and c["weight_pct"] >= 80
              for c in out["concentration"]),
          str(out["concentration"]))


def scenario_coverage_integrity() -> None:
    """ITEM 1: capital-weighted coverage drops portfolio confidence
    when a meaningful share sits in unanalyzable holdings."""
    banner("SCENARIO 10: Coverage Integrity Layer (item 1)")
    cat = "Equity Scheme - Large Cap Fund"
    funds = [F(1001 + i) for i in range(5)]
    eq_rankings = {cat: category_ranking(funds, cat)}

    # Scenario A: 65% ranked, 35% ETF → "partial"
    registry_a = [
        Scheme(1001, funds[0].fund_name, cat, "AMC X"),
        Scheme(7777, "Some Index ETF Direct - Growth",
               "Other Scheme - Index Funds", "AMC Y"),
    ]
    weights_a = {1001: 0.65, 7777: 0.35}
    out_a = run_with_rankings([1001, 7777], registry_a, eq_rankings, {},
                              weights=weights_a)
    cov_a = out_a.get("coverage")
    check("10.1 partial-coverage band exposed", cov_a is not None)
    if cov_a:
        check("10.2 partial band when 35% in ETF",
              cov_a["confidence_band"] == "partial",
              cov_a["confidence_band"])
        check("10.3 analyzed_pct ≈ 65", abs(cov_a["analyzed_pct"] - 65.0) < 0.1,
              str(cov_a["analyzed_pct"]))
        check("10.4 affected_metrics non-empty",
              len(cov_a["affected_metrics"]) > 0)
        check("10.5 note explains caveat",
              "approximate" in cov_a["note"].lower(),
              cov_a["note"][:80])

    # Scenario B: 100% ranked → "full", no banner needed
    registry_b = [Scheme(1001, funds[0].fund_name, cat, "AMC X")]
    out_b = run_with_rankings([1001], registry_b, eq_rankings, {})
    cov_b = out_b.get("coverage")
    check("10.6 full band when no unranked holdings",
          cov_b and cov_b["confidence_band"] == "full",
          cov_b["confidence_band"] if cov_b else "missing")
    check("10.7 full band emits empty note",
          cov_b and cov_b["note"] == "",
          repr(cov_b["note"]) if cov_b else "missing")

    # Scenario C: 30% ranked, 70% ETF → "low"
    weights_c = {1001: 0.30, 7777: 0.70}
    out_c = run_with_rankings([1001, 7777], registry_a, eq_rankings, {},
                              weights=weights_c)
    cov_c = out_c.get("coverage")
    check("10.8 low band when ranked share <50%",
          cov_c and cov_c["confidence_band"] == "low",
          cov_c["confidence_band"] if cov_c else "missing")
    check("10.9 low band note flags misleading risk",
          cov_c and "misleading" in cov_c["note"].lower(),
          cov_c["note"][:80] if cov_c else "missing")


def scenario_hidden_correlation() -> None:
    """SLICE 2 / P3: two cross-category funds whose returns correlate
    above the threshold MUST be flagged in the response, regardless
    of category labels."""
    banner("SCENARIO 9: Hidden overlap across categories (P3)")
    import random
    from datetime import date, timedelta

    def gen(seed: int, days: int = 600):
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

    def scaled(series, factor=1.0):
        out = [{"date": series[0]["date"], "nav": series[0]["nav"]}]
        for i in range(1, len(series)):
            ret = (series[i]["nav"] / series[i-1]["nav"]) - 1.0
            new_ret = factor * ret
            out.append({"date": series[i]["date"],
                        "nav": max(0.01, out[-1]["nav"] * (1 + new_ret))})
        return out

    # Held in two DIFFERENT categories (Large Cap and Flexi Cap), but
    # NAV returns are identical → correlation should be ~1.0.
    nav_a = gen(seed=901)
    nav_b = scaled(nav_a, factor=1.0)
    nav_by_code = {1101: nav_a, 1102: nav_b}

    cat_lc = "Equity Scheme - Large Cap Fund"
    cat_fc = "Equity Scheme - Flexi Cap Fund"
    eq_rankings = {
        cat_lc: category_ranking([F(1101), F(1103), F(1104), F(1105), F(1106)], cat_lc),
        cat_fc: category_ranking([F(1102), F(1107), F(1108), F(1109), F(1110)], cat_fc),
    }
    registry = [
        Scheme(1101, "Big Bank Large Cap Direct Plan - Growth", cat_lc, "AMC X"),
        Scheme(1102, "Big Bank Flexi Cap Direct Plan - Growth", cat_fc, "AMC X"),
    ]
    out = run_with_rankings(
        scheme_codes=[1101, 1102],
        registry=registry, eq_rankings=eq_rankings, dt_rankings={},
        nav_by_code=nav_by_code,
    )
    pairs = out.get("correlations", [])
    check("9.1 high-overlap pair flagged", len(pairs) >= 1, f"pairs={len(pairs)}")
    if pairs:
        p = pairs[0]
        check("9.2 correlation >= 0.85", p["correlation"] >= 0.85,
              str(p["correlation"]))
        check("9.3 cross_category True", p["cross_category"] is True,
              str(p["cross_category"]))
        check("9.4 fund_a + fund_b carry names",
              "fund_name" in p["fund_a"] and "fund_name" in p["fund_b"])
        check("9.5 message mentions diversification",
              "diversification" in (p.get("message") or "").lower(),
              p.get("message"))
    check("9.6 correlation_threshold exposed",
          "correlation_threshold" in out)
    # When correlation is flagged, no_major_issues must be False.
    check("9.7 no_major_issues becomes False with correlation flag",
          out.get("no_major_issues") is False,
          str(out.get("no_major_issues")))


def scenario_neutral_comparable_peers() -> None:
    """OBS-1: Neutral holding where peers are NOT materially better must
    emit the comparable-to-peers note, not the data-limitation note."""
    banner("SCENARIO 8: Neutral holding + comparable peers (OBS-1 / OBS-2 split)")
    cat = "Equity Scheme - Large Cap Fund"
    # 5 funds, all Direct Plan Growth, all High-confidence (8y history).
    # Held is rank 3 (middle → Neutral). Peers above are only marginally
    # better — gate must filter them all and emit the OBS-1 phrasing.
    f1 = F(801, excess_return_pct=2.6, consistency_pct=53,
           max_drawdown_pct=-14.5, volatility_pct=11.5)
    f2 = F(802, excess_return_pct=2.5, consistency_pct=52,
           max_drawdown_pct=-14.7, volatility_pct=11.7)
    held = F(803, excess_return_pct=2.3, consistency_pct=51,
             max_drawdown_pct=-15.0, volatility_pct=12.0)
    f4 = F(804, excess_return_pct=2.0, consistency_pct=50,
           max_drawdown_pct=-15.5, volatility_pct=12.3)
    f5 = F(805, excess_return_pct=1.8, consistency_pct=49,
           max_drawdown_pct=-15.8, volatility_pct=12.5)
    eq_rankings = {cat: category_ranking([f1, f2, held, f4, f5], cat)}
    out = run_with_rankings(
        scheme_codes=[803],
        registry=[Scheme(803, held.fund_name, cat)],
        eq_rankings=eq_rankings, dt_rankings={},
    )
    h = out["holdings"][0]
    check("8.1 status Neutral", h["status"] == "Neutral", h["status"])
    check("8.2 alts empty (gate filters marginal peers)",
          len(h["alternatives"]) == 0,
          str([a["scheme_code"] for a in h["alternatives"]]))
    # OBS-1: explicit "comparable to peers" framing for Neutral case.
    check("8.3 action_note flags comparable holding (OBS-1)",
          "comparable" in (h.get("action_note") or "").lower(),
          h.get("action_note"))


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    scenario_single_fund()
    scenario_all_weak()
    scenario_all_low_confidence()
    scenario_etf()
    scenario_marginal_no_alts()
    scenario_magnitude_calibration()
    scenario_mixed_categories_and_concentration()
    scenario_neutral_comparable_peers()
    scenario_hidden_correlation()
    scenario_coverage_integrity()

    failed: List[Tuple[str, str]] = []
    total = 0
    print()
    for entry in REPORT:
        if "banner" in entry:
            print(f"\n=== {entry['banner']} ===")
            continue
        total += 1
        mark = "PASS" if entry["pass"] else "FAIL"
        print(f"  [{mark}] {entry['case']}"
              + (f"  ({entry['detail']})" if entry["detail"] else ""))
        if not entry["pass"]:
            failed.append((entry["case"], entry["detail"]))

    print(f"\n\n{total - len(failed)}/{total} checks passed")
    if failed:
        print("FAILURES:")
        for c, d in failed:
            print(f"  - {c}: {d}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
