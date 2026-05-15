"""Tests for backend.jobs.watchlist.

Exercises the snapshot store + diff logic with mocked rank_*
functions so the test never hits the network.

AUDIT_PATH is redirected to a tempfile in every test that exercises
run_snapshot — the live audit chain must never be touched by tests.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.jobs import watchlist as wl
from backend.investment_analytics.ranking import (
    CategoryRanking, FundMetrics, RankedFund,
)


def _f(code: int, **kw) -> FundMetrics:
    base = dict(
        scheme_code=code,
        fund_name=f"Fund {code} Direct Plan - Growth",
        fund_house=f"AMC {code // 100}",
        excess_return_pct=2.0, max_drawdown_pct=-15.0, consistency_pct=55.0,
        volatility_pct=12.0, downside_capture_ratio=1.0,
        fund_cagr_pct=10.0, benchmark_cagr_pct=8.0,
        aligned_points=1500, history_years=8.0, drawdown_trough_date=None,
    )
    base.update(kw)
    return FundMetrics(**base)


def _ranking(category: str, codes_in_order: list, conf="High") -> CategoryRanking:
    funds = [_f(c, excess_return_pct=4.0 - i) for i, c in enumerate(codes_in_order)]
    ranked = [
        RankedFund(rank=i + 1, fund=f, dominance_count=4 - i, total_peers=5,
                   confidence_level=conf, strengths=[], weaknesses=[])
        for i, f in enumerate(funds)
    ]
    return CategoryRanking(
        category=category, benchmark_name="Nifty 100", benchmark_code=999,
        benchmark_fallback=False, ranked=ranked, excluded=[],
        computed_at="2026-05-06T00:00:00+00:00", total_funds_in_category=len(funds),
    )


class WatchlistConfigTests(unittest.TestCase):

    def test_load_config_defaults_when_file_missing(self) -> None:
        cfg = wl.load_config(Path("/nonexistent/categories.json"))
        self.assertIn("equity_categories", cfg)
        self.assertIn("debt_categories", cfg)
        self.assertGreater(len(cfg["equity_categories"]), 0)


class SnapshotStoreTests(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self._orig_dir = wl.SNAPSHOTS_DIR
        wl.SNAPSHOTS_DIR = Path(self.tmp.name) / "snapshots"

    def tearDown(self) -> None:
        wl.SNAPSHOTS_DIR = self._orig_dir
        self.tmp.cleanup()

    def test_save_and_load_roundtrip(self) -> None:
        cat = "Equity Scheme - Large Cap Fund"
        payload = {"category": cat, "snapshot_date": "2026-05-06",
                   "ranked": [{"scheme_code": 101, "fund_name": "X", "rank": 1,
                                "dominance": {"beats": 4, "of": 5}}]}
        wl.save_snapshot(cat, "2026-05-06", payload)
        roundtrip = wl.load_snapshot(cat, "2026-05-06")
        self.assertEqual(roundtrip, payload)

    def test_list_dates_sorted(self) -> None:
        cat = "Equity Scheme - Mid Cap Fund"
        for d in ("2026-05-04", "2026-05-06", "2026-05-05"):
            wl.save_snapshot(cat, d, {"snapshot_date": d, "ranked": []})
        self.assertEqual(
            wl.list_snapshot_dates(cat),
            ["2026-05-04", "2026-05-05", "2026-05-06"],
        )

    def test_latest_snapshot_returns_most_recent(self) -> None:
        cat = "Equity Scheme - Small Cap Fund"
        wl.save_snapshot(cat, "2026-05-04", {"snapshot_date": "2026-05-04",
                                              "ranked": [{"scheme_code": 1}]})
        wl.save_snapshot(cat, "2026-05-06", {"snapshot_date": "2026-05-06",
                                              "ranked": [{"scheme_code": 2}]})
        latest = wl.latest_snapshot(cat)
        self.assertIsNotNone(latest)
        d, snap = latest
        self.assertEqual(d, "2026-05-06")
        self.assertEqual(snap["ranked"][0]["scheme_code"], 2)

    def test_latest_snapshot_none_when_empty(self) -> None:
        self.assertIsNone(wl.latest_snapshot("nonexistent_category"))


class DiffSnapshotTests(unittest.TestCase):

    def _snap(self, date: str, codes_ranked: list) -> dict:
        return {
            "category": "Equity Scheme - Large Cap Fund",
            "snapshot_date": date,
            "ranked": [
                {"scheme_code": code, "fund_name": f"F{code}",
                 "rank": i + 1,
                 "dominance": {"beats": 4 - i, "of": 5}}
                for i, code in enumerate(codes_ranked)
            ],
        }

    def test_diff_no_changes(self) -> None:
        prev = self._snap("2026-05-05", [101, 102, 103])
        curr = self._snap("2026-05-06", [101, 102, 103])
        d = wl.diff_snapshots(prev, curr)
        self.assertEqual(d["new_funds"], [])
        self.assertEqual(d["dropped_funds"], [])
        self.assertEqual(d["rank_changes"], {})
        self.assertEqual(d["dominance_changes"], {})

    def test_diff_detects_new_and_dropped(self) -> None:
        prev = self._snap("2026-05-05", [101, 102, 103])
        curr = self._snap("2026-05-06", [102, 103, 104])
        d = wl.diff_snapshots(prev, curr)
        self.assertEqual([f["scheme_code"] for f in d["new_funds"]], [104])
        self.assertEqual([f["scheme_code"] for f in d["dropped_funds"]], [101])

    def test_diff_detects_rank_climbs_and_drops(self) -> None:
        prev = self._snap("2026-05-05", [101, 102, 103])
        # Curr: 102 climbed from rank 2 to rank 1; 101 dropped 1 → 2.
        curr = self._snap("2026-05-06", [102, 101, 103])
        d = wl.diff_snapshots(prev, curr)
        # 102: prev_rank=2, curr_rank=1, delta=+1 (improvement)
        self.assertEqual(d["rank_changes"][102]["delta"], 1)
        # 101: prev_rank=1, curr_rank=2, delta=-1 (regression)
        self.assertEqual(d["rank_changes"][101]["delta"], -1)

    def test_diff_dominance_changes(self) -> None:
        prev = self._snap("2026-05-05", [101, 102])
        curr = self._snap("2026-05-06", [101, 102])
        # Tweak dominance counts to simulate a metric shift.
        curr["ranked"][0]["dominance"]["beats"] = 5
        d = wl.diff_snapshots(prev, curr)
        self.assertIn(101, d["dominance_changes"])
        self.assertEqual(d["dominance_changes"][101]["delta"], 1)


class RunSnapshotTests(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self._orig_dir = wl.SNAPSHOTS_DIR
        self._orig_audit = wl.AUDIT_PATH
        self._orig_sidecar = wl.SIDECAR_PATH
        self._orig_holiday = wl.HOLIDAY_CALENDAR_PATH
        wl.SNAPSHOTS_DIR = Path(self.tmp.name) / "snapshots"
        wl.AUDIT_PATH = Path(self.tmp.name) / "audit" / "audit.jsonl"
        wl.SIDECAR_PATH = Path(self.tmp.name) / "watchlist" / "last_run.json"
        wl.HOLIDAY_CALENDAR_PATH = Path(self.tmp.name) / "reference" / "holidays.json"

    def tearDown(self) -> None:
        wl.SNAPSHOTS_DIR = self._orig_dir
        wl.AUDIT_PATH = self._orig_audit
        wl.SIDECAR_PATH = self._orig_sidecar
        wl.HOLIDAY_CALENDAR_PATH = self._orig_holiday
        self.tmp.cleanup()

    def test_run_snapshot_writes_one_file_per_equity_category(self) -> None:
        cfg = {
            "equity_categories": [
                "Equity Scheme - Large Cap Fund",
                "Equity Scheme - Mid Cap Fund",
            ],
            "debt_categories": [],
            "include_gold": False,
            "top_n_to_persist": 5,
        }

        def fake_rank(category, _registry_path):
            return _ranking(category, [101, 102, 103, 104, 105])

        with patch.object(wl, "rank_category", side_effect=fake_rank), \
             patch.object(wl, "_validate_registry", return_value=None):
            summary = wl.run_snapshot(
                config=cfg, registry_path="ignored", snapshot_date="2026-05-06",
            )

        self.assertEqual(summary["Equity Scheme - Large Cap Fund"], "ok")
        self.assertEqual(summary["Equity Scheme - Mid Cap Fund"], "ok")
        for cat in cfg["equity_categories"]:
            snap = wl.load_snapshot(cat, "2026-05-06")
            self.assertIsNotNone(snap)
            self.assertEqual(snap["snapshot_kind"], "equity")
            self.assertEqual(snap["snapshot_date"], "2026-05-06")
            self.assertEqual(len(snap["ranked"]), 5)
            self.assertEqual(snap["showing_top_n"], 5)

    def test_run_snapshot_swallows_per_category_errors(self) -> None:
        cfg = {
            "equity_categories": [
                "Equity Scheme - Large Cap Fund",
                "Equity Scheme - Mid Cap Fund",
            ],
            "debt_categories": [],
            "include_gold": False,
            "top_n_to_persist": 5,
        }
        # Raise for one category, succeed for the other.

        def fake_rank(category, _path):
            if "Mid Cap" in category:
                raise ValueError("benchmark unavailable")
            return _ranking(category, [101, 102, 103])

        with patch.object(wl, "rank_category", side_effect=fake_rank), \
             patch.object(wl, "_validate_registry", return_value=None):
            summary = wl.run_snapshot(
                config=cfg, registry_path="ignored", snapshot_date="2026-05-06",
            )
        self.assertEqual(summary["Equity Scheme - Large Cap Fund"], "ok")
        self.assertIn("error:", summary["Equity Scheme - Mid Cap Fund"])
        # Successful category still got persisted; failing one didn't.
        self.assertIsNotNone(wl.load_snapshot("Equity Scheme - Large Cap Fund", "2026-05-06"))
        self.assertIsNone(wl.load_snapshot("Equity Scheme - Mid Cap Fund", "2026-05-06"))


class RegistryValidationTests(unittest.TestCase):
    """Audit fix: watchlist runner aborts early on missing / corrupt
    / undersized registry rather than producing N per-category errors
    that obscure the real problem."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self._orig_dir = wl.SNAPSHOTS_DIR
        self._orig_audit = wl.AUDIT_PATH
        self._orig_sidecar = wl.SIDECAR_PATH
        self._orig_holiday = wl.HOLIDAY_CALENDAR_PATH
        wl.SNAPSHOTS_DIR = Path(self.tmp.name) / "snapshots"
        wl.AUDIT_PATH = Path(self.tmp.name) / "audit" / "audit.jsonl"
        wl.SIDECAR_PATH = Path(self.tmp.name) / "watchlist" / "last_run.json"
        wl.HOLIDAY_CALENDAR_PATH = Path(self.tmp.name) / "reference" / "holidays.json"

    def tearDown(self) -> None:
        wl.SNAPSHOTS_DIR = self._orig_dir
        wl.AUDIT_PATH = self._orig_audit
        wl.SIDECAR_PATH = self._orig_sidecar
        wl.HOLIDAY_CALENDAR_PATH = self._orig_holiday
        self.tmp.cleanup()

    def test_empty_registry_aborts_with_single_summary_entry(self) -> None:
        # Stub a registry loader that returns 0 schemes.
        with patch.object(wl, "_validate_registry",
                          return_value="registry is empty"):
            summary = wl.run_snapshot(
                config={"equity_categories": ["X"], "debt_categories": [],
                        "include_gold": False, "top_n_to_persist": 5},
                registry_path="ignored",
                snapshot_date="2026-05-06",
            )
        self.assertIn("__registry__", summary)
        self.assertIn("registry is empty", summary["__registry__"])
        # No per-category errors should be produced.
        self.assertEqual(len(summary), 1)

    def test_undersized_registry_aborts(self) -> None:
        with patch.object(wl, "_validate_registry",
                          return_value="registry has only 42 entries"):
            summary = wl.run_snapshot(
                config={"equity_categories": ["X", "Y"], "debt_categories": ["Z"],
                        "include_gold": True, "top_n_to_persist": 5},
                registry_path="ignored",
                snapshot_date="2026-05-06",
            )
        self.assertEqual(len(summary), 1)
        self.assertIn("__registry__", summary)

    def test_valid_registry_passes_through(self) -> None:
        # With _validate_registry returning None, the runner should
        # proceed to rank categories normally.
        from backend.investment_analytics.ranking import (
            CategoryRanking, FundMetrics, RankedFund,
        )
        with patch.object(wl, "_validate_registry", return_value=None), \
             patch.object(wl, "rank_category") as rc:
            # Build a tiny ranking object.
            def fake_rank(category, _path):
                funds = [
                    FundMetrics(
                        scheme_code=100 + i, fund_name=f"F{i}", fund_house="X",
                        excess_return_pct=4.0 - i, max_drawdown_pct=-12.0,
                        consistency_pct=70.0 - i * 5, volatility_pct=10.0,
                        downside_capture_ratio=1.0, fund_cagr_pct=10.0,
                        benchmark_cagr_pct=8.0, aligned_points=1500,
                        history_years=8.0, drawdown_trough_date=None,
                    ) for i in range(3)
                ]
                ranked = [
                    RankedFund(rank=i + 1, fund=f, dominance_count=2 - i,
                               total_peers=3, confidence_level="High",
                               strengths=[], weaknesses=[])
                    for i, f in enumerate(funds)
                ]
                return CategoryRanking(
                    category=category, benchmark_name="X", benchmark_code=1,
                    benchmark_fallback=False, ranked=ranked, excluded=[],
                    computed_at="2026-05-06T00:00:00+00:00",
                    total_funds_in_category=3,
                )
            rc.side_effect = fake_rank
            summary = wl.run_snapshot(
                config={"equity_categories": ["Equity Scheme - Large Cap Fund"],
                        "debt_categories": [], "include_gold": False,
                        "top_n_to_persist": 5},
                registry_path="ignored", snapshot_date="2026-05-06",
            )
        self.assertEqual(summary["Equity Scheme - Large Cap Fund"], "ok")
        self.assertNotIn("__registry__", summary)


class WatchlistAuditEmissionTests(unittest.TestCase):
    """Step 5: every run_snapshot invocation emits exactly one
    watchlist_run audit event with a by-reference evidence file.
    Fires on success AND on registry-abort. The evidence file must
    land on disk BEFORE the audit ref is appended."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self._orig_snap = wl.SNAPSHOTS_DIR
        self._orig_audit = wl.AUDIT_PATH
        self._orig_sidecar = wl.SIDECAR_PATH
        self._orig_holiday = wl.HOLIDAY_CALENDAR_PATH
        wl.SNAPSHOTS_DIR = Path(self.tmp.name) / "snapshots"
        wl.AUDIT_PATH = Path(self.tmp.name) / "audit" / "audit.jsonl"
        wl.SIDECAR_PATH = Path(self.tmp.name) / "watchlist" / "last_run.json"
        wl.HOLIDAY_CALENDAR_PATH = Path(self.tmp.name) / "reference" / "holidays.json"
        self.data_dir = Path(self.tmp.name)
        self.evidence_dir = self.data_dir / "evidence" / "watchlist_run"

    def tearDown(self) -> None:
        wl.SNAPSHOTS_DIR = self._orig_snap
        wl.AUDIT_PATH = self._orig_audit
        wl.SIDECAR_PATH = self._orig_sidecar
        wl.HOLIDAY_CALENDAR_PATH = self._orig_holiday
        self.tmp.cleanup()

    def _read_audit_records(self) -> list:
        """Return the inner ``event`` dicts from each audit-chain line —
        that's where ``event_type`` and all caller-supplied fields live."""
        if not wl.AUDIT_PATH.exists():
            return []
        with wl.AUDIT_PATH.open("r", encoding="utf-8") as handle:
            return [json.loads(line)["event"] for line in handle if line.strip()]

    def _fake_rank(self, codes_in_order=None):
        codes = codes_in_order or [101, 102, 103]

        def _impl(category, _path):
            return _ranking(category, codes)

        return _impl

    def test_successful_run_emits_one_watchlist_run_event(self) -> None:
        cfg = {
            "equity_categories": [
                "Equity Scheme - Large Cap Fund",
                "Equity Scheme - Mid Cap Fund",
            ],
            "debt_categories": [],
            "include_gold": False,
            "top_n_to_persist": 5,
        }
        with patch.object(wl, "rank_category", side_effect=self._fake_rank()), \
             patch.object(wl, "_validate_registry", return_value=None):
            wl.run_snapshot(config=cfg, registry_path="ignored", snapshot_date="2026-05-15")

        records = self._read_audit_records()
        watchlist_events = [r for r in records if r.get("event_type") == "watchlist_run"]
        self.assertEqual(len(watchlist_events), 1,
                         f"expected exactly 1 watchlist_run event, got {len(watchlist_events)}")

    def test_audit_event_carries_lightweight_counts_and_errored_categories(self) -> None:
        cfg = {
            "equity_categories": [
                "Equity Scheme - Large Cap Fund",
                "Equity Scheme - Mid Cap Fund",
                "Equity Scheme - Small Cap Fund",
            ],
            "debt_categories": ["Debt Scheme - Liquid Fund"],
            "include_gold": True,
            "top_n_to_persist": 5,
        }

        def equity(category, _path):
            if "Mid Cap" in category:
                raise ValueError("benchmark unavailable")
            return _ranking(category, [101, 102, 103])

        def debt(category, _path):
            return _ranking(category, [201, 202])

        def gold(_path):
            return _ranking("Gold Fund (FoF)", [301, 302])

        with patch.object(wl, "rank_category", side_effect=equity), \
             patch.object(wl, "rank_debt_category", side_effect=debt), \
             patch.object(wl, "rank_gold_funds", side_effect=gold), \
             patch.object(wl, "debt_ranking_to_dict", side_effect=wl.ranking_to_dict), \
             patch.object(wl, "gold_ranking_to_dict", side_effect=wl.ranking_to_dict), \
             patch.object(wl, "_validate_registry", return_value=None):
            wl.run_snapshot(config=cfg, registry_path="ignored", snapshot_date="2026-05-15")

        records = self._read_audit_records()
        watchlist_events = [r for r in records if r.get("event_type") == "watchlist_run"]
        self.assertEqual(len(watchlist_events), 1)
        ev = watchlist_events[0]
        self.assertEqual(ev["snapshot_date"], "2026-05-15")
        self.assertEqual(ev["equity_count"], 3)
        self.assertEqual(ev["debt_count"], 1)
        self.assertTrue(ev["gold_included"])
        # 2 equity + 1 debt + 1 gold = 4 ok; 1 equity errored.
        self.assertEqual(ev["ok_count"], 4)
        self.assertEqual(ev["error_count"], 1)
        self.assertEqual(ev["errored_categories"], ["Equity Scheme - Mid Cap Fund"])
        self.assertIsNone(ev["registry_problem"])
        self.assertEqual(ev["schema_version"], "v1")
        # By-reference: heavy payload not inlined on the audit row.
        self.assertNotIn("summary", ev)
        self.assertNotIn("payload", ev)

    def test_evidence_file_exists_and_payload_carries_full_summary(self) -> None:
        cfg = {
            "equity_categories": ["Equity Scheme - Large Cap Fund"],
            "debt_categories": [],
            "include_gold": False,
            "top_n_to_persist": 5,
        }
        with patch.object(wl, "rank_category", side_effect=self._fake_rank()), \
             patch.object(wl, "_validate_registry", return_value=None):
            wl.run_snapshot(config=cfg, registry_path="ignored", snapshot_date="2026-05-15")

        records = self._read_audit_records()
        ev = next(r for r in records if r.get("event_type") == "watchlist_run")
        ref = ev["evidence_ref"]
        ev_file = self.data_dir / ref["path"]
        self.assertTrue(ev_file.exists(), f"evidence file missing at {ev_file}")

        envelope = json.loads(ev_file.read_text(encoding="utf-8"))
        # Run ID consistency: audit and evidence file agree.
        self.assertEqual(envelope["run_id"], ev["run_id"])
        self.assertEqual(ev_file.name, f"{ev['run_id']}.json")
        # Evidence_kind on the envelope.
        self.assertEqual(envelope["evidence_kind"], "watchlist_run")
        # Full summary lives in evidence payload, not on audit row.
        payload = envelope["payload"]
        self.assertEqual(payload["snapshot_date"], "2026-05-15")
        self.assertEqual(
            payload["summary"]["Equity Scheme - Large Cap Fund"], "ok"
        )
        self.assertEqual(
            payload["config"]["equity_categories"],
            ["Equity Scheme - Large Cap Fund"],
        )

    def test_evidence_ref_sha256_matches_file_bytes(self) -> None:
        cfg = {
            "equity_categories": ["Equity Scheme - Large Cap Fund"],
            "debt_categories": [],
            "include_gold": False,
            "top_n_to_persist": 5,
        }
        with patch.object(wl, "rank_category", side_effect=self._fake_rank()), \
             patch.object(wl, "_validate_registry", return_value=None):
            wl.run_snapshot(config=cfg, registry_path="ignored", snapshot_date="2026-05-15")

        ev = next(r for r in self._read_audit_records()
                  if r.get("event_type") == "watchlist_run")
        ref = ev["evidence_ref"]
        ev_file = self.data_dir / ref["path"]
        raw = ev_file.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), ref["sha256"])
        self.assertEqual(len(raw), ref["size_bytes"])

    def test_registry_abort_emits_event_with_registry_problem(self) -> None:
        cfg = {
            "equity_categories": ["X", "Y"],
            "debt_categories": ["Z"],
            "include_gold": True,
            "top_n_to_persist": 5,
        }
        with patch.object(wl, "_validate_registry",
                          return_value="registry has only 42 entries"):
            summary = wl.run_snapshot(
                config=cfg, registry_path="ignored", snapshot_date="2026-05-15",
            )

        self.assertIn("__registry__", summary)
        records = self._read_audit_records()
        watchlist_events = [r for r in records if r.get("event_type") == "watchlist_run"]
        self.assertEqual(len(watchlist_events), 1)
        ev = watchlist_events[0]
        self.assertEqual(ev["registry_problem"], "registry has only 42 entries")
        # Counts reflect what was configured, not what attempted.
        self.assertEqual(ev["equity_count"], 2)
        self.assertEqual(ev["debt_count"], 1)
        self.assertTrue(ev["gold_included"])
        # Nothing actually ran, so ok/errored both zero.
        self.assertEqual(ev["ok_count"], 0)
        self.assertEqual(ev["error_count"], 0)
        self.assertEqual(ev["errored_categories"], [])
        # Evidence file still produced with the abort summary.
        ev_file = self.data_dir / ev["evidence_ref"]["path"]
        envelope = json.loads(ev_file.read_text(encoding="utf-8"))
        self.assertIn("__registry__", envelope["payload"]["summary"])

    def test_cache_fingerprint_captured_on_inputs(self) -> None:
        cfg = {
            "equity_categories": ["Equity Scheme - Large Cap Fund"],
            "debt_categories": [],
            "include_gold": False,
            "top_n_to_persist": 5,
        }
        with patch.object(wl, "rank_category", side_effect=self._fake_rank()), \
             patch.object(wl, "_validate_registry", return_value=None):
            wl.run_snapshot(config=cfg, registry_path="ignored", snapshot_date="2026-05-15")

        ev = next(r for r in self._read_audit_records()
                  if r.get("event_type") == "watchlist_run")
        # cache_tracker was started — fingerprint is the sha256 of an
        # empty read-set (deterministic non-null), not None.
        self.assertIsNotNone(ev["inputs"]["cache_fingerprint"])
        self.assertEqual(len(ev["inputs"]["cache_fingerprint"]), 64)
        # And the same fingerprint shows up in the evidence envelope.
        ev_file = self.data_dir / ev["evidence_ref"]["path"]
        envelope = json.loads(ev_file.read_text(encoding="utf-8"))
        self.assertEqual(
            envelope["inputs"]["cache_fingerprint"],
            ev["inputs"]["cache_fingerprint"],
        )

    def test_evidence_file_written_before_audit_append(self) -> None:
        """Ordering invariant: if emit_evidence wrote the audit row
        without writing the evidence file first, a reader would see
        an audit row pointing at a missing file. Verify both exist
        and the path on the audit row resolves."""
        cfg = {
            "equity_categories": ["Equity Scheme - Large Cap Fund"],
            "debt_categories": [],
            "include_gold": False,
            "top_n_to_persist": 5,
        }
        with patch.object(wl, "rank_category", side_effect=self._fake_rank()), \
             patch.object(wl, "_validate_registry", return_value=None):
            wl.run_snapshot(config=cfg, registry_path="ignored", snapshot_date="2026-05-15")

        ev = next(r for r in self._read_audit_records()
                  if r.get("event_type") == "watchlist_run")
        # Audit row exists AND points at a real file.
        self.assertTrue((self.data_dir / ev["evidence_ref"]["path"]).is_file())

    def test_cache_tracker_stopped_after_run(self) -> None:
        cfg = {
            "equity_categories": ["Equity Scheme - Large Cap Fund"],
            "debt_categories": [],
            "include_gold": False,
            "top_n_to_persist": 5,
        }
        from backend.data_discovery import cache_tracker
        # Ensure clean state going in (defensive — prior test leaks
        # would otherwise pass this silently).
        cache_tracker.stop_tracking()
        self.assertFalse(cache_tracker.is_active())
        with patch.object(wl, "rank_category", side_effect=self._fake_rank()), \
             patch.object(wl, "_validate_registry", return_value=None):
            wl.run_snapshot(config=cfg, registry_path="ignored", snapshot_date="2026-05-15")
        # Tracker stopped via finally even on success path.
        self.assertFalse(cache_tracker.is_active())


class NoChangeDetectionTests(unittest.TestCase):
    """Step 6: short-circuit weekend / holiday / unchanged-NAV runs.

    Detection ladder (each layer can short-circuit independently):
      1. weekend         → audit-only, no compute
      2. holiday         → audit-only, no compute (calendar file required)
      3. no_nav_change   → compute runs, but content_hash matches sidecar
                           → no snapshot writes, no evidence file

    Short-circuits NEVER run without a sidecar baseline. Registry-abort
    is never reclassified as no_change. ``--force`` bypasses all layers.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self._orig_snap = wl.SNAPSHOTS_DIR
        self._orig_audit = wl.AUDIT_PATH
        self._orig_sidecar = wl.SIDECAR_PATH
        self._orig_holiday = wl.HOLIDAY_CALENDAR_PATH
        self.data_dir = Path(self.tmp.name)
        wl.SNAPSHOTS_DIR = self.data_dir / "snapshots"
        wl.AUDIT_PATH = self.data_dir / "audit" / "audit.jsonl"
        wl.SIDECAR_PATH = self.data_dir / "watchlist" / "last_run.json"
        wl.HOLIDAY_CALENDAR_PATH = self.data_dir / "reference" / "india_market_holidays.json"

    def tearDown(self) -> None:
        wl.SNAPSHOTS_DIR = self._orig_snap
        wl.AUDIT_PATH = self._orig_audit
        wl.SIDECAR_PATH = self._orig_sidecar
        wl.HOLIDAY_CALENDAR_PATH = self._orig_holiday
        self.tmp.cleanup()

    # ── fixtures ──

    def _cfg(self, equity=("Equity Scheme - Large Cap Fund",), debt=(),
             include_gold=False, top_n=5) -> dict:
        return {
            "equity_categories": list(equity),
            "debt_categories": list(debt),
            "include_gold": include_gold,
            "top_n_to_persist": top_n,
        }

    def _run_real(self, cfg, snap_date="2026-05-14", codes=(101, 102, 103)) -> None:
        """Helper: do a real (non-short-circuit) watchlist_run via the
        public entry point so the sidecar is written by production code,
        not the test. ``2026-05-14`` is a Thursday (weekday)."""

        def rank(category, _path):
            return _ranking(category, list(codes))

        with patch.object(wl, "rank_category", side_effect=rank), \
             patch.object(wl, "_validate_registry", return_value=None):
            wl.run_snapshot(config=cfg, registry_path="ignored", snapshot_date=snap_date)

    def _audit_events(self) -> list:
        if not wl.AUDIT_PATH.exists():
            return []
        with wl.AUDIT_PATH.open("r", encoding="utf-8") as h:
            return [json.loads(line)["event"] for line in h if line.strip()]

    def _write_holiday_calendar(self, days) -> None:
        wl.HOLIDAY_CALENDAR_PATH.parent.mkdir(parents=True, exist_ok=True)
        wl.HOLIDAY_CALENDAR_PATH.write_text(
            json.dumps({"holidays": list(days)}), encoding="utf-8",
        )

    # ── tests ──

    def test_no_sidecar_means_no_short_circuit_even_on_weekend(self) -> None:
        # 2026-05-16 is a Saturday — the weekend layer would fire IF a
        # baseline existed. Without a sidecar we MUST fall through to
        # a normal watchlist_run, not invent a no_change with no anchor.
        self.assertFalse(wl.SIDECAR_PATH.exists())
        cfg = self._cfg()
        with patch.object(wl, "rank_category",
                          side_effect=lambda c, p: _ranking(c, [101, 102, 103])), \
             patch.object(wl, "_validate_registry", return_value=None):
            summary = wl.run_snapshot(config=cfg, registry_path="ignored",
                                      snapshot_date="2026-05-16")
        self.assertEqual(summary["Equity Scheme - Large Cap Fund"], "ok")
        events = self._audit_events()
        self.assertEqual([e["event_type"] for e in events], ["watchlist_run"])
        # Sidecar written by the real run.
        self.assertTrue(wl.SIDECAR_PATH.exists())

    def test_weekend_short_circuits_to_no_change(self) -> None:
        # Seed a real run on Thursday 2026-05-14 to populate sidecar.
        cfg = self._cfg()
        self._run_real(cfg, snap_date="2026-05-14")
        sidecar_after_real = json.loads(wl.SIDECAR_PATH.read_text())

        # Saturday 2026-05-16 — weekend layer fires.
        with patch.object(wl, "rank_category") as rc, \
             patch.object(wl, "_validate_registry", return_value=None):
            summary = wl.run_snapshot(config=cfg, registry_path="ignored",
                                      snapshot_date="2026-05-16")
            rc.assert_not_called()  # no compute happened

        self.assertEqual(summary, {"__no_change__": "weekend"})
        events = self._audit_events()
        types = [e["event_type"] for e in events]
        self.assertEqual(types, ["watchlist_run", "no_change"])
        nc = events[-1]
        self.assertEqual(nc["reason"], "weekend")
        self.assertEqual(nc["previous_run_id"], sidecar_after_real["run_id"])
        self.assertEqual(nc["parent_run_id"], sidecar_after_real["run_id"])
        self.assertEqual(nc["snapshot_date"], "2026-05-16")
        # Sidecar must NOT have moved to the no_change row.
        sidecar_after_nc = json.loads(wl.SIDECAR_PATH.read_text())
        self.assertEqual(sidecar_after_nc["run_id"], sidecar_after_real["run_id"])

    def test_holiday_short_circuits_when_calendar_present(self) -> None:
        cfg = self._cfg()
        self._run_real(cfg, snap_date="2026-05-14")
        baseline = json.loads(wl.SIDECAR_PATH.read_text())

        # 2026-05-15 is a Friday (weekday). Mark as holiday.
        self._write_holiday_calendar(["2026-05-15"])

        with patch.object(wl, "rank_category") as rc, \
             patch.object(wl, "_validate_registry", return_value=None):
            summary = wl.run_snapshot(config=cfg, registry_path="ignored",
                                      snapshot_date="2026-05-15")
            rc.assert_not_called()

        self.assertEqual(summary, {"__no_change__": "holiday"})
        nc = self._audit_events()[-1]
        self.assertEqual(nc["event_type"], "no_change")
        self.assertEqual(nc["reason"], "holiday")
        self.assertEqual(nc["parent_run_id"], baseline["run_id"])

    def test_holiday_calendar_absent_falls_through_to_content_check(self) -> None:
        # No calendar file. A weekday that COULD be a holiday is treated
        # as a normal trading day — we never claim holiday status without
        # evidence. Content matches → no_nav_change, not "holiday".
        cfg = self._cfg()
        self._run_real(cfg, snap_date="2026-05-14")
        self.assertFalse(wl.HOLIDAY_CALENDAR_PATH.exists())

        # Same content, different date.
        def rank(category, _path):
            return _ranking(category, [101, 102, 103])

        with patch.object(wl, "rank_category", side_effect=rank), \
             patch.object(wl, "_validate_registry", return_value=None):
            summary = wl.run_snapshot(config=cfg, registry_path="ignored",
                                      snapshot_date="2026-05-15")

        self.assertEqual(summary, {"__no_change__": "no_nav_change"})
        nc = self._audit_events()[-1]
        self.assertEqual(nc["reason"], "no_nav_change")

    def test_post_compute_content_match_short_circuits(self) -> None:
        cfg = self._cfg()
        self._run_real(cfg, snap_date="2026-05-14")
        baseline = json.loads(wl.SIDECAR_PATH.read_text())
        original_snapshot = wl.load_snapshot(
            "Equity Scheme - Large Cap Fund", "2026-05-14",
        )

        # Same ranking content next day. Use a different computed_at
        # to prove the volatile-field strip works — the content hash
        # must IGNORE computed_at and snapshot_date.
        def rank(category, _path):
            r = _ranking(category, [101, 102, 103])
            r.computed_at = "2026-05-15T00:00:00+00:00"  # different wall clock
            return r

        with patch.object(wl, "rank_category", side_effect=rank), \
             patch.object(wl, "_validate_registry", return_value=None):
            summary = wl.run_snapshot(config=cfg, registry_path="ignored",
                                      snapshot_date="2026-05-15")

        self.assertEqual(summary, {"__no_change__": "no_nav_change"})
        # Old snapshot file is untouched; no new snapshot was written.
        self.assertIsNone(
            wl.load_snapshot("Equity Scheme - Large Cap Fund", "2026-05-15"),
        )
        unchanged = wl.load_snapshot("Equity Scheme - Large Cap Fund", "2026-05-14")
        self.assertEqual(unchanged, original_snapshot)
        # No new evidence file under data/evidence/watchlist_run/.
        ev_dir = self.data_dir / "evidence" / "watchlist_run"
        before = len(list(ev_dir.iterdir())) if ev_dir.exists() else 0
        self.assertEqual(before, 1)  # only the original real run's file
        # Sidecar still anchored to the original real run.
        sc = json.loads(wl.SIDECAR_PATH.read_text())
        self.assertEqual(sc["run_id"], baseline["run_id"])

    def test_content_change_writes_new_snapshot_and_updates_sidecar(self) -> None:
        cfg = self._cfg()
        self._run_real(cfg, snap_date="2026-05-14")
        baseline = json.loads(wl.SIDECAR_PATH.read_text())

        # Different codes → different content hash → real run.
        def rank(category, _path):
            return _ranking(category, [201, 202, 203])

        with patch.object(wl, "rank_category", side_effect=rank), \
             patch.object(wl, "_validate_registry", return_value=None):
            summary = wl.run_snapshot(config=cfg, registry_path="ignored",
                                      snapshot_date="2026-05-15")

        self.assertEqual(summary["Equity Scheme - Large Cap Fund"], "ok")
        self.assertIsNotNone(
            wl.load_snapshot("Equity Scheme - Large Cap Fund", "2026-05-15"),
        )
        events = self._audit_events()
        self.assertEqual([e["event_type"] for e in events],
                         ["watchlist_run", "watchlist_run"])
        # Sidecar moved to the new run.
        sc_new = json.loads(wl.SIDECAR_PATH.read_text())
        self.assertNotEqual(sc_new["run_id"], baseline["run_id"])
        self.assertEqual(sc_new["snapshot_date"], "2026-05-15")

    def test_force_bypasses_all_short_circuit_layers(self) -> None:
        cfg = self._cfg()
        self._run_real(cfg, snap_date="2026-05-14")
        # Even with weekend AND matching content AND holiday calendar
        # listing today, --force must still produce a real watchlist_run.
        self._write_holiday_calendar(["2026-05-16"])

        def rank(category, _path):
            return _ranking(category, [101, 102, 103])  # same content

        with patch.object(wl, "rank_category", side_effect=rank), \
             patch.object(wl, "_validate_registry", return_value=None):
            summary = wl.run_snapshot(
                config=cfg, registry_path="ignored",
                snapshot_date="2026-05-16",  # Saturday + holiday
                force=True,
            )

        self.assertEqual(summary["Equity Scheme - Large Cap Fund"], "ok")
        events = self._audit_events()
        self.assertEqual([e["event_type"] for e in events],
                         ["watchlist_run", "watchlist_run"])

    def test_corrupt_sidecar_falls_through_to_real_run(self) -> None:
        # No prior real run; instead, plant a corrupt sidecar so the
        # short-circuit path is forced to a definite "skip" decision.
        wl.SIDECAR_PATH.parent.mkdir(parents=True, exist_ok=True)
        wl.SIDECAR_PATH.write_text("{ not valid json", encoding="utf-8")

        cfg = self._cfg()
        with patch.object(wl, "rank_category",
                          side_effect=lambda c, p: _ranking(c, [1, 2, 3])), \
             patch.object(wl, "_validate_registry", return_value=None):
            # Saturday — weekend layer WOULD fire if sidecar were valid.
            summary = wl.run_snapshot(config=cfg, registry_path="ignored",
                                      snapshot_date="2026-05-16")

        # Fell through to real run (corrupt sidecar treated as missing).
        self.assertEqual(summary["Equity Scheme - Large Cap Fund"], "ok")
        events = self._audit_events()
        self.assertEqual([e["event_type"] for e in events], ["watchlist_run"])
        # The successful real run rewrites the sidecar atomically with
        # a valid baseline — corrupt content is replaced, not preserved.
        # (Preservation of corrupt files is the responsibility of the
        # load path's WARN log, not of the write path.)
        sc = json.loads(wl.SIDECAR_PATH.read_text())
        self.assertIn("run_id", sc)

    def test_broken_snapshot_ref_disables_content_short_circuit(self) -> None:
        cfg = self._cfg()
        self._run_real(cfg, snap_date="2026-05-14")
        # Delete the snapshot file the sidecar points at. With a missing
        # reference, the post-compute layer must NOT claim no_change —
        # we'd rather regenerate than emit a row pointing at vanished evidence.
        ref_path = (wl.SNAPSHOTS_DIR / "Equity_Scheme__Large_Cap_Fund"
                    / "2026-05-14.json")
        ref_path.unlink()
        self.assertFalse(ref_path.exists())

        def rank(category, _path):
            return _ranking(category, [101, 102, 103])  # same content

        with patch.object(wl, "rank_category", side_effect=rank), \
             patch.object(wl, "_validate_registry", return_value=None):
            summary = wl.run_snapshot(config=cfg, registry_path="ignored",
                                      snapshot_date="2026-05-15")

        # Fell through to real run despite content match.
        self.assertEqual(summary["Equity Scheme - Large Cap Fund"], "ok")
        events = self._audit_events()
        self.assertEqual([e["event_type"] for e in events],
                         ["watchlist_run", "watchlist_run"])

    def test_registry_abort_never_reclassified_as_no_change(self) -> None:
        # Seed sidecar with a real run.
        cfg = self._cfg()
        self._run_real(cfg, snap_date="2026-05-14")
        baseline = json.loads(wl.SIDECAR_PATH.read_text())

        with patch.object(wl, "_validate_registry",
                          return_value="registry has only 42 entries"):
            summary = wl.run_snapshot(config=cfg, registry_path="ignored",
                                      snapshot_date="2026-05-15")

        self.assertIn("__registry__", summary)
        events = self._audit_events()
        self.assertEqual([e["event_type"] for e in events],
                         ["watchlist_run", "watchlist_run"])
        self.assertIsNotNone(events[-1].get("registry_problem"))
        # Sidecar NOT updated by registry-abort path.
        sc_after = json.loads(wl.SIDECAR_PATH.read_text())
        self.assertEqual(sc_after["run_id"], baseline["run_id"])

    def test_no_change_event_has_watchlist_run_evidence_kind_and_no_evidence_ref(self) -> None:
        # Closed-enum check: we reuse watchlist_run rather than adding
        # a new evidence_kind. no_change rows must NOT carry an
        # evidence_ref (audit-only by design).
        cfg = self._cfg()
        self._run_real(cfg, snap_date="2026-05-14")

        with patch.object(wl, "rank_category") as rc, \
             patch.object(wl, "_validate_registry", return_value=None):
            wl.run_snapshot(config=cfg, registry_path="ignored",
                            snapshot_date="2026-05-16")  # Saturday
            rc.assert_not_called()

        nc = self._audit_events()[-1]
        self.assertEqual(nc["event_type"], "no_change")
        self.assertEqual(nc["evidence_kind"], "watchlist_run")
        self.assertNotIn("evidence_ref", nc)

    def test_audit_chain_remains_valid_across_mixed_sequence(self) -> None:
        from backend.investment_analytics.audit import verify_audit_chain
        cfg = self._cfg()
        # real → weekend no_change → content-match no_change → real
        self._run_real(cfg, snap_date="2026-05-14")  # real (Thursday)

        # Saturday — weekend layer.
        with patch.object(wl, "rank_category") as rc, \
             patch.object(wl, "_validate_registry", return_value=None):
            wl.run_snapshot(config=cfg, registry_path="ignored",
                            snapshot_date="2026-05-16")
            rc.assert_not_called()

        # Monday — same content (no_nav_change).
        with patch.object(wl, "rank_category",
                          side_effect=lambda c, p: _ranking(c, [101, 102, 103])), \
             patch.object(wl, "_validate_registry", return_value=None):
            wl.run_snapshot(config=cfg, registry_path="ignored",
                            snapshot_date="2026-05-18")

        # Tuesday — different content (real run).
        with patch.object(wl, "rank_category",
                          side_effect=lambda c, p: _ranking(c, [201, 202, 203])), \
             patch.object(wl, "_validate_registry", return_value=None):
            wl.run_snapshot(config=cfg, registry_path="ignored",
                            snapshot_date="2026-05-19")

        types = [e["event_type"] for e in self._audit_events()]
        self.assertEqual(types,
                         ["watchlist_run", "no_change", "no_change", "watchlist_run"])
        self.assertTrue(verify_audit_chain(wl.AUDIT_PATH))

    def test_unknown_holiday_reason_rejected_at_emit(self) -> None:
        # Defense in depth — _emit_no_change_event should refuse to
        # write a reason not in the closed enum.
        cfg = self._cfg()
        self._run_real(cfg, snap_date="2026-05-14")
        sc = json.loads(wl.SIDECAR_PATH.read_text())
        with self.assertRaises(ValueError):
            wl._emit_no_change_event("2026-05-15", "bogus_reason", sc, None)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
