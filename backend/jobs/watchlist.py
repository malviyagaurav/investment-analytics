"""Daily watchlist runner — rank a curated set of categories and
persist top-N snapshots so users can see how their selection drifts
day-over-day without clicking through the UI.

Designed to be invoked by:
- `python -m backend.jobs.watchlist run`           — fresh snapshot
- `python -m backend.jobs.watchlist latest <cat>`  — print today's top-N
- `python -m backend.jobs.watchlist diff <cat>`    — diff today vs yesterday
- a cron / launchd job (see scripts/ for templates)

What it does NOT do:
- Pretend to refresh more than once a day. Indian MFs are EOD-priced
  by SEBI rule; intraday recomputation is wasted work. The 24h NAV
  cache already covers that.
- Mutate the existing on-demand /analytics/portfolio-health flow.
  The snapshot store is a separate side-channel; the API endpoint
  remains entirely independent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.data_discovery import cache_tracker
from backend.evidence.store import emit_evidence
from backend.investment_analytics.ranking import (
    DEFAULT_DEBT_CATEGORIES,
    EXCLUDED_CATEGORIES,
    RANKABLE_CATEGORIES,
    debt_ranking_to_dict,
    gold_ranking_to_dict,
    rank_category,
    rank_debt_category,
    rank_gold_funds,
    ranking_to_dict,
)

logger = logging.getLogger("jobs.watchlist")

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = ROOT / "data" / "watchlist" / "categories.json"
SNAPSHOTS_DIR = ROOT / "data" / "snapshots"
REGISTRY_PATH = ROOT / "data" / "registry" / "schemes.json"
AUDIT_PATH = ROOT / "data" / "audit" / "audit.jsonl"
SIDECAR_PATH = ROOT / "data" / "watchlist" / "last_run.json"
HOLIDAY_CALENDAR_PATH = ROOT / "data" / "reference" / "india_market_holidays.json"

# Closed set; widening requires re-thinking the detection ladder, not
# just adding a string.
NO_CHANGE_REASONS = frozenset({"weekend", "holiday", "no_nav_change"})
SIDECAR_SCHEMA_VERSION = "v1"
NO_CHANGE_SCHEMA_VERSION = "v1"

# Fields stripped before content-hashing a per-category trimmed
# snapshot. ``computed_at`` is wall-clock; ``snapshot_date`` is the
# run's logical date. Both change every run; neither reflects whether
# the underlying ranking changed.
_VOLATILE_SNAPSHOT_FIELDS = ("computed_at", "snapshot_date")


# ── Config ───────────────────────────────────────────────────────────


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load watchlist config. Falls back to RANKABLE_CATEGORIES +
    DEFAULT_DEBT_CATEGORIES if no config file exists, so a fresh
    install runs without a config."""
    p = path or DEFAULT_CONFIG_PATH
    if not p.exists():
        return {
            "equity_categories": list(RANKABLE_CATEGORIES),
            "debt_categories": list(DEFAULT_DEBT_CATEGORIES),
            "include_gold": True,
            "top_n_to_persist": 10,
        }
    return json.loads(p.read_text(encoding="utf-8"))


# ── Snapshot store ───────────────────────────────────────────────────


def _safe_dir(category: str) -> Path:
    """Map a category name to a filesystem-safe directory under
    data/snapshots/. Strips spaces / slashes so a single AMFI category
    like 'Equity Scheme - Large Cap Fund' becomes a clean path."""
    safe = (
        category.replace(" - ", "__")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )
    return SNAPSHOTS_DIR / safe


def _snapshot_path(category: str, snapshot_date: str) -> Path:
    return _safe_dir(category) / f"{snapshot_date}.json"


def save_snapshot(
    category: str,
    snapshot_date: str,
    payload: Dict[str, Any],
) -> Path:
    """Atomic write — temp file then rename — so a concurrent reader
    never sees a half-written JSON."""
    target = _snapshot_path(category, snapshot_date)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(target)
    return target


def list_snapshot_dates(category: str) -> List[str]:
    """Sorted list of YYYY-MM-DD dates we have snapshots for."""
    d = _safe_dir(category)
    if not d.exists():
        return []
    out = []
    for child in d.iterdir():
        if child.suffix == ".json" and len(child.stem) == 10:
            out.append(child.stem)
    return sorted(out)


def load_snapshot(category: str, snapshot_date: str) -> Optional[Dict[str, Any]]:
    p = _snapshot_path(category, snapshot_date)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def latest_snapshot(category: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    dates = list_snapshot_dates(category)
    if not dates:
        return None
    last = dates[-1]
    snap = load_snapshot(category, last)
    return (last, snap) if snap else None


# ── Snapshot builder ─────────────────────────────────────────────────


def _trim_for_snapshot(category_dict: Dict[str, Any], top_n: int) -> Dict[str, Any]:
    """Drop the long tail of `ranked` and `excluded` so each daily
    snapshot is small. Keep the trust + benchmark + limitations
    headers intact."""
    out = dict(category_dict)
    ranked = out.get("ranked", [])
    out["ranked"] = ranked[:top_n]
    out["showing_top_n"] = min(top_n, len(ranked))
    out["full_ranked_count"] = len(ranked)
    # Drop noisy fields.
    out.pop("excluded", None)
    return out


MIN_REGISTRY_ENTRIES = 500  # AMFI returns ~10k schemes; <500 means a parse failure


def _validate_registry(registry_path: str) -> Optional[str]:
    """Return None if the registry file is loadable and contains a
    plausible number of schemes; otherwise return a one-line error
    message. Stops the runner from producing a wave of per-category
    'category has fewer than 2 funds' errors when the real problem is
    a missing or corrupt registry."""
    from backend.data_discovery.registry import load_registry  # local import

    try:
        entries = load_registry(Path(registry_path))
    except Exception as exc:
        return f"registry load failed: {exc}"
    n = len(entries)
    if n == 0:
        return (
            f"registry at {registry_path} is empty — "
            f"run POST /discover/refresh-registry before scheduling snapshots"
        )
    if n < MIN_REGISTRY_ENTRIES:
        return (
            f"registry at {registry_path} has only {n} entries "
            f"(<{MIN_REGISTRY_ENTRIES}); likely parse failure or partial download"
        )
    return None


def _emit_watchlist_audit(
    snap_date: str,
    cfg: Dict[str, Any],
    reg: str,
    summary: Dict[str, str],
    registry_problem: Optional[str],
) -> Dict[str, Any]:
    """Emit one watchlist_run audit + evidence event per run_snapshot
    invocation. Fired on success AND on registry-abort so the cron
    job is never silently invisible to the audit chain.

    Returns the audit record (whose ``event.run_id`` the caller uses
    when updating the sidecar pointer to the last real run).
    """
    equity_cats = list(cfg.get("equity_categories", []))
    debt_cats = list(cfg.get("debt_categories", []))
    gold_included = bool(cfg.get("include_gold", False))

    # Per-category outcomes only — strip the synthetic __registry__
    # key used by the abort path so counts reflect real categories.
    category_outcomes = {k: v for k, v in summary.items() if k != "__registry__"}
    ok_count = sum(1 for v in category_outcomes.values() if v == "ok")
    errored_categories = sorted(
        k for k, v in category_outcomes.items() if v != "ok"
    )
    error_count = len(errored_categories)

    audit_event = {
        "event_type": "watchlist_run",
        "snapshot_date": snap_date,
        "equity_count": len(equity_cats),
        "debt_count": len(debt_cats),
        "gold_included": gold_included,
        "ok_count": ok_count,
        "error_count": error_count,
        "errored_categories": errored_categories,
        "registry_problem": registry_problem,
        "schema_version": "v1",
    }

    payload = {
        "snapshot_date": snap_date,
        "config": {
            "equity_categories": equity_cats,
            "debt_categories": debt_cats,
            "include_gold": gold_included,
            "top_n_to_persist": int(cfg.get("top_n_to_persist", 10)),
        },
        "summary": summary,
        "registry_path": str(reg),
        "snapshots_dir": str(SNAPSHOTS_DIR),
    }

    return emit_evidence(
        audit_log_path=AUDIT_PATH,
        evidence_kind="watchlist_run",
        audit_event=audit_event,
        payload=payload,
    )


# ── no_change detection ──────────────────────────────────────────────


def _load_holiday_calendar(
    path: Optional[Path] = None,
) -> Optional[frozenset[str]]:
    """Load the India market holiday calendar if present.

    Shape on disk: ``{"holidays": ["YYYY-MM-DD", ...]}``. Returns
    ``None`` (not an empty set) when the file is absent so callers
    distinguish "no calendar shipped" from "calendar present but
    today not listed". A malformed file logs WARN and returns None;
    we never claim holiday status from a file we cannot read.
    """
    p = path or HOLIDAY_CALENDAR_PATH
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        days = data.get("holidays", [])
        if not isinstance(days, list):
            raise ValueError("holidays must be a list")
        return frozenset(str(d) for d in days)
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        logger.warning("Holiday calendar at %s unreadable: %s", p, exc)
        return None


def _is_weekend(snap_date: str) -> bool:
    return datetime.fromisoformat(snap_date).weekday() >= 5


def _is_holiday(snap_date: str, calendar: Optional[frozenset[str]]) -> bool:
    return calendar is not None and snap_date in calendar


def _strip_volatile(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in payload.items() if k not in _VOLATILE_SNAPSHOT_FIELDS}


def _content_hash(category_payloads: Dict[str, Dict[str, Any]]) -> str:
    """SHA-256 over canonical-JSON of ``{category -> stripped_payload}``.

    Independent of run wall-clock and run date. Two runs producing
    the same per-category ranking content hash to the same value
    regardless of when they ran.
    """
    stripped = {cat: _strip_volatile(p) for cat, p in category_payloads.items()}
    canonical = json.dumps(
        stripped, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_sidecar(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load the last-run sidecar pointer. Returns None when missing
    or unreadable. Corrupt files log WARN but are NOT deleted — left
    for forensic inspection.

    Does NOT validate that the referenced snapshot files still exist;
    that check happens in the post-compute path (it only matters for
    ``no_nav_change``, not for weekend / holiday short-circuits).
    """
    p = path or SIDECAR_PATH
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Sidecar at %s unreadable: %s", p, exc)
        return None
    if not isinstance(data, dict) or "run_id" not in data:
        logger.warning("Sidecar at %s missing run_id field", p)
        return None
    return data


def _write_sidecar(payload: Dict[str, Any], path: Optional[Path] = None) -> Path:
    """Atomic write: tmp + fsync + rename. Matches the discipline
    used by ``save_snapshot`` and ``evidence.store.write_evidence``."""
    target = path or SIDECAR_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    data_bytes = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    with tmp.open("wb") as handle:
        handle.write(data_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(target)
    return target


def _all_refs_exist(refs: Dict[str, str]) -> bool:
    """Sidecar references are repo-relative paths under data/. Check
    each one resolves to an existing file. One missing ref disables
    the post-compute short-circuit (we'd rather regenerate than emit
    a no_change pointing at vanished evidence)."""
    data_dir = ROOT / "data"
    for rel in refs.values():
        # rel is stored as a project-root-relative path string
        # (e.g. "data/snapshots/.../2026-05-14.json"). Resolve and
        # check; an absolute path is also accepted.
        p = Path(rel)
        if not p.is_absolute():
            p = ROOT / rel if rel.startswith("data/") else data_dir / rel
        if not p.exists():
            return False
    return True


def _emit_no_change_event(
    snap_date: str,
    reason: str,
    sidecar: Dict[str, Any],
    registry_hash_now: Optional[str],
) -> Dict[str, Any]:
    """Audit-only append for a no_change run.

    Reuses ``evidence_kind="watchlist_run"`` (no new enum value).
    Sets ``parent_run_id`` on the envelope to the sidecar's run_id
    so replay can chain back to the last content-bearing run.
    Does NOT write an evidence file.
    """
    from backend.investment_analytics.audit import append_audit_record
    from backend.investment_analytics.provenance import capture_provenance_inputs

    if reason not in NO_CHANGE_REASONS:
        raise ValueError(
            f"no_change reason must be one of {sorted(NO_CHANGE_REASONS)}, "
            f"got {reason!r}"
        )

    previous_run_id = sidecar["run_id"]

    audit_event = {
        "event_type": "no_change",
        "snapshot_date": snap_date,
        "reason": reason,
        "previous_run_id": previous_run_id,
        "previous_snapshot_date": sidecar.get("snapshot_date"),
        "previous_content_hash": sidecar.get("content_hash"),
        "registry_hash": registry_hash_now,
        "cache_fingerprint": cache_tracker.cache_fingerprint(),
        "category_snapshot_refs": sidecar.get("category_snapshot_refs", {}),
        "schema_version": NO_CHANGE_SCHEMA_VERSION,
    }

    return append_audit_record(
        AUDIT_PATH,
        audit_event,
        evidence_kind="watchlist_run",
        parent_run_id=previous_run_id,
        inputs=capture_provenance_inputs(),
    )


def run_snapshot(
    config: Optional[Dict[str, Any]] = None,
    registry_path: Optional[str] = None,
    snapshot_date: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Re-rank every category in the watchlist; persist top-N per
    category to data/snapshots/<safe_category>/<YYYY-MM-DD>.json.

    Validates the registry up-front. If the registry is missing,
    empty, or implausibly small, returns immediately with a single
    "registry_error" summary entry instead of producing dozens of
    per-category errors that obscure the real problem.

    Emits exactly one audit event per invocation:
      - ``watchlist_run`` on the real path (success or registry-abort),
        with the full per-category summary persisted by-reference.
      - ``no_change`` (audit-only, no evidence file) when a short-circuit
        layer fires: weekend, holiday (calendar required), or
        post-compute output content equals the sidecar's previous run.

    Short-circuits are disabled when:
      - ``force=True``
      - no sidecar exists (no baseline to compare against)
      - the sidecar is corrupt / unreadable
      - registry validation fails (those paths emit watchlist_run with
        ``registry_problem``; they are never reclassified as no_change)

    Cache reads during the run are aggregated into a single
    ``cache_fingerprint`` recorded on the audit row's provenance.

    Returns a summary {category -> "ok" | error_message}, or
    ``{"__no_change__": "<reason>"}`` for short-circuit runs.
    """
    cfg = config if config is not None else load_config()
    reg = registry_path or str(REGISTRY_PATH)
    snap_date = snapshot_date or datetime.now(timezone.utc).date().isoformat()
    top_n = int(cfg.get("top_n_to_persist", 10))

    summary: Dict[str, str] = {}
    registry_problem: Optional[str] = None

    cache_tracker.start_tracking()
    try:
        # ── Pre-compute short-circuits (require sidecar baseline) ──
        sidecar = None if force else _load_sidecar()
        if sidecar is not None:
            registry_hash_now = _hash_registry_for_provenance(reg)
            if _is_weekend(snap_date):
                _emit_no_change_event(snap_date, "weekend", sidecar, registry_hash_now)
                return {"__no_change__": "weekend"}
            if _is_holiday(snap_date, _load_holiday_calendar()):
                _emit_no_change_event(snap_date, "holiday", sidecar, registry_hash_now)
                return {"__no_change__": "holiday"}

        registry_problem = _validate_registry(reg)
        if registry_problem is not None:
            logger.error("Watchlist run aborted: %s", registry_problem)
            summary["__registry__"] = f"error: {registry_problem}"
            _emit_watchlist_audit(
                snap_date=snap_date,
                cfg=cfg,
                reg=reg,
                summary=summary,
                registry_problem=registry_problem,
            )
            # Do NOT update sidecar on registry-abort: it has no
            # content baseline; using it for future comparisons
            # would let an empty result masquerade as "no change."
            return summary

        # ── Compute everything first; defer disk writes until after
        # the post-compute short-circuit decision. ─────────────────
        category_payloads: Dict[str, Dict[str, Any]] = {}
        for cat in cfg.get("equity_categories", []):
            if cat in EXCLUDED_CATEGORIES:
                summary[cat] = "excluded by ranking policy"
                continue
            try:
                result = rank_category(cat, reg)
                payload = ranking_to_dict(result)
                payload["snapshot_date"] = snap_date
                payload["snapshot_kind"] = "equity"
                category_payloads[cat] = _trim_for_snapshot(payload, top_n)
                summary[cat] = "ok"
            except Exception as exc:
                logger.warning("equity rank failed for %s: %s", cat, exc)
                summary[cat] = f"error: {exc}"

        for cat in cfg.get("debt_categories", []):
            try:
                result = rank_debt_category(cat, reg)
                payload = debt_ranking_to_dict(result)
                payload["snapshot_date"] = snap_date
                payload["snapshot_kind"] = "debt"
                category_payloads[cat] = _trim_for_snapshot(payload, top_n)
                summary[cat] = "ok"
            except Exception as exc:
                logger.warning("debt rank failed for %s: %s", cat, exc)
                summary[cat] = f"error: {exc}"

        if cfg.get("include_gold", False):
            cat = "Gold Fund (FoF)"
            try:
                result = rank_gold_funds(reg)
                payload = gold_ranking_to_dict(result)
                payload["snapshot_date"] = snap_date
                payload["snapshot_kind"] = "gold"
                category_payloads[cat] = _trim_for_snapshot(payload, top_n)
                summary[cat] = "ok"
            except Exception as exc:
                logger.warning("gold rank failed: %s", exc)
                summary[cat] = f"error: {exc}"

        # ── Post-compute short-circuit: identical content vs sidecar ──
        content_hash_now = _content_hash(category_payloads)
        if (
            not force
            and sidecar is not None
            and category_payloads
            and sidecar.get("content_hash") == content_hash_now
            and _all_refs_exist(sidecar.get("category_snapshot_refs", {}))
        ):
            registry_hash_now = _hash_registry_for_provenance(reg)
            _emit_no_change_event(
                snap_date, "no_nav_change", sidecar, registry_hash_now,
            )
            return {"__no_change__": "no_nav_change"}

        # ── Real run: persist per-category snapshots, emit audit,
        # update sidecar to the new run as the canonical baseline. ──
        category_refs: Dict[str, str] = {}
        for cat, payload in category_payloads.items():
            target = save_snapshot(cat, snap_date, payload)
            # Prefer repo-relative ("data/snapshots/...") for portability.
            # Fall back to absolute when the snapshots dir lives outside
            # the repo root (test fixtures monkeypatch this).
            try:
                # ``.as_posix()`` forces forward-slash separators so the
                # path string is byte-canonical across hosts. ``str(Path)``
                # would emit backslash on Windows and end up in both the
                # sidecar and the ``no_change`` audit event's
                # ``category_snapshot_refs`` field — non-canonical chain
                # bytes. POSIX behavior is unchanged (already forward
                # slash). Mirrors the R3 fix in ``write_evidence``.
                category_refs[cat] = target.relative_to(ROOT).as_posix()
            except ValueError:
                # Absolute-path fallback: intrinsically host-bound (drive
                # letter on Windows, leading ``/`` on POSIX). Leaving as
                # ``str(target)`` deliberately — see commit body for why
                # canonicalizing this branch is a separate decision.
                category_refs[cat] = str(target)

        record = _emit_watchlist_audit(
            snap_date=snap_date,
            cfg=cfg,
            reg=reg,
            summary=summary,
            registry_problem=registry_problem,
        )

        if category_payloads:
            # Sidecar tracks ONLY content-bearing watchlist_run rows.
            # no_change rows never update it; the lineage anchor stays
            # the last real run.
            _write_sidecar({
                "schema_version": SIDECAR_SCHEMA_VERSION,
                "run_id": record["event"]["run_id"],
                "snapshot_date": snap_date,
                "generated_at": record["event"]["generated_at"],
                "registry_hash": record["event"]["inputs"].get("registry_hash"),
                "cache_fingerprint": record["event"]["inputs"].get("cache_fingerprint"),
                "content_hash": content_hash_now,
                "category_snapshot_refs": category_refs,
            })
    finally:
        cache_tracker.stop_tracking()

    return summary


def _hash_registry_for_provenance(registry_path: str) -> Optional[str]:
    """Local helper: sha256 of the registry file if it exists.

    Used only for the no_change event's ``registry_hash`` field
    (forensic). Returns None if the path doesn't resolve or is
    unreadable — matching the contract of ``capture_provenance_inputs``.
    """
    from backend.investment_analytics.provenance import _hash_file
    return _hash_file(Path(registry_path))


# ── Diff ─────────────────────────────────────────────────────────────


def diff_snapshots(
    prev: Dict[str, Any],
    curr: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare two snapshots of the same category and return
    structured changes. Pure function — caller supplies two dicts.

    Output sections:
      - new_funds: scheme codes in curr but not prev (entered top-N)
      - dropped_funds: in prev but not in curr (left top-N)
      - rank_changes: scheme code -> (prev_rank, curr_rank, delta)
                      where positive delta = improvement (lower number)
      - dominance_changes: same shape, on dominance_count
    """
    def _by_code(snap: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        return {f["scheme_code"]: f for f in snap.get("ranked", [])}

    prev_map = _by_code(prev)
    curr_map = _by_code(curr)

    new_codes = sorted(set(curr_map) - set(prev_map))
    dropped_codes = sorted(set(prev_map) - set(curr_map))
    common = sorted(set(prev_map) & set(curr_map))

    rank_changes: Dict[int, Dict[str, Any]] = {}
    dom_changes: Dict[int, Dict[str, Any]] = {}
    for code in common:
        p = prev_map[code]
        c = curr_map[code]
        if p.get("rank") != c.get("rank"):
            rank_changes[code] = {
                "fund_name": c.get("fund_name"),
                "prev_rank": p.get("rank"),
                "curr_rank": c.get("rank"),
                "delta": (p.get("rank", 0) - c.get("rank", 0)),  # +ve = climbed
            }
        prev_dom = (p.get("dominance") or {}).get("beats")
        curr_dom = (c.get("dominance") or {}).get("beats")
        if prev_dom != curr_dom and prev_dom is not None and curr_dom is not None:
            dom_changes[code] = {
                "fund_name": c.get("fund_name"),
                "prev_dominance": prev_dom,
                "curr_dominance": curr_dom,
                "delta": curr_dom - prev_dom,
            }

    return {
        "category": curr.get("category") or prev.get("category"),
        "prev_date": prev.get("snapshot_date"),
        "curr_date": curr.get("snapshot_date"),
        "new_funds": [
            {"scheme_code": c,
             "fund_name": curr_map[c].get("fund_name"),
             "rank": curr_map[c].get("rank")}
            for c in new_codes
        ],
        "dropped_funds": [
            {"scheme_code": c,
             "fund_name": prev_map[c].get("fund_name"),
             "prev_rank": prev_map[c].get("rank")}
            for c in dropped_codes
        ],
        "rank_changes": rank_changes,
        "dominance_changes": dom_changes,
    }


def diff_latest_two(category: str) -> Optional[Dict[str, Any]]:
    """Diff today's snapshot against yesterday's (or whatever the
    previous one is). Returns None if fewer than 2 snapshots exist."""
    dates = list_snapshot_dates(category)
    if len(dates) < 2:
        return None
    prev = load_snapshot(category, dates[-2])
    curr = load_snapshot(category, dates[-1])
    if not prev or not curr:
        return None
    return diff_snapshots(prev, curr)


# ── CLI ──────────────────────────────────────────────────────────────


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m backend.jobs.watchlist")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Take a fresh snapshot of every watched category.")
    run_p.add_argument("--date", help="Override snapshot date (YYYY-MM-DD).")
    run_p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Bypass the no_change short-circuit. Forces a full ranking + "
            "snapshot write even when today is a weekend, a holiday, or "
            "the content matches the sidecar's previous run."
        ),
    )

    latest_p = sub.add_parser("latest", help="Print latest snapshot for a category.")
    latest_p.add_argument("category", help="AMFI category name (quote it).")

    diff_p = sub.add_parser("diff", help="Diff latest two snapshots for a category.")
    diff_p.add_argument("category", help="AMFI category name (quote it).")

    list_p = sub.add_parser("list", help="List snapshot dates for a category.")
    list_p.add_argument("category", help="AMFI category name (quote it).")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        summary = run_snapshot(snapshot_date=args.date, force=args.force)
        ok = sum(1 for v in summary.values() if v == "ok")
        print(json.dumps({"ok_count": ok, "total": len(summary), "summary": summary}, indent=2))
        return 0

    if args.cmd == "latest":
        latest = latest_snapshot(args.category)
        if latest is None:
            print(f"No snapshots for {args.category!r}")
            return 1
        d, snap = latest
        print(f"# {args.category}  ({d})")
        for f in snap.get("ranked", []):
            print(
                f"  rank {f['rank']:>2}  {f['fund_name']}  "
                f"({f['fund_house']})  conf={f.get('confidence_level','?')}"
            )
        return 0

    if args.cmd == "diff":
        d = diff_latest_two(args.category)
        if d is None:
            print("Need at least 2 snapshots to diff.")
            return 1
        print(json.dumps(d, indent=2))
        return 0

    if args.cmd == "list":
        for d in list_snapshot_dates(args.category):
            print(d)
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
