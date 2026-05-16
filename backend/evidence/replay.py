"""Semantic replay tool for evidence-bearing audit rows.

Loads a prior audit record, follows ``evidence_ref`` to its by-reference
evidence file, reconstructs the recorded inputs, reruns the underlying
computation, compares recorded vs replayed output, and emits a typed
``replay_result`` evidence artifact.

## States (closed enum, per the architecture decision)

  - ``exact_match``           — byte-equal canonical JSON of sanitized payloads.
  - ``semantically_equivalent`` — equal after volatile-field strip
                                  (computed_at, snapshot_date, generated_at,
                                  registry_path, snapshots_dir).
  - ``expected_divergence``   — content differs AND ≥1 named driver explains
                                it (methodology / code_sha / registry_hash /
                                cache_fingerprint).
  - ``unreproducible``        — prerequisites unmet: audit row absent, no
                                evidence_ref, evidence file missing,
                                registry missing, no handler for kind.
  - ``invalid_replay``        — chain broken, evidence hash mismatch, handler
                                raised, OR content differs with no identified
                                driver (per the approved decision default).

## Side-effect discipline (load-bearing)

Handlers run computation ONLY. They do not append to the audit chain, do
not write snapshots, do not update the watchlist sidecar. The single
side effect of ``replay_run`` is one ``replay_result`` row + one evidence
file under ``data/evidence/replay_result/<run_id>.json``, emitted through
the existing ``emit_evidence`` path.

## Lineage

``parent_run_id`` on the replay_result envelope points at the original
``run_id``. No new linkage field is introduced.

## Scope

Replay only handles the three evidence_kinds that have emitters today:
``ranking_snapshot``, ``portfolio_health_snapshot``, ``watchlist_run``.
Attempting to replay any other kind (including ``replay_result``)
returns ``unreproducible`` ("no handler"). Audit-only events
(``no_change``, ``jurisdiction_gate``) have no ``evidence_ref`` and are
correctly non-replayable.

## What replay is NOT

Replay is not a hermetic time machine. The NAV cache mutates, mfapi
can serve newer NAVs on cache-miss replays, and the AMFI registry
shifts. The diagnostic value of replay is therefore weighted toward
``semantically_equivalent`` (typical good outcome) and
``expected_divergence`` with named drivers (typical interesting
outcome). ``exact_match`` is rare in practice and reserved for the
cases where it genuinely holds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.data_discovery import cache_tracker
from backend.evidence.store import emit_evidence
from backend.investment_analytics.audit import (
    sanitize_audit_event,
    verify_audit_chain,
)

logger = logging.getLogger("evidence.replay")

REPLAY_TOOL_VERSION = "v1"

REPLAY_STATES = frozenset({
    "exact_match",
    "semantically_equivalent",
    "expected_divergence",
    "unreproducible",
    "invalid_replay",
})

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_AUDIT_PATH = ROOT / "data" / "audit" / "audit.jsonl"
DEFAULT_REGISTRY_PATH = ROOT / "data" / "registry" / "schemes.json"

# Fields legitimately volatile between original and replay. Stripped
# recursively from BOTH payloads before semantic comparison. Keep this
# set small and well-justified — over-stripping hides real divergence.
#
#   computed_at, snapshot_date, generated_at — wall-clock identifiers.
#   registry_path, snapshots_dir            — environmental paths, not content.
_VOLATILE_KEYS = frozenset({
    "computed_at",
    "snapshot_date",
    "generated_at",
    "registry_path",
    "snapshots_dir",
})


# ── Exceptions ───────────────────────────────────────────────────────


class EvidenceHashMismatch(RuntimeError):
    """Evidence file's recomputed sha256 disagrees with the audit row's
    recorded evidence_ref.sha256. Strong signal of tamper or filesystem
    corruption — replay classifies as invalid_replay rather than try to
    proceed with suspect content."""


# ── Public helpers ───────────────────────────────────────────────────


def find_record_by_run_id(audit_path: Path, run_id: str) -> Optional[dict]:
    """Linear scan of the JSONL chain; first match wins.

    run_ids are UUID4 per the envelope spec; collisions across the chain
    are not expected. Single-machine + bounded chain size makes a
    forward scan acceptable (no need for an index). Returns ``None``
    when the run_id is absent OR the audit file does not exist.

    Malformed lines (legacy or corrupted rows) are skipped silently —
    chain-validity checks are the responsibility of ``verify_chain``.
    """
    if not audit_path.exists():
        return None
    with audit_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event", {}).get("run_id") == run_id:
                return record
    return None


def load_and_verify_evidence(evidence_ref: dict, data_dir: Path) -> dict:
    """Load an evidence file and verify its sha256 matches the audit ref.

    ``evidence_ref`` is the dict stored in the audit row:
      ``{"path": "<rel-to-data/>", "sha256": "<hex>", "size_bytes": int}``.
    ``write_evidence`` always stores ``path`` relative to ``data/``
    (e.g. ``"evidence/watchlist_run/<run_id>.json"``), so resolution
    is just ``data_dir / path``. Absolute paths pass through.

    Raises ``EvidenceHashMismatch`` if the file's recomputed sha256
    disagrees with ``evidence_ref["sha256"]``. Raises ``FileNotFoundError``
    if the file is absent. Callers map these to invalid_replay /
    unreproducible respectively.
    """
    rel = evidence_ref["path"]
    p = Path(rel)
    if not p.is_absolute():
        p = data_dir / rel
    if not p.exists():
        raise FileNotFoundError(p)
    raw_bytes = p.read_bytes()
    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha != evidence_ref.get("sha256"):
        raise EvidenceHashMismatch(
            f"Evidence file {p} sha256={actual_sha} does not match "
            f"audit ref sha256={evidence_ref.get('sha256')}"
        )
    return json.loads(raw_bytes.decode("utf-8"))


# ── Comparison helpers ───────────────────────────────────────────────


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _strip_volatile(value: Any) -> Any:
    """Recursively remove ``_VOLATILE_KEYS`` from nested dicts/lists.

    Returns a new structure; does not mutate input. Lists preserve
    order — order is content-significant in ranking outputs.
    """
    if isinstance(value, dict):
        return {
            k: _strip_volatile(v)
            for k, v in value.items()
            if k not in _VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [_strip_volatile(item) for item in value]
    return value


def _top_level_diff_keys(a: Any, b: Any) -> List[str]:
    """Return the set of top-level keys whose values differ between
    two (already-stripped) dicts. Non-dict inputs yield ``["<root>"]``
    to mark a structural divergence."""
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return ["<root>"]
    keys = set(a.keys()) | set(b.keys())
    return sorted(k for k in keys if a.get(k) != b.get(k))


def _identify_divergence_drivers(
    recorded_envelope: dict,
    current_inputs: dict,
    current_methodology: dict,
) -> List[dict]:
    """Compare recorded provenance vs current execution context.

    Returns a list of typed drivers in deterministic order:
      methodology_changed, code_sha_changed, registry_hash_changed,
      cache_fingerprint_changed. Each driver carries from/to values
      so the audit row is self-describing.

    ``cache_fingerprint_changed`` is labeled forensic_only because the
    fingerprint includes mtime — it diverges trivially across machines
    and time and is not a load-bearing change signal.
    """
    drivers: List[dict] = []
    rec_methodology = recorded_envelope.get("methodology", {})
    if rec_methodology and rec_methodology != current_methodology:
        drivers.append({
            "kind": "methodology_changed",
            "from": rec_methodology,
            "to": current_methodology,
        })

    # For provenance fields that can legitimately be None (the emit
    # path did not capture them), only flag a CHANGE when both sides
    # carry non-None values. An asymmetric None means "one side did
    # not capture this" — that's a capture-policy mismatch, not a
    # content change, and surfacing it as a driver produces noise
    # that operators have to wade through.
    rec_inputs = recorded_envelope.get("inputs", {})
    for key, kind, note in (
        ("code_sha", "code_sha_changed", None),
        ("registry_hash", "registry_hash_changed", None),
        ("cache_fingerprint", "cache_fingerprint_changed",
         "forensic_only — mtime-based, not load-bearing"),
    ):
        rec_val = rec_inputs.get(key)
        cur_val = current_inputs.get(key)
        if rec_val is None or cur_val is None:
            continue
        if rec_val != cur_val:
            entry: Dict[str, Any] = {"kind": kind, "from": rec_val, "to": cur_val}
            if note:
                entry["note"] = note
            drivers.append(entry)
    return drivers


# Payload-level driver registry. Maps a top-level payload key to the
# driver kind that should fire when the recorded vs current values
# differ. Used by ``_classify`` AFTER envelope-level drivers are
# checked. Payload-level fields are evidence-kind-specific
# provenance that downstream tooling treats as ground truth, so
# differences must surface as typed drivers rather than collapse to
# generic "no driver".
#
# Step 11 adds ``calibration_engine_version`` so a calibration
# methodology bump (engine v1 → v2) is independently observable
# from a taxonomy bump or a regime classifier bump. Each is a
# distinct drift surface; collapsing them into a generic
# "methodology_changed" would erase WHICH inference layer drifted.
_PAYLOAD_LEVEL_DRIVERS: Dict[str, str] = {
    "taxonomy_version":              "taxonomy_changed",
    "regime_classifier_version":     "classifier_methodology_changed",
    "calibration_engine_version":    "calibration_methodology_changed",
}


def _identify_payload_drivers(
    recorded_payload: Any, current_payload: Any,
) -> List[dict]:
    """Surface payload-level provenance drifts (taxonomy / classifier
    versions) as typed drivers. Only fires when BOTH sides carry the
    field — asymmetric None is treated as "this evidence_kind doesn't
    expose that surface" rather than as drift, matching the existing
    convention from envelope-level drivers."""
    if not (isinstance(recorded_payload, dict)
            and isinstance(current_payload, dict)):
        return []
    out: List[dict] = []
    for key, kind in _PAYLOAD_LEVEL_DRIVERS.items():
        rec_val = recorded_payload.get(key)
        cur_val = current_payload.get(key)
        if rec_val is None or cur_val is None:
            continue
        if rec_val != cur_val:
            out.append({"kind": kind, "from": rec_val, "to": cur_val})
    return out


def _classify(
    recorded_envelope: dict,
    current_payload: dict,
    registry_path_for_provenance: Path,
    audit_path: Optional[Path] = None,
) -> dict:
    """Decide which of the five replay states applies and build the
    payload that ``replay_run`` returns and emits.

    State decision table:
      byte-equal canonical JSON                → exact_match
      volatile-stripped equal                  → semantically_equivalent
      stripped differs + ≥1 driver             → expected_divergence
      stripped differs + zero drivers          → invalid_replay (per default)

    ``audit_path`` is consulted for kind-specific driver detection
    (currently regime_dependency_superseded on calibration_report
    payloads). Backward-compatible default of None preserves
    callers that don't have an audit_path handy.
    """
    from backend.investment_analytics.methodology import current_methodology
    from backend.investment_analytics.provenance import capture_provenance_inputs

    recorded_payload = recorded_envelope.get("payload", {})
    rec_sanitized = sanitize_audit_event(recorded_payload)
    cur_sanitized = sanitize_audit_event(current_payload)

    rec_canonical = _canonical_json(rec_sanitized)
    cur_canonical = _canonical_json(cur_sanitized)
    recorded_sha = _sha256_hex(rec_canonical)
    current_sha = _sha256_hex(cur_canonical)

    base = {
        "recorded": {
            "sha256": recorded_sha,
            "size_bytes": len(rec_canonical.encode("utf-8")),
        },
        "current": {
            "sha256": current_sha,
            "size_bytes": len(cur_canonical.encode("utf-8")),
        },
        "differences": [],
        "divergence_drivers": [],
        "missing_inputs": [],
    }

    if rec_canonical == cur_canonical:
        return {
            **base,
            "state": "exact_match",
            "reason": "byte-equal canonical JSON of sanitized payloads",
        }

    rec_stripped = _strip_volatile(rec_sanitized)
    cur_stripped = _strip_volatile(cur_sanitized)
    if rec_stripped == cur_stripped:
        return {
            **base,
            "state": "semantically_equivalent",
            "reason": "differs only in volatile fields (timestamps, paths)",
            "differences": sorted(_VOLATILE_KEYS & (
                set(rec_sanitized.keys()) | set(cur_sanitized.keys())
            )) if (isinstance(rec_sanitized, dict) and isinstance(cur_sanitized, dict)) else [],
        }

    # Stripped content differs. Look for drivers in provenance.
    drivers = _identify_divergence_drivers(
        recorded_envelope=recorded_envelope,
        current_inputs=capture_provenance_inputs(registry_path_for_provenance),
        current_methodology=current_methodology(),
    )
    # Payload-level drivers — evidence-kind-specific provenance that
    # lives in the heavy payload rather than the envelope. Step 10
    # introduced ``taxonomy_version`` and ``regime_classifier_version``
    # on regime_summary and drift_analysis payloads; these are two
    # distinct drift surfaces (ontology vs methodology) and must be
    # distinguishable on replay rather than collapsing into a generic
    # "differs without driver" → invalid_replay outcome.
    drivers.extend(_identify_payload_drivers(rec_sanitized, cur_sanitized))
    # Per-kind custom drivers: walks the chain to detect supersession
    # of regime_summary rows that the recorded payload depends on.
    # First-class driver kept distinct from generic methodology drift
    # because the remediation paths differ (refresh the calibration
    # vs revisit the regime classifier).
    if audit_path is not None and isinstance(recorded_payload, dict):
        derived_from = recorded_payload.get("derived_from_run_ids") or []
        if derived_from:
            drivers.extend(_detect_regime_dependency_superseded(
                audit_path, list(derived_from),
            ))
    diff_keys = _top_level_diff_keys(rec_stripped, cur_stripped)

    if drivers:
        return {
            **base,
            "state": "expected_divergence",
            "reason": f"content differs; {len(drivers)} driver(s) identified",
            "differences": diff_keys,
            "divergence_drivers": drivers,
        }

    return {
        **base,
        "state": "invalid_replay",
        "reason": "content differs and no identified driver explains it",
        "differences": diff_keys,
    }


# ── Handlers (one per replayable evidence_kind) ──────────────────────


def _replay_ranking(
    audit_event: dict,
    recorded_envelope: dict,
    registry_path: Path,
) -> dict:
    """Replay a ranking_snapshot. Dispatches on event_type."""
    event_type = audit_event["event_type"]
    reg = str(registry_path)
    # Lazy imports so test patches against the ranking package's parent-
    # exported names resolve correctly (preserves the established seam
    # documented in backend/investment_analytics/ranking/__init__.py).
    from backend.investment_analytics.ranking import (
        all_assets_to_dict,
        multi_ranking_to_dict,
        rank_all_assets,
        rank_all_categories,
        rank_category,
        ranking_to_dict,
    )

    if event_type == "rank_category":
        category = audit_event["category"]
        result = rank_category(category, reg)
        return ranking_to_dict(result)

    if event_type == "rank_all_categories":
        top_n = int(audit_event.get("top_n", 0))
        recorded_payload = recorded_envelope.get("payload", {})
        # Recover the category filter from the recorded payload's
        # category_results map (the multi-ranking payload key the
        # original wrote). If absent, replay with no filter (full set).
        categories = list(recorded_payload.get("category_results", {}).keys()) or None
        result = rank_all_categories(
            registry_path=reg, top_n=top_n, categories=categories,
        )
        return multi_ranking_to_dict(result)

    if event_type == "rank_all_assets":
        top_n = int(audit_event.get("top_n", 0))
        result = rank_all_assets(registry_path=reg, top_n=top_n)
        return all_assets_to_dict(result)

    raise ValueError(f"Unsupported ranking event_type for replay: {event_type!r}")


def _replay_portfolio_health(
    audit_event: dict,
    recorded_envelope: dict,
    registry_path: Path,
) -> dict:
    """Replay a portfolio_health_snapshot.

    Reconstructs scheme_codes from ``holdings + not_found`` in the
    recorded payload. Weights are reconstructed from
    ``decision_summary[*].weight_pct`` (normalized form — the same
    form ``check_portfolio_health`` accepts and round-trips through
    ``portfolio_health_to_dict``).
    """
    from backend.investment_analytics.portfolio_health import check_portfolio_health
    from backend.investment_analytics.portfolio_health.serializer import (
        portfolio_health_to_dict,
    )

    recorded_payload = recorded_envelope.get("payload", {})
    holdings = recorded_payload.get("holdings", []) or []
    not_found = recorded_payload.get("not_found", []) or []
    scheme_codes: List[int] = [int(h["scheme_code"]) for h in holdings]
    scheme_codes += [int(n) if not isinstance(n, dict) else int(n["scheme_code"])
                     for n in not_found]

    # Weights from decision_summary, summed across buckets — normalized
    # to 0..1 from the recorded weight_pct.
    weights: Dict[int, float] = {}
    decision_summary = recorded_payload.get("decision_summary", {}) or {}
    for bucket_entries in decision_summary.values():
        for entry in bucket_entries:
            try:
                weights[int(entry["scheme_code"])] = float(entry["weight_pct"]) / 100.0
            except (KeyError, TypeError, ValueError):
                continue
    # not_found entries get zero weight by definition — they were
    # never analyzed in the original run.

    result = check_portfolio_health(
        scheme_codes=scheme_codes,
        weights=weights or None,
        registry_path=str(registry_path),
    )
    return portfolio_health_to_dict(result, weights or None)


def _replay_watchlist_run(
    audit_event: dict,
    recorded_envelope: dict,
    registry_path: Path,
) -> dict:
    """Replay a watchlist_run.

    Invokes the leaf ``rank_*`` functions directly per the recorded
    config — does NOT call ``run_snapshot`` (which would emit a real
    watchlist_run audit row, write snapshot files, and update the
    sidecar). Comparison is at the per-category summary level; the
    full per-category ranking content lives in ``data/snapshots/``
    files, not in the watchlist evidence payload.
    """
    from backend.investment_analytics.ranking import (
        EXCLUDED_CATEGORIES,
        rank_category,
        rank_debt_category,
        rank_gold_funds,
    )
    from backend.jobs.watchlist import _validate_registry

    recorded_payload = recorded_envelope.get("payload", {})
    cfg = recorded_payload.get("config", {}) or {}
    snap_date = recorded_payload.get("snapshot_date")
    reg = str(registry_path)

    summary: Dict[str, str] = {}
    registry_problem = _validate_registry(reg)
    if registry_problem is not None:
        summary["__registry__"] = f"error: {registry_problem}"
    else:
        for cat in cfg.get("equity_categories", []) or []:
            if cat in EXCLUDED_CATEGORIES:
                summary[cat] = "excluded by ranking policy"
                continue
            try:
                rank_category(cat, reg)
                summary[cat] = "ok"
            except Exception as exc:  # noqa: BLE001
                summary[cat] = f"error: {exc}"
        for cat in cfg.get("debt_categories", []) or []:
            try:
                rank_debt_category(cat, reg)
                summary[cat] = "ok"
            except Exception as exc:  # noqa: BLE001
                summary[cat] = f"error: {exc}"
        if cfg.get("include_gold", False):
            cat = "Gold Fund (FoF)"
            try:
                rank_gold_funds(reg)
                summary[cat] = "ok"
            except Exception as exc:  # noqa: BLE001
                summary[cat] = f"error: {exc}"

    return {
        "snapshot_date": snap_date,
        "config": cfg,
        "summary": summary,
        "registry_path": reg,
        "snapshots_dir": recorded_payload.get("snapshots_dir"),
    }


def _replay_experiment_run(
    audit_event: dict,
    recorded_envelope: dict,
    registry_path: Path,
) -> dict:
    """Replay an experiment_run.

    Reconstructs the recorded ExperimentConfig, verifies the registry
    contract hasn't drifted (target / allowed_param_keys / callable
    signature), and re-invokes the parameterized callable with the
    recorded inputs + overrides against the current registry. Step 7's
    ``_classify`` handles equality classification on the returned
    payload.

    Raises ``RuntimeError`` with a contract-diff message when the
    registry contract has changed between record and replay — this
    bubbles to ``invalid_replay`` per the surrounding handler-raise
    contract. Diff components surface WHICH dimension drifted (target /
    allowed_param_keys / callable_signature), giving operators the
    forensic signal the simple fingerprint alone would not.
    """
    from backend.experiments.config import ExperimentConfig
    from backend.experiments.registry import (
        REGISTERED_PARAMETERIZED_FUNCS,
        registry_contract,
    )

    recorded_payload = recorded_envelope.get("payload", {})
    recorded_config_dict = recorded_payload.get("config", {})
    recorded_contract = recorded_payload.get("registry_contract", {})
    target = recorded_config_dict.get("target")

    if target not in REGISTERED_PARAMETERIZED_FUNCS:
        raise RuntimeError(
            f"replay refused: target {target!r} is no longer registered. "
            f"current registry: {sorted(REGISTERED_PARAMETERIZED_FUNCS)}"
        )

    current_contract = registry_contract(target)
    drift_diffs = []
    for key in ("target", "allowed_param_keys", "callable_signature"):
        rec_val = recorded_contract.get(key)
        cur_val = current_contract.get(key)
        if rec_val != cur_val:
            drift_diffs.append(f"{key}: recorded={rec_val!r} current={cur_val!r}")
    if drift_diffs:
        raise RuntimeError(
            "replay refused: registry_contract drift detected — "
            + "; ".join(drift_diffs)
        )

    # Reconstruct the config and re-run. The ExperimentConfig validator
    # protects us against malformed recorded payloads (impossible if
    # they were emitted by this codebase, but cheap insurance).
    config = ExperimentConfig(
        target=target,
        target_inputs=recorded_config_dict.get("target_inputs", {}),
        param_overrides=recorded_config_dict.get("param_overrides", {}),
        methodology_kind=recorded_config_dict.get("methodology_kind"),
        experiment_status=recorded_config_dict.get("experiment_status"),
        derived_from_run_ids=tuple(
            recorded_config_dict.get("derived_from_run_ids", [])
        ),
        non_semantic_metadata=recorded_config_dict.get("non_semantic_metadata", {}),
    )

    entry = REGISTERED_PARAMETERIZED_FUNCS[target]
    call_kwargs = dict(config.target_inputs)
    call_kwargs.setdefault("registry_path", str(registry_path))
    call_kwargs.update(config.param_overrides)
    result = entry.callable(**call_kwargs)
    output_payload = entry.serializer(result)

    # Return the same payload shape the runner emits so _classify can
    # compare recorded vs current at the same level of detail.
    return {
        "config":                          config.to_payload(),
        "output":                          output_payload,
        "production_methodology_versions": recorded_payload.get(
            "production_methodology_versions", {}),
        "experiment_overrides":            dict(config.param_overrides),
        "derivation_depth":                recorded_payload.get("derivation_depth", 0),
        "registry_contract":               current_contract,
        "baseline_run_id":                 recorded_payload.get("baseline_run_id"),
        "schema_version":                  recorded_payload.get(
            "schema_version", "v1"),
    }


def _replay_regime_summary(
    audit_event: dict,
    recorded_envelope: dict,
    registry_path: Path,
) -> dict:
    """Replay a regime_summary row.

    Re-invokes ``classify_window`` with the recorded window dates and
    the recorded ``applied_bands`` from ``classification_basis``. The
    cache directory is the runner's default (production cache); replay
    surfaces classifier-methodology bumps (different default bands) as
    expected_divergence via Step 7's _classify, not by failing here.

    Side-effect free: no new regime_summary is emitted, no supersedes
    chain is touched. Pure re-derivation.
    """
    from backend.regimes.classifier import (
        DEFAULT_CACHE_DIR,
        classification_to_payload,
        classify_window,
    )
    from backend.regimes.config import ClassifierParams

    recorded_payload = recorded_envelope.get("payload", {})
    basis = recorded_payload.get("classification_basis", {}) or {}
    applied_bands = basis.get("applied_bands", {}) or {}

    params = ClassifierParams(
        signal_scheme_code=basis.get("signal_scheme_code", 120716),
        low_threshold_pct=applied_bands.get("low_threshold_pct", 12.0),
        high_threshold_pct=applied_bands.get("high_threshold_pct", 20.0),
        crisis_threshold_pct=applied_bands.get("crisis_threshold_pct", 35.0),
        min_coverage_days=basis.get("min_coverage_days", 60),
        boundary_confidence_margin_pct=basis.get(
            "boundary_confidence_margin_pct", 10.0),
    )

    classification = classify_window(
        recorded_payload["window_start_date"],
        recorded_payload["window_end_date"],
        params=params,
        cache_dir=DEFAULT_CACHE_DIR,
    )
    payload = classification_to_payload(classification)
    # Preserve supersedes_run_id from recorded — it's environmental
    # lineage, not derivable from re-classification.
    payload["supersedes_run_id"] = recorded_payload.get("supersedes_run_id")
    return payload


def _detect_regime_dependency_superseded(
    audit_path: Path,
    derived_from_run_ids: List[str],
) -> List[dict]:
    """First-class replay driver for calibration_report and any
    future kind that derives from regime_summary lineage.

    Walks the chain ONCE collecting every regime_summary's
    supersedes_run_id field, then checks whether any cited
    derived_from_run_ids has been superseded by a newer claim.

    Returns a list of driver dicts (one per superseded dependency)
    so calibration consumers can see WHICH regime claim was
    invalidated, not just THAT something was invalidated. Calibration
    authority is contingent on the regime claims it was built on —
    surfacing the specific superseded link is the forensic value
    operators need to decide whether to refresh the calibration.

    Intentionally NOT collapsed into the generic
    ``classifier_methodology_changed`` driver: a calibration whose
    regime CLAIMS were superseded is operationally different from
    a calibration whose regime METHODOLOGY drifted. The first means
    "the partition you used is no longer the current partition";
    the second means "the rules that produce partitions have
    themselves changed." Different remediation paths.
    """
    if not derived_from_run_ids:
        return []
    if not audit_path.exists():
        return []
    # Build superseded-by index in one chain pass.
    superseded_by: Dict[str, str] = {}
    with audit_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = record.get("event", {})
            if event.get("evidence_kind") != "regime_summary":
                continue
            target = event.get("supersedes_run_id")
            if target:
                superseded_by[target] = event["run_id"]
    drivers: List[dict] = []
    for cited in derived_from_run_ids:
        if cited in superseded_by:
            drivers.append({
                "kind":            "regime_dependency_superseded",
                "superseded_run_id":  cited,
                "superseded_by_run_id": superseded_by[cited],
                "note": (
                    "calibration authority is contingent on the cited "
                    "regime_summary; that claim has been superseded by "
                    "a newer regime methodology"
                ),
            })
    return drivers


def _replay_calibration_report(
    audit_event: dict,
    recorded_envelope: dict,
    registry_path: Path,
) -> dict:
    """Replay a calibration_report row.

    Re-runs the calibration engine for the same target with the
    same percentile against the CURRENT chain state. Step 7's
    ``_classify`` then compares recorded vs current and surfaces
    drift via:

      - calibration_methodology_changed (payload-level driver)
      - regime_dependency_superseded (custom driver appended by
        the runner via _detect_regime_dependency_superseded)

    Side-effect free: emit_audit is False through the lazy
    re-invocation path; the replay handler does NOT itself append a
    new calibration_report. The replay_result audit row that
    documents the replay is emitted by ``replay_run`` (existing
    Step 7 machinery), not by this handler.
    """
    from backend.calibration.runner import run_calibration

    recorded_payload = recorded_envelope.get("payload", {})
    target = recorded_payload.get("target_canonical_id") or recorded_payload.get("target")
    basis = recorded_payload.get("calibration_basis") or {}
    percentile = basis.get("percentile_used")

    # The replay's re-derivation runs against the live chain at
    # replay time — that's the whole point. ``run_calibration``
    # would emit a NEW calibration_report; for replay we need the
    # payload WITHOUT emit. Reuse the assembly path directly so
    # replay does not mutate the chain.
    return _rebuild_calibration_payload_for_replay(
        target=target,
        percentile=percentile,
        audit_path_for_chain=_chain_path_from_evidence_path(
            recorded_envelope, registry_path,
        ),
    )


def _chain_path_from_evidence_path(
    recorded_envelope: dict, registry_path: Path,
) -> Path:
    """Resolve the audit chain path that the replay handler should
    consult. The runner's default chain lives at
    ``<repo>/data/audit/audit.jsonl``; tests redirect via env or
    monkey-patching. We trust the runner's default unless a future
    extension passes the path explicitly through replay_run."""
    from backend.calibration.runner import DEFAULT_AUDIT_PATH
    return DEFAULT_AUDIT_PATH


def _rebuild_calibration_payload_for_replay(
    target: str,
    percentile: Optional[float],
    audit_path_for_chain: Path,
) -> dict:
    """Run the same code path as the runner but skip the emit step.
    Returns the payload that WOULD have been emitted — Step 7's
    classifier compares this against the recorded payload."""
    from backend.calibration import config as cal_config
    from backend.calibration.runner import (
        _build_basis, _empty_basis, _recommendation_payload,
        _refusal_payload,
    )
    from backend.calibration.sampling import (
        assemble_per_regime_samples, passes_coverage_floors,
    )
    from backend.calibration.targets import (
        DEFAULT_CACHE_DIR, REGISTERED_TARGETS, get_target,
    )
    from backend.calibration.percentiles import (
        unweighted_median, weighted_percentile,
    )

    if (target not in cal_config.CALIBRATION_TARGETS
            or target not in REGISTERED_TARGETS):
        return _refusal_payload(
            target_id=target, refusal_reason="target_unsupported",
            percentile=percentile or 95.0,
            basis=_empty_basis(target, percentile or 95.0),
            derived_from_run_ids=[], derivation_depth=0,
        )
    target_obj = get_target(target)
    pctl = percentile if percentile is not None else target_obj.default_percentile

    buckets, consulted = assemble_per_regime_samples(
        audit_path=audit_path_for_chain,
        target=target_obj,
        cache_dir=DEFAULT_CACHE_DIR,
    )
    if not buckets:
        return _refusal_payload(
            target_id=target, refusal_reason="insufficient_substrate",
            percentile=pctl,
            basis=_build_basis(
                target_id=target, percentile=pctl,
                buckets=[], consulted=consulted, samples_by_regime={},
            ),
            derived_from_run_ids=[], derivation_depth=0,
        )

    valid_signatures: List[dict] = []
    excluded_signatures: List[dict] = []
    per_regime_recommendations: List[float] = []
    samples_by_regime: Dict[str, dict] = {}
    derived_from_run_ids: List[str] = []
    for bucket in buckets:
        sig_key = json.dumps(bucket.regime_signature,
                             sort_keys=True, separators=(",", ":"))
        passed, refusal = passes_coverage_floors(bucket)
        per_regime_entry: Dict[str, Any] = {
            "regime_class":         bucket.regime_class,
            "n":                    bucket.raw_observation_count,
            "effective_weight":     bucket.effective_weight_total,
            "coverage_quality_mix": bucket.coverage_quality_mix,
            "contributing_run_ids": list(bucket.contributing_run_ids),
        }
        if passed:
            v = round(weighted_percentile(list(bucket.samples), pctl), 6)
            per_regime_entry["percentile_value"] = v
            per_regime_entry["status"] = "included"
            valid_signatures.append(bucket.regime_signature)
            per_regime_recommendations.append(v)
            derived_from_run_ids.extend(bucket.contributing_run_ids)
        else:
            per_regime_entry["percentile_value"] = None
            per_regime_entry["status"] = "excluded"
            per_regime_entry["per_regime_refusal_reason"] = refusal
            excluded_signatures.append(bucket.regime_signature)
        samples_by_regime[sig_key] = per_regime_entry

    basis = _build_basis(
        target_id=target, percentile=pctl,
        buckets=buckets, consulted=consulted,
        samples_by_regime=samples_by_regime,
    )

    if not valid_signatures:
        all_failed_on_confidence = all(
            samples_by_regime[k].get("per_regime_refusal_reason")
            == "confidence_floor_unmet"
            for k in samples_by_regime
        )
        aggregate_refusal = (
            "confidence_floor_unmet" if all_failed_on_confidence
            else "insufficient_coverage"
        )
        payload = _refusal_payload(
            target_id=target, refusal_reason=aggregate_refusal,
            percentile=pctl, basis=basis,
            derived_from_run_ids=[], derivation_depth=0,
        )
        payload["calibration_scope"]["excluded_regimes"] = excluded_signatures
        return payload

    return _recommendation_payload(
        target_id=target,
        recommendation=round(unweighted_median(per_regime_recommendations), 6),
        percentile=pctl,
        valid_signatures=valid_signatures,
        excluded_signatures=excluded_signatures,
        basis=basis,
        derived_from_run_ids=sorted(set(derived_from_run_ids)),
        derivation_depth=1,
    )


def _replay_drift_analysis(
    audit_event: dict,
    recorded_envelope: dict,
    registry_path: Path,
) -> dict:
    """Replay a drift_analysis row.

    Re-classifies both windows under current defaults, recomputes the
    drift composition, returns the typed payload. Methodology drift
    between record and replay surfaces via Step 7's standard
    classification machinery.
    """
    from backend.regimes.classifier import (
        DEFAULT_CACHE_DIR,
        classify_window,
    )
    from backend.regimes.drift import compute_drift, drift_to_payload

    recorded_payload = recorded_envelope.get("payload", {})
    window_a = recorded_payload.get("window_a", {}) or {}
    window_b = recorded_payload.get("window_b", {}) or {}

    a = classify_window(
        window_a["start_date"], window_a["end_date"],
        cache_dir=DEFAULT_CACHE_DIR,
    )
    b = classify_window(
        window_b["start_date"], window_b["end_date"],
        cache_dir=DEFAULT_CACHE_DIR,
    )
    drift = compute_drift(
        a, b,
        drift_kind=recorded_payload.get("drift_kind", "rolling_vol_shift"),
        signal_kind=recorded_payload.get("signal_kind", "nifty50_realized_vol"),
    )
    return drift_to_payload(
        drift,
        window_a_run_id=window_a.get("regime_summary_run_id"),
        window_b_run_id=window_b.get("regime_summary_run_id"),
    )


def _replay_threshold_recommendation(
    audit_event: dict,
    recorded_envelope: dict,
    registry_path: Path,
) -> dict:
    """Replay a threshold_recommendation row.

    Re-locates the cited calibration_report and re-projects its
    recommendation into the same payload shape. Step 7's
    ``_classify`` then compares recorded vs current.

    ## Load-bearing refusal-projection guard (mirrors emit-side)

    If the cited calibration_report has become (or always was) a
    typed REFUSAL (recommendation=null), the handler RAISES. Step 7's
    handler-raises-→-invalid_replay contract maps this to a typed
    invalid_replay state. This keeps the anti-evidence-laundering
    invariant honored at REPLAY as well as at EMIT: a typed wrapper
    cannot turn a non-recommendation into a recommendation, at any
    layer or at any point in time.
    """
    from backend.calibration.config import CALIBRATION_TARGETS
    from backend.research_artifacts.runner import (
        _load_evidence_payload,
        DEFAULT_AUDIT_PATH,
    )

    recorded_payload = recorded_envelope.get("payload", {})
    calibration_run_id = recorded_payload.get(
        "derived_from_calibration_report_run_id",
    )
    if not calibration_run_id:
        raise RuntimeError(
            "replay refused: recorded threshold_recommendation has no "
            "derived_from_calibration_report_run_id"
        )

    cal_record = find_record_by_run_id(DEFAULT_AUDIT_PATH, calibration_run_id)
    if cal_record is None:
        raise RuntimeError(
            f"replay refused: cited calibration_report "
            f"{calibration_run_id!r} not found in current chain"
        )
    cal_event = cal_record.get("event", {})
    if cal_event.get("evidence_kind") != "calibration_report":
        raise RuntimeError(
            f"replay refused: cited run_id {calibration_run_id!r} has "
            f"evidence_kind={cal_event.get('evidence_kind')!r}, "
            f"expected 'calibration_report'"
        )

    # Refusal-projection guard — same rule as emit. A calibration
    # that has become / always was a typed refusal cannot be
    # projected into a threshold_recommendation.
    if cal_event.get("recommendation") is None:
        raise RuntimeError(
            f"replay refused: cited calibration {calibration_run_id!r} "
            f"is a typed refusal (refusal_reason="
            f"{cal_event.get('refusal_reason')!r}); "
            f"threshold_recommendation cannot be projected from a "
            f"non-recommendation"
        )

    target = cal_event.get("target")
    if target not in CALIBRATION_TARGETS:
        raise RuntimeError(
            f"replay refused: cited calibration targets {target!r}, "
            f"not in CALIBRATION_TARGETS"
        )

    cal_evidence_ref = cal_event.get("evidence_ref")
    if not cal_evidence_ref:
        raise RuntimeError(
            "replay refused: cited calibration_report has no evidence_ref"
        )
    cal_payload = _load_evidence_payload(
        DEFAULT_AUDIT_PATH, cal_evidence_ref,
    )

    recommended_value = cal_payload.get("recommendation")
    if recommended_value is None:
        raise RuntimeError(
            "replay refused: audit/evidence mismatch — calibration row "
            "shows non-null recommendation but evidence payload shows null"
        )

    cal_scope = cal_payload.get("calibration_scope", {}) or {}
    recommendation_scope = {
        "valid_within_regimes":  cal_scope.get("valid_within_regimes", []),
        "assumed_stationarity":  cal_scope.get(
            "assumed_stationarity",
            "(not recorded on cited calibration)",
        ),
        "known_limitations":     cal_scope.get("known_limitations", []),
    }

    # Preserve recorded environmental fields the replay can't
    # recompute deterministically.
    return {
        "schema_version":         recorded_payload.get(
            "schema_version", "v1"),
        "target_canonical_id":    target,
        "recommended_value":      recommended_value,
        "recommendation_scope":   recommendation_scope,
        "adoption_status":        recorded_payload.get(
            "adoption_status", "proposed"),
        "derived_from_calibration_report_run_id": calibration_run_id,
        "supersedes_run_id":      recorded_payload.get("supersedes_run_id"),
        "methodology_kind":       "data_driven_variant",
        "non_semantic_metadata":  recorded_payload.get(
            "non_semantic_metadata", {}),
    }


def _replay_reliability_score(
    audit_event: dict,
    recorded_envelope: dict,
    registry_path: Path,
) -> dict:
    """Replay a reliability_score row.

    Re-runs the scoring engine against the CURRENT chain with the
    recorded ``scoring_window_days`` and ``target_run_id``. Step 7's
    ``_classify`` then compares recorded vs current.

    Reliability scores legitimately drift between record and replay
    as new evidence accumulates (more recent rows shift the
    histograms). The replay handler does NOT pin a frozen window —
    it re-derives against the current chain. Drift surfaces via
    existing drivers:
      - ``methodology_changed`` if reliability_weighting bumped
      - ``calibration_methodology_changed`` if engine bumped
      - ``regime_dependency_superseded`` if consulted regime rows
        were superseded after the original score

    Side-effect-free — does NOT emit a new reliability_score row.
    Step 7's existing replay_run is the only writer.
    """
    from backend.reliability.runner import (
        DEFAULT_AUDIT_PATH as RELIABILITY_AUDIT_PATH,
        score_target,
    )

    recorded_payload = recorded_envelope.get("payload", {})
    target_canonical_id = recorded_payload.get("target_canonical_id")
    target_run_id = recorded_payload.get("target_run_id")
    window_days = recorded_payload.get("scoring_window_days", 90)
    if not target_canonical_id or not target_run_id:
        raise RuntimeError(
            "replay refused: recorded reliability_score missing "
            "target_canonical_id or target_run_id"
        )
    # Re-score with emit=False — we want the payload, not a new row.
    result = score_target(
        target_canonical_id=target_canonical_id,
        target_run_id=target_run_id,
        audit_path=RELIABILITY_AUDIT_PATH,
        scoring_window_days=window_days,
        supersedes_run_id=recorded_payload.get("supersedes_run_id"),
        emit=False,
    )
    return result["payload"]


ReplayHandler = Callable[[dict, dict, Path], dict]

REPLAY_HANDLERS: Dict[str, ReplayHandler] = {
    "ranking_snapshot":          _replay_ranking,
    "portfolio_health_snapshot": _replay_portfolio_health,
    "watchlist_run":             _replay_watchlist_run,
    "experiment_run":            _replay_experiment_run,
    "regime_summary":            _replay_regime_summary,
    "drift_analysis":            _replay_drift_analysis,
    "calibration_report":        _replay_calibration_report,
    "threshold_recommendation":  _replay_threshold_recommendation,
    "reliability_score":         _replay_reliability_score,
}


# ── Main entry point ─────────────────────────────────────────────────


def _unreproducible(reason: str, missing_inputs: Optional[List[str]] = None) -> dict:
    return {
        "state": "unreproducible",
        "reason": reason,
        "differences": [],
        "divergence_drivers": [],
        "missing_inputs": missing_inputs or [],
        "recorded": {},
        "current": {},
    }


def _invalid_replay(reason: str) -> dict:
    return {
        "state": "invalid_replay",
        "reason": reason,
        "differences": [],
        "divergence_drivers": [],
        "missing_inputs": [],
        "recorded": {},
        "current": {},
    }


def replay_run(
    audit_path: Path,
    run_id: str,
    *,
    verify_chain: bool = True,
    emit_audit: bool = True,
    registry_path: Optional[Path] = None,
) -> dict:
    """Replay a prior evidence-bearing run and (optionally) append a
    typed ``replay_result`` audit row.

    Args:
      audit_path:    path to ``audit.jsonl``.
      run_id:        the original event's run_id.
      verify_chain:  when True, refuse to replay against a chain that
                     fails ``verify_audit_chain`` (≪100ms on a 10MB
                     chain — safety floor outweighs the cost).
      emit_audit:    when False, classify and return WITHOUT appending
                     a replay_result row. Useful for ad-hoc / CLI
                     inspection that should not leave a trace.
      registry_path: registry to use for the rerun. Defaults to the
                     project's data/registry/schemes.json. The recorded
                     registry_path is intentionally NOT honored — that
                     path was a string from another point in time and
                     may not resolve on the current machine.

    Returns the classification dict (the same payload that's persisted
    when ``emit_audit=True``).
    """
    reg = registry_path or DEFAULT_REGISTRY_PATH

    if verify_chain and not verify_audit_chain(audit_path):
        result = _invalid_replay("audit chain verification failed")
        return _finalize(audit_path, run_id, None, result, 0, emit_audit)

    start = monotonic()
    record = find_record_by_run_id(audit_path, run_id)
    if record is None:
        result = _unreproducible(
            f"run_id {run_id!r} not found in audit chain",
            missing_inputs=["audit_record"],
        )
        return _finalize(audit_path, run_id, None, result,
                         int((monotonic() - start) * 1000), emit_audit)

    event = record.get("event", {})
    evidence_kind = event.get("evidence_kind")
    evidence_ref = event.get("evidence_ref")

    if not evidence_ref:
        result = _unreproducible(
            "audit row has no evidence_ref (audit-only event)",
            missing_inputs=["evidence_ref"],
        )
        return _finalize(audit_path, run_id, evidence_kind, result,
                         int((monotonic() - start) * 1000), emit_audit)

    handler = REPLAY_HANDLERS.get(evidence_kind)
    if handler is None:
        result = _unreproducible(
            f"no replay handler for evidence_kind={evidence_kind!r}",
        )
        return _finalize(audit_path, run_id, evidence_kind, result,
                         int((monotonic() - start) * 1000), emit_audit)

    data_dir = audit_path.parent.parent  # data/
    try:
        recorded_envelope = load_and_verify_evidence(evidence_ref, data_dir)
    except EvidenceHashMismatch as exc:
        result = _invalid_replay(f"evidence file hash mismatch: {exc}")
        return _finalize(audit_path, run_id, evidence_kind, result,
                         int((monotonic() - start) * 1000), emit_audit)
    except FileNotFoundError:
        result = _unreproducible(
            "evidence file missing on disk",
            missing_inputs=["evidence_file"],
        )
        return _finalize(audit_path, run_id, evidence_kind, result,
                         int((monotonic() - start) * 1000), emit_audit)

    # Registry missing → unreproducible (named prerequisite), not invalid_replay.
    if not Path(reg).exists():
        result = _unreproducible(
            f"registry file missing at {reg}",
            missing_inputs=["registry"],
        )
        return _finalize(audit_path, run_id, evidence_kind, result,
                         int((monotonic() - start) * 1000), emit_audit)

    # Cache tracker MUST be active so capture_provenance_inputs (called
    # inside _classify) records a real cache_fingerprint for the replay
    # run. Without this, current.cache_fingerprint would be None while
    # recorded carries a real sha — firing a spurious driver every time.
    cache_tracker.start_tracking()
    try:
        try:
            current_payload = handler(event, recorded_envelope, reg)
        except Exception as exc:  # noqa: BLE001
            result = _invalid_replay(f"replay handler raised: {exc!r}")
            return _finalize(audit_path, run_id, evidence_kind, result,
                             int((monotonic() - start) * 1000), emit_audit)

        classification = _classify(
            recorded_envelope, current_payload, reg,
            audit_path=audit_path,
        )
    finally:
        cache_tracker.stop_tracking()
    duration_ms = int((monotonic() - start) * 1000)

    classification["recorded"]["evidence_ref"] = evidence_ref
    classification["audit_event_ref"] = {
        "run_id": run_id,
        "evidence_kind": evidence_kind,
    }
    classification["duration_ms"] = duration_ms
    classification["replay_tool_version"] = REPLAY_TOOL_VERSION

    return _finalize(audit_path, run_id, evidence_kind, classification,
                     duration_ms, emit_audit)


def _finalize(
    audit_path: Path,
    original_run_id: str,
    evidence_kind: Optional[str],
    classification: dict,
    duration_ms: int,
    emit_audit: bool,
) -> dict:
    """Common tail: ensure ``audit_event_ref``, ``duration_ms``, and
    ``replay_tool_version`` are present on the classification, then
    (optionally) emit the replay_result audit row. Returns the same
    classification dict the caller will receive."""
    classification.setdefault("audit_event_ref", {
        "run_id": original_run_id,
        "evidence_kind": evidence_kind,
    })
    classification.setdefault("duration_ms", duration_ms)
    classification.setdefault("replay_tool_version", REPLAY_TOOL_VERSION)

    if not emit_audit:
        return classification

    audit_event = {
        "event_type": "replay_run",
        "audit_event_ref": classification["audit_event_ref"],
        "state": classification["state"],
        "reason": classification.get("reason"),
        "schema_version": "v1",
    }
    emit_evidence(
        audit_log_path=audit_path,
        evidence_kind="replay_result",
        audit_event=audit_event,
        payload=classification,
        parent_run_id=original_run_id,
    )
    return classification


# ── CLI ──────────────────────────────────────────────────────────────


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.evidence.replay",
        description="Replay a prior evidence-bearing audit row.",
    )
    parser.add_argument("run_id", help="Original event's run_id to replay.")
    parser.add_argument(
        "--audit-path",
        default=str(DEFAULT_AUDIT_PATH),
        help=f"Path to audit.jsonl (default: {DEFAULT_AUDIT_PATH}).",
    )
    parser.add_argument(
        "--registry-path",
        default=str(DEFAULT_REGISTRY_PATH),
        help=f"Path to registry (default: {DEFAULT_REGISTRY_PATH}).",
    )
    parser.add_argument(
        "--no-verify-chain",
        action="store_true",
        help="Skip the chain integrity check before replay.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify without appending a replay_result row.",
    )
    args = parser.parse_args(argv)

    result = replay_run(
        audit_path=Path(args.audit_path),
        run_id=args.run_id,
        verify_chain=not args.no_verify_chain,
        emit_audit=not args.dry_run,
        registry_path=Path(args.registry_path),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    # Map states to exit codes: clean outcomes = 0; problematic = nonzero
    # so ops can chain replay in scripts and react.
    return {
        "exact_match": 0,
        "semantically_equivalent": 0,
        "expected_divergence": 0,
        "unreproducible": 2,
        "invalid_replay": 3,
    }.get(result["state"], 1)


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
