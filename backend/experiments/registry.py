"""Registry of parameterized callables — the experiment surface.

Each entry exposes a production function under a name, declares which
of its module-level parameters can be overridden, and provides the
serializer that turns the output into a payload dict.

## Day-one scope (Step 9)

ONE registered target: ``rank_category`` with two overridable
parameters (``MIN_ALIGNED_POINTS``, ``ROLLING_WINDOW_DAYS``).
Demonstrates the full chain end-to-end without trying to
parameterize everything. Adding more registered functions / more
overridable parameters is mechanical follow-up.

## Parameter-injection mechanism

Production callers continue to use ``rank_category`` unchanged — they
read module-level constants and produce ``ranking_snapshot`` evidence.

Experiments invoke ``_rank_category_with_params`` which:
  1. Acquires a module-level lock (experiments serialize against
     concurrent production calls — see concurrency note below).
  2. Temporarily swaps the module-level constants on the ``equity``
     module for the duration of the call.
  3. Calls the production ``rank_category``.
  4. Restores the originals under ``try/finally``.

The constants are bound INTO ``equity`` at import time (via
``from ... _util import MIN_ALIGNED_POINTS``), so the swap target is
``equity.MIN_ALIGNED_POINTS``, not ``_util.MIN_ALIGNED_POINTS``.

### Concurrency note

The swap is process-global. ``_EXPERIMENT_LOCK`` serializes
experiments against production callers within the same process — a
``rank_category`` request entering during an experiment will block on
the lock and see the production constants when it acquires.
Single-machine deployment + low-concurrency reality makes this
acceptable; a future refactor could replace the lock with
``contextvars`` for true per-call isolation if needed.

## registry_contract — the forensic fingerprint

For each registered target we record (target name, allowed_param_keys,
callable_signature). The sha256 of these components becomes the
``registry_contract_fingerprint`` on every emitted experiment_run.
Replay compares recorded vs current; mismatch → invalid_replay, and
the component-level diff surfaces WHICH dimension drifted.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet


# Global lock — experiments serialize against production rank_category
# calls within the same process. See module docstring concurrency note.
_EXPERIMENT_LOCK = threading.Lock()


@contextmanager
def _temp_module_attrs(module, **overrides):
    """Swap module attributes for the duration of the block.

    Always restores under finally. Skips keys whose override value is
    None (treated as "use the production default"). Raises AttributeError
    immediately if a key doesn't exist on the module — protects against
    typos that would otherwise create new attributes silently.
    """
    originals: Dict[str, Any] = {}
    try:
        for key, value in overrides.items():
            if value is None:
                continue
            if not hasattr(module, key):
                raise AttributeError(
                    f"module {module.__name__} has no attribute {key!r} — "
                    f"refusing to create new attributes via experiment override"
                )
            originals[key] = getattr(module, key)
            setattr(module, key, value)
        yield
    finally:
        for key, value in originals.items():
            setattr(module, key, value)


def _rank_category_with_params(
    category: str,
    registry_path: str,
    *,
    MIN_ALIGNED_POINTS: int | None = None,
    ROLLING_WINDOW_DAYS: int | None = None,
):
    """Production ``rank_category`` with the named constants temporarily
    overridden on the ``equity`` module. None means "do not override."

    Returns a CategoryRanking. The runner serializes via the registered
    serializer (``ranking_to_dict``) before emitting evidence.
    """
    from backend.investment_analytics.ranking import equity, rank_category

    with _EXPERIMENT_LOCK:
        with _temp_module_attrs(
            equity,
            MIN_ALIGNED_POINTS=MIN_ALIGNED_POINTS,
            ROLLING_WINDOW_DAYS=ROLLING_WINDOW_DAYS,
        ):
            return rank_category(category, registry_path)


@dataclass(frozen=True)
class ParameterizedFunc:
    """Registry entry for one parameterized callable."""
    callable: Callable
    allowed_param_keys: FrozenSet[str]
    serializer: Callable
    description: str


def _ranking_to_dict_lazy(result):
    """Lazy serializer wrapper — avoids importing ranking_to_dict at
    module load time (matches the lazy-import seam established
    elsewhere in the codebase)."""
    from backend.investment_analytics.ranking import ranking_to_dict
    return ranking_to_dict(result)


REGISTERED_PARAMETERIZED_FUNCS: Dict[str, ParameterizedFunc] = {
    "rank_category": ParameterizedFunc(
        callable=_rank_category_with_params,
        allowed_param_keys=frozenset({
            "MIN_ALIGNED_POINTS",
            "ROLLING_WINDOW_DAYS",
        }),
        serializer=_ranking_to_dict_lazy,
        description=(
            "Equity category ranking with overridable inclusion gate "
            "(MIN_ALIGNED_POINTS) and rolling-window length "
            "(ROLLING_WINDOW_DAYS)."
        ),
    ),
}


def registry_contract(target: str) -> dict:
    """Return the full contract for a registered target, including its
    sha256 fingerprint. Used by the runner to record the contract
    in-evidence and by the replay handler to detect contract drift.

    Components:
      - target name
      - sorted allowed_param_keys (registry-level surface)
      - sorted callable_signature parameter names (function-level surface)

    Both axes change independently: an operator could widen
    allowed_param_keys without changing the callable signature (loosen
    the gate), or change the signature without changing the registry
    (refactor). The fingerprint covers both so any drift surfaces.
    """
    if target not in REGISTERED_PARAMETERIZED_FUNCS:
        raise KeyError(
            f"unknown experiment target: {target!r}. "
            f"Registered targets: {sorted(REGISTERED_PARAMETERIZED_FUNCS)}"
        )
    entry = REGISTERED_PARAMETERIZED_FUNCS[target]
    sig = inspect.signature(entry.callable)
    callable_signature = sorted(sig.parameters.keys())
    contract = {
        "target":              target,
        "allowed_param_keys":  sorted(entry.allowed_param_keys),
        "callable_signature":  callable_signature,
    }
    canonical = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    )
    contract["fingerprint"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return contract
