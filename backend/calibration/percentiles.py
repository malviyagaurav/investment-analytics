"""Weighted percentile — the only new math Step 11 introduces.

Linear-interpolation percentile over (value, weight) pairs. Pure
function. Deterministic. No external dependencies. Fully replayable.

This is the entire "sophistication" of the calibration engine. The
day-one engine is: sort + weighted percentile + median across
regimes + refuse when coverage fails. Each step inspectable.

Resisting fancier alternatives (kernel density estimates, bootstrap
confidence intervals, Bayesian credible intervals) is deliberate —
the Step 11 win is transparency over impressiveness. Sophistication
arrives when the substrate has accumulated enough that simple
percentiles demonstrably underrepresent something measurable. Until
then, the simple computation IS the correct computation.
"""
from __future__ import annotations

from typing import Sequence, Tuple


def weighted_percentile(
    samples: Sequence[Tuple[float, float]],
    percentile: float,
) -> float:
    """Linear-interpolation percentile over (value, weight) pairs.

    Args:
      samples:    sequence of (value, weight). Weight must be >= 0.
                  Pairs with weight == 0 are ignored (contribute
                  neither to the total mass nor to the ordering).
      percentile: 0..100 inclusive.

    Returns:
      The interpolated value at the requested percentile of the
      cumulative weight distribution.

    Raises:
      ValueError: empty samples, all-zero total weight, negative
                  weight, percentile out of range.

    Determinism: result depends only on the sorted (value, weight)
    pairs and the percentile. No clock, no randomness, no rounding
    quirks beyond Python float arithmetic.
    """
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(
            f"percentile must be in [0, 100], got {percentile!r}"
        )
    if not samples:
        raise ValueError("samples must be non-empty")

    cleaned: list = []
    for v, w in samples:
        if w < 0:
            raise ValueError(f"weight must be >= 0, got {w!r}")
        if w == 0:
            continue
        cleaned.append((float(v), float(w)))

    if not cleaned:
        raise ValueError("all samples have zero weight")

    cleaned.sort(key=lambda x: x[0])
    total_weight = sum(w for _, w in cleaned)
    target = total_weight * percentile / 100.0

    if percentile == 0.0:
        return cleaned[0][0]
    if percentile == 100.0:
        return cleaned[-1][0]

    cumulative = 0.0
    for i, (v, w) in enumerate(cleaned):
        cumulative += w
        if cumulative > target:
            # Linear interpolation between this point and the prior.
            if i == 0:
                return v
            prev_v, _ = cleaned[i - 1]
            prev_cum = cumulative - w
            span = w
            t = (target - prev_cum) / span if span > 0 else 0.0
            return prev_v + t * (v - prev_v)
        if cumulative == target:
            # Exact boundary — average across to avoid the
            # degenerate "snap to lower side" that misrepresents
            # the percentile on equal-weight even-count samples.
            if i + 1 < len(cleaned):
                next_v, _ = cleaned[i + 1]
                return (v + next_v) / 2.0
            return v
    return cleaned[-1][0]


def weighted_median(samples: Sequence[Tuple[float, float]]) -> float:
    """Convenience wrapper for the p50 case. Same semantics as
    ``weighted_percentile(samples, 50.0)``."""
    return weighted_percentile(samples, 50.0)


def unweighted_median(values: Sequence[float]) -> float:
    """Plain median for the top-level cross-regime aggregation
    (median of per-regime recommendations). Pre-existing
    statistics.median would also work; this lives here so the
    cross-regime aggregation has a single named entry point that
    Step 14 reliability scoring can later target if needed."""
    if not values:
        raise ValueError("values must be non-empty")
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0
