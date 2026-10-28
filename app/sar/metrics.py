"""Statistics for comparing planners: find rates with confidence intervals, time-to-find curves, paired differences.

Times are seconds to find, or None when the run ended without a find (censored at the run length).
"""

import math

import numpy as np

Times = list[float | None]


def wilson(found: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion; well-behaved near 0 and 1, unlike the normal approximation."""
    if n == 0:
        return (0.0, 1.0)
    p = found / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def found_by(times: Times, t_s: float) -> int:
    return sum(1 for t in times if t is not None and t <= t_s)


def time_to_find_curve(times: Times) -> list[tuple[float, float]]:
    """Kaplan-Meier estimate of P(found by t), as (t_s, fraction) steps. With one shared run length every unfound run
    is censored at the end, so this equals the plain cumulative find fraction; it stays correct if run lengths differ."""
    finds = sorted(t for t in times if t is not None)
    curve, remaining = [(0.0, 0.0)], len(times)
    still_missing = 1.0
    for t in finds:
        still_missing *= 1 - 1 / remaining
        remaining -= 1
        curve.append((t, 1 - still_missing))
    return curve


def median_time(times: Times) -> float | None:
    """Time by which half the subjects were found, or None if fewer than half were."""
    return next((t for t, f in time_to_find_curve(times) if f >= 0.5), None)


def paired_difference(a: Times, b: Times, t_s: float, rng: np.random.Generator, resamples: int = 2000) -> tuple[float, float, float]:
    """Difference in find rate by `t_s` (a minus b) over the same scenarios, with a 95% bootstrap interval that
    resamples scenarios, so both planners always face the same subjects."""
    hit_a = np.array([t is not None and t <= t_s for t in a], float)
    hit_b = np.array([t is not None and t <= t_s for t in b], float)
    diff = hit_a - hit_b
    idx = rng.integers(0, len(diff), (resamples, len(diff)))
    boot = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
