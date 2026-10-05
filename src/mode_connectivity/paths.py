"""Framework-agnostic mode-connectivity path math.

All functions here use only arithmetic (+, *, **) and so work identically on
Python floats, numpy arrays, and torch tensors. This is deliberate: it lets the
research-novel path/barrier logic be unit-tested with numpy on a CPU-only box,
while the same code drives torch parameter vectors on the GPU.

A "path" between two endpoints theta_a, theta_b is parameterized by t in [0, 1]:

  - linear:  gamma(t) = (1 - t) * theta_a + t * theta_b
  - bezier:  gamma(t) = (1-t)^2 * theta_a + 2(1-t)t * control + t^2 * theta_b
             (quadratic Bezier; `control` is the learnable bend / midpoint).

The *loss barrier* of a path is the maximum amount by which the realized loss
along the path exceeds the straight-line interpolation of the two endpoint
losses (Garipov et al., 2018; Frankle et al., 2020):

  barrier = max_t [ loss(t) - ((1 - t) * loss(0) + t * loss(1)) ]
"""

from __future__ import annotations

from typing import Callable, Sequence


def linear_interp(theta_a, theta_b, t):
    """Point at fraction t in [0,1] along the straight segment a->b."""
    return (1.0 - t) * theta_a + t * theta_b


def bezier_interp(theta_a, theta_b, control, t):
    """Point at fraction t along the quadratic Bezier a -> control -> b.

    At t=0 returns theta_a, at t=1 returns theta_b; `control` bends the middle.
    """
    mt = 1.0 - t
    return (mt * mt) * theta_a + (2.0 * mt * t) * control + (t * t) * theta_b


def bezier_midpoint_init(theta_a, theta_b):
    """Default Bezier control point: the straight-line midpoint (zero bend).

    With this control point the Bezier path is exactly the linear path, so a
    freshly-initialized bend starts as the linear interpolation and can only
    *reduce* the barrier from there.
    """
    return 0.5 * theta_a + 0.5 * theta_b


def linspace(n: int) -> list:
    """n evenly spaced points on [0, 1] inclusive (pure python, no numpy)."""
    if n < 2:
        raise ValueError("need at least 2 points to span [0,1]")
    return [i / (n - 1) for i in range(n)]


def endpoint_baseline(loss_start: float, loss_end: float, t: float) -> float:
    """Straight-line interpolation of the two endpoint losses at fraction t."""
    return (1.0 - t) * loss_start + t * loss_end


def barrier_from_losses(losses: Sequence[float], ts: Sequence[float]) -> float:
    """Loss barrier given sampled losses at fractions ts (must include 0 and 1).

    Returns max_t [ loss(t) - baseline(t) ], clamped at >= 0. A value near 0
    means the two endpoints are mode-connected along this path.
    """
    losses = list(losses)
    ts = list(ts)
    if len(losses) != len(ts):
        raise ValueError("losses and ts must have equal length")
    if len(losses) < 2:
        raise ValueError("need >= 2 samples")
    # endpoints: nearest samples to t=0 and t=1 (paths are sampled inclusively)
    i0 = min(range(len(ts)), key=lambda i: ts[i])
    i1 = max(range(len(ts)), key=lambda i: ts[i])
    l0, l1 = losses[i0], losses[i1]
    worst = 0.0
    for loss, t in zip(losses, ts):
        excess = loss - endpoint_baseline(l0, l1, t)
        if excess > worst:
            worst = excess
    return worst


def argmax_barrier_t(losses: Sequence[float], ts: Sequence[float]) -> float:
    """Fraction t at which the barrier (excess loss) is largest."""
    losses = list(losses)
    ts = list(ts)
    i0 = min(range(len(ts)), key=lambda i: ts[i])
    i1 = max(range(len(ts)), key=lambda i: ts[i])
    l0, l1 = losses[i0], losses[i1]
    best_t, best_excess = ts[i0], -float("inf")
    for loss, t in zip(losses, ts):
        excess = loss - endpoint_baseline(l0, l1, t)
        if excess > best_excess:
            best_excess, best_t = excess, t
    return best_t


def sample_path_losses(
    point_at: Callable[[float], object],
    loss_of: Callable[[object], float],
    n_points: int = 11,
) -> tuple[list, list]:
    """Evaluate loss at n_points along a path.

    point_at(t)  -> the parameter object (state) at fraction t
    loss_of(pt)  -> scalar loss for that parameter object
    Returns (ts, losses).
    """
    ts = linspace(n_points)
    losses = [float(loss_of(point_at(t))) for t in ts]
    return ts, losses
