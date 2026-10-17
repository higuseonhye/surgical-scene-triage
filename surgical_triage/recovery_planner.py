"""Retract the tip until every no-go disk has a recovery clearance.

The planner is a short-horizon search, not a learned diffusion policy. From
the current tip it tries a fixed set of headings, keeps the step that most
increases the worst-zone clearance, and breaks ties toward the incoming
direction so the retract follows the path that entered the field.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .safety_barrier import DT_S, U_MAX_MM_S
from .scene_generator import Zone, min_clearance

SAFE_CLEARANCE_MM = 3.0
RECOVERY_HORIZON_S = 1.5


@dataclass(frozen=True)
class RecoveryResult:
    xy: np.ndarray
    initial_clearance_mm: float
    final_clearance_mm: float
    success: bool
    steps: int


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return None
    return vector / norm


def recovery_headings(retreat: np.ndarray | None) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    headings = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    if retreat is not None:
        headings = np.vstack([retreat, headings])
    return headings


def plan_recovery(
    tip_xy: np.ndarray,
    history_xy: np.ndarray | None,
    zones: tuple[Zone, ...],
    safe_clearance_mm: float = SAFE_CLEARANCE_MM,
    horizon_s: float = RECOVERY_HORIZON_S,
    dt_s: float = DT_S,
    u_max_mm_s: float = U_MAX_MM_S,
) -> RecoveryResult:
    """Integrate the retract until `safe_clearance_mm` or the horizon ends."""
    point = np.array(tip_xy, dtype=float)
    initial = min_clearance(point, zones)
    if initial >= safe_clearance_mm:
        return RecoveryResult(
            xy=point.reshape(1, 2),
            initial_clearance_mm=initial,
            final_clearance_mm=initial,
            success=True,
            steps=0,
        )
    steps = int(round(horizon_s / dt_s))
    path = [point.copy()]
    step_length = u_max_mm_s * dt_s
    retreat = None
    if history_xy is not None and len(history_xy) >= 2:
        retreat = _unit(np.asarray(history_xy[-2], dtype=float) - np.asarray(history_xy[-1], dtype=float))
    for _ in range(steps):
        if min_clearance(point, zones) >= safe_clearance_mm:
            break
        headings = recovery_headings(retreat)
        best_point = point
        best_clearance = min_clearance(point, zones)
        best_align = -1e9
        for heading in headings:
            proposal = point + step_length * heading
            clearance = min_clearance(proposal, zones)
            align = 0.0 if retreat is None else float(heading @ retreat)
            better = clearance > best_clearance + 1e-6
            tie = abs(clearance - best_clearance) <= 1e-6 and align > best_align
            if better or tie:
                best_point = proposal
                best_clearance = clearance
                best_align = align
        if best_clearance <= min_clearance(point, zones) + 1e-6:
            break
        point = best_point
        path.append(point.copy())
    final = min_clearance(point, zones)
    stacked = np.vstack(path)
    return RecoveryResult(
        xy=stacked,
        initial_clearance_mm=initial,
        final_clearance_mm=float(final),
        success=bool(final >= safe_clearance_mm - 1e-6),
        steps=int(stacked.shape[0] - 1),
    )
