"""Discrete control barrier for disk-shaped no-go zones.

The tool is a point in the camera plane with velocity command u (mm/s),
integrated at the control period. The barrier function for zone i is

    h_i(x) = ||x - c_i|| - R_i - m

with m the required clearance. A command is admissible when the straight
step x + Δt u keeps every h_i non-negative. That test is exact for disks.
Before the step, u is the Euclidean projection of the target-attractor
command onto the half-spaces ∇h_i · u >= -α h_i and onto the speed disk.
The half-space is the single-integrator form of the quadratic program
min ||u - u_nom||^2 subject to grad h · u + alpha h >= 0. This module does
not call a solver. Successive projection applies that half-space, and the
exact step test keeps the straight move outside every expanded disk. A
head-on attractor is then given a tangent slide so the feasible command is
not the zero vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scene_generator import MARGIN_MM, Episode, Zone, min_clearance
CONTROL_HZ = 100.0
DT_S = 1.0 / CONTROL_HZ
U_MAX_MM_S = 30.0
ATTRACT_GAIN = 4.0
CBF_ALPHA = 8.0
HORIZON_STEPS = 400


@dataclass(frozen=True)
class BarrierConfig:
    margin_mm: float = MARGIN_MM
    alpha: float = CBF_ALPHA
    u_max_mm_s: float = U_MAX_MM_S
    dt_s: float = DT_S
    gain: float = ATTRACT_GAIN
    steps: int = HORIZON_STEPS


@dataclass(frozen=True)
class Rollout:
    policy: str
    xy: np.ndarray
    clearance_mm: np.ndarray
    min_clearance_mm: float
    penetrated: bool
    margin_hold_rate: float
    final_target_mm: float
    reached: bool


@dataclass(frozen=True)
class InterlockDecision:
    hold: bool
    reason: str


def nominal_velocity(point_xy: np.ndarray, target_xy: np.ndarray, config: BarrierConfig) -> np.ndarray:
    """Saturated attractor. The straight segment is an invariant of this field."""
    delta = target_xy - point_xy
    command = config.gain * delta
    speed = float(np.linalg.norm(command))
    if speed > config.u_max_mm_s:
        command = command * (config.u_max_mm_s / speed)
    return command


def project_velocity(point_xy: np.ndarray, command: np.ndarray, zones: tuple[Zone, ...], config: BarrierConfig) -> np.ndarray:
    """Project `command` onto the CBF half-spaces and the speed limit."""
    velocity = np.array(command, dtype=float)
    speed = float(np.linalg.norm(velocity))
    if speed > config.u_max_mm_s:
        velocity *= config.u_max_mm_s / speed
    for _ in range(8):
        changed = False
        for zone in zones:
            delta = point_xy - zone.center
            dist = float(np.linalg.norm(delta))
            if dist < 1e-9:
                normal = np.array([1.0, 0.0])
                dist = 1e-9
            else:
                normal = delta / dist
            barrier = dist - (zone.radius_mm + config.margin_mm)
            bound = -config.alpha * barrier
            radial = float(normal @ velocity)
            if radial < bound:
                velocity = velocity + (bound - radial) * normal
                changed = True
        speed = float(np.linalg.norm(velocity))
        if speed > config.u_max_mm_s:
            velocity *= config.u_max_mm_s / speed
            changed = True
        if not changed:
            break
    return velocity


def max_safe_scale(point_xy: np.ndarray, velocity: np.ndarray, zones: tuple[Zone, ...], config: BarrierConfig) -> float:
    """Largest s in [0, 1] such that x + s Δt u stays outside every expanded disk."""
    displacement = velocity * config.dt_s
    scale = 1.0
    quad = float(displacement @ displacement)
    if quad < 1e-18:
        return 0.0
    for zone in zones:
        radius = zone.radius_mm + config.margin_mm
        delta = point_xy - zone.center
        linear = float(2.0 * delta @ displacement)
        constant = float(delta @ delta - radius * radius)
        if constant < 0.0:
            return 0.0
        # On the expanded boundary an inward step is not admissible.
        if constant <= 1e-8 and linear < 0.0:
            return 0.0
        discriminant = linear * linear - 4.0 * quad * constant
        if discriminant < 0.0:
            continue
        entry = (-linear - discriminant**0.5) / (2.0 * quad)
        if entry > 1e-8:
            scale = min(scale, entry * (1.0 - 1e-4))
    return max(0.0, scale)


def _circulation(point_xy: np.ndarray, target_xy: np.ndarray, zones: tuple[Zone, ...], config: BarrierConfig, command: np.ndarray) -> np.ndarray:
    """Add a tangent slide while the attractor points into a nearby disk.

    A pure projection of a head-on attractor is the zero vector, so the tip
    would stop on the near boundary. The slide is still passed through the
    half-space projection, and the exact disk test rejects any step that
    would enter an expanded zone.
    """
    if float(np.linalg.norm(command)) < 1.0:
        return command

    def barrier(zone: Zone) -> float:
        return float(np.linalg.norm(point_xy - zone.center) - zone.radius_mm - config.margin_mm)

    zone = min(zones, key=barrier)
    if barrier(zone) > 6.0:
        return command
    delta = point_xy - zone.center
    dist = float(np.linalg.norm(delta))
    if dist < 1e-9:
        return command
    normal = delta / dist
    if float(normal @ command) >= 0.0:
        return command
    tangent = np.array([-normal[1], normal[0]])
    to_target = target_xy - point_xy
    if float(to_target @ tangent) < 0.0:
        tangent = -tangent
    span = float(np.linalg.norm(to_target))
    if span < 1e-6 or abs(float(to_target @ tangent)) < 0.2 * span:
        probe = 4.0
        ahead = min_clearance(point_xy + probe * tangent, zones)
        behind = min_clearance(point_xy - probe * tangent, zones)
        if behind > ahead + 0.25:
            tangent = -tangent
    return command + config.u_max_mm_s * tangent


def barrier_velocity(point_xy: np.ndarray, target_xy: np.ndarray, zones: tuple[Zone, ...], config: BarrierConfig) -> np.ndarray:
    command = nominal_velocity(point_xy, target_xy, config)
    command = _circulation(point_xy, target_xy, zones, config, command)
    return project_velocity(point_xy, command, zones, config)


def integrate(episode: Episode, policy: str, config: BarrierConfig | None = None) -> Rollout:
    """Roll the attractor with or without the barrier. State is ground-truth pose."""
    config = config or BarrierConfig()
    if policy not in {"unconstrained", "cbf"}:
        raise ValueError(policy)
    point = episode.start.copy()
    target = episode.target
    xy = np.empty((config.steps + 1, 2), dtype=float)
    clearance = np.empty(config.steps + 1, dtype=float)
    for index in range(config.steps):
        xy[index] = point
        clearance[index] = min_clearance(point, episode.zones)
        if policy == "cbf":
            velocity = barrier_velocity(point, target, episode.zones, config)
            scale = max_safe_scale(point, velocity, episode.zones, config)
            point = point + scale * config.dt_s * velocity
        else:
            velocity = nominal_velocity(point, target, config)
            point = point + config.dt_s * velocity
    xy[-1] = point
    clearance[-1] = min_clearance(point, episode.zones)
    minimum = float(clearance.min())
    hold = float(np.mean(clearance >= config.margin_mm - 1e-6))
    final_distance = float(np.linalg.norm(xy[-1] - target))
    return Rollout(
        policy=policy,
        xy=xy,
        clearance_mm=clearance,
        min_clearance_mm=minimum,
        penetrated=bool(minimum < -1e-6),
        margin_hold_rate=hold,
        final_target_mm=final_distance,
        reached=bool(final_distance <= 2.0),
    )


def evaluate_interlock(occluded: bool, clearance_mm: float, margin_mm: float = MARGIN_MM) -> InterlockDecision:
    """Hold if the frame is rejected or the tip has already lost the margin."""
    if occluded and clearance_mm < margin_mm:
        return InterlockDecision(True, "occlusion+intrusion")
    if occluded:
        return InterlockDecision(True, "occlusion")
    if clearance_mm < margin_mm:
        return InterlockDecision(True, "intrusion")
    return InterlockDecision(False, "none")
