"""Planar surgical field used by the benchmark.

Coordinates are millimetres in the camera plane, origin at the image centre,
+x to the right, +y up. Pixels are a rendering of that plane. Disks stand in
for structures the tool must not enter. Smoke, lens splash, blood, and a
shaft occluder are photometric overlays, not a validated optical model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

IMAGE_SIZE = 128
FIELD_MM = 80.0
TIP_SIGMA_PX = 1.6
TIP_PEAK = 1.0
MARGIN_MM = 1.0


@dataclass(frozen=True)
class Zone:
    """Closed disk the tool tip is not allowed to enter."""

    name: str
    center_xy: tuple[float, float]
    radius_mm: float

    @property
    def center(self) -> np.ndarray:
        return np.asarray(self.center_xy, dtype=float)


@dataclass(frozen=True)
class Episode:
    start_xy: tuple[float, float]
    target_xy: tuple[float, float]
    zones: tuple[Zone, ...]

    @property
    def start(self) -> np.ndarray:
        return np.asarray(self.start_xy, dtype=float)

    @property
    def target(self) -> np.ndarray:
        return np.asarray(self.target_xy, dtype=float)


@dataclass(frozen=True)
class Frame:
    gray: np.ndarray
    rgb: np.ndarray
    tip_rc: tuple[float, float]
    artifact: str


def signed_clearance(point_xy: np.ndarray, zone: Zone) -> float:
    """Positive when the point is outside the anatomical disk."""
    return float(np.linalg.norm(point_xy - zone.center) - zone.radius_mm)


def min_clearance(point_xy: np.ndarray, zones: tuple[Zone, ...]) -> float:
    return min(signed_clearance(point_xy, zone) for zone in zones)


def mm_to_rc(point_xy: np.ndarray, size: int = IMAGE_SIZE, field_mm: float = FIELD_MM) -> np.ndarray:
    col = (point_xy[0] / field_mm + 0.5) * (size - 1)
    row = (0.5 - point_xy[1] / field_mm) * (size - 1)
    return np.array([row, col], dtype=float)


def rc_grids(size: int = IMAGE_SIZE, field_mm: float = FIELD_MM) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.mgrid[0:size, 0:size]
    x_mm = (cols / (size - 1) - 0.5) * field_mm
    y_mm = (0.5 - rows / (size - 1)) * field_mm
    return x_mm.astype(np.float32), y_mm.astype(np.float32)


def stamp_tip(gray: np.ndarray, tip_rc: np.ndarray, peak: float = TIP_PEAK, sigma_px: float = TIP_SIGMA_PX) -> np.ndarray:
    """Bright specular tip written after photometric degradation.

    A tracker that matches this highlight still fires when the tissue texture
    has already been destroyed.
    """
    rows, cols = gray.shape
    radius = int(np.ceil(3.0 * sigma_px))
    row0 = int(np.floor(tip_rc[0]))
    col0 = int(np.floor(tip_rc[1]))
    r0 = max(0, row0 - radius)
    r1 = min(rows, row0 + radius + 1)
    c0 = max(0, col0 - radius)
    c1 = min(cols, col0 + radius + 1)
    if r0 >= r1 or c0 >= c1:
        return gray
    yy, xx = np.ogrid[r0:r1, c0:c1]
    dist2 = (yy - tip_rc[0]) ** 2 + (xx - tip_rc[1]) ** 2
    blob = peak * np.exp(-dist2 / (2.0 * sigma_px * sigma_px))
    out = gray.copy()
    patch = out[r0:r1, c0:c1]
    np.maximum(patch, blob.astype(patch.dtype), out=patch)
    return out


def _segment_distance(x_mm: np.ndarray, y_mm: np.ndarray, start_xy: np.ndarray, end_xy: np.ndarray) -> np.ndarray:
    delta = end_xy - start_xy
    denom = float(delta @ delta)
    if denom < 1e-12:
        return np.hypot(x_mm - start_xy[0], y_mm - start_xy[1])
    weight = ((x_mm - start_xy[0]) * delta[0] + (y_mm - start_xy[1]) * delta[1]) / denom
    weight = np.clip(weight, 0.0, 1.0)
    proj_x = start_xy[0] + weight * delta[0]
    proj_y = start_xy[1] + weight * delta[1]
    return np.hypot(x_mm - proj_x, y_mm - proj_y)


def _tissue(x_mm: np.ndarray, y_mm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    field = 0.52 + 0.07 * np.sin(x_mm * 0.55 + 0.6 * np.sin(y_mm * 0.45))
    field = field + 0.045 * np.sin(y_mm * 1.15 + 1.2)
    field = field + 0.035 * np.sin((x_mm + y_mm) * 1.8)
    field = field + rng.normal(0.0, 0.04, size=field.shape)
    return np.clip(field, 0.0, 1.0).astype(np.float32)


def _darken_disk(gray: np.ndarray, x_mm: np.ndarray, y_mm: np.ndarray, zone: Zone, amount: float = 0.22) -> np.ndarray:
    dist = np.hypot(x_mm - zone.center[0], y_mm - zone.center[1])
    edge = np.clip((zone.radius_mm + 0.8 - dist) / 0.8, 0.0, 1.0).astype(np.float32)
    return gray * (1.0 - amount * edge)


def colorize(gray: np.ndarray) -> np.ndarray:
    tissue = np.clip(gray, 0.0, 1.0)
    rgb = np.empty(tissue.shape + (3,), dtype=np.float32)
    rgb[..., 0] = np.clip(0.16 + 0.98 * tissue, 0.0, 1.0)
    rgb[..., 1] = np.clip(0.025 + 0.30 * tissue, 0.0, 1.0)
    rgb[..., 2] = np.clip(0.02 + 0.24 * tissue, 0.0, 1.0)
    return rgb


def _apply_artifact(gray: np.ndarray, artifact: str, level: float, rng: np.random.Generator) -> np.ndarray:
    level = float(np.clip(level, 0.0, 1.0))
    if artifact == "clean" or level <= 0.0:
        return gray
    if artifact == "smoke":
        blurred = gaussian_filter(gray, sigma=0.8 + 2.4 * level)
        mix = 0.25 + 0.70 * level
        washed = blurred * (1.0 - mix) + 0.64 * mix
        grain = rng.normal(0.0, 0.012 * (1.0 - 0.7 * level), size=gray.shape)
        return np.clip(washed + grain, 0.0, 1.0).astype(np.float32)
    if artifact == "splash":
        sigma = 0.6 + 3.2 * level
        hazy = gaussian_filter(gray, sigma=sigma)
        rows, cols = gray.shape
        out = hazy
        for _ in range(3):
            center = rng.uniform(0.15 * cols, 0.85 * cols), rng.uniform(0.15 * rows, 0.85 * rows)
            radius = rng.uniform(4.0, 9.0)
            yy, xx = np.ogrid[0:rows, 0:cols]
            blob = np.exp(-((yy - center[1]) ** 2 + (xx - center[0]) ** 2) / (2.0 * radius * radius))
            out = out * (1.0 - 0.35 * level * blob) + blob * level * 0.85
        return np.clip(gaussian_filter(out, sigma=0.6 + 0.8 * level), 0.0, 1.0).astype(np.float32)
    if artifact == "blood":
        out = gray.copy()
        x_mm, y_mm = rc_grids(gray.shape[0])
        for _ in range(4):
            center = rng.uniform(-22.0, 22.0, size=2)
            radius = rng.uniform(7.0, 14.0) * (0.45 + 0.55 * level)
            dist = np.hypot(x_mm - center[0], y_mm - center[1])
            mask = dist < radius
            out = np.where(mask, out * (1.0 - 0.82 * level) + 0.04 * level, out)
        return np.clip(gaussian_filter(out, sigma=0.4 + 0.9 * level), 0.0, 1.0).astype(np.float32)
    if artifact == "occluder":
        out = gray.copy()
        cols = gray.shape[1]
        half = max(1, int(round((6 + 10 * level) / 2)))
        mid = cols // 2
        out[:, mid - half : mid + half] = 0.07
        return out
    raise ValueError(f"unknown artifact: {artifact}")


def render_frame(
    episode: Episode,
    tip_xy: np.ndarray,
    artifact: str,
    rng: np.random.Generator,
    level: float = 1.0,
) -> Frame:
    """Render one camera frame with the tool tip at `tip_xy` millimetres."""
    x_mm, y_mm = rc_grids()
    gray = _tissue(x_mm, y_mm, rng)
    for zone in episode.zones:
        gray = _darken_disk(gray, x_mm, y_mm, zone)
    port = np.array([float(np.clip(tip_xy[0], -18.0, 18.0)), -36.0])
    shaft = _segment_distance(x_mm, y_mm, port, np.asarray(tip_xy, dtype=float))
    shaft_mask = np.clip((0.85 - shaft) / 0.85, 0.0, 1.0).astype(np.float32)
    gray = np.clip(gray * (1.0 - 0.55 * shaft_mask) + 0.78 * shaft_mask, 0.0, 1.0)
    gray = _apply_artifact(gray, artifact, level, rng)
    tip_rc = mm_to_rc(np.asarray(tip_xy, dtype=float))
    gray = stamp_tip(gray, tip_rc)
    return Frame(gray=gray.astype(np.float32), rgb=colorize(gray), tip_rc=(float(tip_rc[0]), float(tip_rc[1])), artifact=artifact)


def _episode_is_feasible(start: np.ndarray, target: np.ndarray, zones: tuple[Zone, ...]) -> bool:
    if min_clearance(start, zones) < 8.0 or min_clearance(target, zones) < 8.0:
        return False
    for zone in zones:
        if np.max(np.abs(zone.center)) > 34.0:
            return False
    return True


def demo_episode() -> Episode:
    """Fixed field whose straight start-target segment bisects the vessel disk."""
    start = np.array([-28.0, -6.0])
    target = np.array([28.0, 8.0])
    delta = target - start
    tangent = delta / np.linalg.norm(delta)
    normal = np.array([-tangent[1], tangent[0]])
    vessel = start + 0.48 * delta
    nerve = start + 0.72 * delta + 16.0 * normal
    duct = start + 0.58 * delta - 14.0 * normal
    return Episode(
        start_xy=(float(start[0]), float(start[1])),
        target_xy=(float(target[0]), float(target[1])),
        zones=(
            Zone("vessel", (float(vessel[0]), float(vessel[1])), 6.0),
            Zone("nerve", (float(nerve[0]), float(nerve[1])), 3.5),
            Zone("duct", (float(duct[0]), float(duct[1])), 3.2),
        ),
    )


def sample_episode(rng: np.random.Generator) -> Episode:
    """Random field with one disk on the straight path and a corridor around it.

    The side disks sit far enough off the line that a point tip can pass the
    blocking disk without entering them. Episodes that place the start or the
    target inside a margin are rejected.
    """
    for _ in range(48):
        start = np.array([-30.0, float(rng.uniform(-8.0, 8.0))])
        target = np.array([30.0, float(rng.uniform(-8.0, 8.0))])
        delta = target - start
        tangent = delta / np.linalg.norm(delta)
        normal = np.array([-tangent[1], tangent[0]])
        sign = 1.0 if rng.random() < 0.5 else -1.0
        vessel = start + float(rng.uniform(0.40, 0.55)) * delta
        nerve = start + float(rng.uniform(0.62, 0.80)) * delta + sign * float(rng.uniform(15.0, 18.0)) * normal
        duct = start + float(rng.uniform(0.32, 0.70)) * delta - sign * float(rng.uniform(14.0, 18.0)) * normal
        zones = (
            Zone("vessel", (float(vessel[0]), float(vessel[1])), 5.5),
            Zone("nerve", (float(nerve[0]), float(nerve[1])), 3.2),
            Zone("duct", (float(duct[0]), float(duct[1])), 3.0),
        )
        if _episode_is_feasible(start, target, zones):
            return Episode(
                start_xy=(float(start[0]), float(start[1])),
                target_xy=(float(target[0]), float(target[1])),
                zones=zones,
            )
    raise RuntimeError("failed to sample a feasible episode")
