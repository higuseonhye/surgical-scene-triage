"""Frame entropy and Laplacian energy for visual occlusion triage.

The baseline is a zero-mean matched filter on the specular tip. It stays
high when that highlight is stamped back onto smoke, splash, or blood.
The occlusion decision does not use the filter. High-frequency loss is the
mean absolute Laplacian, the first-moment counterpart of Var(laplacian I).
Entropy is one 64-bin frame histogram, not a grid of local patches. A dark
mass covers a shaft that leaves the Laplacian high.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .scene_generator import TIP_SIGMA_PX, stamp_tip

CONFIDENCE_MIN = 0.55
ENTROPY_BINS = 64
TIP_CROP = 32


DARK_CUTOFF = 0.12


@dataclass(frozen=True)
class Measurement:
    edge_energy: float
    entropy_nats: float
    tip_confidence: float
    dark_fraction: float


@dataclass(frozen=True)
class Reference:
    """Clean-field scale and the decision cuts frozen on the calibration split."""

    edge_median: float
    entropy_floor: float
    edge_ratio_max: float
    dark_fraction_min: float
    confidence_min: float = CONFIDENCE_MIN

    def edge_ratio(self, measurement: Measurement) -> float:
        return measurement.edge_energy / self.edge_median


@dataclass(frozen=True)
class Decision:
    occluded: bool
    edge_ratio: float
    entropy_nats: float
    tip_confidence: float
    confident_track: bool

    @property
    def rejected_confident_track(self) -> bool:
        """Baseline still reports a tip, and this frame is refused."""
        return self.occluded and self.confident_track


class OcclusionDetector:
    """Single-frame cues. Kernels are fixed; nothing is trained."""

    def __init__(self) -> None:
        laplacian = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        self._laplacian = laplacian.view(1, 1, 3, 3)
        radius = 4
        coords = np.arange(-radius, radius + 1, dtype=np.float32)
        xs, ys = np.meshgrid(coords, coords, indexing="xy")
        blob = np.exp(-(xs**2 + ys**2) / (2.0 * TIP_SIGMA_PX * TIP_SIGMA_PX))
        blob = blob - float(blob.mean())
        kernel = torch.from_numpy(np.ascontiguousarray(blob))
        self._tip_kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1])
        self._tip_pad = radius
        ideal = np.full((TIP_CROP, TIP_CROP), 0.55, dtype=np.float32)
        ideal = stamp_tip(ideal, np.array([TIP_CROP / 2, TIP_CROP / 2], dtype=float))
        self._ideal_peak = self._tip_peak(ideal)

    def _as_tensor(self, gray: np.ndarray) -> torch.Tensor:
        array = np.ascontiguousarray(gray, dtype=np.float32)
        return torch.from_numpy(array).view(1, 1, array.shape[0], array.shape[1])

    def edge_energy(self, gray: np.ndarray) -> float:
        with torch.inference_mode():
            response = F.conv2d(self._as_tensor(gray), self._laplacian, padding=1)
            return float(response.abs().mean())

    def _tip_peak(self, gray: np.ndarray) -> float:
        with torch.inference_mode():
            response = F.conv2d(self._as_tensor(gray), self._tip_kernel, padding=self._tip_pad)
            return float(response.max())

    def _tip_crop(self, gray: np.ndarray, tip_rc: tuple[float, float] | np.ndarray) -> np.ndarray:
        """Window around the tracked tip. The baseline does not rescan the frame."""
        half = TIP_CROP // 2
        patch = np.zeros((TIP_CROP, TIP_CROP), dtype=np.float32)
        row = int(round(float(tip_rc[0])))
        col = int(round(float(tip_rc[1])))
        row0 = row - half
        col0 = col - half
        src_row0 = max(0, row0)
        src_col0 = max(0, col0)
        src_row1 = min(gray.shape[0], row0 + TIP_CROP)
        src_col1 = min(gray.shape[1], col0 + TIP_CROP)
        dst_row0 = src_row0 - row0
        dst_col0 = src_col0 - col0
        patch[dst_row0 : dst_row0 + (src_row1 - src_row0), dst_col0 : dst_col0 + (src_col1 - src_col0)] = gray[
            src_row0:src_row1, src_col0:src_col1
        ]
        return patch

    def tip_confidence(self, gray: np.ndarray, tip_rc: tuple[float, float] | np.ndarray) -> float:
        if self._ideal_peak <= 1e-8:
            return 0.0
        return self._tip_peak(self._tip_crop(gray, tip_rc)) / self._ideal_peak

    def measure(self, gray: np.ndarray, tip_rc: tuple[float, float] | np.ndarray) -> Measurement:
        return Measurement(
            edge_energy=self.edge_energy(gray),
            entropy_nats=shannon_entropy(gray),
            tip_confidence=self.tip_confidence(gray, tip_rc),
            dark_fraction=dark_fraction(gray),
        )

    def decide(self, gray: np.ndarray, reference: Reference, tip_rc: tuple[float, float] | np.ndarray) -> Decision:
        return decide_measurement(self.measure(gray, tip_rc), reference)


def shannon_entropy(gray: np.ndarray) -> float:
    """Shannon entropy, in nats, of a 64-bin histogram on [0, 1]."""
    clipped = np.clip(gray, 0.0, 1.0)
    histogram, _ = np.histogram(clipped, bins=ENTROPY_BINS, range=(0.0, 1.0))
    total = int(histogram.sum())
    if total <= 0:
        return 0.0
    probability = histogram[histogram > 0].astype(np.float64) / total
    return float(-np.sum(probability * np.log(probability)))


def dark_fraction(gray: np.ndarray, cutoff: float = DARK_CUTOFF) -> float:
    """Share of pixels under `cutoff`. A shaft across the lens produces a dark mass."""
    return float(np.mean(gray < cutoff))


def decide_measurement(measurement: Measurement, reference: Reference) -> Decision:
    ratio = reference.edge_ratio(measurement)
    occluded = bool(
        ratio < reference.edge_ratio_max
        or measurement.entropy_nats < reference.entropy_floor
        or measurement.dark_fraction >= reference.dark_fraction_min
    )
    confident = bool(measurement.tip_confidence >= reference.confidence_min)
    return Decision(
        occluded=occluded,
        edge_ratio=float(ratio),
        entropy_nats=measurement.entropy_nats,
        tip_confidence=measurement.tip_confidence,
        confident_track=confident,
    )


def select_edge_threshold(clean_ratios: np.ndarray, occluded_ratios: np.ndarray) -> float:
    """Midpoint between the occluded tail and the clean tail when they separate.

    Calibration frames only. The returned cut is frozen before evaluation.
    """
    occluded_tail = float(np.quantile(occluded_ratios, 0.95))
    clean_tail = float(np.quantile(clean_ratios, 0.05))
    if occluded_tail < clean_tail:
        return 0.5 * (occluded_tail + clean_tail)
    best_threshold = 0.5
    best_score = -1e9
    for threshold in np.linspace(0.15, 0.9, 40):
        true_positive = float(np.mean(occluded_ratios < threshold))
        false_positive = float(np.mean(clean_ratios < threshold))
        score = true_positive - false_positive
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def select_dark_threshold(clean_fraction: np.ndarray, occluder_fraction: np.ndarray) -> float:
    """Smallest dark mass that sits between clean frames and a shaft occluder."""
    clean_tail = float(np.quantile(clean_fraction, 0.95))
    occluder_tail = float(np.quantile(occluder_fraction, 0.05))
    if clean_tail < occluder_tail:
        return 0.5 * (clean_tail + occluder_tail)
    return float(occluder_tail)


def fit_reference(clean: list[Measurement], occluded: list[Measurement], occluder: list[Measurement]) -> Reference:
    clean_edge = np.array([item.edge_energy for item in clean], dtype=float)
    clean_entropy = np.array([item.entropy_nats for item in clean], dtype=float)
    edge_median = float(np.median(clean_edge))
    if edge_median <= 1e-8:
        raise RuntimeError("clean edge energy is degenerate")
    clean_ratios = clean_edge / edge_median
    occluded_ratios = np.array([item.edge_energy for item in occluded], dtype=float) / edge_median
    occluded_entropy = np.array([item.entropy_nats for item in occluded], dtype=float)
    collapsed = occluded_entropy[occluded_entropy < float(clean_entropy.min())]
    if collapsed.size:
        entropy_floor = 0.5 * (float(clean_entropy.min()) + float(collapsed.max()))
    else:
        entropy_floor = float(clean_entropy.min() - 0.05)
    dark_clean = np.array([item.dark_fraction for item in clean], dtype=float)
    dark_bar = np.array([item.dark_fraction for item in occluder], dtype=float)
    return Reference(
        edge_median=edge_median,
        entropy_floor=float(entropy_floor),
        edge_ratio_max=select_edge_threshold(clean_ratios, occluded_ratios),
        dark_fraction_min=select_dark_threshold(dark_clean, dark_bar),
        confidence_min=CONFIDENCE_MIN,
    )
