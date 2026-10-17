"""Run the three synthetic surgical-scene benchmarks and write the report.

The run does not read an endoscope archive. `--synthetic` is required.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path

import numpy as np
import scipy
import torch

from . import __version__
from .occlusion_detector import (
    Decision,
    Measurement,
    OcclusionDetector,
    Reference,
    decide_measurement,
    fit_reference,
)
from .recovery_planner import SAFE_CLEARANCE_MM, plan_recovery
from .safety_barrier import (
    CONTROL_HZ,
    DT_S,
    BarrierConfig,
    evaluate_interlock,
    integrate,
)
from .scene_generator import MARGIN_MM, Episode, demo_episode, render_frame, sample_episode

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ("smoke", "splash", "blood", "occluder")
PRIMARY_ARTIFACTS = ("smoke", "splash")
REJECTION_TARGET = 0.90
LATENCY_TARGET_MS = 10.0
RECOVERY_TARGET = 0.95


def collect_environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def percentile(values: np.ndarray, q: float) -> float:
    array = np.asarray(values, dtype=float)
    try:
        return float(np.percentile(array, q, method="linear"))
    except TypeError:
        return float(np.percentile(array, q, interpolation="linear"))


def summarize_ms(values: np.ndarray) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "mean_ms": float(array.mean()),
        "p50_ms": percentile(array, 50),
        "p99_ms": percentile(array, 99),
        "max_ms": float(array.max()),
    }


def _num(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def _pct(value: float | None, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{100.0 * value:.{digits}f}%"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return value


def _render_measurement(detector: OcclusionDetector, episode: Episode, tip: np.ndarray, artifact: str, rng: np.random.Generator, level: float = 1.0) -> Measurement:
    frame = render_frame(episode, tip, artifact, rng, level=level)
    return detector.measure(frame.gray, frame.tip_rc)


def _cue_flags(measurement: Measurement, reference: Reference) -> tuple[bool, bool, bool]:
    ratio = measurement.edge_energy / reference.edge_median
    return (
        bool(ratio < reference.edge_ratio_max),
        bool(measurement.entropy_nats < reference.entropy_floor),
        bool(measurement.dark_fraction >= reference.dark_fraction_min),
    )


def _kind_row(measurements: list[Measurement], reference: Reference) -> dict:
    decisions = [decide_measurement(item, reference) for item in measurements]
    confident = [item for item in decisions if item.confident_track]
    rejected = sum(item.occluded for item in confident)
    edge_on = entropy_on = dark_on = 0
    for measurement in measurements:
        edge, entropy, dark = _cue_flags(measurement, reference)
        edge_on += int(edge)
        entropy_on += int(entropy)
        dark_on += int(dark)
    count = len(decisions)
    return {
        "frames": count,
        "recall": float(sum(item.occluded for item in decisions) / count),
        "baseline_confident_rate": float(len(confident) / count),
        "false_confidence_rejection": float(rejected / len(confident)) if confident else None,
        "edge_rate": float(edge_on / count),
        "entropy_rate": float(entropy_on / count),
        "dark_rate": float(dark_on / count),
        "mean_edge_ratio": float(np.mean([item.edge_energy / reference.edge_median for item in measurements])),
        "mean_entropy": float(np.mean([item.entropy_nats for item in measurements])),
        "mean_tip_confidence": float(np.mean([item.tip_confidence for item in measurements])),
        "mean_dark_fraction": float(np.mean([item.dark_fraction for item in measurements])),
    }


def _pool_rejection(rows: dict[str, dict], kinds: tuple[str, ...]) -> float | None:
    rejected = 0.0
    confident = 0.0
    for kind in kinds:
        row = rows[kind]
        frames = row["frames"]
        confident_frames = row["baseline_confident_rate"] * frames
        if row["false_confidence_rejection"] is None or confident_frames <= 0:
            continue
        confident += confident_frames
        rejected += row["false_confidence_rejection"] * confident_frames
    if confident <= 0:
        return None
    return float(rejected / confident)


def run_occlusion(detector: OcclusionDetector, rng: np.random.Generator, cal_count: int, eval_count: int) -> tuple[dict, list[tuple[np.ndarray, tuple[float, float]]], list[tuple[np.ndarray, tuple[float, float]]]]:
    """Fit cuts on a calibration split, then score fresh frames."""
    clean_cal: list[Measurement] = []
    smoke_cal: list[Measurement] = []
    splash_cal: list[Measurement] = []
    occluder_cal: list[Measurement] = []
    for _ in range(cal_count):
        episode = sample_episode(rng)
        tip = episode.start
        clean_cal.append(_render_measurement(detector, episode, tip, "clean", rng))
        smoke_cal.append(_render_measurement(detector, episode, tip, "smoke", rng))
        splash_cal.append(_render_measurement(detector, episode, tip, "splash", rng))
        occluder_cal.append(_render_measurement(detector, episode, tip, "occluder", rng))
    reference = fit_reference(clean_cal, smoke_cal + splash_cal, occluder_cal)

    rows: dict[str, dict] = {}
    timed_frames: list[tuple[np.ndarray, tuple[float, float]]] = []
    clean_frames: list[tuple[np.ndarray, tuple[float, float]]] = []
    eval_clean: list[Measurement] = []
    for _ in range(eval_count):
        episode = sample_episode(rng)
        tip = episode.start
        frame = render_frame(episode, tip, "clean", rng)
        eval_clean.append(detector.measure(frame.gray, frame.tip_rc))
        clean_frames.append((frame.gray, frame.tip_rc))
    rows["clean"] = _kind_row(eval_clean, reference)

    for kind in ARTIFACTS:
        measurements: list[Measurement] = []
        for _ in range(eval_count):
            episode = sample_episode(rng)
            tip = episode.start
            frame = render_frame(episode, tip, kind, rng)
            measurements.append(detector.measure(frame.gray, frame.tip_rc))
            if kind in PRIMARY_ARTIFACTS:
                timed_frames.append((frame.gray, frame.tip_rc))
        rows[kind] = _kind_row(measurements, reference)

    for _ in range(20):
        gray, tip_rc = timed_frames[_ % len(timed_frames)]
        detector.decide(gray, reference, tip_rc)
    latencies = np.empty(len(timed_frames), dtype=float)
    for index, (gray, tip_rc) in enumerate(timed_frames):
        started = time.perf_counter()
        decision: Decision = detector.decide(gray, reference, tip_rc)
        latencies[index] = (time.perf_counter() - started) * 1000.0
    levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    sweep = []
    for level in levels:
        flagged = 0
        confident = 0
        rejected = 0
        draws = 24
        for _ in range(draws):
            episode = sample_episode(rng)
            frame = render_frame(episode, episode.start, "smoke", rng, level=level)
            decision = detector.decide(frame.gray, reference, frame.tip_rc)
            flagged += int(decision.occluded)
            if decision.confident_track:
                confident += 1
                rejected += int(decision.occluded)
        sweep.append(
            {
                "level": level,
                "frames": draws,
                "recall": flagged / draws,
                "false_confidence_rejection": (rejected / confident) if confident else None,
            }
        )
    trip = next((row["level"] for row in sweep if row["recall"] >= 0.90), None)

    primary = _pool_rejection(rows, PRIMARY_ARTIFACTS)
    pooled = _pool_rejection(rows, ARTIFACTS)
    return (
        {
            "calibration_frames_per_kind": cal_count,
            "eval_frames_per_kind": eval_count,
            "reference": {
                "edge_median": reference.edge_median,
                "entropy_floor": reference.entropy_floor,
                "edge_ratio_max": reference.edge_ratio_max,
                "dark_fraction_min": reference.dark_fraction_min,
                "confidence_min": reference.confidence_min,
            },
            "kinds": rows,
            "primary_false_confidence_rejection": primary,
            "pooled_false_confidence_rejection": pooled,
            "detection_latency_ms": summarize_ms(latencies),
            "smoke_sweep": sweep,
            "smoke_trip_level": trip,
        },
        timed_frames,
        clean_frames,
    )


def run_barrier(rng: np.random.Generator, episodes: int) -> dict:
    config = BarrierConfig()
    free_pen = []
    guided_pen = []
    free_clear = []
    guided_clear = []
    free_hold = []
    guided_hold = []
    free_goal = []
    guided_goal = []
    free_reach = []
    guided_reach = []
    recovery_flags = []
    recovery_steps = []
    recovery_final = []
    for _ in range(episodes):
        episode = sample_episode(rng)
        free = integrate(episode, "unconstrained", config)
        guided = integrate(episode, "cbf", config)
        free_pen.append(free.penetrated)
        guided_pen.append(guided.penetrated)
        free_clear.append(free.min_clearance_mm)
        guided_clear.append(guided.min_clearance_mm)
        free_hold.append(free.margin_hold_rate)
        guided_hold.append(guided.margin_hold_rate)
        free_goal.append(free.final_target_mm)
        guided_goal.append(guided.final_target_mm)
        free_reach.append(free.reached)
        guided_reach.append(guided.reached)
        deep = int(np.argmin(free.clearance_mm))
        recovery = plan_recovery(free.xy[deep], free.xy[: deep + 1], episode.zones)
        recovery_flags.append(recovery.success)
        recovery_steps.append(recovery.steps)
        recovery_final.append(recovery.final_clearance_mm)

    demo = demo_episode()
    demo_free = integrate(demo, "unconstrained", config)
    demo_guided = integrate(demo, "cbf", config)
    return {
        "episodes": episodes,
        "margin_mm": MARGIN_MM,
        "control_hz": CONTROL_HZ,
        "dt_s": DT_S,
        "steps": config.steps,
        "u_max_mm_s": config.u_max_mm_s,
        "alpha": config.alpha,
        "unconstrained": {
            "penetration_rate": float(np.mean(free_pen)),
            "worst_clearance_mm": float(np.min(free_clear)),
            "median_min_clearance_mm": float(np.median(free_clear)),
            "margin_hold_rate": float(np.mean(free_hold)),
            "goal_reach_rate": float(np.mean(free_reach)),
            "mean_final_target_mm": float(np.mean(free_goal)),
        },
        "cbf": {
            "penetration_rate": float(np.mean(guided_pen)),
            "worst_clearance_mm": float(np.min(guided_clear)),
            "median_min_clearance_mm": float(np.median(guided_clear)),
            "margin_hold_rate": float(np.mean(guided_hold)),
            "goal_reach_rate": float(np.mean(guided_reach)),
            "mean_final_target_mm": float(np.mean(guided_goal)),
        },
        "recovery": {
            "safe_clearance_mm": SAFE_CLEARANCE_MM,
            "success_rate": float(np.mean(recovery_flags)),
            "mean_steps": float(np.mean(recovery_steps)),
            "mean_final_clearance_mm": float(np.mean(recovery_final)),
        },
        "demo": {
            "unconstrained_min_clearance_mm": demo_free.min_clearance_mm,
            "cbf_min_clearance_mm": demo_guided.min_clearance_mm,
            "unconstrained_penetrated": demo_free.penetrated,
            "cbf_penetrated": demo_guided.penetrated,
            "cbf_reached": demo_guided.reached,
            "unconstrained_xy": demo_free.xy,
            "cbf_xy": demo_guided.xy,
            "unconstrained_clearance_mm": demo_free.clearance_mm,
            "cbf_clearance_mm": demo_guided.clearance_mm,
            "zones": [(zone.name, zone.center_xy[0], zone.center_xy[1], zone.radius_mm) for zone in demo.zones],
            "start_xy": demo.start_xy,
            "target_xy": demo.target_xy,
        },
    }


def _hold_case(detector: OcclusionDetector, reference: Reference, case: tuple) -> bool:
    gray, tip_rc, clearance = case
    decision = detector.decide(gray, reference, tip_rc)
    return evaluate_interlock(decision.occluded, clearance).hold


def measure_async(cases: list[tuple], detector: OcclusionDetector, reference: Reference, warmup: int = 20) -> np.ndarray:
    """Safety thread already running. Latency includes the wake and the callback."""
    result = np.empty(len(cases), dtype=float)
    state = {"gen": 0, "seen": -1, "t0": 0.0, "index": 0, "stop": False, "case": None, "ms": 0.0}
    cond = threading.Condition()

    def watchdog() -> None:
        seen = 0
        while True:
            with cond:
                while state["gen"] == seen and not state["stop"]:
                    cond.wait()
                if state["stop"]:
                    return
                seen = state["gen"]
                started = state["t0"]
                case = state["case"]
            held = _hold_case(detector, reference, case)
            elapsed = (time.perf_counter() - started) * 1000.0
            with cond:
                state["ms"] = elapsed if held else float("nan")
                state["seen"] = seen
                cond.notify()

    thread = threading.Thread(target=watchdog, name="safety-interlock", daemon=True)
    thread.start()

    def submit(case: tuple) -> float:
        with cond:
            state["case"] = case
            state["t0"] = time.perf_counter()
            state["gen"] += 1
            generation = state["gen"]
            cond.notify()
            deadline = time.perf_counter() + 5.0
            while state["seen"] != generation:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    raise RuntimeError("safety watchdog did not answer")
                cond.wait(timeout=remaining)
            return float(state["ms"])

    try:
        for _ in range(warmup):
            submit(cases[0])
        for index, case in enumerate(cases):
            result[index] = submit(case)
    finally:
        with cond:
            state["stop"] = True
            cond.notify()
        thread.join(timeout=2.0)
    return result


def measure_sync(cases: list[tuple], detector: OcclusionDetector, reference: Reference, rng: np.random.Generator) -> np.ndarray:
    """Fault arrives at a random phase. The callback runs at the next 100 Hz tick."""
    result = np.empty(len(cases), dtype=float)
    for index, case in enumerate(cases):
        phase = float(rng.uniform(0.0, DT_S))
        started = time.perf_counter()
        while time.perf_counter() - started < phase:
            pass
        held = _hold_case(detector, reference, case)
        result[index] = (time.perf_counter() - started) * 1000.0
        if not held:
            result[index] = float("nan")
    return result


def run_latency(detector: OcclusionDetector, reference: Reference, occluded_frames: list[tuple[np.ndarray, tuple[float, float]]], clean_frames: list[tuple[np.ndarray, tuple[float, float]]], trials: int, rng: np.random.Generator) -> dict:
    cases = []
    for index in range(trials):
        if index % 2 == 0:
            gray, tip_rc = occluded_frames[index % len(occluded_frames)]
            clearance = 12.0
        else:
            gray, tip_rc = clean_frames[index % len(clean_frames)]
            clearance = 0.2
        cases.append((gray, tip_rc, clearance))
    for _ in range(10):
        _hold_case(detector, reference, cases[0])
    async_ms = measure_async(cases, detector, reference)
    sync_ms = measure_sync(cases, detector, reference, rng)
    return {
        "trials": trials,
        "period_ms": DT_S * 1000.0,
        "async": summarize_ms(async_ms),
        "sync": summarize_ms(sync_ms),
        "async_samples_ms": async_ms,
        "sync_samples_ms": sync_ms,
        "async_nan": int(np.isnan(async_ms).sum()),
        "sync_nan": int(np.isnan(sync_ms).sum()),
    }


def render_markdown(report: dict) -> str:
    occ = report["occlusion"]
    barrier = report["barrier"]
    hold = report["interlock"]
    targets = report["targets"]
    kinds = occ["kinds"]
    kind_rows = []
    for name in ("clean",) + ARTIFACTS:
        row = kinds[name]
        kind_rows.append(
            [
                name,
                str(row["frames"]),
                _pct(row["baseline_confident_rate"]),
                _pct(row["recall"]),
                _pct(row["false_confidence_rejection"]),
                _num(row["mean_edge_ratio"], 2),
                _num(row["mean_entropy"], 2),
                _num(row["mean_dark_fraction"], 3),
            ]
        )
    sweep_rows = [
        [f"{row['level']:.1f}", str(row["frames"]), _pct(row["recall"]), _pct(row["false_confidence_rejection"])]
        for row in occ["smoke_sweep"]
    ]
    free = barrier["unconstrained"]
    guided = barrier["cbf"]
    det = occ["detection_latency_ms"]
    lines = [
        "# Surgical scene triage — benchmark results",
        "",
        "One CPU run on a synthetic camera plane. No endoscope recording, no patient study, and no trained manipulation policy is loaded.",
        "",
        f"- Seed {report['seed']}. Package {report['version']}.",
        f"- Occlusion calibration is {occ['calibration_frames_per_kind']} frames per kind. Evaluation is {occ['eval_frames_per_kind']} fresh frames per kind.",
        f"- Barrier rollouts: {barrier['episodes']} episodes, {barrier['steps']} steps at {barrier['control_hz']:.0f} Hz, speed limit {barrier['u_max_mm_s']:.0f} mm/s, margin {barrier['margin_mm']:.1f} mm.",
        f"- Hold trials: {hold['trials']}. Half are a smoky frame with a safe tip. Half are a clean frame whose tip is already inside the margin.",
        "",
        "## 1. Visual occlusion triage",
        "",
        "The baseline is a zero-mean tip filter on a 32×32 window at the tracked tip. A frame is a confident track when that response is at least "
        f"{occ['reference']['confidence_min']:.2f} of the response on an ideal specular tip. The filter does not gate the controller.",
        "",
        "The detector flags a frame when any one cue trips, using cuts frozen on the calibration split:",
        "",
        f"- Laplacian energy below {occ['reference']['edge_ratio_max']:.3f} times the clean median ({occ['reference']['edge_median']:.4f}).",
        f"- Luminance entropy below {occ['reference']['entropy_floor']:.3f} nats.",
        f"- Dark-pixel fraction (luminance < 0.12) at or above {occ['reference']['dark_fraction_min']:.3f}.",
        "",
        "False-confidence rejection is the share of those confident tracks, on ground-truth artifact frames, that the detector refuses. "
        "The primary pool is smoke and splash. Blood and the instrument bar are reported beside them.",
        "",
        _md_table(
            ["Kind", "Frames", "Baseline confident", "Detector recall", "False-confidence rejection", "Mean edge ratio", "Mean entropy", "Mean dark fraction"],
            kind_rows,
        ),
        "",
        f"Primary rejection (smoke + splash): {_pct(occ['primary_false_confidence_rejection'])}. "
        f"All four artifacts: {_pct(occ['pooled_false_confidence_rejection'])}. "
        f"Clean false-rejection rate: {_pct(kinds['clean']['recall'])}. "
        f"Target (primary rejection ≥ {REJECTION_TARGET:.0%}): {'met' if targets['rejection_met'] else 'not met'}.",
        "",
        "Detection latency is the wall time of one decision after the frame is already in memory. It is not exposure, transfer, or display time.",
        "",
        _md_table(
            ["", "N", "Mean ms", "P50 ms", "P99 ms"],
            [["Detector", str(det["n"]), _num(det["mean_ms"], 3), _num(det["p50_ms"], 3), _num(det["p99_ms"], 3)]],
        ),
        "",
        "Smoke strength is a mix toward a flat fog plus blur. The same frozen cuts are applied. "
        f"The lowest tabulated strength with recall at least 90% is {occ['smoke_trip_level'] if occ['smoke_trip_level'] is not None else 'n/a'}.",
        "",
        _md_table(["Smoke strength", "Frames", "Recall", "False-confidence rejection"], sweep_rows),
        "",
        "Edge energy falls on smoke, splash, and blood. It does not fall on the instrument bar, which replaces a vertical strip with a constant and adds two strong edges. "
        "That bar is carried by the dark-fraction cue. Entropy collapses on dense smoke and on splash; blood keeps a broad histogram, so blood is an edge decision.",
        "",
        "## 2. No-go zone margin",
        "",
        "Both policies use the same saturated attractor toward the target. The straight segment is an invariant of that field, and the blocking disk is built on the segment, so the unconstrained tip goes through the disk. "
        "The CBF policy projects the command onto ∇h · u ≥ −α h and onto the speed limit, then refuses any straight step that would enter an expanded disk. "
        "A head-on attractor has no tangential component left after that projection, so the safe policy adds a tangent slide while the attractor points into a disk closer than 6 mm. The slide is projected again. It is not a trained diffusion policy.",
        "",
        _md_table(
            ["Policy", "Penetration", "Worst clearance mm", "Median episode minimum mm", "Margin hold", "Goal reach"],
            [
                [
                    "Unconstrained",
                    _pct(free["penetration_rate"], 1),
                    _num(free["worst_clearance_mm"], 2),
                    _num(free["median_min_clearance_mm"], 2),
                    _pct(free["margin_hold_rate"]),
                    _pct(free["goal_reach_rate"]),
                ],
                [
                    "CBF interlock",
                    _pct(guided["penetration_rate"], 1),
                    _num(guided["worst_clearance_mm"], 2),
                    _num(guided["median_min_clearance_mm"], 2),
                    _pct(guided["margin_hold_rate"]),
                    _pct(guided["goal_reach_rate"]),
                ],
            ],
        ),
        "",
        f"Penetration means anatomical clearance below 0. The margin target is {MARGIN_MM:.1f} mm. "
        f"CBF penetration target (0%): {'met' if targets['penetration_met'] else 'not met'}. "
        f"CBF worst clearance ≥ {MARGIN_MM:.1f} mm: {'met' if targets['margin_met'] else 'not met'}.",
        "",
        "Clearance is ||x − c|| − R for the nearest disk, in millimetres. Goal reach means the last sample is within 2 mm of the target.",
        "",
        f"The illustrated field used by the board has unconstrained clearance {_num(barrier['demo']['unconstrained_min_clearance_mm'], 2)} mm "
        f"and CBF clearance {_num(barrier['demo']['cbf_min_clearance_mm'], 2)} mm. "
        f"CBF reached the illustrated target: {barrier['demo']['cbf_reached']}.",
        "",
        "## 3. Emergency hold and retract",
        "",
        "The synchronous scheduler samples the same callback on the next 100 Hz tick. The fault time is uniform inside the 10 ms period, and the wait is a busy loop up to that tick so the residual is not an operating-system sleep quantum. "
        "The asynchronous scheduler is a thread that is already blocked on a condition variable. The timestamp is taken before the wake, and the hold is recorded after the detector and the clearance check both finish.",
        "",
        _md_table(
            ["Scheduler", "N", "Mean ms", "P50 ms", "P99 ms", "Max ms"],
            [
                [
                    "Next control tick",
                    str(hold["sync"]["n"]),
                    _num(hold["sync"]["mean_ms"], 3),
                    _num(hold["sync"]["p50_ms"], 3),
                    _num(hold["sync"]["p99_ms"], 3),
                    _num(hold["sync"]["max_ms"], 3),
                ],
                [
                    "Async interlock",
                    str(hold["async"]["n"]),
                    _num(hold["async"]["mean_ms"], 3),
                    _num(hold["async"]["p50_ms"], 3),
                    _num(hold["async"]["p99_ms"], 3),
                    _num(hold["async"]["max_ms"], 3),
                ],
            ],
        ),
        "",
        f"Async P99 < {LATENCY_TARGET_MS:.1f} ms: {'met' if targets['latency_met'] else 'not met'}. "
        f"Trials that did not assert a hold: async {hold['async_nan']}, sync {hold['sync_nan']}.",
        "",
        "Recovery starts at the deepest unconstrained sample, inside the blocking disk, and steps at the speed limit along the heading that most increases the worst-zone clearance. Ties prefer the incoming direction. "
        f"Success is clearance at least {barrier['recovery']['safe_clearance_mm']:.1f} mm. "
        f"Success rate {_pct(barrier['recovery']['success_rate'])} over {barrier['episodes']} episodes "
        f"(mean { _num(barrier['recovery']['mean_steps'], 1)} steps, mean final clearance {_num(barrier['recovery']['mean_final_clearance_mm'], 2)} mm). "
        f"Target ≥ {RECOVERY_TARGET:.0%}: {'met' if targets['recovery_met'] else 'not met'}.",
        "",
        "## Targets",
        "",
        _md_table(
            ["Target", "Required", "Measured", "Status"],
            [
                [
                    "Primary false-confidence rejection",
                    f"≥ {REJECTION_TARGET:.0%}",
                    _pct(occ["primary_false_confidence_rejection"]),
                    "met" if targets["rejection_met"] else "not met",
                ],
                [
                    "CBF penetration",
                    "0%",
                    _pct(guided["penetration_rate"], 1),
                    "met" if targets["penetration_met"] else "not met",
                ],
                [
                    "CBF worst clearance",
                    f"≥ {MARGIN_MM:.1f} mm",
                    f"{guided['worst_clearance_mm']:.3f} mm",
                    "met" if targets["margin_met"] else "not met",
                ],
                [
                    "Async hold P99",
                    f"< {LATENCY_TARGET_MS:.1f} ms",
                    f"{hold['async']['p99_ms']:.3f} ms",
                    "met" if targets["latency_met"] else "not met",
                ],
                [
                    "Retract success",
                    f"≥ {RECOVERY_TARGET:.0%}",
                    _pct(barrier["recovery"]["success_rate"]),
                    "met" if targets["recovery_met"] else "not met",
                ],
            ],
        ),
        "",
        "## What this run is not",
        "",
        "The field is a 128×128 rendering of an 80 mm plane. Disks are not vessels segmented from a patient. Smoke, splash, blood, and the bar are composited degradations, not a scattering model of electrosurgical plume. "
        "The robot is a point mass. Latency is inside this process. A later controller would still have to account for camera exposure, bus delay, and the robot's own tracking loop.",
        "",
    ]
    return "\n".join(lines)


def save_dashboard(report: dict, barrier: dict) -> None:
    demo = barrier["demo"]
    episode = demo_episode()
    rng = np.random.default_rng(report["seed"])
    images = {}
    tip = np.asarray(demo["cbf_xy"][len(demo["cbf_xy"]) // 3], dtype=float)
    for kind in ("clean",) + ARTIFACTS:
        frame = render_frame(episode, tip, kind, rng, level=1.0)
        images[kind] = np.clip(frame.rgb * 255.0, 0, 255).astype(np.uint8)
    zones = np.array([[item[1], item[2], item[3]] for item in demo["zones"]], dtype=float)
    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "dashboard.npz",
        clean=images["clean"],
        smoke=images["smoke"],
        splash=images["splash"],
        blood=images["blood"],
        occluder=images["occluder"],
        zones=zones,
        zone_names=np.array([item[0] for item in demo["zones"]]),
        start_xy=np.asarray(demo["start_xy"], dtype=float),
        target_xy=np.asarray(demo["target_xy"], dtype=float),
        free_xy=np.asarray(demo["unconstrained_xy"], dtype=float),
        cbf_xy=np.asarray(demo["cbf_xy"], dtype=float),
        free_clearance_mm=np.asarray(demo["unconstrained_clearance_mm"], dtype=float),
        cbf_clearance_mm=np.asarray(demo["cbf_clearance_mm"], dtype=float),
        async_ms=np.asarray(report["interlock"]["async_samples_ms"], dtype=float),
        sync_ms=np.asarray(report["interlock"]["sync_samples_ms"], dtype=float),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic surgical-scene triage benchmarks")
    parser.add_argument("--synthetic", action="store_true", help="Required. Refuses to run a real archive because none is read.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--cal-per-artifact", type=int, default=36)
    parser.add_argument("--eval-per-artifact", type=int, default=60)
    parser.add_argument("--latency-trials", type=int, default=300)
    args = parser.parse_args(argv)
    if not args.synthetic:
        print(
            "Refusing to run without --synthetic. This benchmark does not read a surgical video archive.",
            file=sys.stderr,
        )
        return 2

    rng = np.random.default_rng(args.seed)
    detector = OcclusionDetector()
    occlusion, occluded_frames, clean_frames = run_occlusion(
        detector,
        rng,
        args.cal_per_artifact,
        args.eval_per_artifact,
    )
    reference = Reference(**occlusion["reference"])
    barrier = run_barrier(np.random.default_rng(args.seed + 1), args.episodes)
    # The barrier uses its own stream so a change in episode count does not reshuffle the vision split.
    interlock = run_latency(
        detector,
        reference,
        occluded_frames,
        clean_frames,
        args.latency_trials,
        np.random.default_rng(args.seed + 2),
    )
    primary = occlusion["primary_false_confidence_rejection"]
    targets = {
        "rejection_met": bool(primary is not None and primary >= REJECTION_TARGET),
        "penetration_met": bool(barrier["cbf"]["penetration_rate"] == 0.0),
        "margin_met": bool(barrier["cbf"]["worst_clearance_mm"] >= MARGIN_MM - 1e-4),
        "latency_met": bool(interlock["async"]["p99_ms"] < LATENCY_TARGET_MS and interlock["async_nan"] == 0),
        "recovery_met": bool(barrier["recovery"]["success_rate"] >= RECOVERY_TARGET),
    }
    report = {
        "seed": args.seed,
        "version": __version__,
        "environment": collect_environment(),
        "occlusion": occlusion,
        "barrier": {key: value for key, value in barrier.items() if key != "demo"},
        "interlock": {key: value for key, value in interlock.items() if not key.endswith("_samples_ms")},
        "targets": targets,
        "demo_summary": {
            "unconstrained_min_clearance_mm": barrier["demo"]["unconstrained_min_clearance_mm"],
            "cbf_min_clearance_mm": barrier["demo"]["cbf_min_clearance_mm"],
            "cbf_penetrated": barrier["demo"]["cbf_penetrated"],
            "cbf_reached": barrier["demo"]["cbf_reached"],
        },
    }
    # Keep the arrays for the dashboard writer, then drop them from JSON.
    report_for_dash = dict(report)
    report_for_dash["interlock"] = interlock
    text = render_markdown(
        {
            **report,
            "barrier": barrier,
            "interlock": interlock,
        }
    )
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "benchmark_report.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    docs = ROOT / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "BENCHMARK_RESULTS.md").write_text(text, encoding="utf-8")
    (ROOT / "BENCHMARK_RESULTS.md").write_text(text, encoding="utf-8")
    save_dashboard(report_for_dash, barrier)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)
    print(f"Wrote {results / 'benchmark_report.json'}")
    return 0 if all(targets.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
