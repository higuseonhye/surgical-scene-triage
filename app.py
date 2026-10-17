"""Local board for the synthetic surgical field. Serves on 127.0.0.1:8505."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from surgical_triage.occlusion_detector import OcclusionDetector, Reference, dark_fraction, fit_reference
from surgical_triage.recovery_planner import SAFE_CLEARANCE_MM, plan_recovery
from surgical_triage.safety_barrier import DT_S, BarrierConfig, integrate
from surgical_triage.scene_generator import (
    FIELD_MM,
    MARGIN_MM,
    demo_episode,
    mm_to_rc,
    render_frame,
    sample_episode,
)

ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "results" / "benchmark_report.json"
DASHBOARD_PATH = ROOT / "results" / "dashboard.npz"
ARTIFACTS = {
    "Clean": "clean",
    "Smoke": "smoke",
    "Lens splash": "splash",
    "Blood": "blood",
    "Instrument bar": "occluder",
}
ARTIFACT_SEED = {"clean": 1, "smoke": 2, "splash": 3, "blood": 4, "occluder": 5}

st.set_page_config(page_title="Surgical Scene Triage", layout="wide")


@st.cache_data
def load_report() -> dict | None:
    if not REPORT_PATH.exists():
        return None
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_latency() -> dict | None:
    if not DASHBOARD_PATH.exists():
        return None
    with np.load(DASHBOARD_PATH) as packed:
        return {"async_ms": np.array(packed["async_ms"]), "sync_ms": np.array(packed["sync_ms"])}


@st.cache_resource
def load_detector() -> OcclusionDetector:
    return OcclusionDetector()


@st.cache_data
def fit_live_reference(seed: int) -> dict:
    """Used only when the benchmark report has not been written yet."""
    detector = load_detector()
    rng = np.random.default_rng(seed)
    clean, occluded, occluder = [], [], []
    for _ in range(16):
        episode = sample_episode(rng)
        for kind, bucket in (("clean", clean), ("smoke", occluded), ("splash", occluded), ("occluder", occluder)):
            frame = render_frame(episode, episode.start, kind, rng)
            bucket.append(detector.measure(frame.gray, frame.tip_rc))
    reference = fit_reference(clean, occluded, occluder)
    return {
        "edge_median": reference.edge_median,
        "entropy_floor": reference.entropy_floor,
        "edge_ratio_max": reference.edge_ratio_max,
        "dark_fraction_min": reference.dark_fraction_min,
        "confidence_min": reference.confidence_min,
    }


@st.cache_data
def load_rollouts() -> dict:
    episode = demo_episode()
    config = BarrierConfig()
    free = integrate(episode, "unconstrained", config)
    guided = integrate(episode, "cbf", config)
    return {
        "zones": tuple((zone.name, zone.center_xy, zone.radius_mm) for zone in episode.zones),
        "start": episode.start_xy,
        "target": episode.target_xy,
        "free_xy": free.xy,
        "free_clearance": free.clearance_mm,
        "cbf_xy": guided.xy,
        "cbf_clearance": guided.clearance_mm,
    }


def reference_from(payload: dict) -> Reference:
    return Reference(
        edge_median=float(payload["edge_median"]),
        entropy_floor=float(payload["entropy_floor"]),
        edge_ratio_max=float(payload["edge_ratio_max"]),
        dark_fraction_min=float(payload["dark_fraction_min"]),
        confidence_min=float(payload["confidence_min"]),
    )


def _blend(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    if not np.any(mask):
        return
    paint = np.array(color, dtype=np.float32)
    view = image[mask].astype(np.float32)
    image[mask] = np.clip(view * (1.0 - alpha) + paint * alpha, 0, 255).astype(np.uint8)


def _disk_mask(shape: tuple[int, int], center_rc: np.ndarray, radius_px: float, ring: float | None = None) -> np.ndarray:
    rows, cols = shape
    yy, xx = np.ogrid[0:rows, 0:cols]
    dist = np.hypot(yy - center_rc[0], xx - center_rc[1])
    if ring is None:
        return dist <= radius_px
    return (dist >= radius_px - ring) & (dist <= radius_px + ring)


def _draw_segment(image: np.ndarray, start_rc: np.ndarray, end_rc: np.ndarray, color: tuple[int, int, int]) -> None:
    count = int(max(np.hypot(*(end_rc - start_rc)), 1))
    rows = np.linspace(start_rc[0], end_rc[0], count)
    cols = np.linspace(start_rc[1], end_rc[1], count)
    height, width = image.shape[:2]
    rr = np.clip(np.rint(rows).astype(int), 0, height - 1)
    cc = np.clip(np.rint(cols).astype(int), 0, width - 1)
    image[rr, cc] = color


def compose(rgb: np.ndarray, rollout: dict, policy: str, step: int, recovery_xy: np.ndarray | None) -> np.ndarray:
    image = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    path = rollout["cbf_xy"] if policy == "cbf" else rollout["free_xy"]
    for name, center_xy, radius in rollout["zones"]:
        del name
        center = mm_to_rc(np.asarray(center_xy, dtype=float))
        radius_px = radius / FIELD_MM * (image.shape[0] - 1)
        margin_px = (radius + MARGIN_MM) / FIELD_MM * (image.shape[0] - 1)
        _blend(image, _disk_mask(image.shape[:2], center, radius_px), (153, 27, 27), 0.28)
        ring = _disk_mask(image.shape[:2], center, margin_px, ring=0.7)
        image[ring] = np.array([245, 158, 11], dtype=np.uint8)
    history = path[: step + 1]
    color = (45, 212, 191) if policy == "cbf" else (203, 213, 225)
    for prev, nxt in zip(history[:-1], history[1:]):
        _draw_segment(image, mm_to_rc(prev), mm_to_rc(nxt), color)
    if recovery_xy is not None and len(recovery_xy) > 1:
        for prev, nxt in zip(recovery_xy[:-1], recovery_xy[1:]):
            _draw_segment(image, mm_to_rc(prev), mm_to_rc(nxt), (251, 146, 60))
    tip = mm_to_rc(path[step])
    target = mm_to_rc(np.asarray(rollout["target"], dtype=float))
    image[_disk_mask(image.shape[:2], target, 2.2)] = np.array([250, 204, 21], dtype=np.uint8)
    image[_disk_mask(image.shape[:2], tip, 2.6)] = np.array([240, 253, 250], dtype=np.uint8)
    return image


def trajectory_figure(rollout: dict, step: int, policy: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=rollout["free_xy"][:, 0],
            y=rollout["free_xy"][:, 1],
            mode="lines",
            name="Unconstrained",
            line={"color": "#cbd5e1", "width": 1.5, "dash": "dot"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=rollout["cbf_xy"][:, 0],
            y=rollout["cbf_xy"][:, 1],
            mode="lines",
            name="CBF interlock",
            line={"color": "#2dd4bf", "width": 2},
        )
    )
    path = rollout["cbf_xy"] if policy == "cbf" else rollout["free_xy"]
    figure.add_trace(
        go.Scatter(
            x=[path[step, 0]],
            y=[path[step, 1]],
            mode="markers",
            name="Tip",
            marker={"size": 10, "color": "#f8fafc", "line": {"color": "#0f172a", "width": 1}},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[rollout["target"][0]],
            y=[rollout["target"][1]],
            mode="markers",
            name="Target",
            marker={"size": 9, "symbol": "x", "color": "#facc15"},
        )
    )
    for name, center_xy, radius in rollout["zones"]:
        figure.add_shape(
            type="circle",
            x0=center_xy[0] - radius,
            x1=center_xy[0] + radius,
            y0=center_xy[1] - radius,
            y1=center_xy[1] + radius,
            line={"color": "#f87171", "width": 1},
            fillcolor="rgba(248, 113, 113, 0.18)",
        )
        grown = radius + MARGIN_MM
        figure.add_shape(
            type="circle",
            x0=center_xy[0] - grown,
            x1=center_xy[0] + grown,
            y0=center_xy[1] - grown,
            y1=center_xy[1] + grown,
            line={"color": "#f59e0b", "width": 1, "dash": "dot"},
        )
        figure.add_annotation(x=center_xy[0], y=center_xy[1], text=name, showarrow=False, font={"size": 11, "color": "#fecaca"})
    figure.update_layout(
        template="plotly_dark",
        height=420,
        margin={"l": 48, "r": 16, "t": 32, "b": 40},
        legend={"orientation": "h", "y": 1.12},
        xaxis_title="x (mm)",
        yaxis_title="y (mm)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    return figure


def clearance_figure(rollout: dict) -> go.Figure:
    time_ms = np.arange(rollout["free_clearance"].shape[0]) * DT_S * 1000.0
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=time_ms, y=rollout["free_clearance"], name="Unconstrained", line={"color": "#cbd5e1", "dash": "dot"}))
    figure.add_trace(go.Scatter(x=time_ms, y=rollout["cbf_clearance"], name="CBF interlock", line={"color": "#2dd4bf"}))
    figure.add_hline(y=0.0, line_dash="dash", line_color="#f87171", annotation_text="anatomy")
    figure.add_hline(y=MARGIN_MM, line_dash="dot", line_color="#f59e0b", annotation_text="1 mm margin")
    figure.update_layout(
        template="plotly_dark",
        height=320,
        margin={"l": 48, "r": 16, "t": 32, "b": 40},
        xaxis_title="Time (ms)",
        yaxis_title="Clearance (mm)",
        legend={"orientation": "h", "y": 1.14},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def latency_figure(samples: dict) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Histogram(x=samples["sync_ms"], name="Next control tick", opacity=0.7, nbinsx=28, marker_color="#94a3b8"))
    figure.add_trace(go.Histogram(x=samples["async_ms"], name="Async interlock", opacity=0.75, nbinsx=28, marker_color="#2dd4bf"))
    figure.add_vline(x=10.0, line_dash="dash", line_color="#f87171", annotation_text="10 ms")
    figure.update_layout(
        barmode="overlay",
        template="plotly_dark",
        height=320,
        margin={"l": 48, "r": 16, "t": 32, "b": 40},
        xaxis_title="Hold latency (ms)",
        yaxis_title="Trials",
        legend={"orientation": "h", "y": 1.14},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def interlock_label(occluded: bool, clearance: float, policy: str, recovering: bool, recovery_ok: bool) -> str:
    if occluded:
        return "VISUAL HOLD"
    if recovering and recovery_ok:
        return "RECOVERY"
    if recovering:
        return "RECOVERY INCOMPLETE"
    if clearance < 0.0:
        return "INTRUSION"
    if clearance < MARGIN_MM:
        return "MARGIN BREACH"
    if policy == "cbf" and clearance < MARGIN_MM + 3.0:
        return "BARRIER SLIDE"
    return "NOMINAL"


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


report = load_report()
rollout = load_rollouts()
detector = load_detector()
if report is not None:
    reference = reference_from(report["occlusion"]["reference"])
else:
    reference = reference_from(fit_live_reference(0))

st.title("Surgical scene triage")
st.caption(
    "Synthetic laparoscopic plane. Disks stand in for structures the tip must not enter. "
    "Smoke, splash, blood, and the instrument bar are composited. This board is not a clinical device."
)

if report is None:
    st.warning("No recorded run yet. Metric cards stay empty until `python -m surgical_triage.runner --synthetic` has been executed.")

occ = None if report is None else report["occlusion"]
guided = None if report is None else report["barrier"]["cbf"]
hold = None if report is None else report["interlock"]
columns = st.columns(4)
columns[0].metric(
    "Smoke/splash rejection",
    "—" if occ is None else _pct(occ["primary_false_confidence_rejection"]),
)
columns[1].metric(
    "CBF penetration",
    "—" if guided is None else _pct(guided["penetration_rate"]),
)
columns[2].metric(
    "CBF worst clearance",
    "—" if guided is None else f"{guided['worst_clearance_mm']:.2f} mm",
)
columns[3].metric(
    "Async hold P99",
    "—" if hold is None else f"{hold['async']['p99_ms']:.2f} ms",
)

left, right = st.columns([1.05, 0.95])
with left:
    artifact_label = st.selectbox("Camera artifact", list(ARTIFACTS), index=0)
    artifact = ARTIFACTS[artifact_label]
    level = 1.0
    if artifact != "clean":
        level = st.slider("Artifact strength", 0.0, 1.0, 1.0, 0.05)
    policy_label = st.radio("Trajectory", ["CBF interlock", "Unconstrained"], horizontal=True)
    policy = "cbf" if policy_label == "CBF interlock" else "unconstrained"
    path = rollout["cbf_xy"] if policy == "cbf" else rollout["free_xy"]
    clearance_trace = rollout["cbf_clearance"] if policy == "cbf" else rollout["free_clearance"]
    default_step = int(np.argmin(rollout["cbf_clearance"]))
    step = st.slider("Control step", 0, int(path.shape[0] - 1), default_step)
    recovering = st.checkbox(f"Plan retract to {SAFE_CLEARANCE_MM:.0f} mm clearance", value=False)

clearance = float(clearance_trace[step])
tip = path[step]
history = path[: step + 1]
recovery = plan_recovery(tip, history, demo_episode().zones) if recovering else None
frame_rng = np.random.default_rng(5000 + step * 17 + int(level * 100) + ARTIFACT_SEED[artifact])
frame = render_frame(demo_episode(), tip, artifact, frame_rng, level=level)
decision = detector.decide(frame.gray, reference, frame.tip_rc)
dark = dark_fraction(frame.gray)
label = interlock_label(
    decision.occluded,
    clearance,
    policy,
    recovering,
    False if recovery is None else recovery.success,
)
view = compose(frame.rgb, rollout, policy, step, None if recovery is None else recovery.xy)

with left:
    st.image(view, caption="Red disks are anatomy. Amber rings are the 1 mm margin. The line is the selected trajectory up to this step. Yellow marks the target.", width="stretch")
    st.markdown(f"**Interlock: {label}**")

with right:
    st.markdown(
        f"Edge ratio `{decision.edge_ratio:.3f}` against cut `{reference.edge_ratio_max:.3f}`  \n"
        f"Entropy `{decision.entropy_nats:.3f}` nats against floor `{reference.entropy_floor:.3f}`  \n"
        f"Dark fraction `{dark:.3f}` against cut `{reference.dark_fraction_min:.3f}`  \n"
        f"Tip confidence `{decision.tip_confidence:.3f}` (baseline track at `{reference.confidence_min:.2f}`)  \n"
        f"Clearance `{clearance:.2f}` mm"
    )
    if recovery is not None:
        st.markdown(
            f"Retract `{recovery.initial_clearance_mm:.2f}` → `{recovery.final_clearance_mm:.2f}` mm "
            f"in `{recovery.steps}` steps. Success: `{recovery.success}`."
        )
    st.plotly_chart(clearance_figure(rollout), width="stretch")

st.plotly_chart(trajectory_figure(rollout, step, policy), width="stretch")

samples = load_latency()
if samples is not None:
    st.subheader("Recorded hold latency")
    st.plotly_chart(latency_figure(samples), width="stretch")
    st.caption("The next-tick distribution is the residual of a 10 ms control period plus the callback. The async trace is the watchdog already waiting on the fault.")

if report is not None:
    targets = report["targets"]
    rows = [
        ("Primary false-confidence rejection", targets["rejection_met"]),
        ("CBF penetration", targets["penetration_met"]),
        ("CBF worst clearance", targets["margin_met"]),
        ("Async hold P99", targets["latency_met"]),
        ("Retract success", targets["recovery_met"]),
    ]
    body = "\n".join(f"| {name} | {'met' if flag else 'not met'} |" for name, flag in rows)
    st.subheader("Recorded targets")
    st.markdown("| Target | Status |\n| --- | --- |\n" + body)
