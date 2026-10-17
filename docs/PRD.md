# Product requirements

This document states what `surgical-scene-triage` has to demonstrate. The repository is a synthetic benchmark. A service built on the same rules would still have to be validated as a clinical device on its own evidence. Measured values are the seed-0 run in [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).

## 1. Problem

Electrosurgical smoke, fluid on the lens, blood, and a tool across the camera can destroy the texture a vision tracker is using while a specular tip highlight remains. A controller that keeps integrating that highlight commands motion from a frame it should have refused.

A nominal command aimed only at a target will cross a critical structure when that structure lies on the straight path. The tip needs a margin, here 1 mm in the camera plane, that the target-seeking term cannot spend.

When the frame is refused or the margin is already gone, the hold has to be issued inside one period of a 100 Hz loop.

## 2. Three mechanisms

### 2.1 Two-cue visual gate

Refuse the track when high-frequency structure collapses or the luminance distribution leaves the clean band.

```math
\operatorname{Var}(\nabla^2 I) \lt \tau_{\mathrm{edge}}, \qquad H = -\sum_k p_k \ln p_k.
```

The scored detector uses mean absolute Laplacian energy in place of the variance, one 64-bin frame histogram in place of a patch grid, and a dark-pixel fraction for a shaft that does not reduce edge energy. The baseline tip filter is not part of the decision.

### 2.2 Disk barrier

Keep the command near $`u_{\mathrm{nom}}`$ subject to forward invariance of the exterior of each disk. The squared barrier

```math
h(p) = \|p - p_{\mathrm{vessel}}\|^2 - r_{\mathrm{safe}}^2
```

has safe set $`\mathcal{C} = \{h \ge 0\}`$. Admissible commands satisfy $`\dot h + \alpha(h) \ge 0`$. The nearest-command filter is

```math
\min_u \|u - u_{\mathrm{nom}}\|^2 \quad \text{subject to} \quad \dot h(x,u) + \alpha(h(x)) \ge 0.
```

The scored barrier is the signed distance $`h_i(x) = \|x - c_i\| - R_i - m`$ with $`m = 1`$ mm, which is the same disk. The code projects onto that half-space and rejects any 10 ms step that would enter the disk. It does not call a quadratic-program solver. A tangent slide is added when a head-on attractor would otherwise be projected to zero.

### 2.3 Hold and retract

```text
NORMAL_TRACKING -> VISUAL_LOSS -> EMERGENCY_HOLD -> SAFE_RETRACT
```

`VISUAL_LOSS` is the occlusion decision. `EMERGENCY_HOLD` is that decision or a tip already inside the margin. `SAFE_RETRACT` increases worst-disk clearance until it is at least 3 mm. The recognition budget stated for the decision is 2 ms. The publish budget for the hold is one 100 Hz period, 10 ms.

## 3. Users of this benchmark

- A local viewer that shows one synthetic camera, the disks, the margin, and the interlock state.
- A batch path that reports occlusion counts, clearance, and hold latency.

## 4. Functional requirements

| Item | Requirement |
| --- | --- |
| Scene | 128×128 rendering of an 80 mm plane. Three disks. One disk lies on the start-target segment |
| Artifacts | Smoke, lens splash, blood, and a vertical instrument bar, composited. A specular tip is stamped after the composite |
| Baseline | Zero-mean tip filter on a 32×32 window at the tracked tip. Confident when the peak is at least 0.55 of the ideal tip |
| Detector | Refuse the frame if Laplacian energy, 64-bin luminance entropy, or the dark-pixel fraction crosses a cut |
| Cuts | Frozen on a calibration split before evaluation. Smoke and splash set the edge cut. The bar sets the dark cut |
| Primary pool | Smoke and splash only. Blood and the bar are reported separately |
| Barrier | Point mass, 100 Hz, speed limit 30 mm/s. Signed-distance disk with a 1 mm margin. Exact step plus a tangent slide |
| Unconstrained | The same attractor with no barrier and no slide |
| Hold | Same callback. Sync: next 10 ms tick. Async: a thread already waiting on the fault |
| Retract | From the deepest unconstrained sample, reach 3 mm of clearance |

## 5. Out of scope

- A cleared device, a patient study, or an endoscope recording.
- A manipulator, a remote center of motion, or a measured robot round trip.
- A trained generative policy. The attractor is a saturated proportional field. The quadratic program in section 2 is the specification. The code uses projection.
- Camera exposure, serialization, and bus delay. Latency starts when the frame and the tip state are already in memory.
- A claim that strength 0.2 on the smoke slider is a faint plume. In this generator that setting is already a blur and a mix toward gray.
- A fixed 5 mm retreat. Success in this run is 3 mm of clearance.

## 6. Acceptance

| Criterion | Target | Seed 0 |
| --- | --- | --- |
| Primary false-confidence rejection | at least 90% | 100% (120 / 120) |
| Clean frames refused | Report | 0 / 60 |
| Decision latency | Report against a 2 ms recognition budget | P99 1.730 ms |
| Barrier penetration | 0% | 0 / 40 |
| Barrier worst clearance | at least 1.0 mm | 1.189 mm |
| Barrier goal reach | Report | 40 / 40 |
| Unconstrained penetration | Report | 40 / 40, worst clearance −5.50 mm |
| Async hold P99 | under 10.0 ms | 1.956 ms (300 trials) |
| Next-tick hold P99 | Report | 11.418 ms |
| Retract success | at least 95% | 100% (40 / 40), mean final clearance 3.19 mm |

Latency moves if the run is repeated. Occlusion counts and penetration counts are fixed for seed 0.

## 7. Documents

| File | Role |
| --- | --- |
| [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) | The measured run |
| [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) | What the code implements |
| [PAPER.md](PAPER.md) | The same run as a technical note |
