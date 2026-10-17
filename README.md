# Surgical scene triage

Synthetic benchmark for three safety questions on a planar surgical tool: when a specular tip track must be refused, when a nominal command must be bent off a disk that stands in for a critical structure, and when the hold that follows has to land inside one 100 Hz period.

This is a benchmark, not a clinical device. It does not read an endoscope recording. The numbers below are one CPU run at seed 0. [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md) is that run, copied to [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md). [docs/PRD.md](docs/PRD.md) is the requirement. [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) is the implementation. [docs/PAPER.md](docs/PAPER.md) is the same run as a technical note.

## Demo

```powershell
pip install -r requirements.txt
python -m surgical_triage.runner --synthetic
streamlit run app.py
```

The board is local only, at [http://127.0.0.1:8505](http://127.0.0.1:8505). Metric cards are the recorded run. The artifact control, the trajectory choice, the step slider, and the retract checkbox re-render the camera and recompute the interlock. They do not refit the cuts.

Seed 0:

| Question | Result |
| --- | --- |
| Occlusion | Smoke and splash false-confidence rejection 100% (120 / 120). Clean frames refused: 0 / 60. Decision P99 1.730 ms. |
| No-go zone | Unconstrained penetration 40 / 40, worst clearance −5.50 mm. Barrier penetration 0 / 40, worst clearance 1.189 mm, goal reach 40 / 40. |
| Hold | Async watchdog P99 1.956 ms. Next 100 Hz tick P99 11.418 ms. Retract to 3 mm clearance 40 / 40. |

## 1. Visual occlusion triage

Electrosurgical smoke, fluid on the lens, and blood flatten tissue texture while a specular tip highlight can remain. A tracker that keeps reporting that highlight will keep issuing a command from a frame that no longer describes the field.

Two cues mark that loss. A clear field still has high-frequency structure: fine surface detail and the metal edge of a tool. Smoke and lens contamination remove it. The usual scalar is the variance of the Laplacian,

$$
\operatorname{Var}(\nabla^2 I) < \tau_{\mathrm{edge}}.
$$

The second cue is the entropy of the luminance distribution. A wash toward a single gray, or a field that has collapsed into uniform noise, moves

$$
H = -\sum_k p_k \ln p_k
$$

off the band measured on clean frames. Either cue is enough to mark the frame `VISUAL_LOSS` and to withhold the track.

This repository scores a discrete form of that pair. High-frequency loss is the mean absolute Laplacian, $E(I) = |\Omega|^{-1}\sum |\nabla^2 I|$, compared with the clean median. Entropy is one 64-bin histogram of the whole frame. A vertical instrument bar does not reduce $E(I)$, because the bar edges are sharp, so a dark-pixel fraction is a third cut for that case. The baseline tip filter is scored only as the thing the gate refuses. On the seed-0 evaluation it stayed confident on every artifact frame, and the gate refused all 120 smoke and splash frames. Decision P99 was 1.730 ms.

## 2. No-go zone barrier

A nominal command $u_{\mathrm{nom}}$ can come from a tracker or from a generative policy. Either source can point the tip at a structure that must not be entered. The filter keeps the command close to $u_{\mathrm{nom}}$ while enforcing a barrier. For a disk of safe radius $r_{\mathrm{safe}}$ about a center $p_{\mathrm{vessel}}$,

$$
h(p) = \|p - p_{\mathrm{vessel}}\|^2 - r_{\mathrm{safe}}^2.
$$

The safe set is $\mathcal{C} = \{p : h(p) \ge 0\}$. The boundary $h = 0$ is the disk. $h < 0$ is the interior. Forward invariance of $\mathcal{C}$ asks for a control that satisfies

$$
\dot h(x, u) + \alpha(h(x)) \ge 0
$$

for a class-$\mathcal{K}$ function $\alpha$. The quadratic program that stays nearest the nominal command is

$$
\begin{aligned}
\min_u \quad & \|u - u_{\mathrm{nom}}\|^2 \\
\text{s.t.} \quad & \dot h(x, u) + \alpha(h(x)) \ge 0.
\end{aligned}
$$

For a point with $\dot p = u$, $\dot h = \nabla h^\top u$.

The scored barrier uses the signed distance to the same disk,

$$
h_i(x) = \|x - c_i\| - R_i - m, \qquad m = 1\,\mathrm{mm},
$$

so $r_{\mathrm{safe}} = R_i + m$. The zero level set matches the squared form. The code does not call a quadratic-program solver. It projects $u_{\mathrm{nom}}$ onto $\nabla h_i^\top u \ge -\alpha h_i$ and onto the speed limit, then shortens the 10 ms step so the straight move stays outside every expanded disk. A head-on attractor has no tangential component left after that projection, so the safe nominal adds a tangent slide while the command points into a disk. On 40 episodes the unconstrained attractor penetrated every blocking disk (worst clearance −5.50 mm). The barrier penetrated none, held a worst clearance of 1.189 mm, and reached the target on every episode.

## 3. Emergency hold and retract

The interlock is a short state list:

```text
NORMAL_TRACKING
    -> VISUAL_LOSS
    -> EMERGENCY_HOLD
    -> SAFE_RETRACT
```

`VISUAL_LOSS` is the occlusion decision. `EMERGENCY_HOLD` is the same callback when the frame is refused or the tip is already inside the margin. `SAFE_RETRACT` walks headings at the speed limit until clearance is restored. A 100 Hz period is 10 ms. On this run the decision P99 was 1.730 ms and the async hold P99 was 1.956 ms, both inside that period. The watchdog is a thread that is already waiting. The comparison scheduler waits for the next tick, and its P99 was 11.418 ms. Success for the retract is clearance of at least 3 mm. All 40 episodes reached it, in 28.8 steps on average.

## Experiments

| Experiment | Comparison | Target |
| --- | --- | --- |
| Occlusion | Tip-filter confidence versus Laplacian, entropy, and dark-mass cuts | False-confidence rejection at least 90% on smoke and splash |
| No-go zone | Saturated attractor versus the same attractor under the disk barrier | Penetration 0%, worst clearance at least 1 mm |
| Hold | Next 100 Hz tick versus a waiting interlock thread | Async P99 under 10 ms |

`--synthetic` is required. Defaults are 36 calibration frames and 60 evaluation frames per artifact, 40 barrier episodes, 300 hold trials, and seed 0. Without the flag the runner exits 2.

## Layout

```text
surgical_triage/     scene, occlusion cues, disk barrier, retract, runner
app.py               camera, clearance, hold latency
results/             seed-0 JSON and dashboard arrays
docs/PRD.md
docs/TECHNICAL_REPORT.md
docs/PAPER.md
docs/BENCHMARK_RESULTS.md
```

## References

Ames, Xu, Grizzle, and Tabuada, "Control barrier function based quadratic programs for safety critical systems," IEEE Transactions on Automatic Control, 2017, is the quadratic-program statement in section 2. This repository does not solve that program numerically. The entropy above is a luminance histogram, not the entropy of a network distribution. No surgical robot interface and no trained visuomotor policy are loaded.
