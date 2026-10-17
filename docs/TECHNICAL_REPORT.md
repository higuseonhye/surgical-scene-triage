# Technical report

This report describes the seed-0 implementation scored in [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md). It separates the standard statement of each safety rule from the discrete form the runner actually times.

## 1. Scene

`scene_generator.py` builds an 80 mm camera plane and renders it at 128×128. The origin is the image center, +x is right, +y is up. Tissue is a sum of sinusoids plus pixel noise. Each no-go zone is a disk that darkens the rendering. The tool shaft is a bright segment from a port at the near edge to the tip.

The photometric artifacts are applied after the shaft and before the tip:

| Artifact | Operation at full strength |
| --- | --- |
| Smoke | Gaussian blur, then a mix toward a flat gray |
| Splash | A heavier blur, a few bright droplets, then another blur |
| Blood | Several dark disks and a light blur |
| Instrument bar | A vertical strip replaced by a near-black constant |

`stamp_tip` then writes a Gaussian highlight of peak 1 and sigma 1.6 px. That order is why a tip filter can stay confident after the field has lost its texture.

Episodes place a disk of radius 5.5 mm on the open segment from start to target, and two smaller disks far enough off that line that a point tip has a corridor around the blocker. The illustrated field in `demo_episode` uses the same construction with a 6 mm blocker. The board draws that field. The 40-episode table does not.

## 2. Visual gate

### 2.1 Standard statement

A usable frame still contains high-frequency structure. The scalar used for that collapse is the variance of the Laplacian,

```math
\mathrm{Var}(\nabla^2 I) \lt \tau_{\mathrm{edge}}.
```

The second scalar is the Shannon entropy of luminance. On a histogram with masses $`p_k`$,

```math
H(I) = -\sum_k p_k \ln p_k.
```

A local-patch version of $`H`$ asks whether a neighborhood has become uniform. The frame-level version asks the same question of the whole image. The gate withholds the vision command when either cue leaves the clean operating range. The tip tracker is not one of the cues. Its confidence is what the gate is designed to refuse.

### 2.2 What is scored

`OcclusionDetector` holds two fixed kernels. Nothing is trained.

Laplacian energy is the mean absolute response of a 3×3 Laplacian, a Torch convolution over the full frame:

```math
E(I) = \frac{1}{|\Omega|}\sum_{p \in \Omega} |(\nabla^2 I)(p)|.
```

$`E`$ and $`\mathrm{Var}(\nabla^2 I)`$ both fall when the high-frequency content is flattened. They are not the same number. The cut is on the ratio of $`E`$ to the clean median, not on a variance threshold. Entropy is the 64-bin histogram of the whole frame on $`[0, 1]`$, in nats. There is no patch grid. The dark fraction is the share of pixels below 0.12. The baseline confidence is a separate Torch convolution: a 9×9 zero-mean Gaussian on a 32×32 crop at the tracked tip, divided by the peak on an ideal tip. A confident track is a ratio of at least 0.55.

Calibration, 36 frames per kind at seed 0, freezes three cuts before any evaluation frame is scored:

- Edge ratio: the midpoint between the 95th percentile of smoke and splash ratios and the 5th percentile of clean ratios, when those tails do not overlap. Clean median energy was 0.1633. The cut was 0.595 of that median.
- Entropy floor: the midpoint between the lowest clean entropy and the highest smoke or splash entropy that still sits below every clean calibration frame. The floor was 2.816 nats.
- Dark fraction: the midpoint between the clean tail and the instrument-bar tail. The cut was 0.0625.

A frame is refused if any one cut trips. Evaluation is 60 fresh frames per kind.

On that split the tip filter was confident on every clean frame and every artifact frame. The detector refused 0 / 60 clean frames and 60 / 60 of each artifact. Smoke and splash, the primary pool, are 120 / 120. Smoke tripped both the edge cut and the entropy floor (mean entropy 0.26 nats). Splash did the same (mean entropy 2.40 nats, mean edge ratio 0.12). Blood tripped the edge cut on every frame and the dark cut on 22 / 60. Its mean entropy stayed at 3.07 nats, above the floor. The instrument bar did not trip the edge cut (mean edge ratio 0.95) or the entropy floor. Its dark fraction was 0.125 on every frame. A Laplacian-only rule would have accepted the bar.

Decision latency was measured on the 120 primary evaluation frames after the pixels were already in memory, with 20 warm-up calls. Mean 1.108 ms, P99 1.730 ms. That P99 is inside a 2 ms recognition budget. It does not include exposure or transfer.

The smoke sweep uses the same frozen cuts on 24 new frames at strengths 0, 0.2, 0.4, 0.6, 0.8, and 1. Strength 0 recalled nothing. Strength 0.2 and above recalled all 24. Strength 0.2 in this generator is already a blur of about 1.3 px and a substantial mix toward gray.

## 3. Disk barrier

### 3.1 Standard statement

Let $`p`$ be the tool tip and let a no-go disk have center $`p_{\mathrm{vessel}}`$ and safe radius $`r_{\mathrm{safe}}`$, the anatomical radius plus the margin. The squared barrier

```math
h(p) = \|p - p_{\mathrm{vessel}}\|^2 - r_{\mathrm{safe}}^2
```

is positive outside the disk, zero on the boundary, and negative inside. The safe set $`\mathcal{C} = \{h \ge 0\}`$ is forward invariant when every applied command satisfies

```math
\dot h(x, u) + \alpha(h(x)) \ge 0
```

for an extended class-$`\mathcal{K}`$ function $`\alpha`$. Among those commands, the quadratic program

```math
\begin{aligned}
\min_u \quad & \|u - u_{\mathrm{nom}}\|^2 \\
\text{s.t.} \quad & \nabla h(x)^\top u + \alpha(h(x)) \ge 0
\end{aligned}
```

picks the one nearest the nominal command, for a single integrator $`\dot p = u`$. Ames, Xu, Grizzle, and Tabuada (IEEE Transactions on Automatic Control, 2017) state this filter. If $`u_{\mathrm{nom}}`$ points at the disk center, the feasible set near $`h = 0`$ is the tangent line, and a solver that only projects $`u_{\mathrm{nom}}`$ returns approximately zero. A safe nominal has to supply the tangent direction. The filter then keeps that direction feasible.

### 3.2 What is scored

The state is the tip position in millimetres. The command is a velocity in mm/s, integrated at $`\Delta t = 10`$ ms for 400 steps. The speed limit is 30 mm/s. The nominal command is a gain of 4 on the vector to the target, saturated at the speed limit. That field leaves the start-target segment invariant, which is why every unconstrained episode penetrates the blocking disk. Worst unconstrained clearance was −5.50 mm. All 40 unconstrained episodes still reached the target.

The scored barrier is the signed distance

```math
h_i(x) = \|x - c_i\| - R_i - m, \qquad m = 1\,\mathrm{mm}.
```

If $`r_{\mathrm{safe}} = R_i + m`$, this $`h_i`$ and the squared barrier vanish on the same circle. Their gradients are positive scalar multiples of each other outside the center, so the half-space $`\nabla h^\top u \ge -\alpha h`$ describes the same family of admissible directions, with a different gain. Here $`\alpha = 8\,\mathrm{s}^{-1}`$.

The command is projected onto every half-space and onto the speed disk by successive correction. No quadratic-program solver is called. The step is then scaled by the largest $`s \in [0, 1]`$ such that $`x + s \Delta t\, u`$ stays outside every expanded disk. That scale is the closed-form entry time of a ray against a circle. An inward step that starts on the expanded boundary gets scale 0.

`barrier_velocity` adds a unit tangent, scaled by the speed limit, while the closest expanded surface is within 6 mm and the nominal command points into that disk. The side agrees with the direction to the target, or, on the colinear case, with the side that has more clearance 4 mm ahead. The summed command is projected again. On the seed-0 episodes this slide is what carries the tip around the blocker. Goal reach was 40 / 40. Penetration was 0 / 40. Worst clearance was 1.189 mm. The illustrated field matches that pattern: unconstrained clearance −5.89 mm, barrier clearance 1.16 mm, target reached.

This is a point mass in the plane. It is not a joint-space quadratic program, and it does not model a remote center of motion.

## 4. Hold and retract

The states used by the benchmark are:

```text
NORMAL_TRACKING -> VISUAL_LOSS -> EMERGENCY_HOLD -> SAFE_RETRACT
```

`VISUAL_LOSS` is `decide`. `EMERGENCY_HOLD` is `evaluate_interlock`: the frame was refused, or clearance is already under 1 mm, or both. `SAFE_RETRACT` is `plan_recovery`.

Each of 300 hold trials is one of two faults, alternating. Even trials pass a smoke frame and a clearance of 12 mm, so only the detector can raise the hold. Odd trials pass a clean frame and a clearance of 0.2 mm, so the clearance check raises the hold. Both checks run on every trial. All 300 trials in each scheduler asserted a hold.

The synchronous scheduler draws the fault phase uniformly from $`[0, 10)`$ ms and busy-waits until that phase has elapsed, then runs the callback. Mean 6.128 ms, P99 11.418 ms. The P99 sits above 10 ms because the residual can be almost a full period and the callback still has to run.

The asynchronous scheduler starts one thread and blocks it on a condition variable. The timestamp is taken on the calling thread immediately before the notify. The hold time is taken after the detector and the clearance check return on the watchdog thread. Twenty warm-up jobs run on that thread before the 300 timed trials. Mean 1.325 ms, P99 1.956 ms, max 2.570 ms. That P99 is inside the 10 ms period. It is an in-process wake and callback, not a robot bus measurement.

`plan_recovery` starts at the sample of minimum clearance on the unconstrained rollout, inside the blocking disk. Each 10 ms step tries 16 headings at the speed limit and keeps the one that most increases the worst-disk clearance. A tie prefers the reverse of the incoming step. The horizon is 1.5 s. Success is clearance of at least 3 mm, not a fixed 5 mm retreat. All 40 episodes succeeded, in 28.8 steps on average, with mean final clearance 3.19 mm.

## 5. What was not run

No surgical video was decoded. No weights were loaded. The nominal command was not a diffusion or flow model. Torch is used for the Laplacian and the tip kernel only. SciPy supplies the Gaussian blur in the renderer. The board reads the seed-0 JSON and replays `demo_episode`. Moving a slider does not refit the cuts.
