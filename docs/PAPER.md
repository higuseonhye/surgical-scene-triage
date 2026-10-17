# A synthetic benchmark for visual-loss gating, disk barriers, and a 100 Hz hold

This note rewrites one repository run. It was not peer reviewed. It does not claim a result on a surgical robot or on an endoscope recording.

## Abstract

On one CPU, a specular tip filter stayed confident on 60 smoke frames and 60 lens-splash frames after the tissue texture had been flattened. A gate on mean absolute Laplacian energy and 64-bin luminance entropy refused all 120, and refused 0 of 60 clean frames. Decision P99 was 1.730 ms. The same tip, driven by a saturated attractor through a disk placed on the start-target segment, entered that disk on 40 of 40 episodes (worst clearance −5.50 mm). Projecting the command onto a signed-distance barrier with a 1 mm margin, then adding a tangent slide when the projection would otherwise stop the tip, entered the disk on 0 of 40 episodes (worst clearance 1.189 mm) and reached the target on all 40. A watchdog thread already waiting on the fault issued the hold with P99 1.956 ms over 300 trials. The same callback deferred to the next 10 ms tick had P99 11.418 ms. From the deepest intrusion, a heading search restored 3 mm of clearance on 40 of 40 episodes. No quadratic-program solver, no trained generative policy, and no surgical video were run.

## 1. Background

### 1.1 A confident track on a ruined frame

Electrosurgical smoke, fluid on the lens, and blood remove the high-frequency structure of a surgical view: fine surface detail and the edge of a metal tool. A highlight on the tool can survive that loss. A tracker matched to the highlight then reports a high score and a pose. The pose is no longer supported by the tissue in the frame.

The two-cue gate used for this failure mode is old signal processing. High-frequency collapse is written

$$
\operatorname{Var}(\nabla^2 I) < \tau_{\mathrm{edge}}.
$$

Distributional collapse is the Shannon entropy of luminance,

$$
H = -\sum_k p_k \ln p_k,
$$

on the frame or on local patches. The command from vision is withheld when either cue leaves the range measured on clean frames. The tracker score is the quantity being refused, not a third feature inside the gate.

Section 3 scores a first-moment version of the Laplacian, $E(I) = |\Omega|^{-1}\sum |\nabla^2 I|$, and one frame histogram. It also scores a dark-mass cut, because a shaft across the lens adds edges and would pass a Laplacian-only rule.

### 1.2 A barrier in front of a nominal command

A nominal velocity $u_{\mathrm{nom}}$ may be a hand-designed attractor or the output of a generative policy. The safety claim does not depend on which of those produced it. It depends on the command that is actually integrated.

For a disk that must not be entered, write

$$
h(p) = \|p - p_{\mathrm{vessel}}\|^2 - r_{\mathrm{safe}}^2
$$

and $\mathcal{C} = \{p : h(p) \ge 0\}$. Forward invariance of $\mathcal{C}$ requires

$$
\dot h(x, u) + \alpha(h(x)) \ge 0.
$$

The quadratic program that stays closest to the nominal command is

$$
\begin{aligned}
\min_u \quad & \|u - u_{\mathrm{nom}}\|^2 \\
\text{s.t.} \quad & \nabla h(x)^\top u + \alpha(h(x)) \ge 0,
\end{aligned}
$$

which is the single-integrator case of the control-barrier quadratic program in Ames, Xu, Grizzle, and Tabuada, IEEE Transactions on Automatic Control, 2017. If $u_{\mathrm{nom}}$ aims at the center of the disk, the projection onto that half-space is approximately zero at the boundary. A tangent component has to be present in the command that enters the filter, or the tip stops.

The signed distance $h_i(x) = \|x - c_i\| - R_i - m$ vanishes on the same circle when $r_{\mathrm{safe}} = R_i + m$. This run uses that form, with $m = 1$ mm, and enforces the half-space by projection rather than by a solver. A tangent slide is added in the nominal command while the attractor points into a nearby disk.

### 1.3 Hold, then retract

The state list is `NORMAL_TRACKING`, `VISUAL_LOSS`, `EMERGENCY_HOLD`, `SAFE_RETRACT`. Recognition of visual loss and publication of the hold share a 100 Hz period of 10 ms. A callback that runs only on the next tick can consume that period. A thread that is already blocked on the fault does not. After the hold, the tip is walked out along a heading that increases clearance. This run calls success at 3 mm of clearance.

## 2. Methods

The plane is 80 mm across and rendered at 128×128. Tissue is a sum of sinusoids and pixel noise. Three disks darken the image. One disk of radius 5.5 mm is centered on the segment from start to target. Smoke is a blur mixed toward gray. Splash is a heavier blur. Blood is a set of dark disks. An instrument bar replaces a vertical strip with a constant. A Gaussian tip of sigma 1.6 px is written after the artifact.

The baseline is a 9×9 zero-mean Gaussian on a 32×32 window at the tip. Confidence is the peak divided by the peak on an ideal tip. The cut is 0.55. The occlusion cuts are frozen on 36 frames per kind and applied to 60 later frames per kind. Edge ratio uses smoke and splash against clean. The dark cut uses the bar against clean. Entropy uses a 64-bin histogram.

The tool is a point mass at 100 Hz for 4.0 s, speed limit 30 mm/s. The unconstrained command is a gain of 4 toward the target, saturated at the speed limit. The barrier command is that vector, plus a tangent slide when the nearest expanded surface is within 6 mm and the command points inward, then projected onto $\nabla h_i^\top u \ge -\alpha h_i$ with $\alpha = 8\,\mathrm{s}^{-1}$ and onto the speed disk. The step is scaled to the largest fraction that does not enter an expanded disk. Penetration is anatomical clearance below zero. Goal reach is a final distance under 2 mm.

Hold trials alternate a smoke frame with a safe tip and a clean frame whose clearance is 0.2 mm. The synchronous path busy-waits a uniform fraction of 10 ms, then runs the callback. The asynchronous path timestamps the notify of a thread that is already waiting, and stamps the hold after the callback returns. Recovery starts at the deepest unconstrained sample and accepts the heading, among 16, that most increases worst-disk clearance, with a tie broken toward the incoming direction.

## 3. Results

The tip filter was confident on all 60 clean, smoke, splash, blood, and bar frames. The gate refused none of the clean frames and all of every artifact class. Primary false-confidence rejection, smoke plus splash, is 120 / 120. Mean edge ratios were 1.00 clean, 0.20 smoke, 0.12 splash, 0.18 blood, and 0.95 on the bar. Mean entropy was 3.05, 0.26, 2.40, 3.07, and 3.03 nats on those five classes. The bar is carried by a dark fraction of 0.125 against a cut of 0.0625. Blood's entropy stays above the floor. Decision P99 on the 120 primary frames was 1.730 ms.

The unconstrained tip entered the blocking disk on 40 / 40 episodes. Worst clearance was −5.50 mm, and the goal was still reached because the horizon is long enough to cross the disk. The barrier entered on 0 / 40, with worst clearance 1.189 mm, margin hold 100%, and goal reach 40 / 40. The illustrated field used by the board repeats the pattern: unconstrained clearance −5.89 mm, barrier clearance 1.16 mm, target reached.

Async hold mean was 1.325 ms and P99 was 1.956 ms, with every one of 300 trials asserting a hold. Next-tick mean was 6.128 ms and P99 was 11.418 ms. Recovery succeeded on 40 / 40 episodes, mean 28.8 steps, mean final clearance 3.19 mm.

## 4. What this run does not show

A variance threshold in place of mean absolute Laplacian energy. A grid of local entropies. A numerical quadratic program. A margin other than 1 mm, or a retract defined as a fixed 5 mm. A trained generative policy as $u_{\mathrm{nom}}$. A manipulator, a remote center of motion, or a time stamp taken on a robot bus. Behavior under a fainter smoke model than the one in `scene_generator.py`. Strength 0.2 of that model already recalls every tabulated frame, and it is already a visible blur.

## References

1. Ames AD, Xu X, Grizzle JW, Tabuada P. Control barrier function based quadratic programs for safety critical systems. IEEE Transactions on Automatic Control. 2017;62(8):3861–3876.
