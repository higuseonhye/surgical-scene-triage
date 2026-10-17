# Surgical scene triage — benchmark results

One CPU run on a synthetic camera plane. No endoscope recording, no patient study, and no trained manipulation policy is loaded.

- Seed 0. Package 0.1.0.
- Occlusion calibration is 36 frames per kind. Evaluation is 60 fresh frames per kind.
- Barrier rollouts: 40 episodes, 400 steps at 100 Hz, speed limit 30 mm/s, margin 1.0 mm.
- Hold trials: 300. Half are a smoky frame with a safe tip. Half are a clean frame whose tip is already inside the margin.

## 1. Visual occlusion triage

The baseline is a zero-mean tip filter on a 32×32 window at the tracked tip. A frame is a confident track when that response is at least 0.55 of the response on an ideal specular tip. The filter does not gate the controller.

The detector flags a frame when any one cue trips, using cuts frozen on the calibration split:

- Laplacian energy below 0.595 times the clean median (0.1633).
- Luminance entropy below 2.816 nats.
- Dark-pixel fraction (luminance < 0.12) at or above 0.062.

False-confidence rejection is the share of those confident tracks, on ground-truth artifact frames, that the detector refuses. The primary pool is smoke and splash. Blood and the instrument bar are reported beside them.

| Kind | Frames | Baseline confident | Detector recall | False-confidence rejection | Mean edge ratio | Mean entropy | Mean dark fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 60 | 100.0% | 0.0% | 0.0% | 1.00 | 3.05 | 0.000 |
| smoke | 60 | 100.0% | 100.0% | 100.0% | 0.20 | 0.26 | 0.000 |
| splash | 60 | 100.0% | 100.0% | 100.0% | 0.12 | 2.40 | 0.000 |
| blood | 60 | 100.0% | 100.0% | 100.0% | 0.18 | 3.07 | 0.050 |
| occluder | 60 | 100.0% | 100.0% | 100.0% | 0.95 | 3.03 | 0.125 |

Primary rejection (smoke + splash): 100.0%. All four artifacts: 100.0%. Clean false-rejection rate: 0.0%. Target (primary rejection ≥ 90%): met.

Detection latency is the wall time of one decision after the frame is already in memory. It is not exposure, transfer, or display time.

|  | N | Mean ms | P50 ms | P99 ms |
| --- | --- | --- | --- | --- |
| Detector | 120 | 1.108 | 1.082 | 1.730 |

Smoke strength is a mix toward a flat fog plus blur. The same frozen cuts are applied. The lowest tabulated strength with recall at least 90% is 0.2.

| Smoke strength | Frames | Recall | False-confidence rejection |
| --- | --- | --- | --- |
| 0.0 | 24 | 0.0% | 0.0% |
| 0.2 | 24 | 100.0% | 100.0% |
| 0.4 | 24 | 100.0% | 100.0% |
| 0.6 | 24 | 100.0% | 100.0% |
| 0.8 | 24 | 100.0% | 100.0% |
| 1.0 | 24 | 100.0% | 100.0% |

Edge energy falls on smoke, splash, and blood. It does not fall on the instrument bar, which replaces a vertical strip with a constant and adds two strong edges. That bar is carried by the dark-fraction cue. Entropy collapses on dense smoke and on splash; blood keeps a broad histogram, so blood is an edge decision.

## 2. No-go zone margin

Both policies use the same saturated attractor toward the target. The straight segment is an invariant of that field, and the blocking disk is built on the segment, so the unconstrained tip goes through the disk. The CBF policy projects the command onto ∇h · u ≥ −α h and onto the speed limit, then refuses any straight step that would enter an expanded disk. A head-on attractor has no tangential component left after that projection, so the safe policy adds a tangent slide while the attractor points into a disk closer than 6 mm. The slide is projected again. It is not a trained diffusion policy.

| Policy | Penetration | Worst clearance mm | Median episode minimum mm | Margin hold | Goal reach |
| --- | --- | --- | --- | --- | --- |
| Unconstrained | 100.0% | -5.50 | -5.44 | 89.2% | 100.0% |
| CBF interlock | 0.0% | 1.19 | 1.20 | 100.0% | 100.0% |

Penetration means anatomical clearance below 0. The margin target is 1.0 mm. CBF penetration target (0%): met. CBF worst clearance ≥ 1.0 mm: met.

Clearance is ||x − c|| − R for the nearest disk, in millimetres. Goal reach means the last sample is within 2 mm of the target.

The illustrated field used by the board has unconstrained clearance -5.89 mm and CBF clearance 1.16 mm. CBF reached the illustrated target: True.

## 3. Emergency hold and retract

The synchronous scheduler samples the same callback on the next 100 Hz tick. The fault time is uniform inside the 10 ms period, and the wait is a busy loop up to that tick so the residual is not an operating-system sleep quantum. The asynchronous scheduler is a thread that is already blocked on a condition variable. The timestamp is taken before the wake, and the hold is recorded after the detector and the clearance check both finish.

| Scheduler | N | Mean ms | P50 ms | P99 ms | Max ms |
| --- | --- | --- | --- | --- | --- |
| Next control tick | 300 | 6.128 | 6.031 | 11.418 | 11.543 |
| Async interlock | 300 | 1.325 | 1.318 | 1.956 | 2.570 |

Async P99 < 10.0 ms: met. Trials that did not assert a hold: async 0, sync 0.

Recovery starts at the deepest unconstrained sample, inside the blocking disk, and steps at the speed limit along the heading that most increases the worst-zone clearance. Ties prefer the incoming direction. Success is clearance at least 3.0 mm. Success rate 100.0% over 40 episodes (mean 28.8 steps, mean final clearance 3.19 mm). Target ≥ 95%: met.

## Targets

| Target | Required | Measured | Status |
| --- | --- | --- | --- |
| Primary false-confidence rejection | ≥ 90% | 100.0% | met |
| CBF penetration | 0% | 0.0% | met |
| CBF worst clearance | ≥ 1.0 mm | 1.189 mm | met |
| Async hold P99 | < 10.0 ms | 1.956 ms | met |
| Retract success | ≥ 95% | 100.0% | met |

## What this run is not

The field is a 128×128 rendering of an 80 mm plane. Disks are not vessels segmented from a patient. Smoke, splash, blood, and the bar are composited degradations, not a scattering model of electrosurgical plume. The robot is a point mass. Latency is inside this process. A later controller would still have to account for camera exposure, bus delay, and the robot's own tracking loop.
