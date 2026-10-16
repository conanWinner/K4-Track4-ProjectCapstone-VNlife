# Known limitations

Research simulation, **not** a safety-certified system. Every limitation below is visible in the logs.

## 1. Constant-velocity model vs sudden turns (main failure case)
`sudden_turn`: the actor almost leaves the corridor, so the CV track says "receding" and the robot resumes;
then the actor turns back at 1.2 m/s just as the robot reaches the crossing line. The Kalman filter needs
several 10 Hz scans before the velocity estimate flips sign (innovation must outweigh the prior), so the
YIELD is issued too late. Result: collision 5/5 seeds for both modes, on AMR and G1 (G1 reactive: knocked
over 5/5). The same reversal while the robot is still yielding far back (`sudden_turn_early`) is handled
100%. Mitigations not implemented (each trades off false stops): innovation/NIS-based maneuver detection,
IMM with constant-turn model, larger margin while a track is young or its NIS is high, shorter horizon
when the track is unstable.

## 2. Ego pose is a simulator oracle
`RobotAdapter.robot_state` reads the base pose from MuJoCo. No localization is claimed. A real system needs
odometry/IMU fusion; pose noise would enter the sensor→world transform and inflate R.

## 3. Detection = visible-surface centroid
A single 2D scan only sees the near side of the actor, so the centroid is biased toward the sensor by up to
one radius (mean track position error vs GT center ≈ 0.20–0.21 m in the crossing scenarios). Compensated by the margin, not modelled.

## 4. Conservative behaviour near static obstacles
Unknown static obstacles (the occluder box) are tracked like any obstacle. If the robot stops close to one,
the "currently overlapping" rule (centers closer than r_robot + r_obs + nominal margin) keeps risk on.
On G1, `brief_occlusion` predictive reaches the goal 5/5 but takes ~32 s vs ~21 s (waiting beside the box).

## 5. G1-specific
- At zero command the pretrained policy keeps stepping in place and drifts (~0.3 m lateral and ~0.7°/s
  yaw per 10 s). YIELD therefore does station keeping (heading hold + small vy back to the yield point).
- Pelvis velocity contains gait sway (~0.075 m/s lateral); an EMA (τ = 0.4 s) is applied before risk.
- Robot footprint approximated by a 0.42 m circle (arms swing outside 0.35 m). Foot sweep not modelled.
- No backing up or sidestepping around people (AVOID_LEFT/RIGHT not implemented): the robot only waits.
- Contact with the actor at speed can knock G1 over (counted as `fall`).

## 6. Sensor model
Ideal 2D fan of 61 rays at one height, Gaussian range noise 2.5 cm, 1% dropout. No reflections, no
multipath, no motion distortion, no people occluding each other in crowds. Only one actor per scenario.

## 7. Evaluation proxies
- Clearance and collision-free success use circle approximations + GT, evaluation only.
- Collision = any MuJoCo contact between robot bodies and the actor geom.
- "False stop" = YIELD entry after which GT clearance stays > 1.5 m for the next horizon; observed while
  the robot is stopped, so it is a proxy.
- 5 seeds per (scenario, mode): enough to show the qualitative gap, not a statistical guarantee.

## 8. Stale-sensor stop
The planner stops (`STOP_SENSOR_FAILURE`) when the last scan is older than 0.3 s. In normal runs scans never
go missing; use `sensor.blackout_s: [t0, t1]` in the config to demonstrate it.
