# Unitree G1 integration (pretrained unitree_rl_gym policy in MuJoCo)

Status: **SIM VERIFIED** (headless, MuJoCo 3.16.0, torch 2.6.0+cpu, Windows 11): the robot alone on
flat ground (this document), and G1 inside the full corridor + perception + planner loop
(`sim/runner.py`, 70 episodes, no falls except when struck by the actor — see `docs/experiments.md`).

## Upstream

| Item | Value |
|---|---|
| Repo | https://github.com/unitreerobotics/unitree_rl_gym |
| Pinned SHA | `276801e46c5d433564f24658bac64f254b7d2d4b` (2025-07-25, "Merge pull request #67"), see `configs/upstream_version.txt` |
| Checkout | `third_party/unitree_rl_gym` (git-ignored; `scripts/setup.ps1` / `setup.sh` fetch exactly this SHA) |
| Sim2Sim entrypoint | `deploy/deploy_mujoco/deploy_mujoco.py g1.yaml` (viewer loop; we reproduce it headless) |
| Deploy config | `deploy/deploy_mujoco/configs/g1.yaml` |
| Policy weights | `deploy/pre_train/g1/motion.pt` (145,745 bytes, TorchScript; shipped in the repo, nothing fake) |
| Scene MJCF | `resources/robots/g1_description/scene.xml` (includes `g1_12dof.xml`, floor plane, light) |
| Training config | `legged_gym/envs/g1/g1_config.py`, base `legged_gym/envs/base/legged_robot_config.py`, obs in `legged_gym/envs/g1/g1_env.py` |

The policy is an **LSTM** (`ActorCriticRecurrent`, LSTM hidden 64, actor MLP [32], ELU) exported by
`PolicyExporterLSTM`: the hidden/cell state are buffers inside the TorchScript module and are
updated on every call. `G1Adapter.reset()` calls `policy.reset_memory()`.

## Rates

| Quantity | Value | Source |
|---|---|---|
| physics dt | 0.002 s (500 Hz) | g1.yaml `simulation_dt` (deploy sets `m.opt.timestep`; adapter sets `spec.option.timestep` and asserts it in `bind`) |
| control_decimation | 10 | g1.yaml |
| policy rate | 0.02 s (50 Hz) | 0.002 x 10; same as training (`sim.dt` 0.005 x `decimation` 4) |
| PD torque | every physics step | deploy_mujoco.py |
| gait clock period | 0.8 s | hard-coded in `g1_env.py` and `deploy_mujoco.py` |

`cfg["g1"]["policy_rate_hz"] = 50` in `configs/default.yaml` is confirmed correct.

## Robot model and joints (policy order = actuator order = `qpos[7:19]`)

| idx | joint | kp [N m/rad] | kd [N m s/rad] | default angle [rad] | torque limit [N m] |
|---|---|---|---|---|---|
| 0 | left_hip_pitch_joint | 100 | 2 | -0.1 | 88 |
| 1 | left_hip_roll_joint | 100 | 2 | 0.0 | 139 |
| 2 | left_hip_yaw_joint | 100 | 2 | 0.0 | 88 |
| 3 | left_knee_joint | 150 | 4 | 0.3 | 139 |
| 4 | left_ankle_pitch_joint | 40 | 2 | -0.2 | 50 |
| 5 | left_ankle_roll_joint | 40 | 2 | 0.0 | 50 |
| 6-11 | right_* (same order) | same | same | same | same |

Actuators are `<motor>` (torque) with limits from the joints' `actuatorfrcrange`. Base body `pelvis`
(free joint `floating_base_joint`, initial height 0.793 m from the MJCF). The 12-DoF model has **no
`torso_link` body**: torso, head and arm meshes are welded to the pelvis. Robot geoms use groups
0 (collision) and 1 (visual) only, so none are in the perceivable group 3 (the adapter would move
any to group 1). Upstream does not change any robot actuator/joint/gain/collision setting.

## Observation (47) — exactly `deploy_mujoco.py`

| slice | content | scale |
|---|---|---|
| 0:3 | base angular velocity, body frame (`qvel[3:6]` of the free joint) | `ang_vel_scale` 0.25 |
| 3:6 | projected gravity from base quat `qpos[3:7]` (w,x,y,z): `[2(-qz qx + qw qy), -2(qz qy + qw qx), 1 - 2(qw^2 + qz^2)]` | 1 |
| 6:9 | command [vx, vy, yaw_rate] (body frame) | `cmd_scale` [2.0, 2.0, 0.25] |
| 9:21 | q - default_angles | `dof_pos_scale` 1.0 |
| 21:33 | dq | `dof_vel_scale` 0.05 |
| 33:45 | previous raw action | 1 |
| 45:47 | sin(2 pi phase), cos(2 pi phase); phase = (t mod 0.8)/0.8 | 1 |

`t` upstream is `counter * simulation_dt` with `counter` = physics steps since start; the adapter
uses `data.time - t_reset` (identical: the obs is built at the start of step k, after k steps).
No base linear velocity in the actor obs (it is only in the 50-dim privileged critic obs).

## Action (12)

`target_q = action * action_scale(0.25) + default_angles`;
`tau = kp * (target_q - q) + kd * (0 - dq)` written to `data.ctrl` every physics step. Targets are
updated at physics steps k = 10, 20, ... (k = 0 uses default angles, action 0) — same as upstream.

## Command range

Training (`G1RoughCfg` does not override base `commands.ranges`): `lin_vel_x` [-1, 1] m/s,
`lin_vel_y` [-1, 1] m/s, `ang_vel_yaw` [-1, 1] rad/s, with `heading_command = True` (yaw-rate =
clip(0.5 * heading error, -1, 1)). Training also **zeroed linear commands with |v_xy| < 0.2**, so
small non-zero speeds were never trained explicitly (they still work, see below). Deploy default
`cmd_init` = [0.5, 0, 0].

`G1Adapter.set_command` clips vx to [-max_vx, max_vx], vy to the same, yaw rate to [-1, 1], with
`max_vx = min(cfg g1.max_vx_mps, 1.0)`.

## Adapter (`src/humanoid_avoidance/sim/g1_adapter.py`)

* `base_spec()`: `MjSpec.from_file(scene.xml)`, absolute `meshdir` (compiles from any cwd and after
  the scene builder adds bodies), `option.timestep = 0.002`.
* `bind()`: resolves the 12 joints by name, their qpos/qvel addresses and actuator ids (does not
  assume the robot is the only free joint), asserts the timestep.
* `reset()`: the only `qpos` writer (enforced by an AST test): xy/yaw from args, z = 0.793,
  joints = default angles, zero velocities and ctrl, LSTM memory and last action cleared, `mj_forward`.
  Upstream starts from all-zero joint angles; we start at the default angles (training init); the
  first-1 s trajectory still matches upstream's loop to < 1e-4 when both start from the same state.
* `pre_step()`: must be called before **every** `mj_step`. Policy step index derived from
  `data.time`, so an extra call without a physics step does not double-step the policy.
* `robot_state()`: pelvis xy/yaw (localization oracle), `v_world` = free-joint `qvel[0:2]`
  (MuJoCo: free-joint linear velocity is world frame; angular velocity is body frame, rotated to
  world for `yaw_rate`), `base_height` = pelvis z.
* `sensor_pose()`: pelvis frame + offset (0.10, 0, 0.30) m (override with `cfg["g1"]["sensor_offset_m"]`),
  body-fixed. Measured sensor height 1.058-1.093 m while walking.
* `robot_body_ids()`: the 13 bodies of the pelvis subtree. `fallen()`: pelvis z < 0.45 m or tilt > 0.8 rad.
* Attributes: `name="g1"`, `radius_m`, `max_vx_mps`, `physics_dt=0.002`, `policy_dt=0.02`,
  `policy_calls`, `last_policy_wall_s`.

## Verified behaviour (headless, flat floor, start (0,0) yaw 0, no obstacles)

Equivalence: the adapter vs a verbatim copy of the upstream deploy loop from the same initial
state: after 10 s at vx 0.3 / 0.5, max |qpos difference| = 1.5e-7 / 2.6e-7 (float32 vs float64).

10 s per command, measured from the pelvis (no qpos writes):

| cmd vx [m/s] | x travelled [m] | mean measured vx [m/s] (steady state, after 2 s) | lateral drift y [m] | yaw drift [deg] | fell |
|---|---|---|---|---|---|
| 0.10 | 1.01 | 0.101 (0.104) | -0.25 | -5.0 | no |
| 0.15 | 1.48 | 0.148 (0.152) | -0.26 | -5.7 | no |
| 0.20 | 1.94 | 0.194 (0.199) | -0.28 | -6.3 | no |
| 0.30 | 2.83 | 0.283 (0.290) | -0.31 | -7.2 | no |
| 0.40 | 3.71 | 0.371 (0.382) | -0.35 | -9.0 | no |
| 0.50 | 4.59 | 0.459 (0.474) | -0.37 | -9.1 | no |
| 0.70 | 6.36 | 0.636 (0.662) | -0.54 | -11.3 | no |
| 1.00 | 8.70 | 0.870 (0.908) | -0.98 | -14.4 | no |

* Lateral sway of the raw pelvis velocity: std(vy) ~0.075 m/s, peak |vy| 0.146 m/s; std(vx) 0.04 m/s at 0.3.
* Tilt max 0.075-0.108 rad; pelvis height 0.753-0.793 m.
* **Stop / resume** (0.3 for 4 s -> 0 for 3 s -> 0.3 for 4 s -> 0 for 10 s): no fall. With cmd 0 the
  robot keeps stepping in place (the gait clock never stops). After a step command 0.3 -> 0 it
  travels a further 0.13 m (0.17 m from 0.5); the 0.8 s centred moving average of vx drops below
  0.05 m/s ~0.6 s after the command. Resume 0 -> 0.3 reaches 90 % of the command in ~0.64 s
  (0.76 s for 0.5). Standing 10 s at cmd 0: drift (+0.11, -0.21) m, yaw -5.8 deg, no fall.
* **Yaw**: cmd (0, 0, 0.4) for 6 s: +126 deg, mean yaw rate 0.364 rad/s, no fall. (0.3, 0, 0.4) for 8 s:
  mean yaw rate 0.357 rad/s; (0.3, 0, -0.3): -0.289 rad/s. No falls.
* **Sim speed**: 6.7-10.6x real time for the robot alone (policy inference 0.4-0.9 ms per call on CPU;
  machine was shared with other jobs, so treat as approximate).
* P0 evidence: `outputs/p0_g1_walk/g1_walk_vx0.3_10s.csv` (pose/velocity at 50 Hz) and
  `outputs/p0_g1_walk/g1_walk_vx0.3_10s.mp4` (offscreen render, tracking camera; frames every 17
  physics steps, so ~2 % faster than real time).
* `tests/test_g1_adapter.py`: 8 tests pass (rates, obs/action shapes, clipping, scene structure,
  qpos-only-in-reset, equivalence with upstream loop, 10 s walk > 1 m, stop/resume).

## Recommendations for the navigation stack

* Command range: vx 0.1-0.5 m/s is well behaved; `navigation.max_vx_mps = 0.3` is conservative,
  0.5 is fine. Measured speed is ~95 % of the command at steady state. `slow_vx_mps = 0.12` works.
* Open-loop yaw drift ~ -0.5 to -1 deg/s and lateral drift ~0.3 m per 10 s: closed-loop heading
  control (k_yaw) is required; yaw-rate commands are tracked at ~90 %.
* The robot does not stand still at cmd 0 (steps in place, ~2 cm/s drift) and needs ~0.15 m /
  ~0.6 s to stop: account for this in the yield distance.
* Raw `v_world_mps` contains gait sway (~0.075 m/s std lateral). For TCA computation consider
  averaging over one gait period (0.8 s) in the planner/runner.

## Known issues / limitations

* Sensor pose is body-fixed on the pelvis, so the horizontal fan tilts with the pelvis sway
  (< 0.11 rad measured). Walls (1.6 m) and actor (1.7 m) remain in the beam at 4 m range.
* No IMU noise/latency model: base quaternion and angular velocity are read from the free joint,
  exactly as upstream `deploy_mujoco.py`.
* Not tested: walking backwards, lateral (vy) commands, contact with obstacles/pushes, rough terrain.
