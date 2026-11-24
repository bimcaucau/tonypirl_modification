# TonyPi high-level navigation — session handoff

> **v1.2 (detours / teacher → 0):** mixed layouts (scatter/wall/u_trap/gate), no fake corridors, no Euclidean geodesic fallback, ram penalty, stretched mix/BC anneal to **exactly 0** at 8M/10M. Spec: `tonypi_rl/implementation/NAV_V1_2.md`. Resume the 4.3M v1.1 zip; do not resume 32×32.
>
> **v1.1 (obs/reward retune, implemented):** polar 18-D vec, occupancy 64×64, arrive 100 / time 0.25 / collision streak. Log: `implementation/NAV_V1_1_IMPLEMENTATION_LOG.md`.

Read this first in a new session. It is the current (v1) design, not a wishlist.

**Goal:** given a body-frame target (example `+2 m X, −0.5 m Y`), walk there while avoiding boxes. Policy sees sensors and outputs a named action group, the same names the real Pi already plays.

**Not the goal:** pixels-to-18-joints. Low-level walking stays in `TonyPiEnv` / `v0_straight_walk.yaml` and must not be rewritten.

---

## 1. Two layers (do not mix)

| Layer | Env | Action | Obs | What it learns |
|---|---|---|---|---|
| Low-level walk (done, leave it) | `TonyPiEnv` / `TonyPi-v0` | `Box(-1,1)^18` joint targets | 63-D proprio | walk +X at 0.20 m/s |
| High-level nav (this work) | `TonyPiNavEnv` / `TonyPiNav-v0` | `Discrete(7)` action-group ids | Dict `vec` + `occupancy` | go to XY, avoid boxes |

On the real robot there is **no `cmd_vel` and no ROS**. Motion is SQLite `.d6a` clips via `AGC.runActionGroup(name)`. One env step = one clip (~0.4 s, same ballpark as `go_forward.d6a` = 10×38 ms).

In sim the robot does **not** learn to walk. Joints stay at stand pose, pelvis XY/yaw is moved kinematically, `mj_forward` updates sensors. Deploy maps the same action id → `runActionGroup`.

```
goal XY + 1 ToF + 1 sonar + IMU + last action + 32×32 occupancy
                         │
                         ▼
              PPO  MultiInputPolicy   (~2–5 Hz)
                         │
                         ▼
     stand | go_forward | back_fast | left_move_fast
           | right_move_fast | turn_left | turn_right
                         │
          sim: kinematic primitive     real: ActionGroupControl
```

---

## 2. Design forks (already decided)

- **Actions:** discrete action groups, not continuous velocity.
- **Mask while training:** software pinhole projection of `obs_*` boxes into `head_camera` (GL-free). Not YOLO, not ResNet.
- **Mask at deploy:** YOLO11-seg (`yolo11n-seg.pt`) → same `(1,32,32)`. On-robot `Functions/model/best.pt` is YOLOv11 **detect** (age/gender), **not** seg. Do not use it.
- **Network:** small CNN on 32×32 + MLP on vec. No LSTM, no ViT, no NatureCNN (NatureCNN dies on 32×32).
- **ToF:** **one** beam, matching stock VL53L0X. A 5-ray fan was tried and removed — the Pi cannot provide it, and the policy would overfit to fake side rays.

---

## 3. Observation

`spaces.Dict`:

```
vec:        (19,) float32
occupancy:  (1, 32, 32) float32 in {0, 1}
```

`vec` layout:

| slice | dims | content |
|---|---|---|
| 0:4 | 4 | goal in **body** frame: `x, y, dist, heading_error` |
| 4:5 | 1 | ToF metres / 1.2 (VL53L0X cutoff) |
| 5:6 | 1 | sonar metres / 5.0 |
| 6:9 | 3 | projected gravity |
| 9:12 | 3 | gyro |
| 12:19 | 7 | previous action one-hot |

Goal is **body frame**, not world. After a `turn_left`, the same world goal rotates in `vec`.

No raw RGB in the policy. RGB is only for the debug preview.

### Occupancy

Training: `MujocoGtOccupancy(use_gl=False)` in `perception.py`. Projects each `obs_*` box’s 8 corners through the head-camera pinhole, fills the convex hull on 64×64, max-pools to 32×32. **No EGL / no `mujoco.Renderer`.** That renderer inside `SubprocVecEnv` spawn was segfaulting on the cluster.

`use_gl=True` still exists for debug (MuJoCo geom-segmentation). Do not turn it on for PPO workers.

Deploy: `YoloSegOccupancy.encode_bgr(camera_bgr)` → same shape. Keep COCO classes that look like obstacles.

---

## 4. Action space

Order is fixed (must match `ActionGroupDict` names on the Pi):

| id | name | sim primitive (body frame) |
|---|---|---|
| 0 | `stand` | 0 |
| 1 | `go_forward` | +0.08 m x |
| 2 | `back_fast` | −0.06 m x |
| 3 | `left_move_fast` | +0.05 m y |
| 4 | `right_move_fast` | −0.05 m y |
| 5 | `turn_left` | +0.35 rad yaw (~+20°) |
| 6 | `turn_right` | −0.35 rad yaw (~−20°) |

Each step interpolates pelvis over `primitive_substeps=16`. Contact with an `obs_*` geom → snap to last free pose, `collision=True`, **do not terminate**. Timeout / success terminate.

---

## 5. Real robot sensors (train must match these)

Stock TonyPi is **not** a 5-beam lidar.

| Sim | Hardware | How to read |
|---|---|---|
| `tof_0` rangefinder, cutoff 1.2 m | **one** VL53L0X, I2C bus 1 addr `0x29` | `VL53L0X.get_distance()` mm. 0 / 8191 = miss → treat as 1.2 m. Already used in `tonypi_inside/TonyPi/Functions/obstacle_avoidance.py` |
| `sonar` rangefinder, cutoff 5.0 m | optional Hiwonder I2C sonar `0x77` | `hiwonder.Sonar.getDistance()` mm. 99999 = unplugged → 5.0 m |
| occupancy 32×32 | USB camera 640×480 | `Camera.read()` → YOLO11-seg |
| gravity / gyro | STM32 IMU | `board.get_imu()` → 6 floats, accel then gyro |

Side / corner obstacles are the **camera mask**, not extra ToFs. Do not re-add `tof_1..4`.

At deploy, fill `vec` as: `tof_0` from VL53L0X, `sonar` from Sonar or 1.0 if missing, occupancy from YOLO, gyro/gravity from IMU, goal from CLI / AprilTag / odometry (odometry is **not** solved yet).

---

## 6. Scene / MuJoCo

- Nav scene: `tonypi_mujoco/scene_nav.xml` (includes `tonypi.xml`).
- Walking scene: `tonypi_mujoco/scene.xml` — unused by nav.
- 4×4 m floor, 8 mocap boxes `obs_0..7` (parked at ~12,12 until `reset` places a subset), mocap green goal (no collision).
- Sites on pelvis in `tonypi.xml`: `tof_0` (+X), `sonar` (+X, slightly lower). Walking env ignores them.
- Head camera: `pos="0.08 0 0.04" fovy="70" xyaxes="0 -1 0 0 0 1"` looking **+X**. `euler="0 1.57 0"` looked **into the torso** — do not revert that.

Demo layout (`layout=demo`, `goal_mode=fixed`): goal `(2.0, -0.5)`, four boxes including one on the ToF centerline at x≈0.95 → ToF≈0.80 m, sonar≈0.81 m.

Random layout: goal in a 1.0–2.5 m annulus, 3–6 boxes, 0.45 m start clearance, 0.35 m goal clearance.

---

## 7. Reward / episode

```
reward = 5.0 * (dist_prev - dist) - 0.05
       + 20.0 if dist < 0.20 m   # arrive, terminate
       - 2.0  if collision this step
```

Config currently uses `max_episode_steps: 240` (~96 s of primitives). Collision does not end the episode; a dumb “always go_forward” heuristic will wedge on a box corner with ToF still reading far (beam thinner than the body). PPO is supposed to learn not to push.

---

## 8. Policy

`NavFeaturesExtractor` (`tonypi_rl/nav_features.py`):

- CNN: 1→8→16→32, 3×3 stride-2, 32×32 → 4×4 → flatten 512
- vec MLP: 19 → 64
- merge → 128, then SB3 `net_arch` `[256, 256]` π and V
- ReLU (not Tanh)

`MultiInputPolicy`. Discrete(7) softmax.

---

## 9. Files

**New (nav)**

| path | role |
|---|---|
| `tonypi_mujoco/scene_nav.xml` | arena, mocap boxes/goal, rangefinders |
| `tonypi_rl/tonypi_nav_env.py` | `TonyPiNavEnv` |
| `tonypi_rl/perception.py` | projective occupancy + YOLO adapter |
| `tonypi_rl/nav_features.py` | CNN+MLP extractor |
| `tonypi_rl/visualize_nav.py` | debug image / GUI / `--expert` |
| `tonypi_rl/configs/v1_nav.yaml` | stock PPO hyperparams (cluster) |
| `tonypi_rl/configs/v1_nav_expert.yaml` | GuidedPPO + geodesic reward |
| `tonypi_rl/nav_planner.py` | world-frame grid: Dijkstra cost-to-go + A* polyline (train-only) |
| `tonypi_rl/nav_expert.py` | carrot-track Discrete(7) chooser on the A* path |
| `tonypi_rl/guided_ppo.py` | mix collect + DAgger CE + BC warmup |
| `tonypi_rl/nav_preview.png` | last debug render |

**Edited**

| path | what changed |
|---|---|
| `tonypi_mujoco/tonypi.xml` | `tof_0` + `sonar` sites; head camera look +X |
| `tonypi_rl/__init__.py` | register `TonyPiNav-v0` (`max_episode_steps=80` gym wrapper; env also has its own cap from yaml) |
| `tonypi_rl/train.py` | `env_class: nav` switch, path bootstrap for spawn workers, `MUJOCO_GL=egl` **before** mujoco import |

**Do not change:** `tonypi_env.py` walking reward, `v0_straight_walk.yaml`, robot `.d6a` files.

**Not built yet:** `deploy_nav.py`, nav path in `eval.py` (eval still assumes `TonyPiEnv`).

---

## 10. Commands

Always from **repo root**. Spawn workers do not inherit `sys.path` from `python tonypi_rl/train.py` — use `-m` and `PYTHONPATH`.

```bash
cd <repo>   # local: /mnt/d/robotics_cccc/tonypi
            # cluster: /AIClub_NAS/nguyenmv/ccc170/HN
export PYTHONPATH=$PWD
export MUJOCO_GL=egl          # if EGL cores, try: export MUJOCO_GL=osmesa

# sanity
python -c "from tonypi_rl.tonypi_nav_env import TonyPiNavEnv; e=TonyPiNavEnv(); o,_=e.reset(seed=0); print(o['vec'].shape, o['occupancy'].shape, float(o['occupancy'].mean())); e.close()"
# expect: (19,) (1, 32, 32)  and occupancy mean ~0.28 on demo

# preview (uses GL only for RGB panels)
python tonypi_rl/visualize_nav.py --layout demo --render-image tonypi_rl/nav_preview.png
python tonypi_rl/visualize_nav.py --gui
python tonypi_rl/visualize_nav.py --layout demo --expert --render-image tonypi_rl/nav_expert_preview.png

# train — MUST be -m, not python tonypi_rl/train.py
python -m tonypi_rl.train --config tonypi_rl/configs/v1_nav.yaml
python -m tonypi_rl.train --config tonypi_rl/configs/v1_nav_expert.yaml
```

Cluster yaml (as last edited): 20M steps, 8 envs, `n_steps=2048`, `batch_size=1024`, `max_episode_steps=240`, `ent_coef=0.002`. Nav uses `SubprocVecEnv(..., start_method="spawn")`.

TensorBoard: `tensorboard --logdir tonypi_rl/logs`.

---

## 11. Bugs already hit (do not reintroduce)

1. **Head camera looked −X** (`euler="0 1.57 0"` / `xyaxes="0 1 0 0 0 1"`). MuJoCo cameras look along **−Z** of the camera frame. Working axes: `xyaxes="0 -1 0 0 0 1"`.
2. **ToF overshot boxes** when boxes were 0.30 m tall and rays sat at ~0.31 m. Boxes are `size z=0.22` (span 0–0.44 m).
3. **Five ToF rays in sim, one on the Pi.** Dropped. Occupancy covers the sides.
4. **`python tonypi_rl/train.py` in spawn workers** → `ModuleNotFoundError: tonypi_rl.tonypi_nav_env`. Script dir lands on `sys.path[0]`, workers bind an old/wrong `tonypi_rl`. Fix: `python -m tonypi_rl.train` plus `_bootstrap_repo()` inside each worker.
8. **Sibling `HN/tonypi_rl_expert` while stock PPO is running in `HN/tonypi_rl`**. `PYTHONPATH=HN` made spawn import the in-flight package, so `TonyPiNavEnv.__init__() got an unexpected keyword argument 'use_geodesic_reward'`. `train.py` now binds `tonypi_rl` to the directory that contains *this* file, regardless of folder name. Do not overwrite the in-flight `tonypi_rl/` tree. Sync expert files into the sibling folder and run *that* `train.py`.
5. **EGL segfault** in 8 spawned workers creating `mujoco.Renderer`. Training occupancy is software now (`use_gl=False`). Do not create a Renderer in `TonyPiNavEnv.__init__`.
6. **NatureCNN on 32×32** shrinks to a 0-size map. Use `NavFeaturesExtractor`.
7. **Collision wedge:** after a slight turn, centerline ToF misses, policy keeps `go_forward`, shoulder clips a box, pose freezes, −2 every step. Expected for a 1-beam bumper; not an env crash.

---

## 12. Status / next

**Done:** nav env, 1 ToF + sonar, software occupancy, small CNN policy, train.py switch, debug preview, cluster import/GL fixes, privileged A* trajectory expert + Dijkstra geodesic + GuidedPPO (BC / mix / DAgger).

**In flight:** cluster PPO run (`python -m tonypi_rl.train --config tonypi_rl/configs/v1_nav.yaml`). That run is **stock PPO** — do not silently swap it to `v1_nav_expert.yaml`. Watch `rollout/ep_rew_mean` and whether episode length drops below 240 as it starts arriving.

**Next, in order:**

1. Confirm training actually steps (no segfault, `fps` printed).
2. Nav eval: `eval.py` still constructs `TonyPiEnv`. Add the same `env_class` switch + occupancy overlay.
3. `deploy_nav.py`: 2–5 Hz loop, `Camera.read()` → YOLO mask, VL53L0X + Sonar, `AGC.runActionGroup(ACTIONS[a])`. Goal XY on hardware is unsolved (no odometry) — first real test needs a visible marker or a short open-loop goal.
4. Optional: terminate after N consecutive collisions so a wedged policy does not burn the rest of the episode.

Python on this WSL box: `/home/bimcaucau/miniconda3/bin/python`. Cluster image: `/usr/local/bin/python` 3.10.

---

## 13. Expert guidance (train-only)

Random PPO wedges because Euclidean `5*(d_prev − d)` punishes going around a box, and the 1-beam ToF misses a shoulder clip. A privileged planner generates a trajectory and a Discrete(7) action that follows it. The **policy never sees the world grid** — obs stay `{vec: (19,), occupancy: (1,32,32)}` (camera pinhole). Deploy is unchanged: no GT boxes, no A*/Dijkstra.

### Planner

`nav_planner.py` rasterizes active mocap boxes (`obs_0..7`, skip `x > 8` parked slots) as AABBs Minkowski-summed with a disk of `expert_robot_radius` (default 0.18 m; shoulders ~±0.082 m, extra margin for mesh clip). Grid: world origin `(−3.2, −3.2)`, size `6.4×6.4` m, `res=0.05` → 128×128. 8-connected, no corner cutting.

- **Dijkstra from the goal** → metres-to-go field (geodesic reward + illegal-next filter). Once per reset; boxes do not move.
- **A\* from the robot** → the XY polyline the controller tracks. Receding-horizon: replanned every `expert_action()` from the current cell (collision snap cannot desync a stored tape). When the Dijkstra field is already filled it is an exact heuristic, so A\* is cheap.

Camera 32×32 is **not** the planning map. Floor XML `size="4 4"` is MuJoCo plane **half-extents** (8×8 m); the planner only covers the 1–2.5 m annulus.

### Chooser

`nav_expert.PrimitiveExpert` scores the 7 action-group primitives with the same body-frame kinematics as `TonyPiNavEnv.step`. Trajectory first, then control — not an open-loop action tape.

- Euclidean dist < `arrive_radius` (0.20) → `stand`
- Desired yaw = carrot on the A\* polyline (`look_ahead=0.30` m), else cost-to-go descent, else `atan2` to goal
- Illegal if next XY is occupied / off-grid (except `stand`)
- Score = `geodesic(next_xy) + heading_weight * |wrap(yaw_next − desired)|`
  - `stand` while not arrived: `+10`
  - `back_fast`: `+0.03` so forward wins ties
- All illegal → turn toward free heading, else `back_fast`

Public env methods (SB3 `env_method` / `get_wrapper_attr` walks Monitor): `expert_action()`, `geodesic_distance()`, `expert_path()`.

### Geodesic reward

`use_geodesic_reward: true` replaces Euclidean progress with `geo_prev − geo` (metres, so `progress_weight: 5.0` stays). Clipped to `±geodesic_progress_clip` (0.15). Arrive / time / collision terms unchanged. If Dijkstra failed, fall back to Euclidean. `info` keys: `expert_action`, `geodesic`, `expert_unreachable`.

### Learning (`GuidedPPO`, `v1_nav_expert.yaml`)

Do **not** wrap `env.step` to swap the action — stock PPO stores the policy sample, not the executed id.

1. **BC warmup** — expert drives the vec env; policy clones `(Dict obs, a*)` for `bc_warmup_steps` (50k) × `bc_epochs` (8).
2. **Decaying mix** — in `collect_rollouts`, with prob `mix_prob` (0.3 → 0 over 1M steps) execute `a*` and store `a*` + `log π(a*|s)`.
3. **DAgger CE** — extra `bc_coef * NLL(a*|s)` on labelled rollout states (0.15 → 0 over 2M steps). Expert labels live on `GuidedDictRolloutBuffer` and are `swap_and_flatten`'d in `get()` so they stay aligned with the minibatch permutation.

Eval callback stays `deterministic=True` with no mix. `v1_nav.yaml` with no `expert:` block is still stock `PPO`.

Resume a GuidedPPO zip with `--config v1_nav_expert.yaml --resume ...` (uses `GuidedPPO.load`).

TensorBoard: `rollout/mix_prob`, `rollout/expert_agree`, `train/bc_loss`, `train/bc_coef`.

### Commands

```bash
# teacher must arrive on the demo box at (0.95, 0.02)  (A* overlay + executed trail)
python tonypi_rl/visualize_nav.py --layout demo --expert --render-image tonypi_rl/nav_expert_preview.png

# 20 random layouts (expect ≥16/20)
python -c "
from tonypi_rl.tonypi_nav_env import TonyPiNavEnv
e = TonyPiNavEnv(layout='random', goal_mode='random', max_episode_steps=240,
                 use_geodesic_reward=True)
ok = 0
for s in range(20):
    e.reset(seed=s)
    for _ in range(240):
        _, _, term, trunc, info = e.step(e.expert_action())
        if term: ok += 1; break
        if trunc: break
print('success', ok, '/20')
e.close()
"

python -m tonypi_rl.train --config tonypi_rl/configs/v1_nav_expert.yaml
```

If the expert still wedges, bump `expert_robot_radius` toward 0.20 — do not change walking or occupancy.
