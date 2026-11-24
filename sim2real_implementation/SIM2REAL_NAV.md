# TonyPiNav sim-to-real — coding-agent implementation spec

**This file is the source of truth.** Read it fully, plus `HARDWARE.md`, before editing. Implement exactly what is specified. Do not invent extra features. Do not rewrite walking. Do not grow the PPO network. Do not call the privileged expert on the robot.

| | |
|---|---|
| Repo root | `/mnt/d/robotics_cccc/tonypi` |
| Layer | **High-level nav only** (Discrete(7) action groups) |
| Checkpoint | `models_ckpt/new_expert_implementation/tonypi_final_model.zip` (GuidedPPO, **10,010,624** steps, SB3 **2.9.0**, teacher already 0: `mix_prob=0`, `bc_coef=0`) |
| Obs | `Dict{ vec: (18,) float32, occupancy: (1, 64, 64) float32 in {0,1} }` |
| Act | `Discrete(7)` → `stand \| go_forward \| back_fast \| left_move_fast \| right_move_fast \| turn_left \| turn_right` |
| Teacher at deploy | **Must be 0.** No `expert_action()`, no A*, no geodesic, no mocap boxes |
| On-robot tree | `tonypi_inside/TonyPi/` (copy of `/home/pi/TonyPi`) |
| Out of scope | `tonypi_env.py`, `v0_straight_walk.yaml`, `.d6a` contents, MuJoCo XML, `nav_features.py` weights/arch, `perception.py` `OCCUPANCY_SIZE`, training yaml, 18-DoF PD walking |

`eval.py` is walk-only. Sim playback of this zip is `python tonypi_rl/visualize_nav.py --model <zip> --gui`. Deploy is this package.

---

## 0. Locked diagnosis (do not reopen)

The trained policy **does not walk**. Walking is a canned `.d6a` clip. The policy is a 2–5 Hz discrete chooser: given polar goal + 1 ToF + 1 sonar + gravity + last action + 64×64 camera mask, pick the next clip.

Sim (`TonyPiNavEnv`) teleports the pelvis by `PRIMITIVES[name]` and `mj_forward`s sensors. Real robot plays the same **name**. Domain gap is sensors + step size, not joint tracking.

`NAV.md` already named the missing piece: `deploy_nav.py`. Goal XY on hardware is unsolved (no odometry). First live test is a **body-frame goal at t=0** with dead-reckon, or an **AprilTag** on the target. Do not build AMCL / visual odometry / LiDAR maps.

Do not deploy `tonypi_rl/tonypi_env.py` 18-DoF actions. That README pulse formula is for a different stack.

---

## 1. Observation contract (must match `_get_obs`)

Copy this. If a field disagrees with `tonypi_nav_env.py`, **the Python env wins** — update this paragraph and the log, do not silently drift.

Source: `tonypi_rl/tonypi_nav_env.py` `TonyPiNavEnv._get_obs` (v1.1/v1.2 polar):

```
vec[0]     = dist / 3.0                         # GOAL_DIST_SCALE = 3.0
vec[1]     = sin(heading_error)
vec[2]     = cos(heading_error)                 # NOT the raw angle
vec[3]     = tof_0_m / 1.2                      # miss → 1.0
vec[4]     = sonar_m / 5.0                      # miss / unplugged → 1.0
vec[5:8]   = projected gravity, unit, pelvis    # sim upright ≈ [0, 0, -1]
vec[8:11]  = gyro rad/s                         # v1 deploy: [0, 0, 0]
vec[11:18] = previous action one-hot            # index = last executed id, stand=0 at reset
```

`heading_error = atan2(gy, gx)` in the **body** frame, X forward, Y left:

```
gx =  cos(yaw) * (goal_x - x) + sin(yaw) * (goal_y - y)
gy = -sin(yaw) * (goal_x - x) + cos(yaw) * (goal_y - y)
```

Occupancy: `float32`, shape `(1, 64, 64)`, values in `{0.0, 1.0}`, channel-first, **row 0 = top of image**, same as `YoloSegOccupancy._resize` (`cv2.INTER_AREA` then `> 0.3`).

`prev_action` is the action **that was just executed**, including safety overrides. Reset → one-hot of `stand` (id 0).

No RGB in the policy. No joint positions. No lin vel. No foot contacts. No VecNormalize.

---

## 2. Action contract

```python
ACTION_NAMES = [
    "stand", "go_forward", "back_fast",
    "left_move_fast", "right_move_fast",
    "turn_left", "turn_right",
]
```

One env step = one `AGC.runActionGroup(name)` (blocking). After the clip returns, **then** sense, **then** `model.predict`. Do not pipeline a second clip while the first is playing. Do not `sleep(0.4)` — clip length is 380–770 ms (see `HARDWARE.md`).

Deterministic: `model.predict(obs, deterministic=True)`.

Refuse any zip whose `data` JSON is not Dict occupancy `(1,64,64)` + vec `(18,)` / Discrete(7). The 32×32+19-D zip (`models_ckpt/149expert/`) and the 63-D walk zip (`models_ckpt/139/`, `models_ckpt/best_model.zip`) **must not load**. Walk zip is a hopping v0 policy — never send it to bus servos as part of this work.

---

## 3. Architecture (locked)

```
sensors (cam, VL53L0X, sonar, IMU) + goal source
                 │
                 ▼
        observations.build()     # exact 18-D + 64×64
                 │
                 ▼
        SB3 GuidedPPO/PPO CPU    # zip above, NavFeaturesExtractor
                 │
                 ▼
        safety.filter(action)    # e-stop / fall / max-steps  (optional tof-guard)
                 │
                 ▼
        AGC.runActionGroup(name)
                 │
                 ▼
        goal.update(name)        # dead-reckon and/or tag
```

Two hardware backends behind the same protocol:

| Backend | When |
|---|---|
| `FakeHardware` | laptop Phase 0, unit tests |
| `TonyPiHardware` | robot; imports `hiwonder` + `VL53L0X` |

Do **not** fork the control loop. `deploy_nav.py` is backend-agnostic.

Compute: **on-robot inference** for v1. If YOLO p95 `> 300 ms`, switch occupancy to `color`, keep PPO on-robot (the CNN is tiny). Remote laptop inference is Phase 6 only.

---

## 4. Goal sources (locked)

Implement three modes. Default for first walk: `deadreckon`.

### 4.1 `deadreckon`

CLI `--goal GX GY` is the goal in **body frame at t = 0**. World frame := that initial body frame. Robot pose starts `(x, y, yaw) = (0, 0, 0)`. After each **executed** action:

```
dx, dy, dyaw = primitives[name]     # from configs/primitives.yaml, else sim table
x +=  cos(yaw)*dx - sin(yaw)*dy
y +=  sin(yaw)*dx + cos(yaw)*dy
yaw = wrap(yaw + dyaw)
```

Then `dist, heading` from §1. This drifts. Cap `--max-steps` (default 20).

### 4.2 `tag`

AprilTag in the head camera. `--tag-id`, `--tag-size` (metres, edge). Solve PnP with `Functions/CameraCalibration/calibration_param.npz`. Tag pose in **camera frame** → transform to **body/pelvis** with a hardcoded cam-from-body from the XML (`pos="0.08 0 0.04"`, look +X). `dist = hypot(gx, gy)` using XY only (ignore tag Z unless it is nonsense, then mark lost).

Lost tag: `--tag-lost hold` (default, keep last body goal) or `stand` (force action 0 until seen). Never extrapolate with vision-only velocity.

### 4.3 `fixed-body`

`--goal GX GY` republished **every step in current body frame** (goal rides with the robot). Only for sensor/policy smoke tests (“heading ~0 and dist constant ⇒ always go_forward on empty floor”). Not a nav test.

Arrive: `dist < --arrive` (default `0.20`, same as train). On arrive: play `stand`, set `arrived=True`, exit 0.

---

## 5. Occupancy backends

`--occupancy` one of: `yolo` | `color` | `file` | `zeros`.

| name | behavior |
|---|---|
| `yolo` | `tonypi_rl.perception.YoloSegOccupancy` with **seg** weights (`yolo11n-seg.pt` or `--yolo-weights`). `imgsz=320`, `conf=0.25` |
| `color` | HSV in-range on BGR, morphology open/close, resize to 64×64, threshold 0.3. Defaults in `configs/deploy.yaml` for a saturated obstacle color. Calibrate with `sensor_probe --tune-hsv` |
| `file` | load a `(64,64)` or `(1,64,64)` npy each step (tests) |
| `zeros` | empty mask (open-floor smoke test) |

Do not use `Functions/model/best.pt`.

---

## 6. Safety (always on unless `--no-safety`)

Order, after `predict`, before `runActionGroup`:

1. **E-stop**: Ctrl+C, SIGTERM, or STM32 key click → `stopActionGroup()`, `runActionGroup("stand")`, exit 2.
2. **Max steps** / **max wall time** (`--max-steps`, `--max-seconds` default 60) → stand, exit 0 if arrived else 1.
3. **Fall**: gravity or raw accel implies `|tilt| > --fall-deg` (default 45°) for 3 consecutive cycles → `stand_up_front` or `stand_up_back` **only if** `--recover-fall`; else stand and exit 3. Reuse the sign logic in `Fall_and_Stand.py` after IMU remap, or call that module’s `atan2` on **raw** STM32 samples (it was tuned on raw). Locked: detect on **raw** STM32 like the stock script; do not require `imu.yaml` for fall.
4. **ToF guard** (off by default): `--tof-guard 0.12` replaces the action with `back_fast` if ToF `< 0.12 m` and the policy picked `go_forward`. Do not enable for Phase 5 until Phase 3 is clean — it hides policy ramming.

`--dry-run`: run the whole loop, print the action name, **do not** call `runActionGroup` (except optional `--dry-run-stand` once at start).

Logging: every step append JSONL `{t, action, vec, tof_m, sonar_m, dist, heading, occ_mean, latency_ms}` to `--log-dir`. Occupancy dumps as png if `--save-debug`.

---

## 7. IMU gravity remap

`configs/imu.yaml`:

```yaml
# real_accel_unit (3,) -> sim_gravity (3,): g_sim = R @ a_real_unit
R:
  - [0.0, 0.0, 1.0]
  - [0.0, 1.0, 0.0]
  - [1.0, 0.0, 0.0]
gyro_mode: zeros          # locked v1
```

The matrix above is a **placeholder**. `calibrate_imu.py` overwrites it:

1. N samples standing still → mean accel `a_stand`, normalize.
2. User tilts **forward** ~30° (robot’s +X) → `a_fwd`.
3. Build orthonormal R so `R @ a_stand ≈ [0,0,-1]` and forward tilt moves gravity toward **+X or −X** consistently with heading. Print the residual. Refuse to write if `||a_stand||` is tiny.

Until `imu.yaml` exists, `observations.py` uses `R = I` and **logs a warning every step**. Gyro always zeros in v1.

---

## 8. Files

### Create (this package)

| path | role |
|---|---|
| `sim2real_implementation/__init__.py` | empty |
| `sim2real_implementation/configs/deploy.yaml` | defaults (model path, arrive, hsv, yolo, max-steps) |
| `sim2real_implementation/configs/imu.yaml` | written by calibrator; ship a stub with `R: identity` + comment |
| `sim2real_implementation/configs/primitives.yaml` | sim table copied; calibrator overwrites |
| `sim2real_implementation/constants.py` | `ACTION_NAMES`, cutoffs, `GOAL_DIST_SCALE`, `OCCUPANCY_SIZE` imported from `tonypi_rl.perception` when possible, else duplicated **with a comment pointing at the env** |
| `sim2real_implementation/hardware.py` | `Hardware` protocol, `FakeHardware`, `TonyPiHardware` |
| `sim2real_implementation/observations.py` | `build_obs(hw, goal, prev_action, occupancy) -> dict` |
| `sim2real_implementation/occupancy_real.py` | yolo / color / file / zeros |
| `sim2real_implementation/policy.py` | dummy gym env + load zip + `predict` |
| `sim2real_implementation/goal.py` | deadreckon / tag / fixed-body |
| `sim2real_implementation/safety.py` | e-stop, fall, tof-guard, step budget |
| `sim2real_implementation/deploy_nav.py` | CLI main loop (`python -m sim2real_implementation.deploy_nav`) |
| `sim2real_implementation/sensor_probe.py` | Phase 1 printer |
| `sim2real_implementation/calibrate_imu.py` | writes `imu.yaml` |
| `sim2real_implementation/calibrate_primitives.py` | tape-measure wizard → `primitives.yaml` |
| `sim2real_implementation/replay_policy.py` | laptop, no hiwonder |
| `sim2real_implementation/requirements-deploy.txt` | Pi extras (torch CPU, sb3, gymnasium, ultralytics optional) |
| `sim2real_implementation/IMPLEMENTATION_LOG.md` | agent fills as they go |

### Do not touch

`tonypi_rl/tonypi_env.py`, `tonypi_rl/tonypi_nav_env.py`, `tonypi_rl/nav_features.py`, `tonypi_rl/perception.py` (import only), `tonypi_rl/guided_ppo.py` (import only), `tonypi_mujoco/*`, `tonypi_inside/TonyPi/ActionGroups/*`, `Functions/model/best.pt`, walking yaml.

Allowed on-robot: **read** hiwonder APIs. Do not patch `ActionGroupControl.py`.

---

## 9. File-by-file implementation

### 9.1 `constants.py`

```python
ACTION_NAMES = [
    "stand", "go_forward", "back_fast",
    "left_move_fast", "right_move_fast",
    "turn_left", "turn_right",
]
GOAL_DIST_SCALE = 3.0
TOF_CUTOFF = 1.2
SONAR_CUTOFF = 5.0
SIM_PRIMITIVES = {
    "stand": (0.00, 0.00, 0.00),
    "go_forward": (0.08, 0.00, 0.00),
    "back_fast": (-0.06, 0.00, 0.00),
    "left_move_fast": (0.00, 0.05, 0.00),
    "right_move_fast": (0.00, -0.05, 0.00),
    "turn_left": (0.00, 0.00, 0.35),
    "turn_right": (0.00, 0.00, -0.35),
}
```

Import `OCCUPANCY_SIZE` from `tonypi_rl.perception`. If that import needs cv2/mujoco, copy `OCCUPANCY_SIZE = 64` and assert it.

### 9.2 `hardware.py`

Protocol (names locked):

```python
class Hardware(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def read_bgr(self) -> np.ndarray | None: ...          # HxWx3 uint8
    def read_tof_m(self) -> float: ...                    # already clipped to [0, 1.2], miss=1.2
    def read_sonar_m(self) -> float: ...                  # [0, 5], miss=5.0
    def read_imu(self) -> tuple[np.ndarray, np.ndarray] | None:
        """(accel_raw_6slice[:3], gyro_raw[3:]). None if never received."""
    def play(self, name: str) -> None: ...
    def stop_motion(self) -> None: ...
    def center_head(self) -> None: ...
    def poll_estop(self) -> bool: ...
```

`TonyPiHardware.start()`:

1. `board = rrc.Board(); board.enable_reception()`
2. `Controller(board)` for PWM head
3. `center_head()` pulses 1500/1500
4. `Camera((640,480)).camera_open()`
5. VL53L0X open + `BETTER` ranging. Import path: add `os.path.join(TONYPI_HOME, "Functions")` to `sys.path` **inside this class**, not at module import (so laptop import of the package does not need the `.so`).
6. `Sonar()` — if `getDistance()==99999` twice, set `self.sonar_ok = False`
7. `play("stand")` unless dry-run

`read_tof_m`: `get_distance()` → mm; `<=0` or `>=8190` → `1.2`.

`FakeHardware`: constructor kwargs for bgr, tof, sonar, accel. `play` appends names to `self.played`.

### 9.3 `observations.py`

```python
def scale_range(metres: float, cutoff: float) -> float:
    if not np.isfinite(metres) or metres < 0:
        metres = cutoff
    return float(np.clip(metres, 0.0, cutoff) / cutoff)

def gravity_from_imu(accel_raw: np.ndarray, R: np.ndarray) -> np.ndarray:
    a = np.asarray(accel_raw, dtype=np.float32).reshape(3)
    n = np.linalg.norm(a)
    if n < 1e-3:
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return (R @ (a / n)).astype(np.float32)

def polar_goal(dist: float, heading: float) -> np.ndarray:
    return np.array([dist / GOAL_DIST_SCALE, np.sin(heading), np.cos(heading)], np.float32)

def one_hot(prev: int) -> np.ndarray:
    v = np.zeros(7, np.float32)
    v[int(prev)] = 1.0
    return v

def build_obs(*, dist, heading, tof_m, sonar_m, gravity, prev_action, occupancy) -> dict:
    vec = np.concatenate([
        polar_goal(dist, heading),
        [scale_range(tof_m, TOF_CUTOFF), scale_range(sonar_m, SONAR_CUTOFF)],
        np.asarray(gravity, np.float32).reshape(3),
        np.zeros(3, np.float32),          # gyro locked zeros
        one_hot(prev_action),
    ]).astype(np.float32)
    assert vec.shape == (18,)
    occ = np.asarray(occupancy, np.float32)
    if occ.shape == (64, 64):
        occ = occ[None, ...]
    assert occ.shape == (1, 64, 64)
    return {"vec": vec, "occupancy": occ}
```

No extra keys. SB3 Dict policy wants numpy arrays, not torch.

### 9.4 `policy.py`

Do **not** construct `TonyPiNavEnv`. Dummy env:

```python
class DummyNavEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Dict({
            "vec": spaces.Box(-np.inf, np.inf, (18,), np.float32),
            "occupancy": spaces.Box(0, 1, (1, 64, 64), np.float32),
        })
    def reset(self, **kw):
        return {"vec": np.zeros(18, np.float32), "occupancy": np.zeros((1,64,64), np.float32)}, {}
    def step(self, a):
        raise RuntimeError("dummy")
```

Inspected load (CPU, no GPU): `GuidedPPO.load` succeeds; spaces match DummyNavEnv; `NavFeaturesExtractor` path is `tonypi_rl.nav_features.NavFeaturesExtractor`. Cluster train used Python 3.10 + torch 2.14 + SB3 2.9. Pi can use torch CPU; if `load` fails on pickle/protocol, install SB3 2.9.x not 2.0.

Load **GuidedPPO first** (this zip is expert training):

```python
from tonypi_rl.nav_features import NavFeaturesExtractor  # must be importable
from tonypi_rl.guided_ppo import GuidedPPO
from stable_baselines3 import PPO

def load_nav_policy(path: str, device: str = "cpu"):
    env = DummyNavEnv()
    try:
        return GuidedPPO.load(path, env=env, device=device, print_system_info=False)
    except Exception:
        return PPO.load(path, env=env, device=device, print_system_info=False)

def act(model, obs, deterministic=True) -> int:
    a, _ = model.predict(obs, deterministic=deterministic)
    return int(np.asarray(a).reshape(-1)[0])
```

Refuse to load if `inspect_checkpoint` (copy the function from `visualize_nav.py`) says not nav.

Device: `cpu`. Do not `.cuda()` on the Pi.

### 9.5 `occupancy_real.py`

Thin wrappers. `YoloSegOccupancy` already returns `(1,64,64)`. Color path:

```python
def color_occupancy(bgr, hsv_lo, hsv_hi) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lo), np.array(hsv_hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    small = cv2.resize(mask, (64, 64), interpolation=cv2.INTER_AREA)
    return (small > 255 * 0.3).astype(np.float32)[None, ...]
```

If `bgr is None`, return zeros and log.

### 9.6 `goal.py`

`class GoalSource` with `reset()`, `current() -> (dist, heading, gx, gy)`, `on_action(name)`, `arrived(radius)`.

Tag path: OpenCV `aruco`/`apriltag` — the robot already has `hiwonder.apriltag`. Prefer that detector for corners, then `cv2.solvePnP` with object points of a square `--tag-size`. Camera matrix from npz. If `hiwonder.apriltag` is missing on the laptop, skip tag tests there.

Cam-from-body (locked, MuJoCo site, metres):

```
t_cam_body = [0.08, 0.00, 0.04]
# camera looks +X, xyaxes="0 -1 0 0 0 1" → camera X = body −Y, camera Y = body +Z, camera Z = body −X
# implement as R_body_from_cam with columns = those axes, then p_body = R @ p_cam + t
```

Write a unit test in `replay_policy.py` comments: a point 1 m in front of the camera (`p_cam` along optical axis) must land near body `+X`. If it lands on −X, the look-axis is wrong — **stop and fix**, this is the same class of bug as NAV.md “head camera looked −X”.

### 9.7 `safety.py`

`class Safety` holds consecutive-fall counter. `filter(action, hw, dist, step) -> (action, reason|None)` where reason in `{None, "estop", "fall", "max_steps", "arrived", "tof_guard"}`.

Ctrl+C: install a handler that sets a flag; the loop checks it. Do not leave the robot mid-clip without `stop_motion`.

### 9.8 `deploy_nav.py` loop

```
load yaml + argparse (CLI overrides yaml)
load policy
hw.start(); center_head
goal.reset(...)
prev = 0
for step in range(max_steps):
    if safety.estop: break
    bgr = hw.read_bgr()
    occ = occupancy.encode(bgr)
    tof, sonar = hw.read_tof_m(), hw.read_sonar_m()
    imu = hw.read_imu()
    gravity = gravity_from_imu(imu[0], R) if imu else [0,0,-1]
    dist, heading, *_ = goal.current()
    obs = build_obs(...)
    a = act(model, obs)
    a, reason = safety.filter(a, ...)
    if reason in ("estop", "fall", "max_steps", "arrived"):
        hw.play("stand"); break
    if not dry_run:
        hw.play(ACTION_NAMES[a])
    goal.on_action(ACTION_NAMES[a])
    prev = a
    log jsonl
hw.stop()
```

CLI (locked names):

```
python -m sim2real_implementation.deploy_nav \
  --model PATH.zip \
  --backend fake|tonypi \
  --goal 2.0 -0.5 \
  --goal-mode deadreckon|tag|fixed-body \
  --occupancy yolo|color|zeros|file \
  --dry-run \
  --max-steps 20 \
  --arrive 0.20 \
  --log-dir sim2real_implementation/logs/run1
```

### 9.9 `replay_policy.py` (Phase 0)

No `hiwonder`. `--synthetic-clear` builds:

- occupancy zeros
- tof 1.2, sonar 5.0
- gravity `[0,0,-1]`
- dist=2.0, heading=0, prev=stand

Print action name + logits if `model.policy` allows. `--synthetic-wall`: occupancy[:, :, 28:36] = 1 (center vertical bar), tof=0.25.

`--image foo.jpg --goal 2 0` runs color or yolo on a still.

### 9.10 `sensor_probe.py`

Infinite loop 5 Hz, print one line. `--save-frame`, `--save-occ`, `--tune-hsv` shows a window with trackbars (skip if `DISPLAY` empty, print HSV defaults).

### 9.11 calibrators

`calibrate_imu.py`: prompts “standing, enter”, “tilt forward, enter”, writes yaml.

`calibrate_primitives.py`: for each of the 6 moving names, play clip N=5 times, user types measured Δx Δy Δyaw_deg on tape. Average, convert yaw to rad, write yaml. `stand` stays zeros.

### 9.12 `configs/deploy.yaml`

```yaml
model: models_ckpt/new_expert_implementation/tonypi_final_model.zip
backend: tonypi
goal: [2.0, -0.5]
goal_mode: deadreckon
occupancy: color
arrive: 0.20
max_steps: 20
max_seconds: 60
yolo_weights: yolo11n-seg.pt
yolo_imgsz: 320
yolo_conf: 0.25
hsv_lo: [0, 80, 80]
hsv_hi: [12, 255, 255]
recover_fall: false
tof_guard: null
dry_run: false
```

HSV defaults assume **red** obstacles (sim boxes). Change for blue tape if that is what you put on the floor.

---

## 10. Tests the coding agent must run on the laptop

Create `sim2real_implementation/tests/test_obs_contract.py` (stdlib `unittest` is enough):

1. `build_obs` vec length 18, occupancy shape, one-hot of 5 has a 1 at index 5.
2. `dist=3, heading=0` → `vec[0]==1`, `vec[1]==0`, `vec[2]==1`.
3. `scale_range(999, 1.2)==1.0`, `scale_range(0.6, 1.2)==0.5`.
4. Fake loop: `FakeHardware` + zeros occupancy + `fixed-body` goal `(2,0)` + **scripted** actions (do not require the zip in CI) updates dead-reckon pose using sim primitives: one `go_forward` → x≈0.08.
5. If the zip is on disk, `load_nav_policy` + synthetic-clear returns an int in `0..6`.

Do not add pytest fixtures that import mujoco.

---

## 11. Copy to the Pi

Document in the log the rsync you used. Suggested layout on the Pi:

```
/home/pi/nav_deploy/
  tonypi_rl/          # nav_features.py, perception.py, guided_ppo.py, __init__.py
  sim2real_implementation/
  models/tonypi_final_model.zip
  yolo11n-seg.pt      # optional
```

`perception.py` imports numpy/cv2/ultralytics only inside `YoloSegOccupancy.__init__` and `encode_bgr`. `MujocoGtOccupancy` imports mujoco **inside methods** — do not instantiate it on the Pi.

`guided_ppo.py` imports SB3 only. Safe.

If SB3 pickle requires the original `features_extractor_class` path `tonypi_rl.nav_features.NavFeaturesExtractor`, the package name must remain `tonypi_rl`.

---

## 12. Bugs already known (do not reintroduce)

1. **18-DoF vs nav zip.** `visualize_nav.inspect_checkpoint` already distinguishes Dict vs Box. Refuse Box zips in `policy.py`.
2. **App ActionGroupDict ids ≠ policy ids.** Map by **name**.
3. **`best.pt` is detect-only.** No masks → empty occupancy → polar homer that rams.
4. **Head PWM not centered** → occupancy geometry ≠ train camera.
5. **`tonypi.service` holds /dev/video.** Stop it.
6. **Gyro unit guessing.** Zeros.
7. **Sonar missing encoded as 1.0 metre.** Encode as cutoff (5 m → scaled 1.0).
8. **Importing `tonypi_nav_env` on the Pi.** Dummy env only.
9. **`python tonypi_rl/train.py` path bug** is irrelevant here; still run as `python -m sim2real_implementation.deploy_nav`.
10. **Calling `expert_action` / loading planner.** Train-only.
11. **VecNormalize.** Not in the zip. Do not wrap.
12. **`times=3` like obstacle_avoidance.py.** Policy outputs one clip. `times=1`.
13. **balance_ankle during gait.** Off.
14. **Camera optical axis sign.** Same failure mode as sim `xyaxes` bug. Unit-test the tag transform.

---

## 13. Implementation order (agent)

Do not write everything then run. After each bullet, run the matching BRINGUP test you can run without the robot (Phase 0) and commit to the log.

1. `constants.py`, `observations.py`, `tests/test_obs_contract.py` (no zip required except test 5).
2. `policy.py` + `replay_policy.py` — **Phase 0.1–0.3 must pass before any hardware file.**
3. `hardware.py` Fake + `goal.py` deadreckon/fixed-body + `safety.py` + `deploy_nav.py` `--backend fake`.
4. `occupancy_real.py` color/zeros/file.
5. `TonyPiHardware` + `sensor_probe.py` + calibrators.
6. Tag goal + cam transform unit test.
7. YAML CLI glue, requirements, IMPLEMENTATION_LOG.

If something in this spec disagrees with `tonypi_nav_env.py` at the time of implementation, **env code wins**. Note the diff in the log. Do not “improve” the obs.

---

## 14. What “done” means

v1 is done when:

- Phase 0 tests pass on the laptop against `tonypi_final_model.zip`
- Phase 1 probe prints live ToF/camera on the Pi
- Phase 3 short-leash walk runs 12 steps without falling, with JSONL logs
- Phase 5 detour: first actions are not a pure `go_forward` ram on a centerline box **or** a captured log shows occupancy/ToF were empty (then occupancy, not policy, is the bug)

v1 is **not** “walks as well as sim on mixed layouts”. That is a later domain-randomization / YOLO-finetune project and is out of this spec.
