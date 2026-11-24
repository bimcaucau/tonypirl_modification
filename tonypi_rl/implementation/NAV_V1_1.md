# TonyPiNav v1.1 — coding-agent implementation spec

**This file is the source of truth for the v1.1 patch.** Read it fully before editing. Implement exactly what is specified. Do not invent extra features. Do not rewrite walking.

| | |
|---|---|
| Repo root | `/mnt/d/robotics_cccc/tonypi` (cluster: `/AIClub_NAS/nguyenmv/ccc170/HN`) |
| Layer | **High-level nav only** (`TonyPiNavEnv`, Discrete(7) action groups) |
| Out of scope | `tonypi_env.py`, `v0_straight_walk.yaml`, `.d6a` files, `tonypi_inside/`, MuJoCo meshes/joints, camera `xyaxes` |
| Playback | **Do not change** GUI `time.sleep`, video `--fps`, frame interpolation, or kinematic teleport speed |
| Old checkpoints | **Incompatible.** New experiment names. Do not `--resume` a 32×32 / 19-D zip |

Current design (do not unlearn it): `tonypi_rl/NAV.md`.

---

## 0. How to run anything (path bugs)

Always from **repo root**. Spawn workers do not inherit `sys.path` from `python tonypi_rl/train.py`.

```bash
cd /mnt/d/robotics_cccc/tonypi
export PYTHONPATH=$PWD
export MUJOCO_GL=egl          # fallback: osmesa
# python: /home/bimcaucau/miniconda3/bin/python  (WSL)

python -m tonypi_rl.train --config tonypi_rl/configs/v1_1_nav_expert.yaml
python tonypi_rl/visualize_nav.py --layout demo --expert --render-image tonypi_rl/nav_expert_preview.png
```

**Forbidden:** `python tonypi_rl/train.py` (puts `tonypi_rl/` on `sys.path[0]`; workers import the wrong package).

**Path resolution rules (apply everywhere you touch):**

- Package dir = `os.path.dirname(os.path.abspath(__file__))` for files inside `tonypi_rl/`.
- Repo root = `os.path.dirname(package_dir)`.
- `xml_path` default already joins repo root + `tonypi_mujoco/scene_nav.xml`. Keep that. Meshes: `tonypi.xml` has `meshdir="../meshes/"` relative to `tonypi_mujoco/` → repo `meshes/`. Do not change.
- YAML `log_dir: "tonypi_rl/logs"` is repo-relative. `train.py` currently does `os.path.join(parent_dir, log_dir)` which is correct **only if** the yaml value is relative. Add helper:

```python
def _resolve_under_repo(path: str, repo_root: str) -> str:
    if not path:
        return os.path.abspath(repo_root)
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(repo_root, path))
```

Use it for `log_dir`, `model_dir`, and `xml_path` if not already absolute.

- One occupancy constant: `from tonypi_rl.perception import OCCUPANCY_SIZE`. No leftover hardcoded `32` for the policy mask.
- Gym register `TonyPiNav-v0` currently `max_episode_steps=80` while yaml used 240. Training constructs `TonyPiNavEnv(...)` directly so yaml wins, but `gym.make` is a trap. Set register cap to **120**.
- `eval.py` is walk-only. If `environment.env_class == nav`, print a clear error pointing at `visualize_nav.py` and exit. Do not silently build `TonyPiEnv`.

---

## 1. Locked design answers (do not reopen)

### 1.1 Occupancy 32 → 64

Today `perception.py` rasterizes at `SEG_RENDER_SIZE=64` then **max-pools 2×2 → `OCCUPANCY_SIZE=32`**. Policy sees a blob.

- Set `OCCUPANCY_SIZE = 64`.
- Project **directly onto 64**. Drop the 2×2 pool (or keep a helper that identity-maps when sizes match).
- `YoloSegOccupancy._resize` already uses `OCCUPANCY_SIZE`; it will follow.
- CNN: one extra stride-2 conv so 64×64 still ends at 4×4 (see §3). **Not NatureCNN.**

### 1.2 Reward / back-and-forth

Current:

```
r = 5.0 * (dist_prev - dist) - 0.05 + 20.0*arrived - 2.0*collision
max_episode_steps = 240
gamma = 0.99
```

Why the prototype jitters (`video_debug/first_prototype.mp4`):

- 240 × 0.05 = 12 < arrive 20. Living is cheap. Discounted arrive at step 100 is `0.99^100 * 20 ≈ 7.4`.
- `back_fast` is free. Forward +8 cm then back −6 cm almost cancels.
- Euclidean progress **punishes going around a box** → ram, reverse, repeat. Geodesic is the correct potential.
- Collision does not end the episode → wedge burns remaining steps.

New reward (implement exactly):

```
r = 5.0 * clip(geo_prev - geo, ±0.15)     # geodesic metres closed; fallback Euclidean if Dijkstra failed
  - 0.25                                   # time / alive, every step
  + 100.0 if arrived                       # then terminate
  - 3.0  if collision this step
  - 0.50 if action == back_fast and dist > 1.5 * arrive_radius
  - 0.30 if action is opposite of previous
         pairs: (go_forward, back_fast), (left_move_fast, right_move_fast), (turn_left, turn_right)
  + 0.15 * cos(heading_error) if action == go_forward
```

- `max_episode_steps = 120`.
- After **8 consecutive collisions**, `truncated=True` (wedge). First bump still does **not** terminate.
- `arrived` remains the only `terminated=True`.
- Training yaml: `use_geodesic_reward: true`. Stock Euclidean PPO is how the prototype learned to ram-and-reverse.

### 1.3 Observation: polar, not world XY, not both Cartesian and polar

**Today** `vec` is 19-D:

| slice | content |
|---|---|
| 0:4 | body-frame goal `gx, gy, dist, heading_error` |
| 4:5 | ToF / 1.2 |
| 5:6 | sonar / 5.0 |
| 6:9 | projected gravity |
| 9:12 | gyro (≈0 in kinematic sim; keep for Pi parity) |
| 12:19 | previous action one-hot (7) |

World robot `(x, y)` is **already not** in the obs. Keep it that way.

`(gx, gy)` **is already** “relative +2 m, −0.5 m” after rotating by pelvis yaw. It is **redundant** with `(dist, heading)` because `gx = dist·cos(θ)`, `gy = dist·sin(θ)`.

For Discrete(7) gait primitives, polar is the control error: heading → turn vs forward; dist → stop vs walk. Cartesian body `(gx, gy)` would matter for continuous `(vx, vy)`, which we do not have.

**New `vec` is 18-D:**

```
vec = [
  dist / 3.0,                 # remaining Euclidean range, scaled
  sin(heading_error),
  cos(heading_error),         # wrap-safe; do NOT use raw angle (±π jump)
  tof_0 / 1.2,
  sonar / 5.0,
  gravity (3),
  gyro (3),
  prev_action one-hot (7),
]
```

`_goal_body_frame()` still computes `gx, gy, dist, heading` for expert, geodesic fallback, debug `info`, and visualize HUD. **Policy obs drops `gx, gy`.** Put `gx`, `gy`, `heading` in `info` only.

### 1.4 Sim speed / physics — **no code change**

Nav is kinematic on purpose: `mj_forward` + pelvis teleport, joints frozen at stand, one env step = one real `.d6a` clip (~0.4 s). Do **not** add `mj_step` walking. Do **not** change viewer sleep, record fps, or primitive interpolation timing. Do **not** add primitive noise.

---

## 2. Files

### Create

| path | role |
|---|---|
| `tonypi_rl/implementation/NAV_V1_1.md` | this spec (already written; do not delete) |
| `tonypi_rl/configs/v1_1_nav.yaml` | stock PPO, new obs/reward/64, **geodesic on** |
| `tonypi_rl/configs/v1_1_nav_expert.yaml` | GuidedPPO + geodesic + new numbers |

Leave `v1_nav.yaml` / `v1_nav_expert.yaml` on disk as history. Do not train them after this patch (obs space changed in code; they would silently use 64×64 + 18-D with old reward numbers if someone runs them). Optional: add a one-line comment at the top of each old yaml: `# SUPERSEDED by v1_1_*.yaml — obs space changed, do not use`.

### Edit

| path | what |
|---|---|
| `tonypi_rl/perception.py` | `OCCUPANCY_SIZE=64`, no 2×2 pool when sizes match |
| `tonypi_rl/nav_features.py` | 4-layer CNN for 64×64 |
| `tonypi_rl/tonypi_nav_env.py` | polar vec, new reward kwargs, collision streak |
| `tonypi_rl/__init__.py` | `max_episode_steps=120` |
| `tonypi_rl/train.py` | `NAV_ENV_KEYS`, path helper, print 64×64 / vec dim |
| `tonypi_rl/visualize_nav.py` | upsample from `OCCUPANCY_SIZE`; HUD from `info` not `vec[0:4]`; default max steps 120. **Do not change sleep/fps.** |
| `tonypi_rl/eval.py` | refuse nav configs |
| `tonypi_rl/nav_planner.py` | comment: camera mask is 64×64, still not the planning grid |
| `tonypi_rl/NAV.md` | add a 10-line “v1.1” pointer at the top to this file; do not rewrite the rest |

### Do not touch

`tonypi_env.py`, `configs/v0_straight_walk.yaml`, `tonypi_mujoco/tonypi.xml` camera axes, `tonypi_inside/`, `guided_ppo.py` (unless a ctor kwarg name must pass through — it should not), `nav_expert.py` / planner logic.

---

## 3. File-by-file implementation

### 3.1 `tonypi_rl/perception.py`

Current:

```python
OCCUPANCY_SIZE = 32
SEG_RENDER_SIZE = 64
```

GL and project paths do:

```python
occupancy = mask64.reshape(OCCUPANCY_SIZE, 2, OCCUPANCY_SIZE, 2).max(axis=(1, 3))
```

That **breaks** if both sizes are 64 (`64 ≠ 32*2`).

Replace with:

```python
OCCUPANCY_SIZE = 64
SEG_RENDER_SIZE = 64   # pinhole / GL working resolution

def _finalize_mask(mask: np.ndarray) -> np.ndarray:
    """(H,W) float -> (1, OCCUPANCY_SIZE, OCCUPANCY_SIZE) binary {0,1}."""
    h, w = mask.shape
    if h == OCCUPANCY_SIZE and w == OCCUPANCY_SIZE:
        occ = mask
    elif h == 2 * OCCUPANCY_SIZE and w == 2 * OCCUPANCY_SIZE:
        occ = mask.reshape(OCCUPANCY_SIZE, 2, OCCUPANCY_SIZE, 2).max(axis=(1, 3))
    else:
        import cv2
        occ = cv2.resize(mask, (OCCUPANCY_SIZE, OCCUPANCY_SIZE), interpolation=cv2.INTER_AREA)
    occ = (occ > 0).astype(np.float32)
    return occ[np.newaxis, ...]
```

Use `_finalize_mask(mask64)` in `_encode_gl` and `_encode_project` (including the empty-geom early return).

Update docstrings: `(1, 64, 64)` not `(1, 32, 32)`.

`rgb()` can stay 64×64. `YoloSegOccupancy._resize` already uses `OCCUPANCY_SIZE`.

### 3.2 `tonypi_rl/nav_features.py`

Replace the 3-conv stack with 4 stride-2 convs. Dummy-tensor flatten size stays the source of `n_cnn` (already there). Keep vec MLP `n_vec → 64` (now `n_vec=18`). Merge unchanged.

```python
self.cnn = nn.Sequential(
    nn.Conv2d(n_in, 8, kernel_size=3, stride=2, padding=1),   # 64 -> 32
    nn.ReLU(),
    nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),     # 32 -> 16
    nn.ReLU(),
    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),    # 16 -> 8
    nn.ReLU(),
    nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),    # 8 -> 4
    nn.ReLU(),
    nn.Flatten(),   # 32*4*4 = 512
)
```

Update the module docstring (NatureCNN still dies on 64; this extractor is required). Do **not** use `CnnPolicy` / NatureCNN.

### 3.3 `tonypi_rl/tonypi_nav_env.py`

**Ctor defaults** (yaml overrides these):

```python
max_episode_steps: int = 120
arrive_radius: float = 0.20
progress_weight: float = 5.0
arrive_bonus: float = 100.0
time_penalty: float = 0.25
collision_penalty: float = 3.0
reverse_penalty: float = 0.50
oscillation_penalty: float = 0.30
heading_align_weight: float = 0.15
collision_streak_limit: int = 8
use_geodesic_reward: bool = False          # yaml turns this on
geodesic_progress_clip: float = 0.15
# existing expert_* kwargs unchanged
```

Store all of them on `self`. Init `self._collision_streak = 0` in `__init__` and reset it in `reset()`.

**Vec dim:**

```python
GOAL_DIST_SCALE = 3.0
self.vec_dim = 3 + len(TOF_NAMES) + 1 + 3 + 3 + len(ACTION_NAMES)  # 18
```

**`_get_obs`:** keep `_goal_body_frame()`; build goal slice as:

```python
gx, gy, dist, heading = self._goal_body_frame()
goal_vec = np.array(
    [dist / GOAL_DIST_SCALE, np.sin(heading), np.cos(heading)],
    dtype=np.float32,
)
# then tof, sonar, gravity, gyro, prev one-hot  — same as now
```

**Opposites** (module-level):

```python
OPPOSITE_NAME = {
    "go_forward": "back_fast",
    "back_fast": "go_forward",
    "left_move_fast": "right_move_fast",
    "right_move_fast": "left_move_fast",
    "turn_left": "turn_right",
    "turn_right": "turn_left",
}
```

**`step()` order** (important: do not overwrite `prev_action` before oscillation check):

```python
prev_action = int(self.prev_action)
# ... kinematic primitive, collided flag ...
if collided:
    self._collision_streak += 1
else:
    self._collision_streak = 0
self.prev_action = action
self.step_count += 1
obs = self._get_obs()
reward, info = self._compute_reward(collided, expert_a, action, prev_action)
terminated = bool(info["arrived"])
truncated = (self.step_count >= self.max_episode_steps) or (
    self._collision_streak >= self.collision_streak_limit
)
```

**`_compute_reward(self, collided, expert_a, action, prev_action)`:**

```python
gx, gy, dist, heading = self._goal_body_frame()
cur = self._progress_metric()
raw_progress = self._prev_progress - cur
if self.use_geodesic_reward:
    progress = float(np.clip(raw_progress, -self.geodesic_progress_clip, self.geodesic_progress_clip))
else:
    progress = float(raw_progress)

reward = self.progress_weight * progress - self.time_penalty
arrived = dist < self.arrive_radius
if arrived:
    reward += self.arrive_bonus
if collided:
    reward -= self.collision_penalty

name = ACTION_NAMES[int(action)]
prev_name = ACTION_NAMES[int(prev_action)]
if name == "back_fast" and dist > 1.5 * self.arrive_radius:
    reward -= self.reverse_penalty
if OPPOSITE_NAME.get(name) == prev_name:
    reward -= self.oscillation_penalty
if name == "go_forward":
    reward += self.heading_align_weight * float(np.cos(heading))

# info: keep existing keys, plus:
# gx, gy, heading, collision_streak
# action_name from ACTION_NAMES[action] (current), not prev
self._prev_dist = dist
self._prev_progress = cur
return float(reward), info
```

`reset()` info should also include `gx, gy, heading, collision_streak=0`.

Docstring at top: occupancy is 64×64; vec is polar 18-D.

### 3.4 `tonypi_rl/__init__.py`

```python
register(
    id="TonyPiNav-v0",
    entry_point="tonypi_rl.tonypi_nav_env:TonyPiNavEnv",
    max_episode_steps=120,
)
```

Training still constructs the class directly; this only fixes `gym.make`.

### 3.5 `tonypi_rl/train.py`

Add to `NAV_ENV_KEYS`:

```python
"reverse_penalty", "oscillation_penalty", "heading_align_weight",
"collision_streak_limit",
```

Keep existing geodesic / expert keys.

Add `_resolve_under_repo` (see §0). Use it for:

```python
base_log_dir = _resolve_under_repo(log_cfg.get("log_dir", "tonypi_rl/logs"), parent_dir)
base_model_dir = _resolve_under_repo(log_cfg.get("model_dir", "tonypi_rl/models"), parent_dir)
```

and the xml_path loop (if relative, join repo root first).

Nav print line currently says `CNN 32x32 + MLP vec`. Change to import `OCCUPANCY_SIZE` and print:

```
Policy: MultiInputPolicy (CNN {OCCUPANCY_SIZE}x{OCCUPANCY_SIZE} + MLP vec)
```

`NAV_REQUIRED_FILES` already lists nav_env / perception / nav_features / scene_nav.xml. Keep it. Fail with the full missing-path list (already the pattern).

Do not change spawn `"spawn"`, `MUJOCO_GL=egl`, `_bind_tonypi_rl`, or GuidedPPO wiring.

### 3.6 `tonypi_rl/visualize_nav.py`

- Module docstring: `64x64` not `32x32`.
- `_upsample(mask)`: `scale = max(1, 256 // int(mask.shape[0]))` so 64 → 4, 32 → 8. Do not hardcode `scale=8`.
- Occupancy panel title: `f"Occupancy mask {occ.shape[0]}x{occ.shape[1]}"`.
- **HUD / `print_obs` currently read `vec[0:4]` as `gx, gy, dist, heading`.** After the obs change that is wrong (`vec[0]=dist/3`, `vec[1]=sin`, `vec[2]=cos`). Read from `info` instead:

```python
gx = info.get("gx", float("nan"))
gy = info.get("gy", float("nan"))
dist = info.get("dist", float("nan"))
heading = info.get("heading", float("nan"))
```

If a code path has no `info` yet, fall back to `env._goal_body_frame()`.

- `default_max_steps`: `120` when expert/model, else 80 is fine for a still preview. Or just 120 always. Prefer 120 for expert/model rollouts.
- **Do not change** `time.sleep(0.04)`, `--fps` default 15, `FrameRecorder`, or interpolation.

### 3.7 `tonypi_rl/eval.py`

After loading yaml, if `cfg["environment"].get("env_class") == "nav"` (case-insensitive):

```
print nav eval is visualize_nav.py; this script is TonyPiEnv / 18-DoF only
sys.exit(1)
```

Do this **before** `TonyPiEnv(**env_kwargs)` so a nav yaml cannot dump unknown kwargs into the walk env.

### 3.8 New YAML

Copy structure from `configs/v1_nav.yaml` / `v1_nav_expert.yaml`. Differences:

`tonypi_rl/configs/v1_1_nav.yaml`:

```yaml
experiment:
  name: "tonypi_v1_1_nav"
  description: "Nav v1.1: 64x64 occupancy, polar goal vec, geodesic progress, time/arrive retune"
  device: "auto"
  seed: 42

curriculum:
  stage: "v1_1_nav"

environment:
  env_class: nav
  xml_path: "tonypi_mujoco/scene_nav.xml"
  layout: random
  goal_mode: random
  n_obstacles: 4
  max_episode_steps: 120
  arrive_radius: 0.20
  progress_weight: 5.0
  arrive_bonus: 100.0
  time_penalty: 0.25
  collision_penalty: 3.0
  reverse_penalty: 0.50
  oscillation_penalty: 0.30
  heading_align_weight: 0.15
  collision_streak_limit: 8
  use_geodesic_reward: true
  expert_robot_radius: 0.18
  expert_grid_res: 0.05
  expert_heading_weight: 0.12

ppo:
  policy: MultiInputPolicy
  timesteps: 20000000
  n_envs: 8
  n_steps: 2048
  batch_size: 1024
  n_epochs: 10
  learning_rate: 0.0003
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.002

logging:
  save_freq: 100000
  eval_freq: 100000
  log_dir: "tonypi_rl/logs"
  model_dir: "tonypi_rl/models"
```

`v1_1_nav_expert.yaml`: same `environment:` and `ppo:` / `logging:`, plus:

```yaml
experiment:
  name: "tonypi_v1_1_nav_expert"
  description: "Nav v1.1 + A* GuidedPPO (BC warmup + mix + DAgger CE)"
  ...

curriculum:
  stage: "v1_1_nav_expert"

expert:
  enabled: true
  bc_warmup_steps: 50000
  bc_epochs: 8
  bc_batch_size: 1024
  bc_coef: 0.15
  bc_anneal_steps: 2000000
  mix_prob: 0.3
  mix_anneal_steps: 1000000
  heading_weight: 0.12
```

**Train this one** (`v1_1_nav_expert.yaml`). Geodesic reward needs the planner rebuild on reset (`use_geodesic_reward: true` already triggers `_rebuild_expert()`). Stock `v1_1_nav.yaml` also has geodesic on so even without BC the progress signal is correct.

### 3.9 `NAV.md` pointer

Insert **above** “Read this first”, a short box:

```markdown
> **v1.1 (obs/reward retune):** polar 18-D vec, occupancy 64×64, arrive 100 / time 0.25 / collision streak.
> Implement and train from `tonypi_rl/implementation/NAV_V1_1.md` and `configs/v1_1_nav_expert.yaml`.
> Do not resume 32×32 checkpoints.
```

Do not rewrite sections 1–13. Those remain the architecture handoff.

---

## 4. What “done” looks like (verify from repo root)

```bash
cd /mnt/d/robotics_cccc/tonypi
export PYTHONPATH=$PWD
export MUJOCO_GL=egl

# A. shapes
python -c "
from tonypi_rl.tonypi_nav_env import TonyPiNavEnv
e = TonyPiNavEnv()
o,_ = e.reset(seed=0)
assert o['vec'].shape == (18,), o['vec'].shape
assert o['occupancy'].shape == (1, 64, 64), o['occupancy'].shape
print('ok', o['vec'].shape, o['occupancy'].shape, float(o['occupancy'].mean()))
e.close()
"

# B. expert still arrives (expert does not use policy vec; must not regress)
python tonypi_rl/visualize_nav.py --layout demo --expert --render-image tonypi_rl/nav_expert_preview.png

python -c "
from tonypi_rl.tonypi_nav_env import TonyPiNavEnv
e = TonyPiNavEnv(layout='random', goal_mode='random', max_episode_steps=120,
                 use_geodesic_reward=True)
ok = 0
for s in range(20):
    e.reset(seed=s)
    for _ in range(120):
        _, _, term, trunc, info = e.step(e.expert_action())
        if term:
            ok += 1
            break
        if trunc:
            break
print('success', ok, '/20')
e.close()
"
# expect >= 16/20. If it wedges, bump expert_robot_radius toward 0.20 — do not change walking.

# C. train smoke (1 env, 1 rollout)
python -m tonypi_rl.train --config tonypi_rl/configs/v1_1_nav_expert.yaml --timesteps 2048 --n_envs 1
```

If A fails: occupancy pool or vec_dim. If B fails: you broke `_goal_body_frame` / primitives / expert rebuild, not the policy. If C fails with `ModuleNotFoundError` / `unexpected keyword`: you ran `python tonypi_rl/train.py` or forgot `NAV_ENV_KEYS`.

---

## 5. Explicit non-goals (reject if a later prompt asks)

- 18-DoF joint control inside nav.
- Playing `.d6a` clips inside MuJoCo.
- Continuous `cmd_vel`.
- Extra ToF rays (`tof_1..4`).
- NatureCNN / LSTM / ViT.
- `mujoco.Renderer` inside `TonyPiNavEnv.__init__` (EGL segfault in spawn).
- Changing head camera `xyaxes`.
- Slowing or speeding visualization / video fps.
- Resuming `models/139/` or `models/149expert/` into v1.1.

---

## 6. After the patch (human, not this coding task)

Retrain from scratch:

```bash
python -m tonypi_rl.train --config tonypi_rl/configs/v1_1_nav_expert.yaml
```

Watch TensorBoard `tonypi_rl/logs`: `rollout/ep_rew_mean`, `rollout/ep_len_mean` (should drop well below 120 as it starts arriving), `rollout/expert_agree`, `train/bc_loss`.
