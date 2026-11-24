# TonyPiNav v1.2 — coding-agent implementation spec

**This file is the source of truth for the v1.2 patch.** Read it fully before editing. Implement exactly what is specified. Do not invent extra features. Do not rewrite walking. Do not grow the PPO network.

> **Post-review (v1.2.1):** do not resume-train until [`NAV_V1_2_FIX.md`](NAV_V1_2_FIX.md) is applied. The v1.2 log is not fully verified: `learn()` still resets the teacher clock, scatter fallback can leave ghost boxes, and `u_trap` is a front fence.

v1.1 is already in the tree (`NAV_V1_1.md` + log). Do not revert occupancy 64×64, polar 18-D, geodesic clip, NaN clamps, or random spawn.

| | |
|---|---|
| Repo root | `/mnt/d/robotics_cccc/tonypi` (cluster: `/AIClub_NAS/nguyenmv/ccc170/HN`) |
| Layer | **High-level nav only** (`TonyPiNavEnv`, Discrete(7)) |
| Out of scope | `tonypi_env.py`, `v0_straight_walk.yaml`, `.d6a`, `tonypi_inside/`, meshes/joints, camera `xyaxes`, `nav_features.py`, `perception.py` size |
| Playback | **Do not change** GUI sleep, `--fps`, interpolation, kinematic teleport |
| Teacher at deploy | **Must be 0.** Final zip infers from occupancy + polar vec only. No `expert_action()` on the Pi |
| Resume | `tonypi_checkpoint_4300000_steps.zip` (v1.1 64×64 / 18-D). **Not** a 32×32 zip |

Evidence: `video_debug/4m3steps.mp4` (open, ARRIVED in 19 steps) vs `video_debug/4m3steps3.mp4` (wall cluster, COLLISION, geo `n/a`).

---

## 0. How to run (path bugs, same as v1.1)

Always from **repo root**:

```bash
cd /mnt/d/robotics_cccc/tonypi
export PYTHONPATH=$PWD
export MUJOCO_GL=egl

python -m tonypi_rl.train --config tonypi_rl/configs/v1_2_nav_expert.yaml
python tonypi_rl/visualize_nav.py --layout wall --expert --render-image tonypi_rl/nav_wall_preview.png
```

**Forbidden:** `python tonypi_rl/train.py`.

Keep `_resolve_under_repo`, `_bind_tonypi_rl`, `NAV_REQUIRED_FILES`. Gym TimeLimit is a trap — env owns the cap (set register to **160**).

YAML keys live under `environment:` (not `env:`). The v1.1 log snippet that says `env:` is wrong; do not copy it.

---

## 1. Locked diagnosis (do not reopen)

The 4.3M policy is a **polar homer**. Open layouts work. Blocked/narrow ones do not.

1. **Fake corridors.** Scatter rejects only center distance `< 0.28 m`. Boxes `hx,hy ≈ 0.08–0.12`, `expert_robot_radius = 0.18` → visual gap ~10 cm. Camera sees a slot; A* and physics do not. Policy `right_move_fast`s into it (`4m3steps3.mp4` steps 10–20).
2. **Easy resets.** Uniform annulus + 3–4 boxes + 0.45 m start clearance. Line of sight is usually free. Occupancy CNN unused.
3. **Teacher died too early.** `mix_anneal_steps: 1e6`, `bc_anneal_steps: 2e6` → mix/BC are **0 by 2M**. That **end** state is required for the real robot (no A*). It must not happen *before* detours are learned. **Stretch the anneal to 0.** Do **not** add `mix_prob_floor`.
4. **Rewards pay for ramming.** `+0.15·cos(heading)` on `go_forward` through a wall. `reverse_penalty` forbids backing out.
5. **Geodesic → Euclidean fallback is a bug.** When the robot wedges, `geodesic` is inf, `_progress_metric` uses Euclidean, dist drops (2.16 → 1.44), clipped progress is **+0.15**. Ramming is rewarded.

Do not expand `NavFeaturesExtractor` / `net_arch`.

---

## 2. Deploy constraint (locked)

At the **end** of training:

- `mix_prob == 0`
- `bc_coef == 0`
- `visualize_nav.py --model <final.zip>` must **not** call `expert_action` to choose actions (expert overlay in the HUD is fine)

Policy obs stays Dict `{vec: (18,), occupancy: (1,64,64)}`. A* / Dijkstra / geodesic are **train-only** (`use_geodesic_reward=True` on the train env). The zip must run on the Pi as Discrete(7) from sensors.

`GuidedPPO._lin` already returns exactly 0 at `t >= t_end`. **Do not add a floor.** Only change yaml durations (see §6).

---

## 3. Layouts that force detours

Keep `goal_mode` / `_sample_goal()` as they are (random annulus or `DEMO_GOAL`).

Add an internal kind, **not** overloaded onto `layout=="demo"`:

```python
# self.layout is the requested mode ("mixed", "wall", "random", ...)
# self._layout_kind is what this episode actually spawned ("scatter"|"wall"|"u_trap"|"gate")
```

Put `layout` and `layout_kind` in `reset`/`step` `info`.

### 3.1 Requested `layout` values

| `layout` | episode kind |
|---|---|
| `demo` | unchanged demo boxes (skip any within 0.35 m of goal, already) |
| `empty` | no boxes |
| `random` / `scatter` | scatter sampler §3.3 |
| `wall` | §3.4 |
| `u_trap` | §3.5 |
| `gate` | §3.6 |
| `mixed` / `curriculum` | each reset: 40% scatter, 30% wall, 15% u_trap, 15% gate. **Train default.** |

`random_spawn=True` (ctor or `reset(options)`): `goal_mode="random"` **and** if `layout` was `demo`/`fixed`-like, set `layout="mixed"`. Do **not** force scatter-only. If caller already passed `layout="wall"`, keep wall.

`n_obstacles` caps **scatter** only. Wall / u_trap / gate use their own counts (≤ `N_OBS_SLOTS=8`).

After placing boxes, if `use_geodesic_reward` or expert will be used: `_rebuild_expert()`. If `self._expert.unreachable`, **resample up to 20 times**, then fall back to scatter.

### 3.2 Geometry helpers

Robot starts at `(0,0)` after `_set_root` (yaw may be random; boxes are in **world** XY). Goal is `self.goal_xy`.

```python
GAP_MERGED = 0.16      # center-to-center: treat as one blob
GAP_PASSABLE = 0.55    # center-to-center: robot can walk between
START_CLEAR = 0.55     # min hypot(box, origin)
GOAL_CLEAR = 0.45
```

`GAP_PASSABLE` may also be computed as `2 * expert_robot_radius + hx + hx_other + 0.10` using the actual geom sizes; 0.55 is the default with current `scene_nav.xml` boxes.

### 3.3 Scatter (fixes fake corridors)

Replace the current `0.28` inter-box reject.

For each new box vs each already placed box, **two regimes only**:

- merged: `dist <= GAP_MERGED`
- passable: `dist >= GAP_PASSABLE`

Reject `GAP_MERGED < dist < GAP_PASSABLE`. Also `hypot(x,y) >= START_CLEAR`, `hypot(x-gx,y-gy) >= GOAL_CLEAR`.

With probability **0.5**, first place a **merged pair** on the start–goal segment (LOS block):

- Midpoint `m = 0.5 * goal`
- Direction `t = (-goal_y, goal_x)` normalized (perpendicular)
- Two boxes at `m ± 0.08 * t` (center dist 0.16), then remaining scatter boxes as usual (not on the robot, not on the goal).

If a candidate fails 50 tries, skip that slot (fewer obstacles OK).

### 3.4 `wall`

4 or 5 boxes (rng) in a straight line through the midpoint of `(0,0)→goal`, perpendicular to that vector, **no passable gap** (spacing `<= GAP_MERGED`).

Leave **one random end** of the line short so the last box is ≥ `0.55 m` from that side’s “would-be” extent — A* must go around that end, not through.

Do not put a box within `START_CLEAR` of origin or `GOAL_CLEAR` of goal. If the wall would hit those, shift along the perpendicular until clear, or shorten.

### 3.5 `u_trap`

Three-sided U. Robot is inside. Opening faces **away from the goal** (must back/strafe out, then around).

- Back wall: 3 merged boxes, perpendicular to goal dir, ~0.55–0.70 m in front of origin toward the goal.
- Two legs: 1–2 boxes each, parallel to goal dir, spacing merged, leaving ~0.45 m inside width (robot fits) but opening opposite the goal.

No box within 0.35 m of origin (spawn collision). Goal stays outside the U.

### 3.6 `gate`

Exactly two boxes, gap **≥ `GAP_PASSABLE`**, gap center **offset** from the LOS (offset 0.35–0.60 m along perpendicular). Robot must notice the hole, not ram the closer box.

### 3.7 Placement API

```python
def _sample_obstacles(self, goal):
    self._hide_all_obstacles()
    kind = self._choose_layout_kind()   # respects self.layout
    self._layout_kind = kind
    if kind == "empty":
        self._active_obs = []
    elif kind == "demo":
        ... existing demo, skip boxes within GOAL_CLEAR of goal ...
    elif kind == "wall":
        self._place_wall(goal)
    elif kind == "u_trap":
        self._place_u_trap(goal)
    elif kind == "gate":
        self._place_gate(goal)
    else:
        self._place_scatter(goal)
```

`_place_*` writes `mocap_pos` and `self._active_obs`.

---

## 4. Geodesic progress — no Euclidean bailout

Replace `_progress_metric`:

```python
def _progress_metric(self) -> float:
    if self.use_geodesic_reward:
        g = self.geodesic_distance()
        if np.isfinite(g):
            return float(g)
        # stuck / inflated-occupied: punish, do NOT switch to Euclidean
        return float(self._prev_progress + self.geodesic_progress_clip)
    _, _, dist, _ = self._goal_body_frame()
    return float(dist)
```

`raw_progress = prev - cur` then becomes `-clip` when geodesic is inf. Keep `np.clip(±geodesic_progress_clip)` as now.

Info: `geodesic: -1.0` when inf (HUD already shows `n/a`).

Stock PPO (`use_geodesic_reward=False`) still uses Euclidean. Train yaml keeps geodesic **on**.

---

## 5. Reward when blocked

Ctor kwargs (yaml overrides). Keep existing numbers except new `ram_penalty` and default `max_episode_steps=160`.

```python
ram_penalty: float = 1.0
max_episode_steps: int = 160
```

In `_compute_reward`, read current ToF **once** (`tof0 = self._read_range("tof_0", TOF_CUTOFF)`):

- `heading_align_weight * cos(heading)` on `go_forward` **only if** `tof0 > 0.50` and not `collided`. Else 0.
- `reverse_penalty` on `back_fast` **only if** `dist > 1.5 * arrive_radius` **and** `tof0 > 0.50`. Backing off a wall is free.
- **New:** if `name == "go_forward"` and `tof0 < 0.35`: `reward -= ram_penalty`.
- Unchanged: geodesic clip × 5.0, time −0.25, arrive +100, collision −3, oscillation −0.30, streak truncate 8.

`NAV_ENV_KEYS` must include `"ram_penalty"`.

Gym register `TonyPiNav-v0`: `max_episode_steps=160`.

---

## 6. Teacher anneal → 0 (no floor)

`GuidedPPO._lin` stays:

```python
def _lin(self, start: float, t_end: int) -> float:
    if t_end <= 0:
        return float(start)
    return float(start) * max(0.0, 1.0 - float(self.num_timesteps) / float(t_end))
```

Add a one-line comment: deploy requires 0 at `t_end`; do not clamp to a floor.

**Do not** add `mix_prob_floor` / `bc_coef_floor`.

Yaml durations (only change that matters):

```
mix_prob: 0.3
mix_anneal_steps: 8000000     # 0.3 → 0 at 8M (was 1M)
bc_coef: 0.15
bc_anneal_steps: 10000000     # 0.15 → 0 at 10M (was 2M)
bc_warmup_steps: 0            # resume 4.3M already cloned
```

`ppo.timesteps: 15000000` → **≥5M pure PPO** after the teacher is gone (10M–15M). If detours collapse in that window, fix layouts/reward — do not turn the teacher back on.

`_lin` uses `self.num_timesteps`. Resume at 4.3M with the new T:

- mix ≈ `0.3 * (1 - 4.3/8) = 0.139`
- bc  ≈ `0.15 * (1 - 4.3/10) = 0.0855`

then both hit **0** on schedule.

Keep v1.1: ratio clamp `log_ratio ∈ [-3,3]`, `ratio ∈ [0.05,20]`, BC NLL `clamp(max=20)`, skip non-finite steps, `target_kl: 0.03`, `n_epochs: 4`.

Print `mix_anneal_steps`, current mix, `bc_anneal_steps` in the train banner.

### 6.1 `--resume` must apply the new schedule

Today `train.py` on resume applies some PPO knobs. **Extend** that block so a 4.3M zip actually gets the stretched anneal:

```python
if resume_path and os.path.exists(resume_path):
    model = Algo.load(...)
    if expert_enabled and isinstance(model, GuidedPPO):
        model.mix_prob0 = float(expert_cfg.get("mix_prob", model.mix_prob0))
        model.mix_anneal_steps = int(expert_cfg.get("mix_anneal_steps", model.mix_anneal_steps))
        model.bc_coef0 = float(expert_cfg.get("bc_coef", model.bc_coef0))
        model.bc_anneal_steps = int(expert_cfg.get("bc_anneal_steps", model.bc_anneal_steps))
    # existing: target_kl, n_epochs, ent_coef, learning_rate from yaml
```

If those attributes have different names in the loaded pickle, set both the yaml fields and whatever `GuidedPPO` uses (`mix_prob0`, `bc_coef0`).

`bc_warmup_steps: 0` so resume does not re-run BC warmup.

---

## 7. New config

Create `tonypi_rl/configs/v1_2_nav_expert.yaml` (copy `v1_1_nav_expert.yaml`, then these diffs):

```yaml
experiment:
  name: "tonypi_v1_2_nav_expert"
  description: "Nav v1.2: mixed detour layouts, stretched expert anneal to 0, ram penalty"

curriculum:
  stage: "v1_2_nav_expert"

environment:
  env_class: nav
  xml_path: "tonypi_mujoco/scene_nav.xml"
  layout: mixed
  goal_mode: random
  random_spawn: true
  random_robot_yaw: true
  n_obstacles: 4
  max_episode_steps: 160
  arrive_radius: 0.20
  progress_weight: 5.0
  arrive_bonus: 100.0
  time_penalty: 0.25
  collision_penalty: 3.0
  reverse_penalty: 0.50
  oscillation_penalty: 0.30
  heading_align_weight: 0.15
  ram_penalty: 1.0
  collision_streak_limit: 8
  use_geodesic_reward: true
  expert_robot_radius: 0.18
  expert_grid_res: 0.05
  expert_heading_weight: 0.12

expert:
  enabled: true
  bc_warmup_steps: 0
  bc_epochs: 8
  bc_batch_size: 1024
  bc_coef: 0.15
  bc_anneal_steps: 10000000
  mix_prob: 0.3
  mix_anneal_steps: 8000000
  heading_weight: 0.12

ppo:
  policy: MultiInputPolicy
  timesteps: 15000000
  n_envs: 8
  n_steps: 2048
  batch_size: 1024
  n_epochs: 4
  learning_rate: 0.0003
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.005
  target_kl: 0.03
```

Optional: `v1_2_nav.yaml` same env but `expert.enabled: false` (stock PPO, no mix). Not required to train.

Mark `v1_1_nav_expert.yaml` with a comment: superseded for **new** runs by v1.2; 4.3M zip is still the resume source.

---

## 8. `visualize_nav.py`

- `--layout` accepts `demo|empty|random|scatter|wall|u_trap|gate|mixed`.
- Keep `--random` / `--random-spawn` / `--random-yaw` / `--episodes`.
- HUD: print `layout_kind`.
- Default max steps 160.
- **Do not** change fps / sleep / interpolation.
- Expert HUD overlay may still call `env.expert_action()` for the debug panel. Policy actions come from the zip.

---

## 9. Files to touch

Create:

- `tonypi_rl/implementation/NAV_V1_2.md` — this file (already being written)
- `tonypi_rl/configs/v1_2_nav_expert.yaml`
- optional `tonypi_rl/configs/v1_2_nav.yaml`

Edit:

- `tonypi_rl/tonypi_nav_env.py` — layouts §3, geodesic §4, reward §5, `ram_penalty`, default 160, `info["layout_kind"]`
- `tonypi_rl/train.py` — `NAV_ENV_KEYS += ram_penalty`; resume overrides §6.1; banner mix/bc anneal
- `tonypi_rl/guided_ppo.py` — comment on `_lin` only (no floor)
- `tonypi_rl/__init__.py` — `TonyPiNav-v0` max 160
- `tonypi_rl/visualize_nav.py` — §8
- `tonypi_rl/implementation/README.md` — index this spec as **not yet applied** until the log says so
- `tonypi_rl/NAV.md` — one-line v1.2 pointer at the top (do not rewrite §§1–13)
- After coding: `tonypi_rl/implementation/NAV_V1_2_IMPLEMENTATION_LOG.md`

Do not touch: `nav_features.py`, `perception.py`, `tonypi_env.py`, `tonypi_mujoco/` meshes, playback.

---

## 10. Verification (coding agent must run)

From repo root:

```bash
export PYTHONPATH=$PWD
export MUJOCO_GL=egl

# A. mixed yields all four kinds; info has layout_kind
python -c "
from collections import Counter
from tonypi_rl.tonypi_nav_env import TonyPiNavEnv
e = TonyPiNavEnv(layout='mixed', goal_mode='random', use_geodesic_reward=True, random_robot_yaw=True)
c = Counter()
for s in range(40):
    _, info = e.reset(seed=s)
    c[info.get('layout_kind', '?')] += 1
print(dict(c))
assert set(c) <= {'scatter', 'wall', 'u_trap', 'gate'}
assert len(c) >= 3
e.close()
"

# B. scatter has no center gap in (0.16, 0.55)
python -c "
import numpy as np
from tonypi_rl.tonypi_nav_env import TonyPiNavEnv
e = TonyPiNavEnv(layout='scatter', goal_mode='random', n_obstacles=4, use_geodesic_reward=True)
bad = 0
for s in range(30):
    e.reset(seed=s)
    xy = np.array([(p[0], p[1]) for p in e._active_obs])
    for i in range(len(xy)):
        for j in range(i+1, len(xy)):
            d = float(np.hypot(*(xy[i]-xy[j])))
            if 0.16 < d < 0.55:
                bad += 1
print('fake_gaps', bad)  # expect 0
e.close()
"

# C. expert solves walls
python -c "
from tonypi_rl.tonypi_nav_env import TonyPiNavEnv
e = TonyPiNavEnv(layout='wall', goal_mode='random', max_episode_steps=160, use_geodesic_reward=True)
ok = 0
for s in range(20):
    e.reset(seed=s)
    for _ in range(160):
        _, _, term, trunc, info = e.step(e.expert_action())
        if term:
            ok += 1
            break
        if trunc:
            break
print('wall expert', ok, '/20')  # expect >= 16
e.close()
"

# D. inf geodesic does not increase Euclidean-style progress
python -c "
import numpy as np
from tonypi_rl.tonypi_nav_env import TonyPiNavEnv
e = TonyPiNavEnv(layout='wall', use_geodesic_reward=True, ram_penalty=1.0)
o, info = e.reset(seed=0)
e._prev_progress = 2.0
# simulate inf geodesic path: monkeypatch
e.geodesic_distance = lambda: float('inf')
m = e._progress_metric()
assert m == 2.0 + e.geodesic_progress_clip, m
e.close()
print('geodesic fallback ok')
"

# E. teacher math hits 0 (no floor)
python -c "
from tonypi_rl.guided_ppo import GuidedPPO
# instantiate is heavy; check _lin on the class if possible, else skip
print('yaml must have mix_anneal_steps=8000000 bc_anneal_steps=10000000 and no mix_prob_floor')
"

# F. train smoke (may need EGL)
python -m tonypi_rl.train --config tonypi_rl/configs/v1_2_nav_expert.yaml --timesteps 2048 --n_envs 1
```

Expert success must not drop because of the new layouts (resample unreachable). If wall expert < 16/20, loosen `expert_robot_radius` toward 0.16 or shorten the wall — do not touch walking.

---

## 11. Human training note

Resume, do not start from zero:

```bash
cd /mnt/d/robotics_cccc/tonypi
export PYTHONPATH=$PWD
export MUJOCO_GL=egl
python -m tonypi_rl.train \
  --config tonypi_rl/configs/v1_2_nav_expert.yaml \
  --resume /path/to/tonypi_checkpoint_4300000_steps.zip
```

Watch:

- `rollout/mix_prob` → 0 at 8M, stays 0
- `train/bc_coef` → 0 at 10M, stays 0
- `eval/mean_reward` on mixed / wall
- `train/approx_kl` (must stay ≪ 1; clamps already in)

After 10M, policy must still detour **without** the teacher. Eval: `visualize_nav.py --model <zip> --layout wall --episodes 20` (no `--expert` for the actor). Open scatter must stay fast (`4m3steps.mp4` style).
