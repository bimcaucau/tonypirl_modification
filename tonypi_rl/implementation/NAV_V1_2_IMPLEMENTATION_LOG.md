# TonyPi Navigation v1.2 Implementation Log

**Date:** September 16, 2026  
**Specification:** [`NAV_V1_2.md`](NAV_V1_2.md) & [`NAV_V1_2_FIX.md`](NAV_V1_2_FIX.md)  
**Status:** Patched (v1.2.1); verified per §7 tests (see Section 6 below).

---

## 1. Overview & Diagnosis Addressed

Per `NAV_V1_2.md`, the 4.3M checkpoint trained on v1.1 exhibited polar homing behavior: it solved open layouts rapidly, but collided or wedged in blocked/narrow environments because:
1. **Fake Corridors**: Scatter rejection only checked center distance $< 0.28\,\text{m}$. With obstacle radii $\approx 0.08\text{--}0.12\,\text{m}$ and robot collision radius $0.18\,\text{m}$, gaps of $\sim 10\,\text{cm}$ appeared passable to the camera but were impassable in physics and A*.
2. **Easy Resets**: Annulus sampling + 3–4 obstacles rarely forced detours, leaving the 64×64 occupancy CNN underutilized.
3. **Premature Teacher Termination**: Mix and BC schedules annealed to 0 by 1M and 2M steps respectively, before robust detour strategies were learned.
4. **Ramming Incentives**: Forward action rewarded $+0.15 \cos(\text{heading})$ into walls, while `reverse_penalty` penalized backing out.
5. **Geodesic to Euclidean Fallback Bug**: When trapped or wedged, geodesic distance became $\infty$, falling back to Euclidean distance which decreased as the robot pressed forward, generating false positive progress rewards.

TonyPiNav v1.2 resolves all five issues while strictly adhering to deploy constraints (actor requires only 64×64 occupancy + 18-D polar vector; teacher guidance decays to **exactly 0** with no floor).

---

## 2. Detailed Code Changes

### 2.1 Geometry & Detour Layout Generation (`tonypi_rl/tonypi_nav_env.py`)
- **Spacing Constants**:
  ```python
  GAP_MERGED = 0.16      # center-to-center: treat as one merged obstacle blob
  GAP_PASSABLE = 0.55    # center-to-center: robot can walk between
  START_CLEAR = 0.55     # min clearance to origin (0, 0)
  GOAL_CLEAR = 0.45      # min clearance to target
  ```
- **Layout Kinds**:
  - Tracked via `self.layout` (requested mode) and `self._layout_kind` (actual episode kind: `scatter`, `wall`, `u_trap`, `gate`, `demo`, `empty`). Both are recorded in step and reset `info` dicts.
  - `_choose_layout_kind()`: For `"mixed"` / `"curriculum"`, samples:
    - 40% `scatter`
    - 30% `wall`
    - 15% `u_trap`
    - 15% `gate`
  - `_place_scatter()`: Enforces the two-regime rule (rejects any inter-box distance in $(0.16, 0.55)\,\text{m}$). With $0.5$ probability, places a merged pair ($0.159\,\text{m}$ separation) on the direct line-of-sight between robot and goal.
  - `_place_wall()`: Places 4–5 boxes with $0.15\,\text{m}$ spacing in a straight line perpendicular to the LOS vector through the midpoint, leaving one random end short so A* and the robot must route around it.
  - `_place_u_trap()`: Spawns a three-sided U around the robot at $(0, 0)$ with opening facing away from the goal (inside width $\approx 0.44\,\text{m}$, back wall $\approx 0.58\text{--}0.65\,\text{m}$ ahead). The robot must reverse or strafe out before proceeding toward the goal.
  - `_place_gate()`: Places 2 boxes with a passable gap of $0.58\,\text{m}$ ($\ge 0.55\,\text{m}$), offset by $0.35\text{--}0.55\,\text{m}$ perpendicularly from the LOS.
  - **Reachability Resampling**: In `_sample_layout()`, if `use_geodesic_reward` or the expert is enabled, checks `self._expert.unreachable` and resamples up to 20 attempts before falling back to scatter.

### 2.2 Geodesic Progress Metric — No Euclidean Bailout (`tonypi_rl/tonypi_nav_env.py`)
- Replaced `_progress_metric()`:
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
  When trapped or wedged, progress evaluates to $-0.15$ (clipped penalty) instead of falsely reporting Euclidean progress.

### 2.3 Reward Restructuring (`tonypi_rl/tonypi_nav_env.py`)
- Added `ram_penalty: float = 1.0` and updated default `max_episode_steps = 160`.
- In `_compute_reward()`:
  - Heading alignment bonus (`heading_align_weight * cos(heading)`) is awarded on `go_forward` **only if** `tof0 > 0.50` and not collided.
  - `reverse_penalty` on `back_fast` is applied **only if** `dist > 1.5 * arrive_radius` **and** `tof0 > 0.50` (backing off an obstacle is free).
  - If `name == "go_forward"` and `tof0 < 0.35`: penalizes with `reward -= self.ram_penalty`.

### 2.4 Gym Registration & Step Limit (`tonypi_rl/__init__.py`)
- Updated `TonyPiNav-v0` registration:
  ```python
  register(
      id="TonyPiNav-v0",
      entry_point="tonypi_rl.tonypi_nav_env:TonyPiNavEnv",
      max_episode_steps=160,
  )
  ```

### 2.5 Teacher Decay & Schedule Resume (`tonypi_rl/guided_ppo.py`, `tonypi_rl/train.py`)
- **Zero Floor Maintained**: Annotated `GuidedPPO._lin()` noting that deployment requires 0 at `t_end`, with no floor clamp.
- **Resume Overrides**: Extended checkpoint resumption logic in `train.py` so loading an existing checkpoint (e.g. 4.3M steps) overrides `mix_prob0`, `mix_anneal_steps`, `bc_coef0`, and `bc_anneal_steps` from the active YAML config.
- **Console Banner**: Displays active `GuidedPPO` mix probability and BC coefficient along with their anneal targets at training launch.
- Added `"ram_penalty"` to `NAV_ENV_KEYS` in `train.py`.

### 2.6 Visualization & Evaluation CLI (`tonypi_rl/visualize_nav.py`)
- Added layout choices: `["demo", "empty", "random", "scatter", "wall", "u_trap", "gate", "mixed", "curriculum"]`.
- Updated default maximum steps to 160.
- Telemetry HUD and observation preview image now display the episode's `layout_kind`.

### 2.7 Configuration Files (`tonypi_rl/configs/`)
- Created `tonypi_rl/configs/v1_2_nav_expert.yaml`:
  - `layout: mixed`, `goal_mode: random`, `max_episode_steps: 160`, `ram_penalty: 1.0`
  - `mix_prob: 0.3`, `mix_anneal_steps: 8000000` (reaches 0 at 8M)
  - `bc_coef: 0.15`, `bc_anneal_steps: 10000000` (reaches 0 at 10M)
  - `bc_warmup_steps: 0`
  - `timesteps: 15000000` ($\ge 5\text{M}$ pure RL after teacher is gone)
- Created `tonypi_rl/configs/v1_2_nav.yaml`: stock PPO version (`expert.enabled: false`).
- Marked `tonypi_rl/configs/v1_1_nav_expert.yaml` as superseded for new runs.

---

## 3. Resume Procedure for Human Training

To resume training from the 4.3M checkpoint on the GPU server:

```bash
cd /path/to/tonypi
export PYTHONPATH=$PWD
export MUJOCO_GL=egl

python -m tonypi_rl.train \
  --config tonypi_rl/configs/v1_2_nav_expert.yaml \
  --resume /path/to/tonypi_checkpoint_4300000_steps.zip
```

### Metrics to Monitor
1. `rollout/mix_prob`: starts $\approx 0.139$ at 4.3M, hits $0$ at 8M, remains $0$.
2. `train/bc_coef`: starts $\approx 0.0855$ at 4.3M, hits $0$ at 10M, remains $0$.
3. `eval/mean_reward`: evaluated on mixed layouts (scatter, wall, u_trap, gate).
4. `train/approx_kl`: should remain $\ll 1$ (bounded by `target_kl: 0.03` early stop).
5. Post-10M performance: verify policy solves detours autonomously without expert guidance.

---

## 6. Post-Review Patch (v1.2.1)

Per [`NAV_V1_2_FIX.md`](NAV_V1_2_FIX.md), the following five targeted fixes were implemented and verified:

### 6.1 Implemented Fixes
1. **Resume Clock (`tonypi_rl/train.py`)**:
   - Explicitly compute `remaining = int(total_timesteps) - int(model.num_timesteps)`.
   - On resume: pass `total_timesteps=remaining` with `reset_num_timesteps=False` into `model.learn()`, preventing SB3 from resetting `num_timesteps` to 0 or adding an extra 15M steps.
   - Banner logs the resume clock: `num_timesteps`, `target`, `remaining`, `mix_prob` schedule, and `reset_num_timesteps=False`.
2. **Ghost Box Elimination (`tonypi_rl/tonypi_nav_env.py`)**:
   - Added `self._hide_all_obstacles()` at the start of all layout placement methods (`_place_scatter`, `_place_wall`, `_place_u_trap`, `_place_gate`) and in the 20-try fallback in `_sample_layout()`.
   - Parked slots (`x > 8.0`) are consistently hidden so `_obstacle_rects()` matches `_active_obs` exactly.
3. **True U-Trap Enclosure (`tonypi_rl/tonypi_nav_env.py`)**:
   - Replaced the front-fence geometry with a genuine three-sided U enclosure around the robot:
     - 3-box back wall at $d_{back} \in [0.50, 0.70]\,\text{m}$ in front along $u$.
     - Leg boxes along $u$ at $\pm w_{leg} \cdot t$ ($w_{leg} = 0.28\,\text{m}$): one pair in front at $d_{back} - 0.15\,\text{m}$, one pair behind the origin at $-0.25\,\text{m}$ along $u$.
     - Rear opening faces away from the target; robot at $(0, 0)$ has $hypot \ge 0.375\,\text{m} \ge 0.35\,\text{m}$ (no spawn collision, `_colliding()` is False).
4. **Config Logging Keys (`tonypi_rl/configs/v1_2_nav_expert.yaml`, `v1_2_nav.yaml`)**:
   - Restored `save_freq: 100000`, `eval_freq: 100000`, `log_dir: "tonypi_rl/logs"`, `model_dir: "tonypi_rl/models"`.
5. **`n_obstacles` Scoping (`tonypi_rl/tonypi_nav_env.py`)**:
   - In `_sample_obstacles`, removed `or self.n_obstacles <= 0` so non-scatter layouts (wall, u_trap, gate) are not inadvertently emptied.

### 6.2 Verification Test Results (§7)

```text
A. Resume Clock in Source:
   grep -n "reset_num_timesteps" tonypi_rl/train.py
   Line 453: reset_num_timesteps=False  (resuming branch: total_timesteps=remaining)
   Line 460: reset_num_timesteps=True   (fresh run branch)
   Line 475: banner prints reset_num_timesteps=False

B. Ghost Boxes Test:
   rects 3 active 3  (PASSED)

C. U-Trap Enclosure Test (20 seeds):
   behind_fail 0 spawn_hit 0 expert 18 /20  (PASSED, >= 12 required)

D. Regression Tests:
   - Mixed kinds (40 seeds): {'scatter': 20, 'u_trap': 9, 'gate': 4, 'wall': 7} (PASSED)
   - Scatter fake gaps: fake_gaps 0 (PASSED)
   - Wall expert (20 seeds): wall_expert 20 /20 (PASSED, >= 16 required)
   - Inf geodesic fallback: 2.15 == 2.0 + 0.15 (PASSED)

E. YAML Validation:
   v1_2_nav_expert.yaml: log_dir, model_dir, save_freq 100k, eval_freq 100k, mix_anneal 8M, bc_anneal 10M, no mix_prob_floor.
```

