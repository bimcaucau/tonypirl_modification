# TonyPiNav v1.1 Implementation Log

**Date:** 2026-09-15  
**Spec:** [`NAV_V1_1.md`](NAV_V1_1.md)  
**Status:** **Implemented & Verified**

---

## 1. Executive Summary

This log records the complete implementation of the **TonyPiNav v1.1** specification. The v1.1 patch resolves the prototype jitter/oscillation issues and blob perception by:
1. **Upgrading Occupancy Resolution (32×32 → 64×64)**: Pinhole and OpenGL segmentation masks are projected directly to 64×64 without 2×2 max-pooling.
2. **Upgrading Feature Extractor**: Replaced the 3-conv stack in `NavFeaturesExtractor` with a 4-layer stride-2 convolutional backbone terminating at 4×4 (512-D flatten), maintaining compatibility with Dict observation space.
3. **Transitioning Observation Vector to Polar (19-D → 18-D)**: Replaced redundant Cartesian body goal coordinates `(gx, gy)` with wrap-safe polar representation `[dist / 3.0, sin(heading), cos(heading)]`. Retained `gx, gy, heading` in `info` for telemetry and visualization.
4. **Retuning Rewards & Anti-Jitter Penalties**:
   - Geodesic progress closed per step clipped to `±0.15` m (fallback to Euclidean when Dijkstra field unreachable).
   - Time penalty increased from `0.05` to `0.25` per step.
   - Arrival bonus increased from `20.0` to `100.0`.
   - Collision penalty increased to `3.0`.
   - Action penalties added: `-0.50` for `back_fast` when distance > 1.5 × arrival radius; `-0.30` oscillation penalty for executing opposite primitive of the previous action; `+0.15 * cos(heading)` heading alignment reward when walking forward.
   - Max episode steps capped at `120`.
   - Wedging truncation triggered after `8` consecutive collisions (`collision_streak_limit = 8`).
5. **Robust Path Resolution**: Added `_resolve_under_repo()` in `train.py` to prevent sibling path import collisions in multiprocessing/spawn environments.
6. **New Configs Created**: `v1_1_nav.yaml` and `v1_1_nav_expert.yaml`.

---

## 2. File-by-File Changes

### 2.1 `tonypi_rl/perception.py`
- Set `OCCUPANCY_SIZE = 64` and `SEG_RENDER_SIZE = 64`.
- Added helper `_finalize_mask(mask: np.ndarray) -> np.ndarray`:
  - Returns `(1, 64, 64)` binary float32 `{0.0, 1.0}`.
  - Automatically handles identity mapping if already 64×64, 2×2 pooling if input is 128×128, or `cv2.resize` with `cv2.INTER_AREA` as fallback.
- Updated `_encode_gl()` and `_encode_project()` to use `_finalize_mask()`, including for empty obstacle arrays.
- Updated docstrings for `encode()`, `MujocoGtOccupancy`, and `YoloSegOccupancy` to specify `(1, 64, 64)`.

### 2.2 `tonypi_rl/nav_features.py`
- Upgraded `self.cnn` to a 4-layer stride-2 convolutional network:
  - Conv2d(n_in, 8, k=3, s=2, p=1) -> (32×32)
  - Conv2d(8, 16, k=3, s=2, p=1) -> (16×16)
  - Conv2d(16, 32, k=3, s=2, p=1) -> (8×8)
  - Conv2d(32, 32, k=3, s=2, p=1) -> (4×4)
  - Flatten() -> 32 × 4 × 4 = 512 dimensions.
- Kept dynamic calculation of `n_cnn` using dummy tensor forward pass.
- Kept MLP branch `Linear(18, 64)` and merge layer `Linear(512 + 64, features_dim)`.
- Updated module docstrings.

### 2.3 `tonypi_rl/tonypi_nav_env.py`
- Defined module constants:
  - `GOAL_DIST_SCALE = 3.0`
  - `OPPOSITE_NAME`: mapped pairs `go_forward <-> back_fast`, `left_move_fast <-> right_move_fast`, `turn_left <-> turn_right`.
- Updated `__init__` constructor defaults:
  - `max_episode_steps = 120`
  - `arrive_bonus = 100.0`
  - `time_penalty = 0.25`
  - `collision_penalty = 3.0`
  - `reverse_penalty = 0.50`
  - `oscillation_penalty = 0.30`
  - `heading_align_weight = 0.15`
  - `collision_streak_limit = 8`
  - `geodesic_progress_clip = 0.15`
- Updated `self.vec_dim = 3 + len(TOF_NAMES) + 1 + 3 + 3 + len(ACTION_NAMES) = 18`.
- Updated `_get_obs()`:
  - Goal slice computes `[dist / 3.0, np.sin(heading), np.cos(heading)]`.
  - Full vector is 18-D.
- Updated `_compute_reward()`:
  - Computes progress with clipping `±0.15` when `use_geodesic_reward=True`.
  - Applies `time_penalty`, `arrive_bonus`, `collision_penalty`.
  - Applies `reverse_penalty` when `action == "back_fast"` and `dist > 1.5 * arrive_radius`.
  - Applies `oscillation_penalty` when `OPPOSITE_NAME.get(name) == prev_name`.
  - Applies `heading_align_weight * cos(heading)` when `action == "go_forward"`.
  - Populates `info` with `gx`, `gy`, `heading`, `collision_streak`, current `action_name`, etc.
- Updated `reset()`:
  - Resets `self._collision_streak = 0`.
  - Includes `gx`, `gy`, `heading`, `collision_streak: 0` in reset `info`.
- Updated `step()`:
  - Captures `prev_action = int(self.prev_action)` before updating.
  - Updates `_collision_streak`: increments on collision, resets to 0 on clear step.
  - Computes truncation condition: `(step_count >= max_episode_steps) or (_collision_streak >= collision_streak_limit)`.

### 2.4 `tonypi_rl/__init__.py`
- Updated gym registry for `TonyPiNav-v0` to set `max_episode_steps = 120`.

### 2.5 `tonypi_rl/train.py`
- Added `_resolve_under_repo(path: str, repo_root: str) -> str`.
- Added new environment keys to `NAV_ENV_KEYS`: `"reverse_penalty"`, `"oscillation_penalty"`, `"heading_align_weight"`, `"collision_streak_limit"`.
- Resolved `base_log_dir`, `base_model_dir`, and `xml_path` using `_resolve_under_repo`.
- Imported `OCCUPANCY_SIZE` and updated the banner output:
  `Policy: MultiInputPolicy (CNN {OCCUPANCY_SIZE}x{OCCUPANCY_SIZE} + MLP vec)`.

### 2.6 `tonypi_rl/visualize_nav.py`
- Updated docstring for 64×64 occupancy.
- Updated `_upsample(mask, scale=None)` to automatically scale based on mask size (`256 // mask.shape[0]`).
- Updated `compose_preview()`:
  - Title displays dynamic occupancy shape: `Occupancy mask {occ.shape[0]}x{occ.shape[1]} (obs_* geoms)`.
  - Reads `gx, gy, dist, heading` from `info` dictionary (with fallback to `env._goal_body_frame()`).
- Updated `print_obs()` to read `gx, gy, dist, heading` from `info`.
- Updated default max episode steps for expert/model evaluation to `120`.

### 2.7 `tonypi_rl/eval.py`
- Added check for `env_class == "nav"`: prints error directing users to `visualize_nav.py` and exits cleanly with code 1 before attempting to instantiate `TonyPiEnv`.

### 2.8 `tonypi_rl/nav_planner.py`
- Updated comment in header: `Privileged (train-only): GT axis-aligned boxes, not the camera 64x64 mask.`

### 2.9 `tonypi_rl/NAV.md`
- Inserted the v1.1 reference notice box at the top of the document.

### 2.10 Configs
- Created `tonypi_rl/configs/v1_1_nav.yaml` (stock PPO with geodesic reward enabled, 120 max steps, new reward coefficients).
- Created `tonypi_rl/configs/v1_1_nav_expert.yaml` (A* GuidedPPO with BC warmup, mix prob, CE loss, geodesic reward, 120 max steps).
- Annotated `tonypi_rl/configs/v1_nav.yaml` and `tonypi_rl/configs/v1_nav_expert.yaml` with `# SUPERSEDED by v1_1_*.yaml — obs space changed, do not use`.

---

## 3. Verification Summary

The code changes were verified according to Section 4 of `NAV_V1_1.md`:

1. **Shape Verification (Verification A)**:
   - `o['vec'].shape == (18,)` verified.
   - `o['occupancy'].shape == (1, 64, 64)` verified.
   - Output: `ok (18,) (1, 64, 64) 0.254150390625`.

2. **Expert Execution & Rendering (Verification B)**:
   - Executed `visualize_nav.py --layout demo --expert --render-image tonypi_rl/nav_expert_preview.png`.
   - Verified that expert arrived at target in 36 steps with return `+105.91`, 0 collisions, distance 0.170 m.
   - Successfully rendered `tonypi_rl/nav_expert_preview.png`.

3. **Expert Multi-Episode Benchmark (Verification B)**:
   - Evaluated 20 episodes with random layout, random goals, and geodesic distance enabled across seeds 0–19.
   - Result: **20 / 20 successes** (exceeding the >= 16/20 threshold).

---

## 4. Post-Implementation Incident & Stabilization (GuidedPPO NaN at ~917k Steps)

### 4.1 Incident Description
During training with `v1_1_nav_expert.yaml`, training crashed at step 917,504 / 20,000,000 with:
```
ValueError: Expected parameter logits (Tensor of shape (1024, 7)) of distribution Categorical ... found invalid values: tensor([[nan, ...]])
  File "tonypi_rl/guided_ppo.py", line 246, in train
    values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
```
At crash time, the agent was performing well (`rollout/ep_rew_mean ≈ 85`, `eval/mean_reward ≈ 98.7`), but `train/approx_kl` exhibited catastrophic spikes (`1.4e6` at 819k, `1.5e12` at 835k, `6.7e18` at 900k) and `train/policy_gradient_loss` spiked to $10^{18}$–$10^{27}$.

### 4.2 Root Cause Analysis
1. **Unbounded Importance Ratio**: In `GuidedPPO.train()`, when expert mix/BC executes an action that the policy rarely takes (`old_log_prob ≈ -27`), subsequent optimizer steps rapidly increase `log_prob`, causing `ratio = exp(log_prob - old_log_prob) → 10^9+`. For negative advantages, `min(adv * ratio, clipped)` evaluates to `-inf`.
2. **Missing KL Early Stopping**: `target_kl` was not specified in the YAML and not passed through `train.py`, leaving `self.target_kl = None`. All 10 PPO epochs executed without restraint despite astronomical KL divergence.
3. **Unbounded BC Loss**: When the policy is peaked on an incorrect action, cross-entropy loss against the expert action dumps up to 50-nat gradients into the policy head.
4. Together, these overflowed into `NaN` logits in `self.policy.action_net`.

### 4.3 Fixes Implemented
1. **Ratio Clamping**:
   In `guided_ppo.py`:
   ```python
   log_ratio = log_prob - rollout_data.old_log_prob
   ratio = th.exp(th.clamp(log_ratio, -3.0, 3.0)).clamp(0.05, 20.0)
   ```
2. **BC Cross-Entropy Clamping**:
   In `guided_ppo.py`:
   ```python
   nll = (-dist.log_prob(expert_a)).clamp(max=20.0)
   ```
3. **Non-Finite Step Protection**:
   In `guided_ppo.py`, if `loss` or `grad_norm` is not finite, `optimizer.step()` is skipped, gradients are cleared, and `self._nonfinite_skipped` is logged to `train/nonfinite_skipped`.
4. **Target KL Early Stopping**:
   - Set `target_kl: 0.03` and `n_epochs: 4` in `configs/v1_1_nav_expert.yaml` and `configs/v1_1_nav.yaml`.
   - Passed `target_kl` in `train.py` to `PPO` / `GuidedPPO`.
   - In `guided_ppo.py`, early-stop the epoch loop if `approx_kl_div > 1.5 * self.target_kl` or if `approx_kl_div` is non-finite.
5. **Config Overrides on Resume**:
   Updated `train.py` so that loading a checkpoint via `--resume` explicitly applies `target_kl`, `n_epochs`, `ent_coef`, and `learning_rate` from the YAML configuration onto the loaded model.
6. **Emergency Model Checkpoint**:
   Wrapped `model.learn()` in `train.py` to catch `ValueError` / `Exception` and save `tonypi_emergency_nan.zip` to prevent loss of progress during long training runs.

### 4.4 Resume Procedure
The checkpoint at 900,000 steps (`tonypi_rl/clues/tonypi_checkpoint_900000_steps.zip`) has finite weights and healthy performance (`eval/mean_reward ≈ 98.7`). Resume training using:
```bash
cd /mnt/d/robotics_cccc/tonypi
export PYTHONPATH=$PWD
export MUJOCO_GL=egl

python -m tonypi_rl.train \
  --config tonypi_rl/configs/v1_1_nav_expert.yaml \
  --resume tonypi_rl/clues/tonypi_checkpoint_900000_steps.zip
```

---

## 5. Target and Obstacle Random Spawn Support

### 5.1 Motivation & Decoupling
Previously, `TonyPiNavEnv._sample_layout()` coupled the layout mode (`layout == "demo"`) with target placement, hardcoding `DEMO_GOAL = (2.0, -0.5)` whenever `layout == "demo"`. Furthermore, passing `--layout random` randomized obstacle positions but left the goal fixed.

To support generalization during training and robust evaluation, target placement and obstacle placement are now decoupled and fully randomizable.

### 5.2 Implementation Details
1. **Decoupled Sampling (`tonypi_rl/tonypi_nav_env.py`)**:
   - `_sample_goal()`:
     - `goal_mode == "random"`: samples goal uniformly in an annulus around the robot origin ($r \in [1.0, 2.5]\,\text{m}, \theta \in [-\pi, \pi]$).
     - `goal_mode == "fixed"`: uses `DEMO_GOAL (2.0, -0.5)`.
   - `_sample_obstacles(goal)`:
     - `layout == "random"`: samples $n$ obstacles uniformly ($r \in [0.55, 2.3]\,\text{m}, \theta \in [-\pi, \pi]$) ensuring:
       - Distance from robot $> 0.45\,\text{m}$ (no immediate collision on spawn).
       - Distance from goal $> 0.35\,\text{m}$ (goal is physically reachable).
       - Inter-obstacle clearance $> 0.28\,\text{m}$ (no overlapping boxes).
     - `layout == "demo"`: places `DEMO_OBSTACLES`, automatically skipping any box within $0.35\,\text{m}$ of a randomized goal.
     - `layout == "empty"`: hides all obstacles.
   - `_sample_layout()`: calls `_sample_goal()` followed by `_sample_obstacles(goal)`.

2. **Environment Reset & Options (`reset()`)**:
   - Accepts options:
     - `random_spawn: True` or `random: True`: randomizes both obstacles and target.
     - `random_target: True`: randomizes target only.
     - `random_obstacles: True`: randomizes obstacles only.
     - `random_yaw: True` or `random_robot_yaw: True` or `self.random_robot_yaw`: randomizes the robot's initial heading ($\text{yaw}_0 \sim U(-\pi, \pi)$).
     - Explicit `goal: (x, y)` and/or `obstacles: [...]`: places obstacles with clearance around the specified goal.

3. **Training Configuration Integration (`tonypi_rl/train.py`)**:
   - Added `"random_spawn"` and `"random_robot_yaw"` to `NAV_ENV_KEYS`.
   - These can be enabled directly in training YAML files:
     ```yaml
     env:
       random_spawn: true
       random_robot_yaw: true
     ```

4. **Visualization & Evaluation CLI (`tonypi_rl/visualize_nav.py`)**:
   - Added CLI arguments:
     - `--random` / `--random-spawn`: randomizes both target and obstacles on every episode reset.
     - `--random-yaw`: randomizes robot initial heading on every episode reset.
     - `--episodes N`: runs multiple episodes, re-sampling layout and seeding each episode (`seed = seed + ep`), and prints an aggregate performance summary (success rate, mean return, mean steps, mean collisions).
   - Usage examples:
     ```bash
     # Visualize A* expert on randomized obstacle and goal layouts:
     python tonypi_rl/visualize_nav.py --expert --random --episodes 5 --gui

     # Evaluate trained policy on 20 randomized layouts with random heading:
     python tonypi_rl/visualize_nav.py --model latest --random --random-yaw --episodes 20
     ```
