# TonyPiNav v1.2.1 — post-review patch spec

**This file is the source of truth for the coding agent that fixes v1.2.** v1.2 is already in the tree. Do **not** re-implement layouts, polar obs, ram penalty, or the anneal yaml. Do **not** grow the PPO net. Do **not** change playback fps/sleep. Do **not** add `mix_prob_floor`.

Read first: [`NAV_V1_2.md`](NAV_V1_2.md) (original spec) and [`NAV_V1_2_IMPLEMENTATION_LOG.md`](NAV_V1_2_IMPLEMENTATION_LOG.md) (what was coded). The log oversells “Implemented & Verified”; byte-compile is not §10. Review ran A/B/C/D: mixed kinds OK, scatter fake-gaps 0, wall expert 20/20, inf geodesic uses `prev+clip`. Two real bugs still break the **teacher calendar** and can leave **ghost boxes**. `u_trap` is a front fence, not a U.

| | |
|---|---|
| Repo root | `/mnt/d/robotics_cccc/tonypi` |
| Run | `cd` repo root, `PYTHONPATH=$PWD`, `MUJOCO_GL=egl`, `python -m tonypi_rl.train` — never `python tonypi_rl/train.py` |
| Resume | `tonypi_checkpoint_4300000_steps.zip` (v1.1 64×64 / 18-D). Not a 32×32 zip |
| Teacher | Must still hit **exactly 0** at 8M mix / 10M BC on the **checkpoint clock** (`num_timesteps`) |

---

## 0. Locked diagnosis (do not reopen)

| # | Issue | Why it matters |
|---|---|---|
| 1 | `model.learn()` uses SB3 default `reset_num_timesteps=True` | Banner prints mix≈0.139 at 4.3M, then SB3 **zeros** `num_timesteps`. Mix jumps to 0.3. Yaml `timesteps: 15e6` becomes **15M extra** steps, not continue-until-15M. Mix/BC “0 at 8M/10M” on TensorBoard will not match the spec. |
| 2 | Unreachable fallback does not hide leftover mocaps | `_sample_layout` `else:` calls `_place_scatter` without `_hide_all_obstacles()`. Wall/u_trap leftovers stay in `mocap_pos`. `_obstacle_rects()` / A* / geodesic see them; `_active_obs` does not. Reproduced: 4 wall rects → scatter-without-hide still 4 rects, `active_obs` 3. |
| 3 | `_place_u_trap` does not enclose the robot | All 7 boxes sit in front of origin (`along ∈ [+0.30, +0.65]`), none behind. Robot stands at the mouth. Spec: robot **inside**, opening **away** from the goal, must back/strafe out then around. Log text is wrong. Expert still 15/15 because it is a short fence. |
| 4 | v1.2 yaml dropped `log_dir` / `model_dir` | `train.py` defaults to `tonypi_rl/logs` and `tonypi_rl/models` so local runs work; restore the keys so cluster copies match v1.1. `save_freq: 50000` / `eval_freq: 20000` were not in the spec (v1.1 was 100k/100k). |
| 5 | `n_obstacles <= 0` empties **all** kinds | `_sample_obstacles`: `if kind == "empty" or self.n_obstacles <= 0`. Spec: `n_obstacles` caps **scatter only**. Yaml has 4, so train is fine; still a footgun. |

Do not expand `NavFeaturesExtractor`. Do not touch `perception.py`, `tonypi_env.py`, playback.

---

## 1. Resume clock (required)

File: `tonypi_rl/train.py`.

SB3 `OnPolicyAlgorithm._setup_learn`:

- `reset_num_timesteps=True` (default): sets `self.num_timesteps = 0`. Loop until `total_timesteps`. Mix uses t=0.
- `reset_num_timesteps=False`: **adds** `total_timesteps += self.num_timesteps`, then loops until that sum.

So to stop at yaml 15M **on the checkpoint clock** from a 4.3M zip:

```python
learn_kwargs = dict(
    callback=[checkpoint_callback, eval_callback],
    progress_bar=True,
)
resuming = bool(resume_path and os.path.exists(resume_path))
if resuming:
    already = int(model.num_timesteps)
    remaining = int(total_timesteps) - already
    if remaining <= 0:
        print(f"[!] checkpoint already at {already:,} >= target {total_timesteps:,}; nothing to train")
        remaining = 0
    learn_kwargs["total_timesteps"] = remaining
    learn_kwargs["reset_num_timesteps"] = False
    # SB3 will do remaining + already → loop until yaml timesteps
else:
    learn_kwargs["total_timesteps"] = total_timesteps
    learn_kwargs["reset_num_timesteps"] = True

model.learn(**learn_kwargs)
```

Keep the existing post-load overrides (`mix_prob0`, `mix_anneal_steps`, `bc_coef0`, `bc_anneal_steps`, `target_kl`, `n_epochs`, `ent_coef`, `learning_rate`).

**Banner after those overrides, before `learn()`**, must print the **clock**, not only yaml starts:

```
GuidedPPO clock:     num_timesteps=4,300,000  target=15,000,000  remaining=10,700,000
GuidedPPO schedule:  mix_prob=0.1388 (0.3 → 0 at mix_anneal=8,000,000)
                     bc_coef=0.0855 (0.15 → 0 at bc_anneal=10,000,000)
                     reset_num_timesteps=False
```

Numbers: mix `0.3 * (1 - 4.3e6/8e6) = 0.13875`, bc `0.15 * (1 - 4.3e6/10e6) = 0.0855`.

After `learn()` starts, TensorBoard `rollout/mix_prob` must **not** jump to 0.3. If it does, the clock was reset — fail the patch.

If `remaining == 0`, skip `learn()` and still save/close cleanly.

Do **not** pass `total_timesteps=15_000_000` with only `reset_num_timesteps=False` and no remaining subtraction: SB3 would train until **19.3M**.

---

## 2. Hide leftover boxes (required)

File: `tonypi_rl/tonypi_nav_env.py`.

`_sample_obstacles` already starts with `_hide_all_obstacles()`. The bug is only the 20-try fallback:

```python
        else:
            self._layout_kind = "scatter"
            self._place_scatter(self.goal_xy)
```

Change to:

```python
        else:
            self._hide_all_obstacles()
            self._layout_kind = "scatter"
            self._place_scatter(self.goal_xy)
            if self.use_geodesic_reward or self._expert is not None:
                self._rebuild_expert()
```

Also belt-and-suspenders: at the **start** of `_place_scatter`, `_place_wall`, `_place_u_trap`, `_place_gate`, call `_hide_all_obstacles()` (or hide slots `len(placed):N_OBS_SLOTS` at the end of each). `_sample_obstacles` may then hide twice; that is fine.

`_obstacle_rects()` must equal `_active_obs` XY after every reset (parked slots `x>8` skipped).

---

## 3. Real U-trap (required)

File: `tonypi_rl/tonypi_nav_env.py` `_place_u_trap`.

Current geometry (do not keep): 3-box back wall at `d_back∈[0.58,0.65]` toward the goal, 4 “legs” that only come **0.28 m back toward the origin**, still all `along > 0`. Robot is at the opening.

Required: robot **inside** a three-sided U. Opening faces **away from the goal** (`-u`). Must back or strafe out, then go around.

World: robot will sit at `(0,0)` after `_set_root`. `u = goal / ||goal||`, `t = (-u_y, u_x)`.

Place (≤ `N_OBS_SLOTS=8`):

1. **Back wall** (blocks LOS): 3 merged boxes, center spacing `≤ GAP_MERGED` (use `0.15`), at `d_back * u + k * 0.15 * t` for `k ∈ {-1,0,1}`. `d_back ~ U(0.55, 0.70)`, then `min` with `goal_dist - GOAL_CLEAR - 0.05`. `d_back` must stay `≥ 0.50`.
2. **Legs** (two sides, parallel to `u`): box centers at `± w_leg * t` with `w_leg = 0.28` (inside width between **faces** ≈ `0.56 − 2*hx ≈ 0.36–0.40 m` with current geoms; robot must fit without spawn collision). Along `u`, at least:
   - one pair near the back wall (`d_back - 0.15`)
   - one pair **behind the origin** (`along ≤ -0.20`, e.g. `-0.25 * u ± w_leg * t`) so the opening is on the far side, away from the goal.
3. **Spawn clearance:** every box `hypot(x,y) ≥ 0.35` (spec). Prefer `≥ START_CLEAR` if it still encloses; 0.35 is the hard floor. If a candidate is closer, push it along the leg (do not drop the behind-origin pair).
4. **Goal clearance:** every box `≥ GOAL_CLEAR` from the goal. Goal stays **outside** the U.
5. Do not put a box on `(0,0)`. After `_set_root` the env must not be in contact (`_colliding()` is False at reset).

If the U would hit the goal (short `goal_dist`), shorten `d_back` first; if still impossible, the 20-try resample in `_sample_layout` will fall back to scatter — that is OK.

After a successful place, this must hold:

```python
along = boxes @ u   # (n,2) @ (2,)
assert (along > 0.20).any()    # back wall in front
assert (along < -0.05).any()   # legs behind origin  → opening away from goal
assert origin_d.min() >= 0.35
```

Log line “U around the robot … must reverse or strafe out” becomes true only after this change. If wall-style expert success on `layout='u_trap'` drops below **12/20**, widen `w_leg` toward `0.32` or shorten the behind-origin pair to `-0.18` — do not go back to a front fence.

---

## 4. Yaml logging (required)

`tonypi_rl/configs/v1_2_nav_expert.yaml` and `v1_2_nav.yaml` — restore v1.1 logging keys. Keep the v1.2 env/expert diffs.

```yaml
logging:
  save_freq: 100000
  eval_freq: 100000
  log_dir: "tonypi_rl/logs"
  model_dir: "tonypi_rl/models"
```

Drop `tensorboard: true` (unused by `train.py`; SB3 writes TB when `tensorboard_log` is set). Do not change expert anneal numbers.

---

## 5. `n_obstacles` only caps scatter (required, one line)

In `_sample_obstacles`:

```python
if kind == "empty":
    self._active_obs = []
```

Do **not** treat `n_obstacles <= 0` as empty for wall / u_trap / gate. Scatter still uses `n = clip(n_obstacles, 1, N_OBS_SLOTS)` inside `_place_scatter`.

---

## 6. Files to touch

Edit only:

- `tonypi_rl/train.py` — §1
- `tonypi_rl/tonypi_nav_env.py` — §2, §3, §5
- `tonypi_rl/configs/v1_2_nav_expert.yaml` — §4
- `tonypi_rl/configs/v1_2_nav.yaml` — §4
- `tonypi_rl/implementation/NAV_V1_2_IMPLEMENTATION_LOG.md` — add **§6 Post-review patch (v1.2.1)**; change the top status from “Verified (byte-compile)” to **patched; re-run §7 below**. Do not rewrite §§1–5.
- `tonypi_rl/implementation/README.md` — mark this spec implemented only after §7 passes

Do not touch: `nav_features.py`, `perception.py`, `guided_ppo.py` math (`_lin` already has no floor), `tonypi_env.py`, playback, occupancy size.

---

## 7. Verification (coding agent must run and paste output in the log)

From repo root:

```bash
export PYTHONPATH=$PWD
export MUJOCO_GL=egl
```

**A. Resume clock (no MuJoCo).** Assert the learn kwargs, not a full train:

```python
# After your train.py change, this must be true in source:
# resuming → reset_num_timesteps is False
# remaining = yaml_timesteps - model.num_timesteps
# grep train.py for reset_num_timesteps
```

```bash
grep -n "reset_num_timesteps" tonypi_rl/train.py
# expect False on the resume branch, and remaining = total - already
```

**B. Ghost boxes**

```python
from tonypi_rl.tonypi_nav_env import TonyPiNavEnv
e = TonyPiNavEnv(layout="wall", goal_mode="random", n_obstacles=3, use_geodesic_reward=True)
e.reset(seed=1)
e._hide_all_obstacles()  # simulate prior wall already placed; then:
# force the fallback path by calling the else-body equivalent:
e._sample_layout.__wrapped__ if False else None
# Direct:
from tonypi_rl.tonypi_nav_env import N_OBS_SLOTS
e.reset(seed=1)
n_wall = len(e._obstacle_rects())
e._hide_all_obstacles()
e._layout_kind = "scatter"
e._place_scatter(e.goal_xy)
# After the patch, _place_scatter hides first, so:
assert len(e._obstacle_rects()) == len(e._active_obs)
print("rects", len(e._obstacle_rects()), "active", len(e._active_obs))
e.close()
```

Expect `rects == active`. Before the patch this was `4 vs 3`.

**C. U-trap encloses**

```python
import numpy as np
from tonypi_rl.tonypi_nav_env import TonyPiNavEnv
e = TonyPiNavEnv(layout="u_trap", goal_mode="random", max_episode_steps=160, use_geodesic_reward=True)
bad_behind = 0
spawn_hit = 0
expert_ok = 0
for s in range(20):
    _, info = e.reset(seed=s)
    goal = e.goal_xy
    u = goal / max(1e-9, float(np.hypot(*goal)))
    boxes = np.array([(p[0], p[1]) for p in e._active_obs], dtype=np.float64)
    along = boxes @ u
    if not (along < -0.05).any():
        bad_behind += 1
    if e._colliding():
        spawn_hit += 1
    term = False
    for _ in range(160):
        _, _, term, trunc, _ = e.step(e.expert_action())
        if term or trunc:
            break
    if term:
        expert_ok += 1
print("behind_fail", bad_behind, "spawn_hit", spawn_hit, "expert", expert_ok, "/20")
e.close()
# expect behind_fail==0, spawn_hit==0, expert>=12
```

**D. Regression (must still pass)**

Reuse `NAV_V1_2.md` §10 A (mixed kinds, ≥3 of 4 in 40 seeds), B (scatter fake_gaps==0), C (wall expert ≥16/20), D (inf geodesic `== prev + clip`).

**E. Yaml**

`v1_2_nav_expert.yaml` has `log_dir`, `model_dir`, `save_freq: 100000`, `mix_anneal_steps: 8000000`, `bc_anneal_steps: 10000000`, no `mix_prob_floor`.

**F. Optional smoke** (EGL):

```bash
python -m tonypi_rl.train --config tonypi_rl/configs/v1_2_nav_expert.yaml --timesteps 2048 --n_envs 1
```

---

## 8. Human training (after this patch only)

```bash
cd /mnt/d/robotics_cccc/tonypi
export PYTHONPATH=$PWD
export MUJOCO_GL=egl
python -m tonypi_rl.train \
  --config tonypi_rl/configs/v1_2_nav_expert.yaml \
  --resume /path/to/tonypi_checkpoint_4300000_steps.zip
```

Launch banner must show `num_timesteps=4300000`, `reset_num_timesteps=False`, mix≈0.14, remaining≈10.7M.

TensorBoard:

- `rollout/mix_prob` starts ~0.14, hits **0 at 8M**, stays 0 (no jump to 0.3 at step 0 of the new run)
- `train/bc_coef` hits **0 at 10M**, stays 0
- do not train if mix jumps to 0.3 after `learn()` starts — the clock patch failed

---

## 9. What not to do

- Do not set a mix/BC floor.
- Do not `learn(..., total_timesteps=15_000_000, reset_num_timesteps=False)` without subtracting `already` (that trains to 19.3M).
- Do not “fix” u_trap by adding more boxes only in front of the robot.
- Do not rewrite walking, occupancy size, or the PPO CNN.
- Do not claim verified from `python -m compileall` alone. Paste §7 numbers in the log.
