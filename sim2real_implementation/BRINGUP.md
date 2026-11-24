# Live bring-up runbook

Run **in this order**. A later phase that is green while an earlier one is red is not a pass. Log each test in `IMPLEMENTATION_LOG.md` with command + one-line result.

Laptop (WSL) Python: `/home/bimcaucau/miniconda3/bin/python`. Robot: `python3` on the Pi.

Always from **repo root** on the laptop:

```bash
cd /mnt/d/robotics_cccc/tonypi
export PYTHONPATH=$PWD
```

On the Pi, copy `sim2real_implementation/`, `tonypi_rl/` (needed: `nav_features.py`, `perception.py`, `guided_ppo.py`, `__init__.py` — not the mujoco envs at runtime), and the zip. Set:

```bash
export PYTHONPATH=/home/pi/nav_deploy:$PYTHONPATH
export TONYPI_HOME=/home/pi/TonyPi
cd /home/pi/nav_deploy
sudo systemctl stop tonypi
```

Put `Functions/` on `sys.path` so `import VL53L0X` works, or import it by file path as the spec says.

---

## Phase 0 — laptop, no robot

| # | Test | Pass |
|---|---|---|
| 0.1 | `python sim2real_implementation/replay_policy.py --model models_ckpt/new_expert_implementation/tonypi_final_model.zip --print-space` | prints `vec (18,) occupancy (1, 64, 64) Discrete(7)` and loads without mujoco |
| 0.2 | Same, `--synthetic-clear` | action is usually `go_forward` or `stand` with goal 2 m ahead, empty mask, tof=1.2 |
| 0.3 | `--synthetic-wall` (center column of occupancy = 1, tof=0.25, goal still +x) | action is **not** always `go_forward` (turn or side-step or back). If it always rams, stop and report — that is a policy issue, not deploy. |
| 0.4 | `python -c "from tonypi_rl.perception import YoloSegOccupancy"` on laptop | imports; downloading `yolo11n-seg.pt` is OK |

Do not start Phase 1 until 0.1–0.3 pass.

---

## Phase 1 — robot sensors, robot **does not walk**

Hold the robot or leave it in `stand`. `--dry-run` never calls `runActionGroup` except one initial `stand`.

| # | Test | Pass |
|---|---|---|
| 1.1 | `python3 -m sim2real_implementation.sensor_probe` | prints ToF mm, sonar mm or `MISSING`, IMU 6-tuple or `NONE`, camera `HxWx3` |
| 1.2 | Cover ToF with a hand at ~20 cm | ToF drops to ~0.20 m, scaled ~0.17 |
| 1.3 | Uncover | ToF returns to miss (~1.2 m) |
| 1.4 | IMU: stand, then tilt forward ~30°, then left | gravity unit vector **changes**; dump three means into the log. Fill `configs/imu.yaml` with `R` that maps stand → `[0,0,-1]` ±0.15 |
| 1.5 | Camera preview `--save-frame /tmp/cam.jpg` | 640×480, head centered, not upside down |
| 1.6 | Occupancy `--occupancy color --save-occ /tmp/occ.png` with a red/blue box in view | 64×64 has a blob, not all zeros |
| 1.7 | If YOLO installed: `--occupancy yolo --save-occ` | blob on a person/chair/box; record **latency ms** |

**Gate:** if YOLO `> 300 ms` p95, lock `--occupancy color` for the first walking tests. Do not block the project on NCNN.

---

## Phase 2 — clips only (no policy)

| # | Test | Pass |
|---|---|---|
| 2.1 | `python3 -m sim2real_implementation.deploy_nav --script stand,go_forward,stand` | plays those three clips, returns to stand |
| 2.2 | Ctrl+C or onboard key mid-`go_forward` | `stopActionGroup` + `stand`, process exits |
| 2.3 | Tip the robot over in `--dry-run` with safety on | fall detector fires, **no** get-up yet (get-up is opt-in `--recover-fall`) |

---

## Phase 3 — open-loop goal, color occupancy, short leash

Clear floor, one obstacle the color mask can see, goal `+1.2 0` (body, metres) at t=0. Max 12 steps.

```bash
python3 -m sim2real_implementation.deploy_nav \
  --model /home/pi/nav_deploy/models/tonypi_final_model.zip \
  --goal 1.2 0.0 \
  --goal-mode deadreckon \
  --occupancy color \
  --max-steps 12 \
  --arrive 0.25
```

| # | Pass |
|---|---|
| 3.1 | Robot takes `go_forward` / small turns, does not freeze in `stand` for 12 steps unless arrived |
| 3.2 | HUD / log prints `vec[:5]` each step (dist/3, sin, cos, tof, sonar) in range |
| 3.3 | E-stop still works |
| 3.4 | After 12 steps, `stand` |

If it spins in place: heading sign is flipped — invert yaw in dead-reckon (`turn_left` should increase heading toward a goal on the **left**).

If it always walks forward into the box: occupancy is empty (check 1.6) or ToF is stuck at miss.

---

## Phase 4 — AprilTag goal

Tape a tag on the goal box. `--goal-mode tag --tag-id 0 --tag-size 0.08`.

| # | Pass |
|---|---|
| 4.1 | When tag is visible 1.5 m ahead, `dist` ≈ 1.5 ± 0.4, `|heading| < 25°` |
| 4.2 | Walk until `dist < 0.25` or 20 steps, then `stand` |
| 4.3 | Hide tag → hold last pose **or** `stand` (flag `--tag-lost stand`). Do not sprint. |

---

## Phase 5 — detour (the actual policy test)

Same as sim demo intent: goal ~2 m ahead, a box on the ToF centerline at ~0.9 m.

| # | Pass |
|---|---|
| 5.1 | ToF reads ~0.8–1.0 m at start |
| 5.2 | First actions include turn or side-step, not 8× `go_forward` |
| 5.3 | Does not fall. If it falls, enable `--recover-fall` and rerun; if it still rams, stop and capture `--log-dir` |

---

## Phase 6 — optional speed

Only if Phase 5 works and YOLO was skipped:

- Export `yolo11n-seg` to NCNN / ONNX and add `--occupancy yolo-ncnn`
- Or `--remote tcp://<laptop>:5555` so the laptop runs YOLO+PPO and the Pi only senses + plays clips

Do not start Phase 6 to “make it nicer” before 5.2 is true.
