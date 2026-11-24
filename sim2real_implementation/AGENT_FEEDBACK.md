# Agent feedback — TonyPi nav sim2real

Read this **before** changing code. The stack in this folder is already implemented and **Phase 0 laptop tests PASS**. Do not rewrite walking, do not call `expert_action()` on the Pi, do not load onboard `Functions/model/best.pt` (YOLOv11 **detect** age/gender, not seg).

**Occupancy default:** `yolo26n-sem-ade20k.pt` (semantic, table legs as `table` stuff). Instance fallback: `--yolo-weights yolo26n-seg.pt`. Do **not** load `yolov5n.pt` or `Functions/model/best.pt`.

**Ground look (2026-09-24):** before each translation the head pitches down, then looks ahead and to each side at about 36° down (not at the feet — a steeper look saw the shins and backed up on every step), vetoes a step whose footprint contains a wire or small block, and uses the center view for a metric range (`min` with ToF) plus a floor-shift correction of the dead-reckon step. The policy still sees the **level** camera. Handoff, robot checks, and the yaml knobs are `GROUND_LOOK.md`. Do not fold the look-down mask into `obs["occupancy"]`.

---

## Live Pi layout (this robot)

`sim2real_implementation` and `TonyPi` are **siblings in the home directory**, not under `/home/pi/nav_deploy/`:

```
~/TonyPi/                  stock Hiwonder tree (Functions/, ActionGroups/, HiwonderSDK/)
~/sim2real_implementation/ this package
~/tonypi_rl/               if rsynced; required for policy load
```

```bash
export PYTHONPATH="$HOME:$PYTHONPATH"
export TONYPI_HOME="$HOME/TonyPi"
sudo systemctl stop tonypi
cd "$HOME"
python3 -m sim2real_implementation.sensor_probe
python3 -m sim2real_implementation.view_segmentation --backend tonypi --mode yolo --http-port 8080
```

ToF: stock `Functions/VL53L0X.py` does `CDLL("./vl53l0x_python.so")` (CWD-relative). `hardware.import_vl53l0x()` now chdirs to `TonyPi/Functions` and remaps that CDLL to the absolute `.so` path. Do **not** "fix" it with `--allow-missing-tof` when the sensor is present — that reports 1.2 m clear and will ram.

---

Locked deploy path: `TonyPiNavEnv` GuidedPPO zip → `Dict{occupancy:(1,64,64), vec:(18,)}` → `Discrete(7)` named `.d6a` clips via `AGC.runActionGroup(name)`.

Checkpoint: `models_ckpt/new_expert_implementation/tonypi_final_model.zip` (10,010,624 steps, teacher already 0).

---

## Status (do not regress)

| Phase | Status | Notes |
|---|---|---|
| 0 laptop contract / synthetic / 8 tests | **PASS** | `replay_policy.py --print-space`, `--synthetic-clear` → `go_forward`, `--synthetic-wall` → `left_move_fast` |
| 1 sensor probe / IMU / primitives tools | coded, **not run on robot** | |
| 2 scripted clips + safety on FakeHardware | PASS mock | live clips not run |
| 3 closed-loop deadreckon | coded | live not run |
| 4 AprilTag | coded, **tag-lost stand is a no-op** | see P0 |
| Live 3-pane segmentation | **done this pass** | `python -m sim2real_implementation.view_segmentation` |

`README.md` previously said the spec was “to implement”. Code exists. Spec files (`SIM2REAL_NAV.md`, `HARDWARE.md`, `BRINGUP.md`) still win on contract; this file wins on **bugs in the current code**.

---

## Do not touch

- `tonypi_rl/tonypi_env.py` 18-DoF walk, hopping zip `139/best_model.zip`, `149expert` (32×32 / 19-D).
- On-robot `Functions/model/best.pt`.
- Action name order (frozen): `stand, go_forward, back_fast, left_move_fast, right_move_fast, turn_left, turn_right`. Map by **name**, never `ActionGroupDict` numeric ids (policy 5 = `turn_left`, app `'5'` = `push_ups`).
- Do not `import tonypi_rl.tonypi_nav_env` on the Pi (pulls mujoco). Dummy env in `policy.py` is the loader.
- Do not `time.sleep(0.4)` after clips. Durations are 380–770 ms and `runActionGroup` already blocks.
- Do not enable `balance_ankle` during gait.
- Gyro stays zeros at deploy (`vec[8:11]`).
- Occupancy policy tensor stays `(1, 64, 64)` float32 in `{0,1}`. Vis overlays are extra; do not change `encode()`.

---

## P0 — fix before any live walk

### 1. `--tag-lost stand` does nothing

`goal.py` `GoalSource.current()` hold and stand branches return the **same** `last_tag_goal`. `deploy_nav.py` never inspects `goal.tag_visible` / `tag_lost_policy`.

Required: when `mode=="tag"` and tag not visible and `--tag-lost stand`, force `stand` (action 0) **before** `safety.filter_action` plays a walk clip. Hold last goal only for `--tag-lost hold`. Add a unit test.

### 2. Camera / ToF fail-open

`TonyPiHardware.start()` logs a warning and continues if camera or VL53L0X fail. Then:

- `read_bgr()` → `None` → occupancy all zeros (policy thinks open floor).
- `read_tof_m()` → `1.2` (policy thinks ToF clear).

If `--occupancy` is `yolo` or `color`, **abort start** when camera open fails. If ToF init fails, refuse to run unless `--allow-missing-tof` (and log every step). Empty occupancy + missing ToF is a ram-into-wall recipe.

### 3. YAML model path vs rsync destination

`configs/deploy.yaml` has `model: models_ckpt/new_expert_implementation/tonypi_final_model.zip`.

`IMPLEMENTATION_LOG.md` rsyncs the zip as a **file** into `/home/pi/nav_deploy/`, so on the Pi the file is `/home/pi/nav_deploy/tonypi_final_model.zip`, not the yaml relative path.

Fix one of: rsync into `nav_deploy/models_ckpt/new_expert_implementation/`, or change yaml to `tonypi_final_model.zip`, or require `--model` and fail loudly if the zip is missing (`FileNotFoundError` already exists in `inspect_checkpoint` — call it before `hw.start()` so the robot is not standing with a bad path).

### 4. `tonypi.service` owns the camera

Bring-up already says `sudo systemctl stop tonypi`. `TonyPiHardware.start()` should detect “camera in use” and print that exact command. Do not start walking if `read()` never returns a frame after N tries.

---

## P1 — wrong or incomplete, fix next

### 5. Dual `Board()` serial clients

`TonyPiHardware` constructs `rrc.Board()` + `enable_reception()` for IMU/PWM. `AGC.runActionGroup` uses the **module-level** `Board()` created at `import hiwonder.ActionGroupControl`. Stock apps do this too; it still races the UART.

Do not “share one Board” as a drive-by refactor unless IMU `get_imu()` still streams while clips play. Probe both in Phase 1: IMU samples during `play("stand")` must not freeze.

### 6. HSV red wrap

`color_occupancy` is a single `inRange`. Default yaml `[0,80,80]–[12,255,255]` only catches hue 0–12. OpenCV red also lives at 170–179. Cardboard / orange cones often miss both.

`hsv_in_range` now wraps when `H_lo > H_hi` (e.g. `--hsv-lo 170 80 80 --hsv-hi 12 255 255`). Wire that into `deploy.yaml` after `sensor_probe --tune-hsv` on the **actual** boxes. Do not assume saturated red.

### 7. `imu.yaml` silences the uncalibrated warning

`load_imu_remap()` treats any existing `R` as calibrated. `configs/imu.yaml` is a **guess** that maps standing `[-1,0,0] → [0,0,-1]` from one vector; it is not a 3-axis cal.

Add `calibrated: false` to the placeholder. Warn every N steps until `calibrate_imu.py` has been run on the robot and flipped the flag. Identity `R` on this X-up mount would feed gravity `[-1,0,0]` into a policy trained on `[0,0,-1]`.

### 8. `goal.py` logger format bug

```python
logger.warning("Failed loading calibration from %s: %e", calib_path, e)
```

`%e` is scientific-notation float, not exception. Use `%s`. Same file: AprilTag `obj_pts` order must match detector corner order (hiwonder vs aruco TL-TR-BR-BL). If PnP `tvec[2]` is nonsense on a known tag distance, swap corner winding before changing `R_BODY_FROM_CAM` (optical-axis unit test already PASSes: `p_cam=[0,0,1] → body ≈ [1.08, 0, 0.04]`).

### 9. `--dry-run` still `hw.play("stand")` on terminate / Ctrl+C

Loop skips `hw.play` when `args.dry_run`, but the `reason in (...arrived...)` and `KeyboardInterrupt` paths call `hw.play("stand")`. Harmless if `TonyPiHardware(dry_run=True)`, **not** harmless if someone sets yaml `dry_run: true` and constructs hardware with `dry_run=False`. One flag, one place.

### 10. COCO filter vs cardboard boxes

`YoloSegOccupancy.COCO_OBSTACLE_NAMES` has no `"box"` / `"cardboard"`. YOLO11-seg will often miss the training-style mocap boxes, or label them `suitcase`. Phase 1 vis (`view_segmentation --mode yolo --keep-all`) is the gate: if the binary pane is empty on a box in view, do not run `--occupancy yolo` on the robot; use tuned `--occupancy color` or expand `keep_cls`.

### 11. `inspect_checkpoint` `json.loads` on SB3 `data`

Works on this zip (Phase 0). Do not assume it works on every SB3 2.9 dump (some are cloudpickle). If it throws, fall back to `GuidedPPO.load` + space asserts, still refuse 32×32 / 19-D / Box(18) walk zips.

---

## P2 — hygiene

- `FakeHardware` in `deploy_nav` has `bgr=None` → zeros occupancy. Fine for smoke; for occupancy tests use `replay_policy` or pass a frame.
- `one_hot(prev)` will `IndexError` if `prev` not in `0..6`. Clamp.
- `fall_recover` plays `stand_up_front/back` then continues the nav loop. Confirm those `.d6a` names exist on the Pi before `--recover-fall`.
- `IMPLEMENTATION_LOG.md` Phase 4 claims tag-lost stand is implemented — it is not. Update the log when you fix P0.1.
- Sonar miss → scaled `1.0` (5 m cutoff), not “1 metre”. Probe print already says MISSING at 5.0 m.

---

## Pi bring-up gates (do not skip)

Copy of `BRINGUP.md` with extra vis gate. Stop after the first FAIL.

1. `sudo systemctl stop tonypi`
2. `export PYTHONPATH=/home/pi/nav_deploy:$PYTHONPATH` and `TONYPI_HOME=/home/pi/TonyPi`
3. `python3 -m sim2real_implementation.sensor_probe --iterations 20` — camera frames, ToF not stuck at 1.20 m when a hand is 20 cm in front, IMU `A:[-1,…]` standing, `g` near `[0,0,-1]` **after** real IMU cal.
4. **Segmentation vis:** follow [YOLO26_SEG.md](YOLO26_SEG.md).  
   `python3 -m sim2real_implementation.view_segmentation --backend tonypi --mode yolo --http-port 8080`  
   Open `http://<ROBOT_IP>:8080`. People/chairs/**table including legs** must be **red** in pane 3. HUD `yolo26n-sem-ade20k`, not `det=hsv`. Instance fallback `--yolo-weights yolo26n-seg.pt`. `best.pt` and `yolov5n.pt` are detect-only. HSV `--mode color` is only for painted-red boxes.
5. `calibrate_imu.py` on the real standing robot; set `calibrated: true`.
6. `--script stand,go_forward,stand` with a leash, `--max-steps 3`.
7. Short-leash `--goal 0.4 0.0 --occupancy zeros --max-steps 8` (open floor).
8. Same with tuned color occupancy and a box in view — policy must **not** pick `go_forward` into a filled center column (laptop analogue: `--synthetic-wall`).

---

## 3-pane vis contract (already implemented)

Script: `sim2real_implementation/view_segmentation.py`

```
python -m sim2real_implementation.view_segmentation \
  --backend tonypi|webcam|fake \
  --mode yolo|color \
  --yolo-weights yolo26n-sem-ade20k.pt \
  --http-port 8080
```

Panes, left to right:

1. **Camera** — raw BGR.
2. **Segmentation** — YOLO instance colors + labels, or HSV mask tint.
3. **Binary** — obstacle = red, road/free = green. This is the full-res mask **before** 64×64 downsample; a small HUD shows the 64×64 policy occupancy so you can see what the CNN will get.

Rules:

- Default `TonyPiHardware(dry_run=True)` — **never** play walk clips from this script.
- Do not use `best.pt`.
- Headless Pi: `--http-port` MJPEG and/or `--save-dir` (`latest_camera.jpg`, `latest_seg.jpg`, `latest_mask.jpg`, `latest_triple.jpg`).
- `OccupancyEncoder.encode()` still returns only `(1,64,64)`. Views go through `encode_views()`.

---

## Suggested patch order for the next agent

1. P0.1 tag-lost stand + test.
2. P0.2 camera/ToF hard-fail.
3. P0.3 model path exists-check before `hw.start()`.
4. P1.7 `calibrated:` flag on imu yaml.
5. P1.8 `%e` → `%s`.
6. Live Phase 1–2 on the robot with vis running in a second ssh session (vis holds the camera — **do not** run vis and `deploy_nav` at the same time; camera is exclusive).
7. Only then Phase 3 leash walk.
