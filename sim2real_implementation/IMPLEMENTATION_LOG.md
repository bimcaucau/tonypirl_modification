# TonyPi Nav Sim-to-Real Implementation Log

**Specification**: `SIM2REAL_NAV.md`, `HARDWARE.md`, `BRINGUP.md`  
**Checkpoint**: `models_ckpt/new_expert_implementation/tonypi_final_model.zip`  
**Architecture**: GuidedPPO, 10,010,624 steps (teacher already 0: `mix_prob=0`, `bc_coef=0`)  
**Observation**: `Dict{ vec: (18,) float32, occupancy: (1, 64, 64) float32 in {0,1} }`  
**Action**: `Discrete(7)` -> `stand | go_forward | back_fast | left_move_fast | right_move_fast | turn_left | turn_right`  
**Hardware Motion**: Canned SQLite `.d6a` clips via `AGC.runActionGroup(name)` (walking is NOT rewritten, no MuJoCo / no VecNormalize on robot).

---

## Phase 0: Laptop Testing & Contract Verification (PASS)

### 0.1 Contract & Space Check
- **Command**:  
  `/home/bimcaucau/miniconda3/bin/python sim2real_implementation/replay_policy.py --model models_ckpt/new_expert_implementation/tonypi_final_model.zip --print-space`
- **Output**:  
  `vec (18,) occupancy (1, 64, 64) Discrete(7)`  
  `Metadata: type='Nav (Dict 64x64+18 -> Discrete(7))', timesteps=10010624, size=3.59 MB`  
  `Successfully loaded model without MuJoCo.`
- **Result**: **PASS**

### 0.2 Synthetic Clear Observation
- **Command**:  
  `/home/bimcaucau/miniconda3/bin/python sim2real_implementation/replay_policy.py --model models_ckpt/new_expert_implementation/tonypi_final_model.zip --synthetic-clear`
- **Observation**: `dist=2.0m, heading=0.0 rad, ToF=1.2m, sonar=5.0m, gravity=[0, 0, -1], occupancy=zeros, prev=stand(0)`
- **Output**:  
  `Action: id=1 (go_forward)`  
  `Logits: stand: -508.82 | go_forward: -0.0000 (PREDICTED) | back_fast: -59.63 | left_move_fast: -12.10 | right_move_fast: -18.96 | turn_left: -16.00 | turn_right: -11.00`
- **Result**: **PASS** (Policy selects `go_forward` with maximum logit)

### 0.3 Synthetic Wall Observation
- **Command**:  
  `/home/bimcaucau/miniconda3/bin/python sim2real_implementation/replay_policy.py --model models_ckpt/new_expert_implementation/tonypi_final_model.zip --synthetic-wall`
- **Observation**: `dist=2.0m, heading=0.0 rad, ToF=0.25m, sonar=5.0m, gravity=[0, 0, -1], occupancy[:, :, 28:36]=1.0, prev=stand(0)`
- **Output**:  
  `Action: id=3 (left_move_fast)`  
  `Logits: stand: -482.39 | go_forward: -3.08 | back_fast: -46.00 | left_move_fast: -0.0981 (PREDICTED) | right_move_fast: -12.16 | turn_left: -4.53 | turn_right: -3.31`  
  `[PASS] Policy successfully avoided ramming wall by picking 'left_move_fast'.`
- **Result**: **PASS** (Policy successfully avoids forward ramming when central corridor is blocked)

### 0.4 Perception Module Import
- **Command**:  
  `/home/bimcaucau/miniconda3/bin/python -c "from tonypi_rl.perception import YoloSegOccupancy; print('YoloSegOccupancy imported successfully')"`
- **Result**: **PASS**

### 0.5 Full Observation Contract & Unit Test Suite
- **Command**:  
  `/home/bimcaucau/miniconda3/bin/python sim2real_implementation/tests/test_obs_contract.py`
- **Tests Executed**:
  1. `test_1_build_obs_shapes_and_one_hot`: vec shape (18,), occupancy (1, 64, 64), one-hot index 16 for action 5.
  2. `test_2_polar_goal`: dist=3.0, heading=0.0 -> vec[0]=1.0, vec[1]=0.0, vec[2]=1.0.
  3. `test_3_scale_range`: ToF & sonar scaling with NaN/Inf/out-of-bounds protection.
  4. `test_4_fake_loop_deadreckon`: dead-reckoning pose integration with sim primitives.
  5. `test_5_policy_load_and_synthetic_predict`: checkpoint loading & deterministic inference.
  6. `test_6_optical_axis_cam_to_body_transform`: optical axis point `p_cam=[0, 0, 1]` transforms to body `+X` (+1.08m, not -X).
  7. `test_7_safety_checks`: e-stop, arrival cutoff, budget cutoff, 3-cycle fall detection (`stand_up_front`), and ToF guard override (`go_forward` -> `back_fast`).
  8. `test_8_occupancy_encoders`: zeros and color HSV segmentation.
- **Result**: **PASS** (8/8 tests OK in 0.935s)

---

## Phase 1: Sensor Diagnostics & Calibration Tools (READY FOR ROBOT)

### 1.1 Sensor Probe (`sim2real_implementation/sensor_probe.py`)
- Reads ToF (`VL53L0X`), Sonar, STM32 IMU, and head camera.
- Tested on laptop with `--backend fake --iterations 3`:
  `[0000] ToF:1.20m (1.00) | Sonar:MISSING | A:[-1.00,+0.00,+0.00] G:[+0.0,+0.0,+0.0] | g:[+0.00,+0.00,-1.00] | Cam:640x480 | Occ:0.00 ( 20ms)`
- Tested `--save-frame /tmp/test_cam.jpg` and `--save-occ /tmp/test_occ.png`:
  `[PASS] Saved frame (640x480 BGR) to /tmp/test_cam.jpg`  
  `[PASS] Saved occupancy (64x64, mean=0.0000, latency=5.1ms) to /tmp/test_occ.png`
- Live on robot command:
  `python3 -m sim2real_implementation.sensor_probe`

### 1.2 IMU Calibrator (`sim2real_implementation/calibrate_imu.py`)
- Derives rotation matrix `R` mapping raw STM32 accel unit vector to sim gravity `g_sim = R @ a_unit`.
- Tested on laptop with `--backend fake --output /tmp/test_imu.yaml`:
  `Residual ||R @ a_stand - [0, 0, -1]|| = 0.000000`  
  `Calibrated Rotation Matrix R: [[0, 0, -1], [0, 1, 0], [1, 0, 0]]`  
  `[SUCCESS] Saved calibrated matrix to /tmp/test_imu.yaml`
- Live on robot command:
  `python3 -m sim2real_implementation.calibrate_imu`

### 1.3 Action Primitives Calibrator (`sim2real_implementation/calibrate_primitives.py`)
- Step-by-step wizard measuring `(dx, dy, dyaw)` on actual floor per action clip over N repeats.
- Tested on laptop with `--backend fake --output /tmp/test_primitives.yaml`:
  Verified accurate scaling, averaging, and YAML emission.
- Live on robot command:
  `python3 -m sim2real_implementation.calibrate_primitives --repeats 5`

---

## Phase 2: Action Clips & Safety Verification (PASS on Mock / READY FOR ROBOT)

### 2.1 Script Execution
- **Command**:  
  `/home/bimcaucau/miniconda3/bin/python -m sim2real_implementation.deploy_nav --backend fake --script stand,go_forward,stand --log-dir sim2real_implementation/logs/script_test`
- **Output**:  
  `Step 0 | Act: stand`  
  `Step 1 | Act: go_forward`  
  `Step 2 | Act: stand`  
  `[SCRIPT] Completed scripted actions.`
- **Result**: **PASS** (Exit code 0)

### 2.2 Safety & Fail-Safes Verification
- E-Stop (Ctrl+C / SIGINT / SIGTERM / board button): stops motion, sets stand, exits code 2.
- Fall Detection: raw accel `math.atan2(ax, az)` tilt threshold checked over 3 consecutive cycles. Exits code 3 or recovers via `stand_up_front`/`stand_up_back` if `--recover-fall`.
- ToF Guard: if `--tof-guard <dist>` specified and ToF drops below cutoff during `go_forward`, action is replaced by `back_fast`.

---

## Phase 3: Short-Leash Navigation & Closed Loop (READY FOR ROBOT)

- Configured in `configs/deploy.yaml`.
- Verified dead-reckoning integration with `SIM_PRIMITIVES`:
  Goal `(2.0, -0.5)` with `go_forward` actions decreased goal distance from `2.06m` to `1.75m` in 5 steps with JSONL telemetry recorded per step.
- Goal Arrival logic verified: `--goal 0.15 0.0 --arrive 0.20` triggered instant arrival and clean exit code 0.

---

## Phase 4: AprilTag Goal Tracking (READY FOR ROBOT & UNIT TESTED)

- Optical axis camera-to-body coordinate transform verified:
  `R_body_from_cam = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]`, `T_cam_body = [0.08, 0.0, 0.04]`.
  Point 1m ahead in camera optical frame (`p_cam=[0, 0, 1]`) transforms to body frame `p_body=[1.08, 0.0, 0.04]` (+X direction).
- Both `hiwonder.apriltag` (on robot) and `cv2.aruco` (standard OpenCV) supported with `cv2.solvePnP` using camera calibration `calibration_param.npz`.
- **P0.1 Tag-Lost Stand Implementation**:
  When `mode=="tag"` and tag is not visible (`not goal.tag_visible`):
  - With `--tag-lost stand`, `goal.should_force_stand()` returns True, forcing action 0 (`stand`) before any locomotion clip is played.
  - With `--tag-lost hold`, `goal.should_force_stand()` returns False, and `goal.current()` holds the last valid body goal.
  - Unit tested in `test_10_tag_lost_stand_and_clamping` (**PASS**).

---

## Agent Feedback Fixes (AGENT_FEEDBACK.md Complete)

| Issue | Severity | Status | Solution |
|---|---|---|---|
| **1. Tag-lost stand no-op** | P0 | **RESOLVED** | Added `GoalSource.should_force_stand()` and wired it into `deploy_nav.py` action selection; added unit test `test_10_tag_lost_stand_and_clamping`. |
| **2. Camera/ToF fail-open** | P0 | **RESOLVED** | `TonyPiHardware.start()` aborts start if camera fails when occupancy is `yolo` or `color`, and aborts if ToF fails unless `--allow-missing-tof`. |
| **3. Model path vs rsync** | P0 | **RESOLVED** | Added `resolve_model_path()` to search multiple candidate paths (`models_ckpt/`, current dir, `/home/pi/nav_deploy/`); pre-validates and loads checkpoint before `hw.start()`. |
| **4. tonypi.service check** | P0 | **RESOLVED** | Detects frame read failure after 10 attempts and displays explicit instruction: `sudo systemctl stop tonypi`. |
| **5. Dual Board() serial clients** | P1 | **DOCUMENTED** | Documented rationale in `hardware.py` and log; independent Board instances preserved to prevent breaking AGC module-level singleton. |
| **6. HSV red wrap** | P1 | **RESOLVED** | `hsv_in_range()` wraps across hue boundary when `H_lo > H_hi` (e.g. 170–179 + 0–H_hi); wired into `occupancy_real.py`, `view_segmentation.py`, and `sensor_probe.py`. |
| **7. imu.yaml calibrated flag** | P1 | **RESOLVED** | Added `calibrated: false` to default placeholder; `calibrate_imu.py` flips it to `true`; `deploy_nav.py` warns every 5 steps if uncalibrated. |
| **8. goal.py %e logger bug** | P1 | **RESOLVED** | Replaced `%e` with `%s` in `goal.py` exception logger. |
| **9. dry-run stand calls** | P1 | **RESOLVED** | Added `if not args.dry_run` guards to `hw.play("stand")` in safety termination break, `TonyPiHardware.stop()`, and `KeyboardInterrupt`. |
| **10. COCO filter vs boxes** | P1 | **RESOLVED** | Added `--yolo-keep-all` option to `OccupancyEncoder`, `sensor_probe.py`, and `deploy_nav.py` for cardboard box obstacles. |
| **11. inspect_checkpoint fallback** | P1 | **RESOLVED** | Added fallback load and space verification in `inspect_checkpoint()` if `data` JSON parsing throws. |
| **12. Hygiene / index clamp** | P2 | **RESOLVED** | Clamped `one_hot(prev)` to `[0..6]`; updated documentation and verified sonar miss cutoff semantics. |

---

## Live robot: VL53L0X `./vl53l0x_python.so` (FIXED)

Stock `TonyPi/Functions/VL53L0X.py` loads `CDLL("./vl53l0x_python.so")` from **process CWD**. Running `python3 -m sim2real_implementation.*` from `$HOME` (where `TonyPi/` and `sim2real_implementation/` are siblings) raised:

```
RuntimeError: VL53L0X ToF sensor failed to initialize: ./vl53l0x_python.so: cannot open shared object file
```

Fix: `hardware.find_tonypi_home()` prefers `$TONYPI_HOME`, then sibling `../TonyPi`, then `~/TonyPi`; `hardware.import_vl53l0x()` chdirs into `Functions/` and remaps that CDLL to the absolute `.so` path, then restores CWD. Viewer passes `allow_missing_tof=True` only as a camera-only fallback; deploy still refuses missing ToF unless `--allow-missing-tof`.

---

## Ground look (2026-09-24) — laptop PASS, robot NOT run

Wires and small floor blocks sit under the level camera and under the chest-height ToF beam, and dead-reckon distance is a fixed primitive per clip. Fix is a veto around the policy, not a new observation. Spec and the live check order: `GROUND_LOOK.md`.

- `ground_look.py`: pitch down (pulse 900, ~54°), black-hat + color-deviation clutter, footprint veto, `min(ToF, ground p10)` range, phase-correlate floor shift.
- `deploy_nav.py` runs the survey after `safety.filter_action` and before `hw.play`. One policy re-query when the fused range is shorter. Reason `ground_look` is not terminal. `--script` skips the look.
- `GoalSource.correct_last_translation` replaces the last translation primitive (dead-reckon only). `LiveTriple.invalidate` drops any frame grabbed while the head was down; the next policy frame is level.
- **Tests**: `/home/bimcaucau/miniconda3/bin/python -m unittest sim2real_implementation.tests.test_ground_look` — 17 tests OK. Existing `test_obs_contract` still passes (29 together).
- Fake `--dry-run --max-steps 2` exits 1 because it does not arrive (expected). It logs `ground look center: no frame` because `FakeHardware.bgr` is None. Not a crash.
- **Not verified on the robot:** pulse 900 really looks down, yaw sign, clear-floor false veto, floor-shift vs tape. Do those checks in `GROUND_LOOK.md` before tuning thresholds.

---

## YOLO26n-sem ADE20K default (2026-09-19)

Instance-seg (`yolo26n-seg.pt`) often masks only the tabletop. Default is now **semantic** `yolo26n-sem-ade20k.pt` (`result.semantic_mask`, ADE20K 150 classes). Floor/wall/sky → occupancy 0; table/desk/chair/person → 1 (whole stuff region, including legs). Occupancy tensor still `(1,64,64)`. Fallback: `--yolo-weights yolo26n-seg.pt`. Cityscapes `yolo26n-sem.pt` is urban — do not use it indoors unless ADE20K weights fail to download.

---

## YOLO26n-seg default (2026-09-18)

Defaults now `yolo26n-seg.pt` (`YoloSegOccupancy`, viewer, `deploy.yaml`, `sensor_probe`). Detect checkpoints (`yolov5n.pt`, `Functions/model/best.pt`, `yolo26n.pt` without `-seg`) raise at init. HUD prints weights basename. **Revert to YOLO11n-seg:** `--yolo-weights yolo11n-seg.pt`. Needs `ultralytics>=8.4.0` so `YOLO("yolo26n-seg.pt")` actually resolves YOLO26.

---

## Occupancy vis diagnosis (2026-09-18) — HSV, not YOLOv5

`debug/Screenshot 2026-09-18 145737.png` HUD: `color 640x480 fps=36.9 enc=10.5ms occ_mean=0.025 det=hsv`.

That session ran `--mode color`. HSV red on skin; people/table/chairs = green road. Onboard `Functions/model/best.pt` and `third_party/yolov5/yolov5n.pt` are **detect**, not seg. Next implement: **`YOLO26_SEG.md`** (`yolo26n-seg.pt`). Do not treat this PNG as a YOLO quality baseline.

---

## 3-Pane Segmentation Viewer (`sim2real_implementation/view_segmentation.py`)

- Script: `python -m sim2real_implementation.view_segmentation`
- Three panes: camera | segmentation overlay | red=obstacle green=road
- Backends: `fake` (synthetic moving red box), `webcam`, `tonypi` (`TonyPiHardware(dry_run=True)` — never plays walk clips)
- Headless: `--http-port 8080` MJPEG and/or `--save-dir`
- `OccupancyEncoder.encode()` still returns only `(1, 64, 64)`; vis uses `encode_views()`
- Tested on fake backend at 30+ FPS.

---

## Deployment to Robot Runbook

### 1. Rsync Files to Raspberry Pi
```bash
# From laptop repo root (/mnt/d/robotics_cccc/tonypi):
rsync -avz --exclude '__pycache__' --exclude '*.pyc' \
  sim2real_implementation \
  tonypi_rl \
  models_ckpt/new_expert_implementation/tonypi_final_model.zip \
  pi@<ROBOT_IP>:/home/pi/nav_deploy/
```

### 2. On-Robot Setup
```bash
ssh pi@<ROBOT_IP>
cd /home/pi/nav_deploy
export PYTHONPATH=/home/pi/nav_deploy:$PYTHONPATH
export TONYPI_HOME=/home/pi/TonyPi

# Stop background app owning camera
sudo systemctl stop tonypi

# Install deploy requirements (no mujoco!)
pip3 install -r sim2real_implementation/requirements-deploy.txt
```

### 3. Run Sensor Probe (Phase 1)
```bash
python3 -m sim2real_implementation.sensor_probe
```

### 4. Calibrate IMU (Phase 1.4)
```bash
python3 -m sim2real_implementation.calibrate_imu
```

### 5. Run Short-Leash Dead-Reckon Walk (Phase 3)
```bash
python3 -m sim2real_implementation.deploy_nav \
  --model /home/pi/nav_deploy/tonypi_final_model.zip \
  --goal 1.2 0.0 \
  --goal-mode deadreckon \
  --occupancy color \
  --max-steps 12 \
  --arrive 0.25
```
