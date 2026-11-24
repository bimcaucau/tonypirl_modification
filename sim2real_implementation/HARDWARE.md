# TonyPi onboard hardware cheat sheet

Facts for the deploy agent. Paths below are **on the robot** (`/home/pi/TonyPi/...`). The repo copy lives at `tonypi_inside/TonyPi/`. Import the real package on the Pi, do not vendor a second copy of `hiwonder`.

The stock app (`TonyPi.py`) is a systemd service that **owns the USB camera**. Stop it before deploy or `Camera.read()` will be empty / fight MJPG.

```bash
sudo systemctl stop tonypi          # this session
# sudo systemctl disable tonypi     # only if you want it off across reboots
```

---

## 1. Motion: action groups, not joints

There is **no `cmd_vel` and no ROS**. Walking is SQLite `.d6a` clips:

```python
import hiwonder.ActionGroupControl as AGC
AGC.runActionGroup("go_forward")          # blocking, ~0.4 s, times=1
AGC.stopActionGroup()                     # cooperative cancel
```

Signature (`HiwonderSDK/hiwonder/ActionGroupControl.py`):

```python
def runActionGroup(
    actName,
    times=1,
    with_stand=False,
    lock_servos="",
    path="/home/pi/TonyPi/ActionGroups/",
): ...
```

`go_forward` / `back_fast` auto-play `go_forward_start` / `*_end` when you switch away from them. That is stock behavior. Call **one clip per policy step**. Do not set `times>1` from the policy.

`runActionGroup` is **blocking** and durations are **not** all ~0.4 s (read from the `.d6a` SQLite `Time` column). The control loop must wait for return; do not add your own 0.4 s sleep.

| name | typical duration | notes |
|---|---|---|
| `stand` | ~385 ms | 1 frame |
| `go_forward` | ~380 ms | 10×38 ms, **plus** `go_forward_start` the first time you enter the gait |
| `back_fast` | ~768 ms | 4×192 ms, plus `back_start` on gait enter |
| `left_move_fast` / `right_move_fast` | ~500 ms | 5×100 ms |
| `turn_left` / `turn_right` | ~770 ms | 5 frames |

Policy action id → clip name (order is frozen, must match `tonypi_rl/tonypi_nav_env.py::ACTION_NAMES`):

| id | name | sim primitive (body) | real clip |
|---|---|---|---|
| 0 | `stand` | 0 | `stand` |
| 1 | `go_forward` | +0.08 m x | `go_forward` |
| 2 | `back_fast` | −0.06 m x | `back_fast` |
| 3 | `left_move_fast` | +0.05 m y | `left_move_fast` |
| 4 | `right_move_fast` | −0.05 m y | `right_move_fast` |
| 5 | `turn_left` | +0.35 rad yaw | `turn_left` |
| 6 | `turn_right` | −0.35 rad yaw | `turn_right` |

App dict (`ActionGroupDict.py`) numbers **do not** match policy ids (`turn_left` is app `'7'`, policy `5`). Never index through the app dict.

Do **not** edit `.d6a` files. Do **not** turn on `balance_ankle.py` while clips play — it writes ankle pulses and fights the gait.

---

## 2. Camera

```python
from hiwonder.Camera import Camera
cam = Camera(resolution=(640, 480))
cam.camera_open()
ok, bgr = cam.read()          # latest frame from a grab thread; may be (False, None)
```

- USB, YUYV, 640×480, ~30 FPS grab thread.
- Flip comes from `/boot/camera_setting.yaml`.
- Alternate: MJPG `http://127.0.0.1:8080/?action=stream` **only if** `tonypi.service` is running. Deploy should open the device itself.
- Calibration npz: `Functions/CameraCalibration/calibration_param.npz` (`mtx_array`, `dist_array`). Use it for AprilTag pose, **not** for the occupancy mask (training occupancy is undistorted pinhole at 70° FOV — see § occupancy).
- Head look is PWM servos 1 (pitch) and 2 (yaw), pulse 1500 = center (`Head_Control.py`, `ApriltagDetect.initMove`). Freeze them at 1500 for deploy so the camera matches the MuJoCo `head_camera` look-along-+X.

MuJoCo camera (`tonypi_mujoco/tonypi.xml`):

```
camera name="head_camera" pos="0.08 0 0.04" fovy="70" xyaxes="0 -1 0 0 0 1"
```

---

## 3. ToF — one VL53L0X

Already used in `Functions/obstacle_avoidance.py`:

```python
import VL53L0X   # Functions/VL53L0X.py + vl53l0x_python.so
tof = VL53L0X.VL53L0X(i2c_bus=1, i2c_address=0x29)
tof.open()
tof.start_ranging(VL53L0X.Vl53l0xAccuracyMode.BETTER)  # ~66 ms, 1.2 m
mm = tof.get_distance()   # int millimetres
```

`VL53L0X.py` does `CDLL("./vl53l0x_python.so")` — **process CWD**, not the module dir. Stock `obstacle_avoidance.py` is launched with cwd=`TonyPi/Functions`. Deploy must use `hardware.import_vl53l0x()` (chdir + absolute CDLL remap). Putting `Functions/` on `sys.path` alone is not enough. On this robot `TonyPi` is a sibling of `sim2real_implementation` (`~/TonyPi`); set `TONYPI_HOME` if it lives elsewhere.

Sim cutoff `TOF_CUTOFF = 1.2` m. Mapping:

- `mm <= 0` or `mm >= 8191` → treat as **miss** → 1.2 m
- else `metres = clip(mm / 1000.0, 0.0, 1.2)`
- obs value = `metres / 1.2`

Stock avoider treats `>350 mm` as clear, `150–350` turn, `<150` back. Policy sees the continuous scaled range; do not pre-binarize it.

---

## 4. Sonar — optional Hiwonder I2C `0x77`

```python
from hiwonder.Sonar import Sonar
s = Sonar()
mm = s.getDistance()   # 99999 = unplugged; >5000 clipped to 5000
```

Sim cutoff `SONAR_CUTOFF = 5.0` m. Mapping:

- `99999` or exception → **1.0** in the **scaled** slot? No. NAV.md said “Sonar or 1.0 if missing”. Locked here: missing → **raw 5.0 m** (scaled `1.0`), i.e. “no echo / infinitely far”, matching sim miss. Do not put `1.0` metres (that would be 1 m obstacle).
- else `metres = clip(mm / 1000.0, 0.0, 5.0)`, obs = `metres / 5.0`

If the sensor is physically absent, log once and keep the miss value. Policy was trained with a working beam; a stuck `1.0` scaled is the honest “I don’t have this”.

---

## 5. IMU — STM32 via `Board.get_imu()`

```python
import hiwonder.ros_robot_controller_sdk as rrc
board = rrc.Board()
board.enable_reception()     # required; otherwise get_imu() is always None
time.sleep(0.3)
sample = board.get_imu()     # 6 floats (ax, ay, az, gx, gy, gz) or None
```

- Queue `maxsize=1`, **non-blocking**. `None` means no new packet — hold last.
- `balance_ankle.py` documents mount: **X-up**, gravity ≈ `(-1, 0, 0)` when standing. Units are **not** documented (likely g for accel). Gyro units unknown (deg/s vs rad/s).
- Fall script (`Functions/Fall_and_Stand.py`) uses `atan2(ax, az)` and `stand_up_front` / `stand_up_back`.

Sim projected gravity is a **unit vector in the MuJoCo pelvis frame**, Z-up, X-forward:

```python
rot = data.xmat[pelvis].reshape(3, 3)
gravity = (-rot[2, :]).astype(np.float32)   # upright → ≈ [0, 0, -1]
gyro = data.sensor("imu_gyro").data         # rad/s, IMU site on pelvis (identity)
```

Kinematic nav **zeros qvel**, so **gyro ≈ 0 in every training sample**. Locked for v1 deploy: **send zeros for gyro**. Do not invent a unit conversion. Gravity still needs an axis remap; that is `calibrate_imu.py`, not a guess in `hardware.py`.

---

## 6. AprilTag (goal, not occupancy)

`Functions/Tag_Detect.py` and `Functions/ApriltagDetect.py` detect id + pixel center. They do **not** return metric XYZ. For a body-frame goal you need pose from tag size + camera matrix (`calibration_param.npz`). Implement that in `goal.py`; do not call the stock “play a clip per tag id” path.

---

## 7. Occupancy on the Pi

Training mask is software pinhole of `obs_*` boxes → `(1, 64, 64)` float32 in `{0, 1}`.

Deploy adapter already exists in sim code:

```python
from tonypi_rl.perception import YoloSegOccupancy, OCCUPANCY_SIZE  # 64
enc = YoloSegOccupancy(weights="yolo26n-sem-ade20k.pt", conf=0.25, imgsz=320)
occ = enc.encode_bgr(bgr)   # (1, 64, 64)
```

**Do not** use `Functions/model/best.pt` (YOLOv11 **detect** age/gender) or `third_party/yolov5/yolov5n.pt` (COCO **detect**, no masks). Occupancy default is semantic `*-sem-ade20k.pt` (table legs as stuff). Instance `*-seg.pt` is a fallback. Plan: `YOLO26_SEG.md`.

The 2026-09-18 debug PNG is `--mode color` HSV (`det=hsv`, `occ_mean=0.025`): skin speckles, furniture labeled road. That is not a YOLO quality issue.

COCO classes kept by `YoloSegOccupancy.COCO_OBSTACLE_NAMES` include `person`, `chair`, `dining table`, `suitcase`, … Cardboard boxes often miss. First live test: `view_segmentation --mode yolo` until people/furniture are red in pane 3. HSV `--occupancy color` is a fallback for painted-red boxes only.

Pi 4 PyTorch nano-seg @320 is often **>300 ms**. Measure in the vis HUD. Fallback: NCNN/ONNX export in `YOLO26_SEG.md` §7, then HSV, then zeros.

---

## 8. Compute / Python

Robot image: Python 3, packages under `hiwonder` (installed via `HiwonderSDK/setup.py`), OpenCV, existing `ultralytics` for the age/gender demo.

Deploy extra deps (install in a venv if the system Python is messy):

```
numpy
torch          # CPU wheel
gymnasium
stable-baselines3
opencv-python  # already there
PyYAML
ultralytics    # only if occupancy=yolo
```

Do **not** install mujoco on the Pi.

`tonypi_rl/__init__.py` only registers gym ids (lazy entry points). Safe to import. **Never** import `tonypi_rl.tonypi_nav_env` on the Pi (pulls mujoco).

SB3 zip has no `vecnormalize.pkl`. Do not wrap observations in `VecNormalize`.

---

## 9. Safety hooks that already exist

| Need | Stock code |
|---|---|
| Fall get-up | `Functions/Fall_and_Stand.py` + clips `stand_up_front` / `stand_up_back` |
| Buzzer | `board.set_buzzer(1900, 0.1, 0.9, 1)` |
| Stop clip | `AGC.stopActionGroup()` |
| Head center | PWM 1 & 2 = 1500 |

STM32 key events: `board.get_key()` / `PacketReportKeyEvents` in `ros_robot_controller_sdk.py`. Use the onboard button as e-stop if it reports.

---

## 10. Primitive magnitudes (dead-reckon)

Sim numbers are **not** the real step sizes. `calibrate_primitives.py` must measure them on carpet/wood and write `configs/primitives.yaml`. Until that file exists, dead-reckon with the sim table and treat pose as **open-loop garbage after ~5 steps**. That is acceptable for the first “does it walk toward the tag” test only if AprilTag refreshes the goal every step.
