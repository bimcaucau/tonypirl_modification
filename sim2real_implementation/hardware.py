"""Hardware abstraction layer for TonyPi navigation sim-to-real.

Defines Hardware protocol and two backends:
  1. FakeHardware: offline testing, unit tests, laptop simulation.
  2. TonyPiHardware: runs on Raspberry Pi with Hiwonder SDK & sensors.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import Protocol, runtime_checkable
import numpy as np

from sim2real_implementation.constants import SONAR_CUTOFF, TOF_CUTOFF

logger = logging.getLogger(__name__)

# On the robot the stock tree is ~/TonyPi, sibling of ~/sim2real_implementation
# (not necessarily /home/pi/nav_deploy/TonyPi). VL53L0X.py does CDLL("./vl53l0x_python.so")
# which is CWD-relative, so we must import it from Functions/.


def find_tonypi_home() -> str:
    """Locate the stock TonyPi tree (the directory that contains Functions/)."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(pkg_dir)
    env = os.environ.get("TONYPI_HOME")
    candidates = [
        env,
        os.path.join(parent, "TonyPi"),          # ~/TonyPi next to ~/sim2real_implementation
        os.path.expanduser("~/TonyPi"),
        "/home/pi/TonyPi",
        os.path.join(parent, "tonypi_inside", "TonyPi"),  # laptop repo copy
        "/home/pi/nav_deploy/TonyPi",
    ]
    seen: set[str] = set()
    ordered: list[str] = []
    for c in candidates:
        if not c:
            continue
        ap = os.path.abspath(c)
        if ap in seen:
            continue
        seen.add(ap)
        ordered.append(ap)
        functions = os.path.join(ap, "Functions")
        so_path = os.path.join(functions, "vl53l0x_python.so")
        if os.path.isdir(functions) and os.path.isfile(so_path):
            return ap
    for ap in ordered:
        if os.path.isdir(os.path.join(ap, "Functions")):
            return ap
    return ordered[0] if ordered else "/home/pi/TonyPi"


def import_vl53l0x(functions_dir: str | None = None):
    """Import stock VL53L0X.py so CDLL('./vl53l0x_python.so') resolves.

    Hiwonder's wrapper loads the .so from process CWD (`./vl53l0x_python.so`),
    not from the module directory. Stock scripts run with cwd=Functions/.
    """
    import ctypes

    if functions_dir is None:
        functions_dir = os.path.join(find_tonypi_home(), "Functions")
    functions_dir = os.path.abspath(functions_dir)
    so_path = os.path.join(functions_dir, "vl53l0x_python.so")
    py_path = os.path.join(functions_dir, "VL53L0X.py")
    if not os.path.isfile(so_path):
        raise FileNotFoundError(
            f"vl53l0x_python.so not found at {so_path}. "
            "Set TONYPI_HOME to the stock tree that contains Functions/ "
            "(on this robot: ~/TonyPi, sibling of sim2real_implementation)."
        )
    if not os.path.isfile(py_path):
        raise FileNotFoundError(f"VL53L0X.py not found at {py_path}")

    real_cdll = ctypes.CDLL

    def _cdll(name, *args, **kwargs):
        if isinstance(name, (str, bytes, os.PathLike)):
            basename = os.path.basename(os.fsdecode(name))
            if basename == "vl53l0x_python.so":
                name = so_path
        return real_cdll(name, *args, **kwargs)

    prev_cwd = os.getcwd()
    ctypes.CDLL = _cdll  # type: ignore[assignment]
    try:
        os.chdir(functions_dir)
        if functions_dir not in sys.path:
            sys.path.insert(0, functions_dir)
        sys.modules.pop("VL53L0X", None)
        import VL53L0X  # noqa: WPS433 — stock Hiwonder module
        return VL53L0X
    finally:
        ctypes.CDLL = real_cdll  # type: ignore[assignment]
        try:
            os.chdir(prev_cwd)
        except OSError:
            pass


TONYPI_HOME = find_tonypi_home()


@runtime_checkable
class Hardware(Protocol):
    """Abstract protocol for robot hardware interaction."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def read_bgr(self) -> np.ndarray | None: ...
    def read_tof_m(self) -> float: ...
    def read_sonar_m(self) -> float: ...
    def read_imu(self) -> tuple[np.ndarray, np.ndarray] | None: ...
    def play(self, name: str) -> None: ...
    def stop_motion(self) -> None: ...
    def center_head(self) -> None: ...
    def poll_estop(self) -> bool: ...


class FakeHardware:
    """Mock hardware backend for laptop testing without physical robot."""

    def __init__(
        self,
        bgr: np.ndarray | None = None,
        tof_m: float = 1.2,
        sonar_m: float = 5.0,
        accel: np.ndarray | None = None,
        gyro: np.ndarray | None = None,
    ):
        self.bgr = bgr
        self.tof_m = float(tof_m)
        self.sonar_m = float(sonar_m)
        # Raw STM32 standing accel is [-1.0, 0.0, 0.0] (X-up mount)
        self.accel = np.asarray(accel if accel is not None else [-1.0, 0.0, 0.0], dtype=np.float32)
        self.gyro = np.asarray(gyro if gyro is not None else [0.0, 0.0, 0.0], dtype=np.float32)

        self.is_started = False
        self.head_centered = False
        self.head_pitch_pulse = 1500
        self.head_yaw_pulse = 1500
        self.head_moves: list[tuple[int, int, int]] = []
        self.instant_head = True
        self.estop_triggered = False
        self.played: list[str] = []

    def start(self) -> None:
        self.is_started = True
        self.center_head()

    def stop(self) -> None:
        self.is_started = False

    def read_bgr(self) -> np.ndarray | None:
        return self.bgr

    def read_tof_m(self) -> float:
        return float(np.clip(self.tof_m, 0.0, TOF_CUTOFF))

    def read_sonar_m(self) -> float:
        return float(np.clip(self.sonar_m, 0.0, SONAR_CUTOFF))

    def read_imu(self) -> tuple[np.ndarray, np.ndarray] | None:
        return (self.accel.copy(), self.gyro.copy())

    def play(self, name: str) -> None:
        self.played.append(name)

    def stop_motion(self) -> None:
        self.played.append("stop")

    def center_head(self) -> None:
        self.set_head(1500, 1500, 400)
        self.head_centered = True

    def set_head(self, pitch_pulse: int, yaw_pulse: int, duration_ms: int = 400) -> bool:
        """PWM1 pitch, PWM2 yaw. Tests read the pulses back; nothing sleeps."""
        self.head_pitch_pulse = int(pitch_pulse)
        self.head_yaw_pulse = int(yaw_pulse)
        self.head_centered = abs(self.head_pitch_pulse - 1500) < 15 and abs(self.head_yaw_pulse - 1500) < 15
        self.head_moves.append((self.head_pitch_pulse, self.head_yaw_pulse, int(duration_ms)))
        return True

    def poll_estop(self) -> bool:
        return self.estop_triggered


# NOTE on P1.5 (Dual Board serial clients):
# TonyPiHardware constructs rrc.Board() for IMU/PWM reception.
# Hiwonder ActionGroupControl (AGC) creates its own module-level Board() internally upon import.
# Stock apps do this too. Do not merge them without verifying IMU streaming during motion.


class TonyPiHardware:
    """Hardware driver for TonyPi robot running on Raspberry Pi."""

    def __init__(
        self,
        dry_run: bool = False,
        occupancy_mode: str = "color",
        allow_missing_tof: bool = False,
    ):
        self.dry_run = dry_run
        self.occupancy_mode = occupancy_mode
        self.allow_missing_tof = allow_missing_tof
        self.board = None
        self.controller = None
        self.camera = None
        self.tof = None
        self.sonar = None
        self.sonar_ok = True
        self.last_imu_sample: tuple[np.ndarray, np.ndarray] | None = None
        self.estop_pressed = False
        self.tonypi_home = find_tonypi_home()

    def start(self) -> None:
        """Initialize sensors and controllers on the robot."""
        # 1. STM32 Board controller
        try:
            import hiwonder.ros_robot_controller_sdk as rrc
            self.board = rrc.Board()
            self.board.enable_reception()
            time.sleep(0.3)
            logger.info("Board reception enabled.")
        except Exception as e:
            logger.error("Failed to initialize Board: %s", e)
            raise

        # 2. PWM Controller for head servos
        try:
            from hiwonder.Controller import Controller
            self.controller = Controller(self.board)
        except Exception as e:
            logger.warning("Failed to initialize Controller for PWM: %s", e)

        # 3. Center head servos
        self.center_head()

        # 4. Camera
        camera_error = None
        try:
            from hiwonder.Camera import Camera
            self.camera = Camera(resolution=(640, 480))
            self.camera.camera_open()
            time.sleep(0.3)
            # Verify camera actually returns frames (detect tonypi.service locking /dev/video0)
            frame_ok = False
            for _ in range(10):
                ret, test_frame = self.camera.read()
                if ret and test_frame is not None:
                    frame_ok = True
                    break
                time.sleep(0.1)
            if not frame_ok:
                camera_error = (
                    "Camera opened but failed to produce frames after 10 attempts.\n"
                    "tonypi.service may be holding /dev/video0.\n"
                    "Please run: sudo systemctl stop tonypi"
                )
            else:
                logger.info("Camera opened at (640, 480) and verified streaming.")
        except Exception as e:
            camera_error = str(e)
            self.camera = None

        if camera_error:
            if self.occupancy_mode in ("yolo", "color"):
                logger.critical("[CAMERA FAIL] %s", camera_error)
                raise RuntimeError(
                    f"Camera initialization failed: {camera_error}\n"
                    f"Cannot run with --occupancy {self.occupancy_mode} without a working camera stream.\n"
                    "If tonypi.service is running, stop it with: sudo systemctl stop tonypi"
                )
            else:
                logger.warning(
                    "Camera initialization failed (%s), but occupancy is '%s'. Continuing.",
                    camera_error,
                    self.occupancy_mode,
                )

        # 5. VL53L0X ToF (stock wrapper CDLL's ./vl53l0x_python.so — import from Functions/)
        tof_error = None
        try:
            self.tonypi_home = find_tonypi_home()
            functions_dir = os.path.join(self.tonypi_home, "Functions")
            VL53L0X = import_vl53l0x(functions_dir)
            logger.info("Loaded VL53L0X from %s", functions_dir)
            self.tof = VL53L0X.VL53L0X(i2c_bus=1, i2c_address=0x29)
            self.tof.open()
            self.tof.start_ranging(VL53L0X.Vl53l0xAccuracyMode.BETTER)
            test_mm = self.tof.get_distance()
            logger.info("VL53L0X ToF sensor started in BETTER ranging mode (initial reading: %s mm).", test_mm)
        except Exception as e:
            tof_error = str(e)
            self.tof = None

        if tof_error or self.tof is None:
            if not self.allow_missing_tof:
                logger.critical("[TOF FAIL] VL53L0X ToF sensor failed to initialize: %s", tof_error)
                raise RuntimeError(
                    f"VL53L0X ToF sensor failed to initialize: {tof_error}\n"
                    "Refusing to run without ToF sensor (empty occupancy + missing ToF will ram into obstacles).\n"
                    "To override deliberately, pass --allow-missing-tof."
                )
            else:
                logger.warning(
                    "[WARNING] Running WITHOUT ToF sensor (--allow-missing-tof enabled). ToF will report cutoff 1.2m."
                )

        # 6. Sonar
        try:
            from hiwonder.Sonar import Sonar
            self.sonar = Sonar()
            # Test read
            d1 = self.sonar.getDistance()
            d2 = self.sonar.getDistance()
            if d1 == 99999 and d2 == 99999:
                self.sonar_ok = False
                logger.warning("Sonar returned 99999 twice; marked absent.")
            else:
                self.sonar_ok = True
                logger.info("Sonar sensor initialized.")
        except Exception as e:
            self.sonar_ok = False
            logger.warning("Sonar sensor unavailable: %s", e)

        # 7. Initial stand
        if not self.dry_run:
            self.play("stand")

    def stop(self) -> None:
        """Shut down sensors and place robot in safe stand."""
        try:
            self.stop_motion()
            if not self.dry_run:
                self.play("stand")
        except Exception:
            pass

        if self.tof is not None:
            try:
                self.tof.stop_ranging()
                self.tof.close()
            except Exception:
                pass

        if self.camera is not None:
            try:
                self.camera.camera_close()
            except Exception:
                pass

    def read_bgr(self) -> np.ndarray | None:
        """Read latest BGR frame from camera."""
        if self.camera is None:
            return None
        try:
            ret, frame = self.camera.read()
            if ret and frame is not None:
                return frame
        except Exception as e:
            logger.debug("Camera read error: %s", e)
        return None

    def read_tof_m(self) -> float:
        """Read ToF in metres, clipped to [0, 1.2], miss/error -> 1.2."""
        if self.tof is None:
            return TOF_CUTOFF
        try:
            mm = self.tof.get_distance()
            if mm <= 0 or mm >= 8190:
                return TOF_CUTOFF
            return float(np.clip(mm / 1000.0, 0.0, TOF_CUTOFF))
        except Exception as e:
            logger.debug("ToF read error: %s", e)
            return TOF_CUTOFF

    def read_sonar_m(self) -> float:
        """Read sonar in metres, clipped to [0, 5.0], miss/unplugged -> 5.0."""
        if not self.sonar_ok or self.sonar is None:
            return SONAR_CUTOFF
        try:
            mm = self.sonar.getDistance()
            if mm == 99999 or mm <= 0:
                return SONAR_CUTOFF
            return float(np.clip(mm / 1000.0, 0.0, SONAR_CUTOFF))
        except Exception as e:
            logger.debug("Sonar read error: %s", e)
            return SONAR_CUTOFF

    def read_imu(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Read 6-tuple IMU from STM32. Returns (accel_3, gyro_3) or None if not ready."""
        if self.board is None:
            return None
        try:
            sample = self.board.get_imu()
            if sample is not None and len(sample) >= 6:
                accel = np.array(sample[:3], dtype=np.float32)
                gyro = np.array(sample[3:6], dtype=np.float32)
                self.last_imu_sample = (accel, gyro)
                return self.last_imu_sample
        except Exception as e:
            logger.debug("IMU read error: %s", e)
        return self.last_imu_sample

    def play(self, name: str) -> None:
        """Run named action group clip synchronously."""
        if self.dry_run:
            logger.info("[dry-run] hw.play('%s')", name)
            return

        import hiwonder.ActionGroupControl as AGC
        AGC.runActionGroup(name, times=1)

    def stop_motion(self) -> None:
        """Cooperative cancel of current action group."""
        try:
            import hiwonder.ActionGroupControl as AGC
            AGC.stopActionGroup()
        except Exception as e:
            logger.debug("stopActionGroup error: %s", e)

    def center_head(self) -> None:
        """Set head pitch (PWM1) and yaw (PWM2) to center pulse 1500 (level, training view)."""
        self.set_head(1500, 1500, 500)

    def set_head(self, pitch_pulse: int, yaw_pulse: int, duration_ms: int = 400) -> bool:
        """Move PWM head servos. Pulse 1500 is level / forward. Returns False if no controller.

        Servo 1 below 1500 looks down (hand_nod.py). Clipped to 600..2400.
        Does not sleep; the caller waits for the servo.
        """
        pitch_pulse = int(np.clip(pitch_pulse, 600, 2400))
        yaw_pulse = int(np.clip(yaw_pulse, 600, 2400))
        self.head_pitch_pulse = pitch_pulse
        self.head_yaw_pulse = yaw_pulse
        if self.controller is None:
            logger.warning("set_head skipped: PWM controller is not initialized")
            return False
        try:
            self.controller.set_pwm_servo_pulse(1, pitch_pulse, int(duration_ms))
            self.controller.set_pwm_servo_pulse(2, yaw_pulse, int(duration_ms))
            return True
        except Exception as e:
            logger.warning("set_head error: %s", e)
            return False

    def poll_estop(self) -> bool:
        """Poll onboard button or hardware key event."""
        if self.board is not None:
            try:
                # Check for button press event from STM32 if supported
                key = getattr(self.board, "get_key", None)
                if callable(key):
                    val = key()
                    if val is not None and val != 0:
                        self.estop_pressed = True
                        return True
            except Exception:
                pass
        return self.estop_pressed
