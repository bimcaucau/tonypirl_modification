"""Safety layer and action filtering for TonyPi navigation sim-to-real.

Execution order (locked in SIM2REAL_NAV.md section 6):
  1. E-stop: Ctrl+C, SIGTERM, or STM32 key click -> stopActionGroup(), stand, exit 2.
  2. Arrived / Max steps / max wall time -> stand, exit 0 if arrived else 1.
  3. Fall: raw accel tilt > --fall-deg (default 45 deg) for 3 consecutive cycles:
     -> stand_up_front / stand_up_back if --recover-fall, else stand and exit 3.
  4. ToF guard: --tof-guard 0.12 replaces go_forward with back_fast if ToF < 0.12m.
"""
from __future__ import annotations

import logging
import math
import signal
import sys
import numpy as np

from sim2real_implementation.constants import ACTION_NAMES, ACTION_TO_ID
from sim2real_implementation.hardware import Hardware

logger = logging.getLogger(__name__)


class Safety:
    """Monitors robot health, handles e-stop signals, falls, and sensor guards."""

    def __init__(
        self,
        max_steps: int = 20,
        max_seconds: float = 60.0,
        arrive_radius: float = 0.20,
        fall_deg: float = 45.0,
        recover_fall: bool = False,
        tof_guard: float | None = None,
        no_safety: bool = False,
    ):
        self.max_steps = max_steps
        self.max_seconds = max_seconds
        self.arrive_radius = arrive_radius
        self.fall_deg = fall_deg
        self.recover_fall = recover_fall
        self.tof_guard = tof_guard
        self.no_safety = no_safety

        self.estop = False
        self.fall_counter = 0
        self.fall_direction = "none"  # "front", "back", "none"

        # Register signal handlers for clean e-stop
        try:
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
        except Exception as e:
            logger.debug("Could not attach signal handler: %s", e)

    def _handle_signal(self, signum, frame):
        logger.warning("\n[E-STOP] Signal %d received! Halting robot immediately.", signum)
        self.estop = True

    def check_fall(self, accel: np.ndarray | None) -> tuple[bool, str]:
        """Detect fall from raw accelerometer reading using stock Fall_and_Stand logic."""
        if accel is None or self.no_safety:
            self.fall_counter = 0
            return False, "none"

        try:
            ax, _, az = float(accel[0]), float(accel[1]), float(accel[2])
            # angle_y is angle in X-Z plane
            angle_y = math.degrees(math.atan2(ax, az))

            # Upright standing on TonyPi STM32 mount has ax ~ -1g, az ~ 0g -> angle ~ -90 deg
            # Fall forward: angle approaches 0 deg (|angle| < fall_deg)
            # Fall backward: angle approaches +/- 180 deg (|angle| > 180 - fall_deg)
            if abs(angle_y) < self.fall_deg:
                self.fall_counter += 1
                self.fall_direction = "front"
            elif abs(angle_y) > (180.0 - self.fall_deg):
                self.fall_counter += 1
                self.fall_direction = "back"
            else:
                self.fall_counter = 0
                self.fall_direction = "none"

            if self.fall_counter >= 3:
                return True, self.fall_direction
        except Exception as e:
            logger.debug("Error computing fall check: %s", e)

        return False, "none"

    def filter_action(
        self,
        action_idx: int,
        hw: Hardware,
        dist: float,
        step: int,
        elapsed_s: float,
        tof_m: float,
    ) -> tuple[int, str, str | None]:
        """Apply safety checks in specified priority order.

        Returns:
            (filtered_action_idx, action_to_play_str, reason)
        """
        # If safety disabled completely
        if self.no_safety:
            return action_idx, ACTION_NAMES[action_idx], None

        # 1. E-STOP Check
        if self.estop or hw.poll_estop():
            self.estop = True
            return 0, "stand", "estop"

        # 2. Arrived Check
        if dist < self.arrive_radius:
            return 0, "stand", "arrived"

        # 3. Max Steps / Wall Time Check
        if step >= self.max_steps:
            return 0, "stand", "max_steps"
        if elapsed_s >= self.max_seconds:
            return 0, "stand", "max_seconds"

        # 4. Fall Check
        imu_sample = hw.read_imu()
        accel = imu_sample[0] if imu_sample is not None else None
        is_fall, direction = self.check_fall(accel)
        if is_fall:
            if self.recover_fall:
                recovery_clip = "stand_up_front" if direction == "front" else "stand_up_back"
                logger.warning("[FALL DETECTED] Attempting recovery: %s", recovery_clip)
                return 0, recovery_clip, "fall_recover"
            else:
                logger.critical("[FALL DETECTED] Tilt exceeded %s deg. Halting!", self.fall_deg)
                return 0, "stand", "fall"

        # 5. ToF Guard Check
        current_name = ACTION_NAMES[action_idx]
        if self.tof_guard is not None and tof_m < self.tof_guard:
            if current_name == "go_forward":
                back_idx = ACTION_TO_ID["back_fast"]
                logger.warning(
                    "[TOF GUARD] ToF %.2fm < %.2fm guard threshold; overriding 'go_forward' with 'back_fast'",
                    tof_m,
                    self.tof_guard,
                )
                return back_idx, "back_fast", "tof_guard"

        return action_idx, current_name, None
