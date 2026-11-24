"""Goal management and coordinate transforms for TonyPi navigation sim-to-real.

Supported modes (locked specification):
  1. deadreckon:
     --goal GX GY is body-frame goal at t=0.
     Dead-reckons robot pose (x, y, yaw) using calibrated or sim primitives table.
  2. tag:
     AprilTag detection via head camera.
     Solves PnP with calibration_param.npz and transforms camera pose to body pose.
  3. fixed-body:
     --goal GX GY republished every step in current body frame (smoke testing).
"""
from __future__ import annotations

import logging
import math
import os
import sys
import numpy as np

from sim2real_implementation.constants import SIM_PRIMITIVES

logger = logging.getLogger(__name__)

# MuJoCo camera pos="0.08 0 0.04", xyaxes="0 -1 0 0 0 1"
# OpenCV camera frame: X right, Y down, Z forward (optical axis)
# Body frame: X forward, Y left, Z up
T_CAM_BODY = np.array([0.08, 0.00, 0.04], dtype=np.float32)
R_BODY_FROM_CAM = np.array([
    [0.0, 0.0, 1.0],   # body X = cam Z (optical axis forward)
    [-1.0, 0.0, 0.0],  # body Y = -cam X (cam right -> body -Y / right)
    [0.0, -1.0, 0.0],  # body Z = -cam Y (cam down -> body -Z / down)
], dtype=np.float32)


def wrap_angle(angle: float) -> float:
    """Wrap angle in radians to [-pi, pi]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def body_goal_from_polar(dist_m: float, angle_deg: float) -> tuple[float, float]:
    """Relative goal in the t=0 body frame: +x forward, +y left.

    angle_deg 0 is straight ahead. Positive angle is left (same as heading = atan2(y, x)).
    """
    dist_m = float(dist_m)
    if not np.isfinite(dist_m) or dist_m < 0.0:
        raise ValueError(f"dist_m must be >= 0, got {dist_m}")
    angle = math.radians(float(angle_deg))
    return (dist_m * math.cos(angle), dist_m * math.sin(angle))


def load_primitives(path: str | None = None) -> dict[str, tuple[float, float, float]]:
    """Load primitives table (dx, dy, dyaw) from yaml, fallback to SIM_PRIMITIVES."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "configs", "primitives.yaml")

    if os.path.exists(path):
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data and isinstance(data, dict):
                prims = {}
                for k, v in data.items():
                    if isinstance(v, (list, tuple)) and len(v) == 3:
                        prims[k] = (float(v[0]), float(v[1]), float(v[2]))
                if all(k in prims for k in SIM_PRIMITIVES):
                    return prims
        except Exception as e:
            logger.warning("Failed to load primitives from %s: %s", path, e)

    return SIM_PRIMITIVES.copy()


def transform_cam_to_body(p_cam: np.ndarray | list[float]) -> np.ndarray:
    """Transform a 3D point in OpenCV camera frame to robot body frame."""
    p_c = np.asarray(p_cam, dtype=np.float32).reshape(3)
    p_b = R_BODY_FROM_CAM @ p_c + T_CAM_BODY
    return p_b.astype(np.float32)


class GoalSource:
    """Manages goal position and heading calculation across various modes."""

    def __init__(
        self,
        mode: str = "deadreckon",
        initial_goal: tuple[float, float] = (2.0, -0.5),
        primitives_path: str | None = None,
        tag_id: int = 0,
        tag_size: float = 0.08,
        tag_lost: str = "hold",
        calib_path: str | None = None,
    ):
        self.mode = mode
        self.initial_goal = (float(initial_goal[0]), float(initial_goal[1]))
        self.primitives = load_primitives(primitives_path)

        # Pose in dead-reckon world frame (t=0 body frame)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self._last_name: str | None = None
        self._last_dx = 0.0
        self._last_dy = 0.0
        self._yaw_before = 0.0

        # Goal in world frame (for deadreckon)
        self.world_goal_x = self.initial_goal[0]
        self.world_goal_y = self.initial_goal[1]

        # Tag configuration
        self.tag_id = tag_id
        self.tag_size = float(tag_size)
        self.tag_lost_policy = tag_lost  # "hold" or "stand"
        self.tag_visible = False
        self.last_tag_goal: tuple[float, float, float, float] = (
            float(np.hypot(initial_goal[0], initial_goal[1])),
            float(np.arctan2(initial_goal[1], initial_goal[0])),
            initial_goal[0],
            initial_goal[1],
        )

        # AprilTag camera calibration
        self.mtx: np.ndarray | None = None
        self.dist_coeff: np.ndarray | None = None
        self._load_calibration(calib_path)

        # Detector
        self._tag_detector = None
        self._init_tag_detector()

    def _load_calibration(self, calib_path: str | None):
        if calib_path is None:
            # Check default on-robot and repo paths
            candidates = [
                os.path.join(os.environ.get("TONYPI_HOME", "/home/pi/TonyPi"), "Functions", "CameraCalibration", "calibration_param.npz"),
                os.path.join(os.path.dirname(__file__), "..", "tonypi_inside", "TonyPi", "Functions", "CameraCalibration", "calibration_param.npz"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    calib_path = c
                    break

        if calib_path and os.path.exists(calib_path):
            try:
                data = np.load(calib_path)
                self.mtx = data["mtx_array"]
                self.dist_coeff = data["dist_array"]
                logger.info("Loaded camera calibration from %s", calib_path)
            except Exception as e:
                logger.warning("Failed loading calibration from %s: %s", calib_path, e)

        if self.mtx is None:
            # Fallback pinhole with reasonable 70 deg FOV defaults
            self.mtx = np.array([
                [425.0, 0.0, 320.0],
                [0.0, 425.0, 240.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            self.dist_coeff = np.zeros(5, dtype=np.float64)

    def _init_tag_detector(self):
        try:
            import hiwonder.apriltag as apriltag
            self._tag_detector = apriltag.Detector(searchpath=apriltag._get_demo_searchpath())
            logger.info("Initialized hiwonder.apriltag Detector.")
            return
        except Exception:
            pass

        try:
            import cv2
            if hasattr(cv2, "aruco") and hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
                dict_36h11 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
                params = cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, "DetectorParameters") else cv2.aruco.DetectorParameters_create()
                self._tag_detector = ("cv2_aruco", dict_36h11, params)
                logger.info("Initialized cv2.aruco AprilTag detector.")
        except Exception:
            pass

    def reset(self, goal_xy: tuple[float, float] | None = None) -> None:
        """Reset pose to origin and set world goal."""
        if goal_xy is not None:
            self.initial_goal = (float(goal_xy[0]), float(goal_xy[1]))
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self._last_name = None
        self._last_dx = 0.0
        self._last_dy = 0.0
        self._yaw_before = 0.0
        self.world_goal_x = self.initial_goal[0]
        self.world_goal_y = self.initial_goal[1]
        self.tag_visible = False
        dist0 = float(np.hypot(self.initial_goal[0], self.initial_goal[1]))
        head0 = float(np.arctan2(self.initial_goal[1], self.initial_goal[0]))
        self.last_tag_goal = (dist0, head0, self.initial_goal[0], self.initial_goal[1])

    def update_vision(self, bgr: np.ndarray | None) -> None:
        """Process head camera frame if in tag mode."""
        if self.mode != "tag" or bgr is None or self._tag_detector is None:
            return

        import cv2

        corners = None
        # Try hiwonder.apriltag
        if hasattr(self._tag_detector, "detect"):
            try:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                detections = self._tag_detector.detect(gray, return_image=False)
                for d in detections:
                    if int(d.tag_id) == self.tag_id:
                        corners = np.array(d.corners, dtype=np.float32)
                        break
            except Exception as e:
                logger.debug("apriltag detect error: %s", e)

        # Try cv2.aruco
        elif isinstance(self._tag_detector, tuple) and self._tag_detector[0] == "cv2_aruco":
            try:
                _, dictionary, params = self._tag_detector
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                if hasattr(cv2.aruco, "ArucoDetector"):
                    detector = cv2.aruco.ArucoDetector(dictionary, params)
                    corners_list, ids, _ = detector.detectMarkers(gray)
                else:
                    corners_list, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
                if ids is not None:
                    for i, cid in enumerate(ids.flatten()):
                        if int(cid) == self.tag_id:
                            corners = corners_list[i].reshape(-1, 2).astype(np.float32)
                            break
            except Exception as e:
                logger.debug("aruco detect error: %s", e)

        if corners is None:
            self.tag_visible = False
            return

        # SolvePnP
        s = self.tag_size
        obj_pts = np.array([
            [-s / 2.0, -s / 2.0, 0.0],
            [s / 2.0, -s / 2.0, 0.0],
            [s / 2.0, s / 2.0, 0.0],
            [-s / 2.0, s / 2.0, 0.0],
        ], dtype=np.float32)

        ok, rvec, tvec = cv2.solvePnP(
            obj_pts,
            corners,
            self.mtx.astype(np.float64),
            self.dist_coeff.astype(np.float64),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok or tvec is None:
            self.tag_visible = False
            return

        p_cam = tvec.reshape(3)
        # Check for valid positive depth along optical axis
        if p_cam[2] <= 0.05 or p_cam[2] > 10.0:
            self.tag_visible = False
            return

        p_body = transform_cam_to_body(p_cam)
        gx, gy = float(p_body[0]), float(p_body[1])
        dist = float(np.hypot(gx, gy))
        heading = float(np.arctan2(gy, gx))

        self.tag_visible = True
        self.last_tag_goal = (dist, heading, gx, gy)

    def on_action(self, name: str) -> None:
        """Update dead-reckoning pose after action clip execution."""
        dx, dy, dyaw = self.primitives.get(name, (0.0, 0.0, 0.0))
        self._last_name = name
        self._last_dx = dx
        self._last_dy = dy
        self._yaw_before = self.yaw
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        self.x += c * dx - s * dy
        self.y += s * dx + c * dy
        self.yaw = wrap_angle(self.yaw + dyaw)

    def correct_last_translation(self, dx_body: float, dy_body: float) -> bool:
        """Replace the last translation primitive with a measured body-frame step.

        Only dead-reckon, only pure translations, only a plausible step
        (|dx| ≤ 0.20 m, |dy| ≤ 0.15 m). Yaw is left on the primitive.
        Returns False when the measurement is ignored.
        """
        if self.mode != "deadreckon":
            return False
        if self._last_name not in ("go_forward", "back_fast", "left_move_fast", "right_move_fast"):
            return False
        dx_body = float(dx_body)
        dy_body = float(dy_body)
        if not (math.isfinite(dx_body) and math.isfinite(dy_body)):
            return False
        if abs(dx_body) > 0.20 or abs(dy_body) > 0.15:
            return False
        ddx = dx_body - self._last_dx
        ddy = dy_body - self._last_dy
        c = math.cos(self._yaw_before)
        s = math.sin(self._yaw_before)
        self.x += c * ddx - s * ddy
        self.y += s * ddx + c * ddy
        self._last_dx = dx_body
        self._last_dy = dy_body
        return True

    def current(self) -> tuple[float, float, float, float]:
        """Return (dist, heading, gx, gy) in current body frame."""
        if self.mode == "fixed-body":
            gx, gy = self.initial_goal[0], self.initial_goal[1]
            dist = float(np.hypot(gx, gy))
            heading = float(np.arctan2(gy, gx))
            return dist, heading, gx, gy

        if self.mode == "tag":
            if self.tag_visible or self.tag_lost_policy == "hold":
                return self.last_tag_goal
            # If tag lost and policy is stand: returns last goal coordinates,
            # but should_force_stand() will signal deploy_nav to play stand.
            return self.last_tag_goal

        # mode == "deadreckon"
        dx_w = self.world_goal_x - self.x
        dy_w = self.world_goal_y - self.y
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        gx = c * dx_w + s * dy_w
        gy = -s * dx_w + c * dy_w
        dist = float(np.hypot(gx, gy))
        heading = float(np.arctan2(gy, gx))
        return dist, heading, float(gx), float(gy)

    def should_force_stand(self) -> bool:
        """Return True if in tag mode, tag is currently not visible, and policy is 'stand'."""
        return (self.mode == "tag") and (not self.tag_visible) and (self.tag_lost_policy == "stand")

    def arrived(self, radius: float = 0.20) -> bool:
        """Check whether robot has reached goal."""
        dist, _, _, _ = self.current()
        return dist < radius
