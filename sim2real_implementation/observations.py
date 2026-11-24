"""Observation building and normalization for TonyPi navigation.

Observation contract strictly matches tonypi_rl/tonypi_nav_env.py::_get_obs:
  vec[0]     = dist / 3.0                         # GOAL_DIST_SCALE = 3.0
  vec[1]     = sin(heading_error)
  vec[2]     = cos(heading_error)
  vec[3]     = tof_0_m / 1.2                      # miss -> 1.0
  vec[4]     = sonar_m / 5.0                      # miss / unplugged -> 1.0
  vec[5:8]   = projected gravity, unit, pelvis    # sim upright ~ [0, 0, -1]
  vec[8:11]  = gyro rad/s                         # v1 deploy: [0, 0, 0]
  vec[11:18] = previous action one-hot            # index = last executed id, stand=0 at reset
  occupancy  = (1, 64, 64) float32 in {0.0, 1.0}
"""
from __future__ import annotations

import logging
import os
import numpy as np

from sim2real_implementation.constants import (
    GOAL_DIST_SCALE,
    OCCUPANCY_SIZE,
    SONAR_CUTOFF,
    TOF_CUTOFF,
)

logger = logging.getLogger(__name__)


def scale_range(metres: float, cutoff: float) -> float:
    """Scale range sensor reading to [0, 1]. Miss or invalid -> 1.0."""
    if not np.isfinite(metres) or metres < 0:
        metres = cutoff
    return float(np.clip(metres, 0.0, cutoff) / cutoff)


def gravity_from_imu(accel_raw: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Convert raw accelerometer 3-vector into unit gravity vector in pelvis frame.

    R maps real_accel_unit -> sim_gravity (3,): g_sim = R @ a_real_unit.
    If accel norm is near zero, defaults to upright [0, 0, -1].
    """
    a = np.asarray(accel_raw, dtype=np.float32).reshape(3)
    n = float(np.linalg.norm(a))
    if n < 1e-3:
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return (R @ (a / n)).astype(np.float32)


def polar_goal(dist: float, heading: float) -> np.ndarray:
    """Encode polar goal matching sim env: [dist / 3.0, sin(heading), cos(heading)]."""
    return np.array([dist / GOAL_DIST_SCALE, np.sin(heading), np.cos(heading)], dtype=np.float32)


def one_hot(prev: int) -> np.ndarray:
    """One-hot encode previous action (length 7). Clamps index to [0..6]."""
    v = np.zeros(7, dtype=np.float32)
    idx = int(np.clip(int(prev), 0, 6))
    v[idx] = 1.0
    return v


def build_obs(
    *,
    dist: float,
    heading: float,
    tof_m: float,
    sonar_m: float,
    gravity: np.ndarray | list[float],
    prev_action: int,
    occupancy: np.ndarray,
) -> dict:
    """Build exact observation dictionary for the SB3 nav policy."""
    vec = np.concatenate([
        polar_goal(dist, heading),
        [scale_range(tof_m, TOF_CUTOFF), scale_range(sonar_m, SONAR_CUTOFF)],
        np.asarray(gravity, dtype=np.float32).reshape(3),
        np.zeros(3, dtype=np.float32),  # gyro locked zeros in v1
        one_hot(prev_action),
    ]).astype(np.float32)

    assert vec.shape == (18,), f"vec shape must be (18,), got {vec.shape}"

    occ = np.asarray(occupancy, dtype=np.float32)
    if occ.ndim == 2 and occ.shape == (OCCUPANCY_SIZE, OCCUPANCY_SIZE):
        occ = occ[np.newaxis, ...]
    assert occ.shape == (1, OCCUPANCY_SIZE, OCCUPANCY_SIZE), (
        f"occupancy shape must be (1, {OCCUPANCY_SIZE}, {OCCUPANCY_SIZE}), got {occ.shape}"
    )

    return {"vec": vec, "occupancy": occ}


def is_imu_calibrated(path: str | None = None) -> bool:
    """Return True if imu.yaml exists and has calibrated: true."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "configs", "imu.yaml")
    if os.path.exists(path):
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return bool(data and data.get("calibrated", False))
        except Exception:
            return False
    return False


def load_imu_remap(path: str | None = None, return_status: bool = False) -> np.ndarray | tuple[np.ndarray, bool]:
    """Load IMU rotation matrix R from yaml, or return identity with warning.

    If return_status is True, returns (R, is_calibrated).
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "configs", "imu.yaml")

    calibrated = False
    if os.path.exists(path):
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data and "R" in data:
                R = np.asarray(data["R"], dtype=np.float32)
                calibrated = bool(data.get("calibrated", False))
                if R.shape == (3, 3):
                    if not calibrated:
                        logger.warning(
                            "imu.yaml has 'calibrated: false' (placeholder guess). "
                            "Run calibrate_imu.py on the robot before live walking!"
                        )
                    if return_status:
                        return R, calibrated
                    return R
        except Exception as e:
            logger.warning("Failed to load imu.yaml from %s: %s. Using identity.", path, e)

    logger.warning("imu.yaml not calibrated or missing; using identity rotation matrix R=I")
    identity = np.eye(3, dtype=np.float32)
    if return_status:
        return identity, False
    return identity
