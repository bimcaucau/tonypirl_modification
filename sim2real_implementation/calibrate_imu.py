"""IMU Calibration Wizard for TonyPi.

Derives the rotation matrix R mapping raw STM32 accelerometer unit vector
to the MuJoCo simulation gravity vector:
  g_sim = R @ a_real_unit
  Upright standing: R @ a_stand ~ [0, 0, -1]
  Forward tilt: tilts gravity in body X direction.

Usage:
  python3 -m sim2real_implementation.calibrate_imu
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import numpy as np
import yaml

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sim2real_implementation.hardware import FakeHardware, Hardware, TonyPiHardware


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate TonyPi IMU gravity remap matrix R.")
    parser.add_argument("--backend", type=str, choices=["tonypi", "fake"], default="tonypi")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to average")
    parser.add_argument(
        "--output",
        type=str,
        default="sim2real_implementation/configs/imu.yaml",
        help="Path to output yaml file",
    )
    return parser.parse_args()


def collect_accel_samples(hw: Hardware, n_samples: int = 50) -> np.ndarray:
    """Collect and average accelerometer readings."""
    accel_list = []
    print(f"Collecting {n_samples} IMU samples...", end="", flush=True)
    while len(accel_list) < n_samples:
        sample = hw.read_imu()
        if sample is not None:
            accel_list.append(sample[0])
        time.sleep(0.02)
    print(" Done.")
    mean_a = np.mean(accel_list, axis=0).astype(np.float64)
    return mean_a


def compute_rotation_matrix(a_stand: np.ndarray, a_fwd: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute orthonormal 3x3 matrix R such that R @ a_stand_unit = [0, 0, -1]."""
    norm_stand = np.linalg.norm(a_stand)
    if norm_stand < 0.1:
        raise ValueError(f"Standing acceleration norm too small: {norm_stand:.4f}")

    u_stand = a_stand / norm_stand

    # In sim, gravity is [0, 0, -1]. Therefore:
    # row 2 of R dotted with u_stand must equal -1.
    # Hence row 2 of R = -u_stand.
    row2 = -u_stand

    # Forward tilt vector
    diff = a_fwd - a_stand
    # Project diff onto plane orthogonal to u_stand
    diff_orth = diff - np.dot(diff, u_stand) * u_stand
    norm_diff = np.linalg.norm(diff_orth)
    if norm_diff < 0.05:
        raise ValueError(f"Tilt difference too small ({norm_diff:.4f}). Tilt more forward (~30 deg).")

    u_diff = diff_orth / norm_diff

    # Pitching forward causes gravity in body frame to point in -X direction
    row0 = -u_diff

    # Complete right-handed coordinate system: row1 = row2 x row0
    row1 = np.cross(row2, row0)
    norm1 = np.linalg.norm(row1)
    if norm1 < 1e-4:
        raise ValueError("Degenerate cross product during IMU calibration.")
    row1 = row1 / norm1

    # Re-orthogonalize row0: row0 = row1 x row2
    row0 = np.cross(row1, row2)
    row0 = row0 / np.linalg.norm(row0)

    R = np.vstack([row0, row1, row2]).astype(np.float32)

    # Check determinant
    det = float(np.linalg.det(R))
    assert abs(det - 1.0) < 1e-3, f"R is not a proper rotation (det={det:.4f})"

    # Check residual on standing
    g_pred = R @ (a_stand / norm_stand)
    g_target = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    residual = float(np.linalg.norm(g_pred - g_target))

    return R, residual


def main():
    args = parse_args()
    print("=" * 70)
    print("TonyPi IMU Gravity Remap Calibration Wizard")
    print("=" * 70)

    if args.backend == "tonypi":
        hw: Hardware = TonyPiHardware(dry_run=True)
    else:
        # Fake backend simulating standing [-1, 0, 0] and forward tilt [-0.86, 0, +0.5]
        hw = FakeHardware(accel=np.array([-1.0, 0.0, 0.0]))

    hw.start()

    try:
        # Step 1: Standing Still
        print("\n--- STEP 1: Stand Still ---")
        if args.backend != "fake":
            input("Place robot standing still on flat ground. Press ENTER to sample...")
        a_stand = collect_accel_samples(hw, args.samples)
        print(f"Mean standing accel: [{a_stand[0]:+.4f}, {a_stand[1]:+.4f}, {a_stand[2]:+.4f}], norm={np.linalg.norm(a_stand):.4f}")

        # Step 2: Tilt Forward
        print("\n--- STEP 2: Tilt Forward ---")
        if args.backend != "fake":
            input("Tilt robot FORWARD ~30 degrees (along robot's +X heading). Press ENTER to sample...")
            a_fwd = collect_accel_samples(hw, args.samples)
        else:
            # Simulate forward tilt for fake backend testing
            a_fwd = np.array([-0.866, 0.0, 0.500], dtype=np.float64)

        print(f"Mean forward tilt accel: [{a_fwd[0]:+.4f}, {a_fwd[1]:+.4f}, {a_fwd[2]:+.4f}]")

        # Step 3: Compute R
        R, residual = compute_rotation_matrix(a_stand, a_fwd)
        print("\n--- Calibration Result ---")
        print(f"Residual ||R @ a_stand - [0, 0, -1]|| = {residual:.6f}")
        print("Calibrated Rotation Matrix R:")
        for r in R:
            print(f"  [{r[0]:+8.4f}, {r[1]:+8.4f}, {r[2]:+8.4f}]")

        # Save to YAML
        out_data = {
            "calibrated": True,
            "R": [[round(float(val), 6) for val in row] for row in R],
            "gyro_mode": "zeros",
            "metadata": {
                "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "residual": round(residual, 6),
                "a_stand_norm": round(float(np.linalg.norm(a_stand)), 4),
            },
        }

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            yaml.dump(out_data, f, default_flow_style=None)

        print(f"\n[SUCCESS] Saved calibrated matrix to {args.output}")

    finally:
        hw.stop()


if __name__ == "__main__":
    main()
