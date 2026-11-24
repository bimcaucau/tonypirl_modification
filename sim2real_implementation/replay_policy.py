"""Laptop policy replay and sanity tester (no robot / hiwonder required).

Phase 0 tests from BRINGUP.md:
  0.1: python sim2real_implementation/replay_policy.py --model models_ckpt/new_expert_implementation/tonypi_final_model.zip --print-space
       prints 'vec (18,) occupancy (1, 64, 64) Discrete(7)' and loads without mujoco.
  0.2: Same, --synthetic-clear
       action is usually go_forward or stand with goal 2 m ahead, empty mask, tof=1.2.
  0.3: --synthetic-wall (center column of occupancy = 1, tof=0.25, goal still +x)
       action is not always go_forward (turn, side-step, or back).
  0.4: python -c "from tonypi_rl.perception import YoloSegOccupancy" on laptop
       imports; downloading yolo11n-seg.pt is OK.

Optical axis verification comment / rule:
  Cam-from-body (MuJoCo site, metres):
    pos = [0.08, 0, 0.04], xyaxes="0 -1 0 0 0 1"
    camera X = body -Y
    camera Y = body +Z
    camera Z = body -X (look vector in MuJoCo optical frame) or standard OpenCV Z-forward.
  A point 1 m in front of the camera (along optical axis) must land near body +X.
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sim2real_implementation.constants import (
    ACTION_NAMES,
    GOAL_DIST_SCALE,
    OCCUPANCY_SIZE,
    SONAR_CUTOFF,
    TOF_CUTOFF,
)
from sim2real_implementation.observations import build_obs
from sim2real_implementation.policy import DummyNavEnv, inspect_checkpoint, load_nav_policy, act


def parse_args():
    parser = argparse.ArgumentParser(description="Replay and test TonyPi nav policy offline.")
    parser.add_argument(
        "--model",
        type=str,
        default="models_ckpt/new_expert_implementation/tonypi_final_model.zip",
        help="Path to zip checkpoint",
    )
    parser.add_argument(
        "--print-space",
        action="store_true",
        help="Print space specification and verify checkpoint loads without mujoco",
    )
    parser.add_argument(
        "--synthetic-clear",
        action="store_true",
        help="Run inference on clear synthetic observation (goal +2m, clear view)",
    )
    parser.add_argument(
        "--synthetic-wall",
        action="store_true",
        help="Run inference on synthetic obstacle wall (center column blocked, ToF 0.25m)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image file for real still-frame testing",
    )
    parser.add_argument(
        "--goal",
        type=float,
        nargs=2,
        default=[2.0, 0.0],
        help="Goal (x, y) in body frame (default: 2.0 0.0)",
    )
    parser.add_argument(
        "--occupancy",
        type=str,
        choices=["yolo", "color", "zeros"],
        default="color",
        help="Occupancy encoder when using --image",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default="yolo11n-seg.pt",
        help="YOLO weights file if using --occupancy yolo",
    )
    parser.add_argument(
        "--hsv-lo",
        type=int,
        nargs=3,
        default=[0, 80, 80],
        help="HSV low bounds",
    )
    parser.add_argument(
        "--hsv-hi",
        type=int,
        nargs=3,
        default=[12, 255, 255],
        help="HSV high bounds",
    )
    return parser.parse_args()


def get_policy_logits(model, obs: dict) -> np.ndarray | None:
    """Extract action logits from SB3 discrete policy if supported."""
    try:
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        distr = model.policy.get_distribution(obs_tensor)
        return distr.distribution.logits.detach().cpu().numpy()[0]
    except Exception:
        return None


def run_replay():
    args = parse_args()

    # Phase 0.1: Check space specification
    dummy = DummyNavEnv()
    if args.print_space:
        print("vec (18,) occupancy (1, 64, 64) Discrete(7)")

    meta = inspect_checkpoint(args.model)
    if not meta["valid"]:
        print(f"[!] Checkpoint invalid: {meta['reason']}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading checkpoint: {args.model}")
    print(f"Metadata: type='{meta['type']}', timesteps={meta['timesteps']}, size={meta['size_mb']:.2f} MB")
    model = load_nav_policy(args.model, device="cpu")
    print("Successfully loaded model without MuJoCo.")

    # Phase 0.2: Synthetic Clear
    if args.synthetic_clear:
        print("\n--- Running Phase 0.2: Synthetic Clear ---")
        occ_clear = np.zeros((1, OCCUPANCY_SIZE, OCCUPANCY_SIZE), dtype=np.float32)
        obs_clear = build_obs(
            dist=2.0,
            heading=0.0,
            tof_m=1.2,
            sonar_m=5.0,
            gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            prev_action=0,
            occupancy=occ_clear,
        )
        action_idx = act(model, obs_clear, deterministic=True)
        action_name = ACTION_NAMES[action_idx]
        logits = get_policy_logits(model, obs_clear)
        print(f"Action: id={action_idx} ({action_name})")
        if logits is not None:
            print("Action logits:")
            for idx, name in enumerate(ACTION_NAMES):
                marker = " <-- PREDICTED" if idx == action_idx else ""
                print(f"  [{idx}] {name:15s}: {logits[idx]:10.4f}{marker}")

    # Phase 0.3: Synthetic Wall
    if args.synthetic_wall:
        print("\n--- Running Phase 0.3: Synthetic Wall ---")
        occ_wall = np.zeros((1, OCCUPANCY_SIZE, OCCUPANCY_SIZE), dtype=np.float32)
        # Center column obstacle (columns 28:36)
        occ_wall[:, :, 28:36] = 1.0
        obs_wall = build_obs(
            dist=2.0,
            heading=0.0,
            tof_m=0.25,
            sonar_m=5.0,
            gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            prev_action=0,
            occupancy=occ_wall,
        )
        action_idx = act(model, obs_wall, deterministic=True)
        action_name = ACTION_NAMES[action_idx]
        logits = get_policy_logits(model, obs_wall)
        print(f"Action: id={action_idx} ({action_name})")
        if logits is not None:
            print("Action logits:")
            for idx, name in enumerate(ACTION_NAMES):
                marker = " <-- PREDICTED" if idx == action_idx else ""
                print(f"  [{idx}] {name:15s}: {logits[idx]:10.4f}{marker}")

        if action_name == "go_forward":
            print("[WARNING] Policy chose 'go_forward' into a synthetic wall!")
        else:
            print(f"[PASS] Policy successfully avoided ramming wall by picking '{action_name}'.")

    # Image test
    if args.image is not None:
        import cv2
        print(f"\n--- Running Still Image Inference: {args.image} ---")
        bgr = cv2.imread(args.image)
        if bgr is None:
            print(f"[!] Could not read image: {args.image}", file=sys.stderr)
            sys.exit(1)

        if args.occupancy == "yolo":
            from tonypi_rl.perception import YoloSegOccupancy
            encoder = YoloSegOccupancy(weights=args.yolo_weights, conf=0.25, imgsz=320)
            occ = encoder.encode_bgr(bgr)
        elif args.occupancy == "color":
            from sim2real_implementation.occupancy_real import color_occupancy
            occ = color_occupancy(bgr, args.hsv_lo, args.hsv_hi)
        else:
            occ = np.zeros((1, OCCUPANCY_SIZE, OCCUPANCY_SIZE), dtype=np.float32)

        gx, gy = args.goal
        dist = float(np.hypot(gx, gy))
        heading = float(np.arctan2(gy, gx))
        obs = build_obs(
            dist=dist,
            heading=heading,
            tof_m=1.2,
            sonar_m=5.0,
            gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            prev_action=0,
            occupancy=occ,
        )
        action_idx = act(model, obs, deterministic=True)
        action_name = ACTION_NAMES[action_idx]
        logits = get_policy_logits(model, obs)
        print(f"Goal: ({gx:.2f}, {gy:.2f}), dist: {dist:.2f}m, heading: {np.rad2deg(heading):.1f} deg")
        print(f"Action: id={action_idx} ({action_name}), Occupancy mean: {float(occ.mean()):.4f}")
        if logits is not None:
            print("Action logits:")
            for idx, name in enumerate(ACTION_NAMES):
                marker = " <-- PREDICTED" if idx == action_idx else ""
                print(f"  [{idx}] {name:15s}: {logits[idx]:10.4f}{marker}")


if __name__ == "__main__":
    run_replay()
