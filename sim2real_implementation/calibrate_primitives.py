"""Primitives Calibration Wizard for TonyPi.

Measures actual displacement and heading change per locomotion clip on real flooring.
Runs each moving clip N times, prompts for tape-measured displacement (dx, dy, dyaw_deg),
computes average per single clip, and writes to configs/primitives.yaml.

Usage:
  python3 -m sim2real_implementation.calibrate_primitives
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

from sim2real_implementation.constants import SIM_PRIMITIVES
from sim2real_implementation.hardware import FakeHardware, Hardware, TonyPiHardware

MOVING_CLIPS = [
    "go_forward",
    "back_fast",
    "left_move_fast",
    "right_move_fast",
    "turn_left",
    "turn_right",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate TonyPi action primitives.")
    parser.add_argument("--backend", type=str, choices=["tonypi", "fake"], default="tonypi")
    parser.add_argument("--repeats", type=int, default=5, help="Number of repetitions per clip (default: 5)")
    parser.add_argument(
        "--output",
        type=str,
        default="sim2real_implementation/configs/primitives.yaml",
        help="Path to output yaml file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("TonyPi Action Primitives Calibration Wizard")
    print("=" * 70)
    print(f"Calibrating on {args.backend} with N={args.repeats} repetitions per clip.")
    print("Mark robot position on floor with tape before starting.")

    if args.backend == "tonypi":
        hw: Hardware = TonyPiHardware(dry_run=False)
    else:
        hw = FakeHardware()

    hw.start()

    calibrated_table: dict[str, list[float]] = {
        "stand": [0.00, 0.00, 0.00],
    }

    try:
        for clip in MOVING_CLIPS:
            sim_dx, sim_dy, sim_dyaw = SIM_PRIMITIVES[clip]
            print(f"\n>>> Calibrating action: '{clip}'")
            print(f"    (Sim baseline: dx={sim_dx:+.3f}m, dy={sim_dy:+.3f}m, dyaw={sim_dyaw:+.3f}rad)")

            if args.backend != "fake":
                input(f"Align robot with starting tape mark. Press ENTER to execute '{clip}' {args.repeats} times...")

            print(f"Playing '{clip}' x{args.repeats}...")
            for i in range(args.repeats):
                hw.play(clip)
                time.sleep(0.1)

            if args.backend != "fake":
                print(f"Robot finished {args.repeats} steps.")
                raw_dx = float(input(f"Enter total measured delta X (forward/backward) in METRES: "))
                raw_dy = float(input(f"Enter total measured delta Y (left/right) in METRES: "))
                raw_dyaw_deg = float(input(f"Enter total measured delta YAW (turn) in DEGREES: "))
            else:
                # Fake backend: simulate measured values close to sim primitives * repeats
                raw_dx = sim_dx * args.repeats
                raw_dy = sim_dy * args.repeats
                raw_dyaw_deg = float(np.rad2deg(sim_dyaw)) * args.repeats

            # Average per single action
            avg_dx = float(raw_dx / args.repeats)
            avg_dy = float(raw_dy / args.repeats)
            avg_dyaw_rad = float(np.deg2rad(raw_dyaw_deg / args.repeats))

            calibrated_table[clip] = [
                round(avg_dx, 4),
                round(avg_dy, 4),
                round(avg_dyaw_rad, 4),
            ]
            print(f"  Result for '{clip}': dx={avg_dx:+.4f}m, dy={avg_dy:+.4f}m, dyaw={avg_dyaw_rad:+.4f}rad")

        # Save to YAML
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            yaml.dump(calibrated_table, f, default_flow_style=None)

        print("\n" + "=" * 70)
        print(f"[SUCCESS] Calibrated primitives written to {args.output}")
        print("=" * 70)
        for k, v in calibrated_table.items():
            print(f"  {k:15s}: dx={v[0]:+7.4f}m, dy={v[1]:+7.4f}m, dyaw={v[2]:+7.4f}rad")

    finally:
        hw.stop()


if __name__ == "__main__":
    main()
