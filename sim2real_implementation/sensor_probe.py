"""Sensor probe and hardware diagnostics utility for TonyPi.

Used in Phase 1 of BRINGUP.md to verify sensors without moving the robot:
  - ToF distance (VL53L0X)
  - Sonar distance (I2C)
  - IMU 6-tuple and gravity vector (STM32)
  - Camera frame capture and resolution (USB)
  - Live occupancy mask encoding and latency measurement

Usage:
  python3 -m sim2real_implementation.sensor_probe
  python3 -m sim2real_implementation.sensor_probe --save-frame /tmp/cam.jpg
  python3 -m sim2real_implementation.sensor_probe --occupancy color --save-occ /tmp/occ.png
  python3 -m sim2real_implementation.sensor_probe --tune-hsv
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import cv2
import numpy as np

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sim2real_implementation.constants import DEFAULT_YOLO_WEIGHTS, OCCUPANCY_SIZE, TOF_CUTOFF
from sim2real_implementation.hardware import FakeHardware, Hardware, TonyPiHardware
from sim2real_implementation.observations import gravity_from_imu, load_imu_remap, scale_range
from sim2real_implementation.occupancy_real import OccupancyEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Probe TonyPi sensors at ~5 Hz.")
    parser.add_argument("--backend", type=str, choices=["tonypi", "fake"], default="tonypi")
    parser.add_argument("--rate", type=float, default=5.0, help="Sampling rate in Hz (default: 5.0)")
    parser.add_argument("--iterations", type=int, default=None, help="Stop after N iterations (default: infinite)")
    parser.add_argument("--save-frame", type=str, default=None, help="Save latest camera BGR frame to path and exit")
    parser.add_argument("--save-occ", type=str, default=None, help="Save latest occupancy mask to path and exit")
    parser.add_argument("--occupancy", type=str, choices=["color", "yolo", "zeros"], default="color")
    parser.add_argument("--yolo-weights", type=str, default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--hsv-lo", type=int, nargs=3, default=[0, 80, 80])
    parser.add_argument("--hsv-hi", type=int, nargs=3, default=[12, 255, 255])
    parser.add_argument("--allow-missing-tof", action="store_true", default=False, help="Allow running without ToF sensor")
    parser.add_argument("--yolo-keep-all", action="store_true", default=False, help="YOLO: keep all COCO classes")
    parser.add_argument("--tune-hsv", action="store_true", help="Open interactive HSV tuner window (needs DISPLAY)")
    return parser.parse_args()


def interactive_hsv_tuner(hw: Hardware, initial_lo: list[int], initial_hi: list[int]):
    """Interactive GUI trackbar tool to tune HSV thresholds."""
    if not os.environ.get("DISPLAY"):
        print("[!] No DISPLAY environment variable set; cannot open GUI window.", file=sys.stderr)
        print(f"Current default HSV: LO={initial_lo}, HI={initial_hi}")
        return

    window_name = "TonyPi HSV Tuner (press q to save & exit)"
    cv2.namedWindow(window_name)

    def nothing(x):
        pass

    cv2.createTrackbar("H_min", window_name, initial_lo[0], 179, nothing)
    cv2.createTrackbar("S_min", window_name, initial_lo[1], 255, nothing)
    cv2.createTrackbar("V_min", window_name, initial_lo[2], 255, nothing)
    cv2.createTrackbar("H_max", window_name, initial_hi[0], 179, nothing)
    cv2.createTrackbar("S_max", window_name, initial_hi[1], 255, nothing)
    cv2.createTrackbar("V_max", window_name, initial_hi[2], 255, nothing)

    print("Opening HSV tuner. Adjust trackbars. Press 'q' in the window to finish.")
    while True:
        frame = hw.read_bgr()
        if frame is None:
            time.sleep(0.05)
            continue

        h_min = cv2.getTrackbarPos("H_min", window_name)
        s_min = cv2.getTrackbarPos("S_min", window_name)
        v_min = cv2.getTrackbarPos("V_min", window_name)
        h_max = cv2.getTrackbarPos("H_max", window_name)
        s_max = cv2.getTrackbarPos("S_max", window_name)
        v_max = cv2.getTrackbarPos("V_max", window_name)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        from sim2real_implementation.occupancy_real import hsv_in_range
        mask = hsv_in_range(hsv, [h_min, s_min, v_min], [h_max, s_max, v_max])
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        res = cv2.bitwise_and(frame, frame, mask=mask)

        small = cv2.resize(mask, (OCCUPANCY_SIZE, OCCUPANCY_SIZE), interpolation=cv2.INTER_AREA)
        occ = (small > 255 * 0.3).astype(np.uint8) * 255
        occ_display = cv2.resize(occ, (200, 200), interpolation=cv2.INTER_NEAREST)
        occ_bgr = cv2.cvtColor(occ_display, cv2.COLOR_GRAY2BGR)

        combined = np.hstack([cv2.resize(res, (320, 240)), cv2.resize(occ_bgr, (320, 240))])
        cv2.imshow(window_name, combined)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            print(f"\nFinal Tuned HSV: LO=[{h_min}, {s_min}, {v_min}], HI=[{h_max}, {s_max}, {v_max}]")
            break

    cv2.destroyAllWindows()


def run_probe():
    args = parse_args()
    print("=" * 70)
    print(f"TonyPi Sensor Probe (Backend: {args.backend})")
    print("=" * 70)

    if args.backend == "tonypi":
        hw: Hardware = TonyPiHardware(
            dry_run=True,
            occupancy_mode=args.occupancy,
            allow_missing_tof=args.allow_missing_tof,
        )
    else:
        dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        hw = FakeHardware(bgr=dummy_bgr, tof_m=1.2, sonar_m=5.0)

    hw.start()
    hw.center_head()
    R = load_imu_remap()

    occupancy = OccupancyEncoder(
        mode=args.occupancy,
        yolo_weights=args.yolo_weights,
        hsv_lo=args.hsv_lo,
        hsv_hi=args.hsv_hi,
        keep_all=args.yolo_keep_all,
    )

    if args.tune_hsv:
        try:
            interactive_hsv_tuner(hw, args.hsv_lo, args.hsv_hi)
        finally:
            hw.stop()
        return

    # Single-shot save frame / save occ mode
    if args.save_frame or args.save_occ:
        time.sleep(0.5)  # allow camera to capture a frame
        bgr = hw.read_bgr()
        if bgr is None:
            print("[!] Failed to capture frame from camera!", file=sys.stderr)
            hw.stop()
            sys.exit(1)

        if args.save_frame:
            cv2.imwrite(args.save_frame, bgr)
            print(f"[PASS] Saved frame ({bgr.shape[1]}x{bgr.shape[0]} BGR) to {args.save_frame}")

        if args.save_occ:
            t0 = time.time()
            occ = occupancy.encode(bgr)
            lat_ms = (time.time() - t0) * 1000.0
            occ_img = (occ[0] * 255.0).astype(np.uint8)
            cv2.imwrite(args.save_occ, occ_img)
            occ_mean = float(occ.mean())
            print(
                f"[PASS] Saved occupancy (64x64, mean={occ_mean:.4f}, latency={lat_ms:.1f}ms) to {args.save_occ}"
            )

        hw.stop()
        return

    print("Running continuous probe loop (Ctrl+C to stop). Format:")
    print("  [Step] ToF (raw/scaled) | Sonar | IMU Accel & Gyro | Gravity Unit Vec | Cam Frame | Latency")
    print("-" * 70)

    count = 0
    interval = 1.0 / max(0.1, args.rate)

    try:
        while True:
            t_start = time.time()

            # 1. ToF
            tof_m = hw.read_tof_m()
            tof_scaled = scale_range(tof_m, TOF_CUTOFF)

            # 2. Sonar
            sonar_m = hw.read_sonar_m()
            sonar_str = f"{sonar_m:4.2f}m" if sonar_m < 5.0 else "MISSING"

            # 3. IMU
            imu_data = hw.read_imu()
            if imu_data is not None:
                accel, gyro = imu_data
                g_sim = gravity_from_imu(accel, R)
                imu_str = f"A:[{accel[0]:+5.2f},{accel[1]:+5.2f},{accel[2]:+5.2f}] G:[{gyro[0]:+4.1f},{gyro[1]:+4.1f},{gyro[2]:+4.1f}]"
                g_str = f"g:[{g_sim[0]:+4.2f},{g_sim[1]:+4.2f},{g_sim[2]:+4.2f}]"
            else:
                imu_str = "IMU: NONE"
                g_str = "g: [ 0.0, 0.0,-1.0]"

            # 4. Camera & Occupancy
            bgr = hw.read_bgr()
            cam_str = f"{bgr.shape[1]}x{bgr.shape[0]}" if bgr is not None else "NO_FRAME"

            t0 = time.time()
            occ = occupancy.encode(bgr)
            occ_lat = (time.time() - t0) * 1000.0

            total_lat = (time.time() - t_start) * 1000.0

            print(
                f"[{count:04d}] ToF:{tof_m:4.2f}m ({tof_scaled:4.2f}) | Sonar:{sonar_str:>7s} | {imu_str} | {g_str} | Cam:{cam_str} | Occ:{float(occ.mean()):4.2f} ({occ_lat:3.0f}ms)"
            )

            count += 1
            if args.iterations is not None and count >= args.iterations:
                break

            dt = time.time() - t_start
            sleep_time = interval - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nProbe stopped by user.")
    finally:
        hw.stop()


if __name__ == "__main__":
    run_probe()
