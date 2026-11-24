"""Main deployment entry point for TonyPi high-level navigation.

Can be run on laptop with --backend fake, or on robot with --backend tonypi.
Usage:
  python -m sim2real_implementation.deploy_nav \
    --model models_ckpt/new_expert_implementation/tonypi_final_model.zip \
    --backend fake \
    --goal 2.0 -0.5 \
    --goal-mode deadreckon \
    --occupancy color \
    --dry-run \
    --max-steps 20 \
    --arrive 0.20 \
    --log-dir sim2real_implementation/logs/run1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
import numpy as np
import yaml

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sim2real_implementation.constants import (
    ACTION_NAMES,
    ACTION_TO_ID,
    DEFAULT_YOLO_WEIGHTS,
    OCCUPANCY_SIZE,
    SIM_PRIMITIVES,
)
from sim2real_implementation.goal import GoalSource
from sim2real_implementation.ground_look import MOTION_ACTIONS, GroundLook, fused_tof
from sim2real_implementation.hardware import FakeHardware, Hardware, TonyPiHardware
from sim2real_implementation.observations import build_obs, gravity_from_imu, load_imu_remap
from sim2real_implementation.occupancy_real import OccupancyEncoder
from sim2real_implementation.policy import act, load_nav_policy
from sim2real_implementation.safety import Safety

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("deploy_nav")


def load_yaml_config(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("Error loading config %s: %s", path, e)
    return {}


def resolve_model_path(path: str) -> str:
    """Resolve model checkpoint path from multiple common candidate locations."""
    candidates = [
        path,
        os.path.join(REPO_ROOT, path),
        os.path.join(os.getcwd(), path),
        os.path.basename(path),
        os.path.join("models", os.path.basename(path)),
        os.path.join("/home/pi/nav_deploy", os.path.basename(path)),
        os.path.join("/home/pi/nav_deploy", path),
        os.path.join(REPO_ROOT, "models_ckpt/new_expert_implementation", os.path.basename(path)),
    ]
    for c in candidates:
        if c and os.path.exists(c) and os.path.isfile(c):
            return os.path.abspath(c)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy TonyPi navigation policy.")
    parser.add_argument("--config", type=str, default="sim2real_implementation/configs/deploy.yaml")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--backend", type=str, choices=["fake", "tonypi"], default=None)
    parser.add_argument("--goal", type=float, nargs=2, default=None)
    parser.add_argument("--goal-mode", type=str, choices=["deadreckon", "tag", "fixed-body"], default=None)
    parser.add_argument("--occupancy", type=str, choices=["yolo", "color", "zeros", "file"], default=None)
    parser.add_argument("--yolo-weights", type=str, default=None)
    parser.add_argument("--yolo-imgsz", type=int, default=None)
    parser.add_argument("--yolo-conf", type=float, default=None)
    parser.add_argument("--yolo-keep-all", action="store_true", default=None, help="YOLO: keep all COCO classes")
    parser.add_argument("--hsv-lo", type=int, nargs=3, default=None)
    parser.add_argument("--hsv-hi", type=int, nargs=3, default=None)
    parser.add_argument("--arrive", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--recover-fall", action="store_true", default=None)
    parser.add_argument("--fall-deg", type=float, default=None)
    parser.add_argument("--tof-guard", type=float, default=None)
    parser.add_argument("--allow-missing-tof", action="store_true", default=None, help="Allow running without ToF sensor")
    parser.add_argument("--dry-run", action="store_true", default=None)
    parser.add_argument("--dry-run-stand", action="store_true", default=False)
    parser.add_argument("--no-safety", action="store_true", default=False)
    parser.add_argument("--log-dir", type=str, default="sim2real_implementation/logs/run_latest")
    parser.add_argument("--save-debug", action="store_true", default=False)
    parser.add_argument(
        "--http-port",
        type=int,
        default=0,
        help="If >0, stream the 3-pane camera|seg|binary view at http://0.0.0.0:PORT/ while walking",
    )
    parser.add_argument("--tag-id", type=int, default=0)
    parser.add_argument("--tag-size", type=float, default=0.08)
    parser.add_argument("--tag-lost", type=str, choices=["hold", "stand"], default="hold")
    parser.add_argument("--script", type=str, default=None, help="Comma-separated clips to run without policy, e.g. stand,go_forward,stand")
    parser.add_argument("--no-ground-look", action="store_true", help="Do not look down before a step")
    parser.add_argument("--no-look-sweep", action="store_true", help="Look down at center only, skip left/right")
    parser.add_argument("--look-pulse", type=int, default=None, help="Head pitch pulse for the ground look (1500 is level, lower looks down)")
    parser.add_argument("--look-yaw-sign", type=int, choices=[-1, 1], default=None, help="Flip if a higher yaw pulse looks right instead of left")

    args = parser.parse_args()

    # Load defaults from deploy.yaml
    cfg = load_yaml_config(args.config)

    # CLI args override YAML config
    raw_model = args.model or cfg.get("model", "models_ckpt/new_expert_implementation/tonypi_final_model.zip")
    final_args = {
        "model": resolve_model_path(raw_model),
        "backend": args.backend or cfg.get("backend", "tonypi"),
        "goal": args.goal if args.goal is not None else cfg.get("goal", [2.0, -0.5]),
        "goal_mode": args.goal_mode or cfg.get("goal_mode", "deadreckon"),
        "occupancy": args.occupancy or cfg.get("occupancy", "color"),
        "yolo_weights": args.yolo_weights or cfg.get("yolo_weights", DEFAULT_YOLO_WEIGHTS),
        "yolo_imgsz": args.yolo_imgsz or cfg.get("yolo_imgsz", 320),
        "yolo_conf": args.yolo_conf or cfg.get("yolo_conf", 0.25),
        "yolo_keep_all": args.yolo_keep_all if args.yolo_keep_all is not None else cfg.get("yolo_keep_all", False),
        "hsv_lo": args.hsv_lo or cfg.get("hsv_lo", [0, 80, 80]),
        "hsv_hi": args.hsv_hi or cfg.get("hsv_hi", [12, 255, 255]),
        "arrive": args.arrive or cfg.get("arrive", 0.20),
        "max_steps": args.max_steps or cfg.get("max_steps", 20),
        "max_seconds": args.max_seconds or cfg.get("max_seconds", 60.0),
        "recover_fall": args.recover_fall if args.recover_fall is not None else cfg.get("recover_fall", False),
        "fall_deg": args.fall_deg or cfg.get("fall_deg", 45.0),
        "tof_guard": args.tof_guard if args.tof_guard is not None else cfg.get("tof_guard", None),
        "allow_missing_tof": args.allow_missing_tof if args.allow_missing_tof is not None else cfg.get("allow_missing_tof", False),
        "dry_run": args.dry_run if args.dry_run is not None else cfg.get("dry_run", False),
        "dry_run_stand": args.dry_run_stand,
        "no_safety": args.no_safety,
        "log_dir": args.log_dir,
        "save_debug": args.save_debug,
        "http_port": args.http_port,
        "tag_id": args.tag_id,
        "tag_size": args.tag_size,
        "tag_lost": args.tag_lost,
        "script": args.script,
        "ground_look": False if args.no_ground_look else bool(cfg.get("ground_look", True)),
        "look_sweep": False if args.no_look_sweep else bool(cfg.get("look_sweep", True)),
        "look_pulse": args.look_pulse if args.look_pulse is not None else int(cfg.get("look_pulse", 900)),
        "look_yaw_sign": args.look_yaw_sign if args.look_yaw_sign is not None else int(cfg.get("look_yaw_sign", 1)),
        "look_yaw_amp": int(cfg.get("look_yaw_amp", 250)),
        "look_pitch_sign": int(cfg.get("look_pitch_sign", -1)),
    }
    return argparse.Namespace(**final_args)


def main():
    args = parse_args()
    logger.info("Starting TonyPi Nav Deploy with configuration:")
    logger.info("  Backend: %s | Model: %s", args.backend, args.model)
    logger.info("  Goal: %s (%s) | Occupancy: %s | Dry-run: %s", args.goal, args.goal_mode, args.occupancy, args.dry_run)

    # Prepare log dir
    os.makedirs(args.log_dir, exist_ok=True)
    jsonl_path = os.path.join(args.log_dir, "nav_log.jsonl")
    jsonl_file = open(jsonl_path, "a", encoding="utf-8")

    # 1. Pre-validate model before initializing or moving hardware
    model = None
    script_actions = None
    if args.script:
        script_actions = [s.strip() for s in args.script.split(",") if s.strip()]
        logger.info("Running script mode with actions: %s", script_actions)
    else:
        if not os.path.exists(args.model):
            raise FileNotFoundError(
                f"Model checkpoint not found: '{args.model}'.\n"
                "Refusing to start hardware. Please pass --model with a valid .zip path."
            )
        logger.info("Pre-validating and loading policy checkpoint: %s", args.model)
        model = load_nav_policy(args.model, device="cpu")

    # 2. Initialize Hardware (fail-fast on sensor stream if needed)
    if args.backend == "tonypi":
        hw: Hardware = TonyPiHardware(
            dry_run=args.dry_run,
            occupancy_mode=args.occupancy,
            allow_missing_tof=args.allow_missing_tof,
        )
    else:
        hw = FakeHardware()

    # 3. Initialize Goal Source
    goal = GoalSource(
        mode=args.goal_mode,
        initial_goal=tuple(args.goal),
        tag_id=args.tag_id,
        tag_size=args.tag_size,
        tag_lost=args.tag_lost,
    )

    # 4. Initialize Safety
    safety = Safety(
        max_steps=args.max_steps,
        max_seconds=args.max_seconds,
        arrive_radius=args.arrive,
        fall_deg=args.fall_deg,
        recover_fall=args.recover_fall,
        tof_guard=args.tof_guard,
        no_safety=args.no_safety,
    )

    # 5. Initialize Occupancy Encoder
    occupancy = OccupancyEncoder(
        mode=args.occupancy,
        yolo_weights=args.yolo_weights,
        yolo_conf=args.yolo_conf,
        yolo_imgsz=args.yolo_imgsz,
        hsv_lo=args.hsv_lo,
        hsv_hi=args.hsv_hi,
        keep_all=args.yolo_keep_all,
    )

    # Load IMU remap matrix and check calibration status
    R, is_calibrated = load_imu_remap(return_status=True)
    if not is_calibrated:
        logger.warning(
            "[WARNING] imu.yaml is UNCALIBRATED (calibrated: false). Run calibrate_imu.py on the robot!"
        )

    # Start hardware
    hw.start()
    hw.center_head()

    live = None
    hud_state = {"text": "starting"}
    if args.http_port:
        from sim2real_implementation.view_segmentation import LiveTriple

        live = LiveTriple(occupancy, args.http_port, args.log_dir)
        live.start(hw.read_bgr, lambda: hud_state["text"])
        logger.info("3-pane view at http://0.0.0.0:%s/ (camera | seg | binary)", args.http_port)
    if args.dry_run and args.dry_run_stand:
        hw.play("stand")

    goal.reset(tuple(args.goal))
    prev_action_idx = 0
    start_time = time.time()
    exit_code = 0
    # Look-down is a veto around the policy, not a new observation. Scripted
    # clips are a calibration tool and must not grow a head motion of their own.
    look = GroundLook(
        enabled=bool(args.ground_look) and script_actions is None,
        look_pulse=args.look_pulse,
        sweep=args.look_sweep,
        yaw_amp=args.look_yaw_amp,
        yaw_sign=args.look_yaw_sign,
        pitch_sign=args.look_pitch_sign,
        mtx=getattr(goal, "mtx", None),
    )
    if look.enabled:
        logger.info(
            "Ground look on: pitch pulse %d (%s sweep), yaw sign %+d",
            args.look_pulse,
            "3-pose" if args.look_sweep else "center",
            args.look_yaw_sign,
        )
    terminal_reasons = {"estop", "fall", "fall_recover", "arrived", "max_steps", "max_seconds"}

    try:
        step_limit = len(script_actions) if script_actions else args.max_steps
        for step in range(step_limit):
            step_start = time.time()
            elapsed_s = step_start - start_time

            if not is_calibrated and (step % 5 == 0):
                logger.warning(
                    "[WARNING] Running with UNCALIBRATED IMU (step %d)! Run calibrate_imu.py on the robot.",
                    step,
                )

            # Sense. The live view thread owns the camera when --http-port is set.
            if live is not None:
                views = live.wait_views(timeout=8.0)
                bgr = None if views is None else views["camera"]
            else:
                views = None
                bgr = hw.read_bgr()
            if args.goal_mode == "tag":
                goal.update_vision(bgr)

            dist, heading, gx, gy = goal.current()
            tof_m = hw.read_tof_m()
            sonar_m = hw.read_sonar_m()
            imu_data = hw.read_imu()
            gravity = gravity_from_imu(imu_data[0], R) if imu_data is not None else np.array([0.0, 0.0, -1.0], dtype=np.float32)

            # Occupancy (reuse the frame the 3-pane view just segmented)
            if views is None:
                occ = occupancy.encode(bgr)
            else:
                occ = views["occupancy"]

            # Build observation
            obs = build_obs(
                dist=dist,
                heading=heading,
                tof_m=tof_m,
                sonar_m=sonar_m,
                gravity=gravity,
                prev_action=prev_action_idx,
                occupancy=occ,
            )

            # Choose Action
            if script_actions is not None:
                action_name = script_actions[step]
                action_idx = ACTION_TO_ID.get(action_name, 0)
            elif goal.should_force_stand():
                action_idx = 0
                action_name = "stand"
                logger.info("[TAG LOST] Tag not visible and --tag-lost is 'stand'. Forcing action 'stand'.")
            else:
                action_idx = act(model, obs, deterministic=True)
                action_name = ACTION_NAMES[action_idx]

            # Filter through Safety Layer
            filtered_idx, name_to_play, reason = safety.filter_action(
                action_idx=action_idx,
                hw=hw,
                dist=dist,
                step=step,
                elapsed_s=elapsed_s,
                tof_m=tof_m,
            )
            if reason is None and goal.should_force_stand():
                reason = "tag_lost"

            look_fields: dict = {}
            if look.enabled and reason not in terminal_reasons and name_to_play in MOTION_ACTIONS:
                surveyed = look.survey(hw, occupancy.encode_views, name_to_play)
                look_fields = surveyed.log_fields()
                if live is not None:
                    # The stream grabbed during the look-down. Don't feed that
                    # frame to the policy; it was trained with the head level.
                    live.invalidate()
                if surveyed.aborted:
                    safety.estop = True
                    filtered_idx, name_to_play, reason = 0, "stand", "estop"
                else:
                    if surveyed.odom_dx is not None and goal.correct_last_translation(surveyed.odom_dx, surveyed.odom_dy):
                        dist, heading, gx, gy = goal.current()
                        logger.info(
                            "Floor shift corrected last step to dx=%+.3f dy=%+.3f (response %.2f)",
                            surveyed.odom_dx,
                            surveyed.odom_dy,
                            surveyed.odom_response,
                        )
                    new_tof = fused_tof(tof_m, surveyed.forward_range_m)
                    look_fields["tof_fused"] = round(new_tof, 3)
                    if (
                        reason is None
                        and script_actions is None
                        and not goal.should_force_stand()
                        and (abs(new_tof - tof_m) > 0.05 or surveyed.odom_dx is not None)
                    ):
                        # Policy was trained on a chest-height beam. A low object the
                        # beam missed shows up here as a shorter range, one re-query.
                        obs = build_obs(
                            dist=dist,
                            heading=heading,
                            tof_m=new_tof,
                            sonar_m=sonar_m,
                            gravity=gravity,
                            prev_action=prev_action_idx,
                            occupancy=occ,
                        )
                        action_idx = act(model, obs, deterministic=True)
                        action_name = ACTION_NAMES[action_idx]
                        filtered_idx, name_to_play, reason = safety.filter_action(
                            action_idx=action_idx,
                            hw=hw,
                            dist=dist,
                            step=step,
                            elapsed_s=elapsed_s,
                            tof_m=new_tof,
                        )
                    if reason is None:
                        safe_name, veto_reason = surveyed.avoid(name_to_play)
                        if veto_reason is not None:
                            logger.info(
                                "[GROUND LOOK] %s footprint blocked; playing %s instead (clearer side %s)",
                                name_to_play,
                                safe_name,
                                surveyed.clearer_side,
                            )
                            name_to_play = safe_name
                            filtered_idx = ACTION_TO_ID.get(safe_name, 0)
                            reason = veto_reason

            latency_ms = (time.time() - step_start) * 1000.0

            # Step Logging
            occ_mean = float(occ.mean())
            log_entry = {
                "t": round(elapsed_s, 3),
                "step": step,
                "action": name_to_play,
                "action_id": filtered_idx,
                "raw_action": action_name,
                "safety_reason": reason,
                "vec": [round(float(v), 4) for v in obs["vec"]],
                "tof_m": round(tof_m, 3),
                "sonar_m": round(sonar_m, 3),
                "dist": round(dist, 3),
                "heading": round(heading, 4),
                "gx": round(gx, 3),
                "gy": round(gy, 3),
                "occ_mean": round(occ_mean, 4),
                "latency_ms": round(latency_ms, 1),
            }
            log_entry.update(look_fields)
            jsonl_file.write(json.dumps(log_entry) + "\n")
            jsonl_file.flush()

            logger.info(
                "Step %2d | Act: %-15s | Goal: dist=%.2fm head=%+5.1f° | ToF: %.2fm | Lat: %4.0fms%s",
                step,
                name_to_play,
                dist,
                np.rad2deg(heading),
                tof_m,
                latency_ms,
                f" [{reason}]" if reason else "",
            )

            # Save debug frames if requested
            if args.save_debug and bgr is not None:
                import cv2
                cv2.imwrite(os.path.join(args.log_dir, f"frame_{step:03d}.jpg"), bgr)
                occ_vis = (occ[0] * 255.0).astype(np.uint8)
                cv2.imwrite(os.path.join(args.log_dir, f"occ_{step:03d}.png"), occ_vis)

            # Check termination from safety reason
            if reason in ("estop", "fall", "max_steps", "max_seconds", "arrived"):
                if not args.dry_run:
                    hw.play("stand")
                if reason == "arrived":
                    logger.info("[SUCCESS] Arrived at goal (dist: %.2fm < %.2fm)!", dist, args.arrive)
                    exit_code = 0
                elif reason == "estop":
                    logger.warning("[STOP] E-stop triggered.")
                    exit_code = 2
                elif reason == "fall":
                    logger.critical("[STOP] Robot fell over.")
                    exit_code = 3
                elif reason in ("max_steps", "max_seconds"):
                    is_arrived = dist < args.arrive
                    logger.info("[STOP] Reached budget limit (%s). Arrived: %s", reason, is_arrived)
                    exit_code = 0 if is_arrived else 1
                look.note_executed("stand")
                break

            ground_txt = ""
            if look_fields.get("ground_range_m") is not None:
                ground_txt = f"  ground={look_fields['ground_range_m']:.2f}m"
            hud_state["text"] = (
                f"step {step} {name_to_play}  dist={dist:.2f}m head={np.rad2deg(heading):+.0f}deg  "
                f"tof={tof_m:.2f}m occ={occ_mean:.3f}{ground_txt}"
            )

            # Execute action (view thread keeps refreshing the mosaic during the clip)
            if not args.dry_run:
                hw.play(name_to_play)

            # Update goal dead reckoning
            goal.on_action(name_to_play)
            look.note_executed(name_to_play)
            prev_action_idx = filtered_idx

        # If loop finished normally without break:
        if exit_code == 0 and not safety.estop:
            dist, _, _, _ = goal.current()
            if dist < args.arrive:
                logger.info("[SUCCESS] Arrived at goal!")
                exit_code = 0
            elif script_actions:
                logger.info("[SCRIPT] Completed scripted actions.")
                exit_code = 0
            else:
                exit_code = 1

    except KeyboardInterrupt:
        logger.warning("\n[KEYBOARD INTERRUPT] Ctrl+C captured. Stopping motion...")
        hw.stop_motion()
        if not args.dry_run:
            hw.play("stand")
        exit_code = 2

    finally:
        if live is not None:
            live.stop()
        jsonl_file.close()
        hw.stop()
        logger.info("Navigation session ended. Log saved to %s (exit code: %d)", jsonl_path, exit_code)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
