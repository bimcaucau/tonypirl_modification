"""Walk to a relative goal with the nav policy and live semantic occupancy.

The robot is the origin at the moment you start. +x is forward, +y is left,
matching the sim body frame. Each step reads the camera, builds the semantic
occupancy mask, and plays one action-group clip.

Examples (on the robot, after `sudo systemctl stop tonypi`):

  # 0.8 m straight ahead
  python3 -m sim2real_implementation.go_to --dist 0.8 --angle-deg 0

  # 0.6 m, 30 degrees to the robot's left
  python3 -m sim2real_implementation.go_to --dist 0.6 --angle-deg 30

  # same frame as --goal in deploy_nav / the sim
  python3 -m sim2real_implementation.go_to --xy 0.6 -0.2

  # laptop, no motion
  python3 -m sim2real_implementation.go_to --dist 0.4 --backend fake --dry-run --max-steps 3
"""
from __future__ import annotations

import argparse
import sys

from sim2real_implementation.constants import DEFAULT_YOLO_WEIGHTS
from sim2real_implementation.goal import body_goal_from_polar


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Navigate to a relative goal (robot at origin, +x forward, +y left)."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--xy", type=float, nargs=2, metavar=("X", "Y"), help="Body-frame metres")
    g.add_argument("--dist", type=float, help="Distance to the goal in metres")
    p.add_argument(
        "--angle-deg",
        type=float,
        default=0.0,
        help="Heading of --dist. 0 = forward, positive = left (degrees)",
    )
    p.add_argument("--backend", choices=["tonypi", "fake"], default="tonypi")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--arrive", type=float, default=0.25)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--max-seconds", type=float, default=120.0)
    p.add_argument("--occupancy", choices=["yolo", "color", "zeros"], default="yolo")
    p.add_argument("--yolo-weights", type=str, default=DEFAULT_YOLO_WEIGHTS)
    p.add_argument("--dry-run", action="store_true", help="Sense and log, do not play clips")
    p.add_argument("--allow-missing-tof", action="store_true")
    p.add_argument("--log-dir", type=str, default="sim2real_implementation/logs/go_to")
    p.add_argument(
        "--http-port",
        type=int,
        default=8080,
        help="Stream the 3-pane view here while walking (0 disables)",
    )
    p.add_argument("--no-view", action="store_true", help="Do not open the 3-pane stream")
    p.add_argument("--no-ground-look", action="store_true", help="Do not look down before a step")
    p.add_argument("--no-look-sweep", action="store_true", help="Look down at center only")
    args = p.parse_args(argv)
    if args.xy is None and args.dist is None:
        p.error("pass --xy X Y or --dist METRES")
    if args.xy is not None and args.angle_deg != 0.0:
        p.error("--angle-deg is only used with --dist")
    if args.dist is not None and args.dist < 0:
        p.error("--dist must be >= 0")
    return args


def goal_xy(args: argparse.Namespace) -> tuple[float, float]:
    if args.xy is not None:
        return (float(args.xy[0]), float(args.xy[1]))
    return body_goal_from_polar(args.dist, args.angle_deg)


def deploy_argv(args: argparse.Namespace, xy: tuple[float, float]) -> list[str]:
    argv = [
        "deploy_nav",
        "--goal",
        f"{xy[0]:.6f}",
        f"{xy[1]:.6f}",
        "--goal-mode",
        "deadreckon",
        "--backend",
        args.backend,
        "--occupancy",
        args.occupancy,
        "--yolo-weights",
        args.yolo_weights,
        "--arrive",
        str(args.arrive),
        "--max-steps",
        str(args.max_steps),
        "--max-seconds",
        str(args.max_seconds),
        "--log-dir",
        args.log_dir,
    ]
    if args.model:
        argv.extend(["--model", args.model])
    if args.dry_run:
        argv.append("--dry-run")
    if args.allow_missing_tof:
        argv.append("--allow-missing-tof")
    if args.no_ground_look:
        argv.append("--no-ground-look")
    if args.no_look_sweep:
        argv.append("--no-look-sweep")
    port = 0 if args.no_view else args.http_port
    argv.extend(["--http-port", str(port)])
    return argv


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    xy = goal_xy(args)
    print(
        f"goal body xy=({xy[0]:+.3f}, {xy[1]:+.3f}) m  "
        f"occupancy={args.occupancy} weights={args.yolo_weights} backend={args.backend}"
    )
    port = 0 if args.no_view else args.http_port
    if port:
        print(f"3-pane view: http://<ROBOT_IP>:{port}/  (camera | segmentation | obstacle/road)")
    print("+x forward, +y left. Ctrl+C stops and stands.")
    sys.argv = deploy_argv(args, xy)
    from sim2real_implementation.deploy_nav import main as deploy_main

    deploy_main()


if __name__ == "__main__":
    main()
