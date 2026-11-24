#!/usr/bin/env python3
"""
TonyPi Humanoid MuJoCo Visualization & Motion Analysis Script.

Enables viewing and evaluating the TonyPi robot's locomotion in MuJoCo:
1. Live Interactive 3D Viewer (GLFW passive viewer with real-time sync & slow-mo).
2. Automatic Model & Checkpoint Loading (latest run, best_model.zip, or specific step).
3. Checkpoint Explorer (--list-checkpoints to view all saved weights).
4. Offscreen Video & GIF Recording (Tracking, Side, Front, and Multi-View split screen).
5. Telemetry HUD & Diagnostics Plotting (Trajectory, Cadence, Velocity, Stability).

Usage Examples:
  # 1. Interactive 3D MuJoCo Window (auto-loads latest best model):
  python tonypi_rl/visualize_robot.py --gui

  # 2. View specific checkpoint in 3D:
  python tonypi_rl/visualize_robot.py --model tonypi_rl/models/tonypi_v0_20260912_150109/best_model.zip --gui

  # 3. View in slow-motion (0.5x) or continuous looping:
  python tonypi_rl/visualize_robot.py --gui --speed 0.5 --loop

  # 4. List all available checkpoints:
  python tonypi_rl/visualize_robot.py --list-checkpoints

  # 5. Record MP4 video (ideal for remote headless server or archiving):
  python tonypi_rl/visualize_robot.py --record walk.mp4 --overlay

  # 6. Record animated GIF:
  python tonypi_rl/visualize_robot.py --record walk.gif --overlay
"""

import os
import sys
import glob
import time
import argparse
import numpy as np

# Ensure tonypi_rl and root are in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import mujoco
from tonypi_rl.tonypi_env import TonyPiEnv


def list_available_checkpoints(models_dir: str):
    """List all available checkpoints and training runs found on disk."""
    if not os.path.exists(models_dir):
        print(f"[!] No models directory found at: {models_dir}")
        return

    runs = sorted(glob.glob(os.path.join(models_dir, "tonypi_*")))
    if not runs:
        print(f"[!] No training runs found in: {models_dir}")
        return

    print("=" * 76)
    print(f"  Available Trained Runs & Checkpoints ({models_dir})")
    print("=" * 76)
    for run in runs:
        run_name = os.path.basename(run)
        print(f"\n📂 Run: {run_name}")

        # Check for best_model.zip
        best_p = os.path.join(run, "best_model.zip")
        if os.path.exists(best_p):
            size_mb = os.path.getsize(best_p) / (1024 * 1024)
            print(f"   ★ best_model.zip ({size_mb:.2f} MB)")

        # Check for final model
        final_p = os.path.join(run, "tonypi_final_model.zip")
        if os.path.exists(final_p):
            size_mb = os.path.getsize(final_p) / (1024 * 1024)
            print(f"   ✓ tonypi_final_model.zip ({size_mb:.2f} MB)")

        # Check for intermediate checkpoints
        checkpoints = sorted(glob.glob(os.path.join(run, "tonypi_checkpoint_*_steps.zip")))
        for cp in checkpoints:
            cp_name = os.path.basename(cp)
            size_mb = os.path.getsize(cp) / (1024 * 1024)
            print(f"   • {cp_name} ({size_mb:.2f} MB)")

    print("\n" + "=" * 76)
    print("To visualize a specific checkpoint, run:")
    print("  python tonypi_rl/visualize_robot.py --model <path_to_checkpoint.zip> --gui")
    print("=" * 76)


def find_latest_model(models_dir: str) -> str | None:
    """Find the most recent trained checkpoint or best_model in models_dir."""
    if not os.path.exists(models_dir):
        return None

    run_folders = sorted(glob.glob(os.path.join(models_dir, "tonypi_v*")))
    if not run_folders:
        zip_files = sorted(glob.glob(os.path.join(models_dir, "**", "*.zip"), recursive=True))
        return zip_files[-1] if zip_files else None

    for run_dir in reversed(run_folders):
        best_model = os.path.join(run_dir, "best_model.zip")
        if os.path.exists(best_model):
            return best_model
        final_model = os.path.join(run_dir, "tonypi_final_model.zip")
        if os.path.exists(final_model):
            return final_model
        checkpoints = sorted(glob.glob(os.path.join(run_dir, "tonypi_checkpoint_*_steps.zip")))
        if checkpoints:
            return checkpoints[-1]

    return None


def setup_tracking_camera(model, data, cam_type: str = "tracking") -> mujoco.MjvCamera:
    """Configure a programmatically tracked camera following the robot COM/pelvis."""
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)

    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = pelvis_id

    if cam_type == "side":
        cam.distance = 1.05
        cam.elevation = -12.0
        cam.azimuth = 90.0
    elif cam_type == "front":
        cam.distance = 1.10
        cam.elevation = -10.0
        cam.azimuth = 180.0
    elif cam_type == "top":
        cam.distance = 1.40
        cam.elevation = -88.0
        cam.azimuth = 90.0
    else:  # default 'tracking' / 3/4 isometric perspective
        cam.distance = 1.15
        cam.elevation = -18.0
        cam.azimuth = 135.0

    return cam


def draw_telemetry_overlay(frame: np.ndarray, telemetry: dict) -> np.ndarray:
    """Draw a clean semi-transparent HUD overlay on video frame."""
    try:
        import cv2
    except ImportError:
        return frame

    h, w, _ = frame.shape
    img = frame.copy()

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 64), (18, 22, 28), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    step_str = f"Ep {telemetry.get('episode', 1)} | Step: {telemetry.get('step', 0):04d} | Time: {telemetry.get('sim_time', 0.0):.2f}s"
    loco_str = f"Vx: {telemetry.get('vx', 0.0):+.2f} m/s | Dist: {telemetry.get('dist_x', 0.0):.2f}m | Drift: {telemetry.get('drift_y', 0.0):+.2f}m | Height: {telemetry.get('height_z', 0.0):.3f}m"

    stance = telemetry.get("stance", "UNKNOWN")
    switches = telemetry.get("switches", 0)
    stance_str = f"Gait: [{stance}] | Switches: {switches}"

    cv2.putText(img, step_str, (16, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 225, 230), 1, cv2.LINE_AA)
    cv2.putText(img, loco_str, (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 220, 140), 1, cv2.LINE_AA)

    badge_color = (0, 190, 255) if "DOUBLE" in stance else ((80, 210, 100) if "LEFT" in stance else (255, 120, 80))
    cv2.putText(img, stance_str, (w - 290, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.50, badge_color, 1, cv2.LINE_AA)

    return img


def generate_performance_plots(history: dict, save_path: str = "walk_performance.png"):
    """Generate diagnostic multi-panel locomotion plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] matplotlib not installed. Skipping diagnostic plot generation.")
        return

    time_steps = history["time"]
    if not time_steps:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("TonyPi Humanoid Locomotion Performance Analysis", fontsize=14, fontweight="bold")

    # 1. 2D Trajectory (X vs Y path)
    ax1 = axes[0, 0]
    ax1.plot(history["pos_x"], history["pos_y"], color="royalblue", lw=2, label="Robot Pelvis Path")
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.7, label="Center Target Line")
    ax1.set_title("2D Walking Trajectory (Top-Down)")
    ax1.set_xlabel("Forward Distance X [m]")
    ax1.set_ylabel("Lateral Drift Y [m]")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Forward Velocity vs Time
    ax2 = axes[0, 1]
    ax2.plot(time_steps, history["vel_x"], color="forestgreen", lw=1.8, label="Forward Velocity (Vx)")
    ax2.plot(time_steps, history["vel_y"], color="crimson", lw=1.2, alpha=0.7, label="Lateral Velocity (Vy)")
    ax2.axhline(0.20, color="orange", linestyle=":", lw=1.5, label="Target Velocity (0.20 m/s)")
    ax2.set_title("Body Velocities Over Time")
    ax2.set_xlabel("Simulation Time [s]")
    ax2.set_ylabel("Velocity [m/s]")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Posture (Pelvis Height & Uprightness)
    ax3 = axes[1, 0]
    ax3.plot(time_steps, history["pos_z"], color="darkviolet", lw=1.8, label="Pelvis Height Z")
    ax3.axhline(0.262, color="gray", linestyle="--", alpha=0.7, label="Nominal Stand Height (0.262m)")
    ax3.set_title("Pelvis Height & Stability")
    ax3.set_xlabel("Simulation Time [s]")
    ax3.set_ylabel("Height [m]")
    ax3.set_ylim([0.15, 0.32])
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. Foot Contacts Timeline (Gait Diagram)
    ax4 = axes[1, 1]
    l_contact = np.array(history["l_force"]) > 2.0
    r_contact = np.array(history["r_force"]) > 2.0
    ax4.plot(time_steps, l_contact.astype(int) * 0.9 + 1.0, color="dodgerblue", lw=2, label="Left Foot Contact")
    ax4.plot(time_steps, r_contact.astype(int) * 0.9, color="darkorange", lw=2, label="Right Foot Contact")
    ax4.set_title("Gait Sequence & Foot Contact Timeline")
    ax4.set_xlabel("Simulation Time [s]")
    ax4.set_yticks([0.45, 1.45])
    ax4.set_yticklabels(["Right Stance", "Left Stance"])
    ax4.set_ylim([-0.2, 2.2])
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
    print(f"\n[+] Diagnostic performance plot saved to: {os.path.abspath(save_path)}")


def run_visualization():
    parser = argparse.ArgumentParser(
        description="Visualize and analyze TonyPi humanoid performance in MuJoCo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="latest",
                        help="Path to trained PPO .zip model, or 'latest' to auto-detect")
    parser.add_argument("--list-checkpoints", action="store_true",
                        help="List all saved models/checkpoints and exit")
    parser.add_argument("--config", type=str, default="tonypi_rl/configs/v0_straight_walk.yaml",
                        help="YAML config path")
    parser.add_argument("--scene", type=str, default=None,
                        help="Override path to scene.xml (default: tonypi_mujoco/scene.xml)")
    parser.add_argument("--policy", type=str, default="trained", choices=["trained", "zero", "sine", "random"],
                        help="Policy to execute if trained model is not used")
    parser.add_argument("--gui", action="store_true",
                        help="Launch interactive 3D MuJoCo viewer")
    parser.add_argument("--record", type=str, default=None,
                        help="Save motion to MP4 or GIF video (e.g. walk.mp4, gait.gif)")
    parser.add_argument("--camera", type=str, default="tracking",
                        choices=["tracking", "side", "front", "top", "multiview"],
                        help="Camera perspective for viewing/rendering")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier for GUI viewer (e.g. 0.5 for slow-motion)")
    parser.add_argument("--overlay", action="store_true", default=True,
                        help="Render telemetry HUD overlay on recorded video frames")
    parser.add_argument("--no-overlay", dest="overlay", action="store_false",
                        help="Disable telemetry HUD overlay")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to evaluate")
    parser.add_argument("--max-steps", type=int, default=600,
                        help="Maximum simulation steps per episode (~12 seconds at frame_skip=10)")
    parser.add_argument("--loop", action="store_true",
                        help="Continuously loop simulation in GUI mode")
    parser.add_argument("--plot", type=str, default=None,
                        help="Save locomotion diagnostics plot to image path (e.g. walk_analysis.png)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frame rate for recorded video / GIF")
    parser.add_argument("--resolution", type=str, default="960x720",
                        help="Video render resolution WIDTHxHEIGHT (e.g. 960x720, 1280x720)")
    args = parser.parse_args()

    models_base = os.path.join(parent_dir, "tonypi_rl", "models")
    if not os.path.exists(models_base):
        models_base = os.path.join(current_dir, "models")

    # Handle --list-checkpoints
    if args.list_checkpoints:
        list_available_checkpoints(models_base)
        return

    # Default to GUI mode if neither record nor plot is specified
    if not args.gui and not args.record and not args.plot:
        args.gui = True

    # Parse resolution
    try:
        width, height = map(int, args.resolution.split("x"))
    except ValueError:
        width, height = 960, 720

    # 1. Resolve Trained Model Checkpoint
    model_path = args.model
    if model_path == "latest":
        auto_found = find_latest_model(models_base)
        if auto_found:
            model_path = auto_found
            print(f"[+] Auto-detected latest trained model: {model_path}")
        else:
            print("[!] No trained models found in models directory. Falling back to policy mode.")
            model_path = None
    elif model_path and not os.path.exists(model_path):
        candidates = [
            os.path.join(parent_dir, model_path),
            os.path.join(current_dir, model_path),
            os.path.join(models_base, model_path),
        ]
        found = False
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                found = True
                break
        if not found:
            print(f"[!] Warning: Model path '{model_path}' not found. Falling back to policy mode: {args.policy}")
            model_path = None

    # 2. Resolve Config File
    config_path = args.config
    if not os.path.exists(config_path):
        for base in [parent_dir, current_dir]:
            c = os.path.join(base, config_path)
            if os.path.exists(c):
                config_path = c
                break

    env_kwargs = {}
    if os.path.exists(config_path):
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        env_kwargs.update(cfg.get("environment", {}))
        env_kwargs.update(cfg.get("rewards", {}))
        env_kwargs.pop("max_episode_steps", None)

    # Resolve scene XML path
    scene_xml = args.scene or env_kwargs.get("xml_path", "tonypi_mujoco/scene.xml")
    if not os.path.isabs(scene_xml):
        for base in [parent_dir, current_dir]:
            candidate = os.path.join(base, scene_xml)
            if os.path.exists(candidate):
                scene_xml = os.path.abspath(candidate)
                break
    env_kwargs["xml_path"] = scene_xml

    # Instantiate Environment
    env = TonyPiEnv(**env_kwargs)
    dt = env.model.opt.timestep * env.frame_skip

    # 3. Load Trained SB3 Policy
    rl_model = None
    if model_path and os.path.exists(model_path):
        from stable_baselines3 import PPO
        print(f"[+] Loading PPO Policy Checkpoint: {model_path}")
        rl_model = PPO.load(model_path, env=env)
        policy_label = f"Trained Checkpoint ({os.path.basename(model_path)})"
    else:
        policy_label = f"Baseline '{args.policy}'"

    print("=" * 72)
    print("      Hiwonder TonyPi Humanoid Locomotion Visualizer")
    print("=" * 72)
    print(f"Policy:          {policy_label}")
    print(f"Scene XML:       {env.xml_path}")
    print(f"Action Scale:    {env.action_scale} * pi ({env.action_mode})")
    print(f"Simulation dt:   {dt:.3f} s ({1.0 / dt:.1f} Hz control rate)")
    print(f"Max Steps/Ep:    {args.max_steps} (~{args.max_steps * dt:.1f} seconds)")
    print(f"Viewer Mode:     {'Interactive 3D GUI' if args.gui else ('Headless / Record' if args.record else 'Telemetry Only')}")
    if args.record:
        print(f"Recording To:    {args.record} (Camera: {args.camera}, Res: {width}x{height}, FPS: {args.fps})")
    print("=" * 72)

    # 4. Offscreen Renderers Setup
    record_frames = []
    renderer_main = None
    renderer_side = None
    cam_main = None
    cam_side = None

    if args.record:
        if args.camera == "multiview":
            half_w = width // 2
            renderer_main = mujoco.Renderer(env.model, height=height, width=half_w)
            renderer_side = mujoco.Renderer(env.model, height=height, width=half_w)
            cam_main = setup_tracking_camera(env.model, env.data, "tracking")
            cam_side = setup_tracking_camera(env.model, env.data, "side")
        else:
            renderer_main = mujoco.Renderer(env.model, height=height, width=width)
            cam_main = setup_tracking_camera(env.model, env.data, args.camera)

    # 5. Interactive GUI Viewer Setup
    gui_viewer = None
    if args.gui:
        try:
            from mujoco import viewer
            gui_viewer = viewer.launch_passive(env.model, env.data)
            pelvis_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            gui_viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            gui_viewer.cam.trackbodyid = pelvis_id

            if args.camera == "side":
                gui_viewer.cam.distance = 1.10
                gui_viewer.cam.elevation = -12.0
                gui_viewer.cam.azimuth = 90.0
            elif args.camera == "front":
                gui_viewer.cam.distance = 1.15
                gui_viewer.cam.elevation = -10.0
                gui_viewer.cam.azimuth = 180.0
            else:
                gui_viewer.cam.distance = 1.25
                gui_viewer.cam.elevation = -18.0
                gui_viewer.cam.azimuth = 135.0

            print("\n[+] Interactive 3D GUI launched!")
            print("    Controls:")
            print("      • Left Click + Drag:   Rotate view")
            print("      • Right Click + Drag:  Zoom in / out")
            print("      • Middle Click + Drag: Pan camera")
            print("      • Spacebar:            Pause / Unpause")
            print("      • Close Window:        Exit visualizer\n")
        except Exception as e:
            print(f"[!] Notice: Interactive GUI window cannot open in this environment ({e}).")
            print("    -> This is expected on headless servers / terminals without a desktop screen.")
            print("    -> To visualize locally on your computer with a screen, run:")
            print("       python tonypi_rl/visualize_robot.py --gui")
            print("    -> To visualize on this terminal server, record a video with:")
            print("       python tonypi_rl/visualize_robot.py --record walk.mp4\n")
            gui_viewer = None

    history = {
        "time": [], "pos_x": [], "pos_y": [], "pos_z": [],
        "vel_x": [], "vel_y": [], "l_force": [], "r_force": [], "rewards": []
    }

    ep = 0
    total_episodes = args.episodes if not args.loop else 999999

    try:
        while ep < total_episodes:
            if gui_viewer is not None and not gui_viewer.is_running():
                print("[*] MuJoCo viewer window closed by user.")
                break

            ep += 1
            obs, info = env.reset(seed=42 + ep)
            ep_reward = 0.0
            steps = 0
            start_time = time.time()

            print(f">>> Episode {ep} Started ...")
            print(f"{'Step':<6} {'Time [s]':<10} {'Vx [m/s]':<10} {'Dist [m]':<10} {'Drift [m]':<10} {'Height [m]':<12} {'Gait Stance':<14} {'Switches'}")
            print("-" * 80)

            for step in range(args.max_steps):
                if gui_viewer is not None and not gui_viewer.is_running():
                    break

                step_start = time.time()

                # Action selection
                if rl_model is not None:
                    action, _ = rl_model.predict(obs, deterministic=True)
                elif args.policy == "zero":
                    action = np.zeros(18, dtype=np.float32)
                elif args.policy == "random":
                    action = env.action_space.sample()
                elif args.policy == "sine":
                    t = step * dt
                    action = np.zeros(18, dtype=np.float32)
                    freq = 1.4
                    action[9] = 0.25 * np.sin(2.0 * np.pi * freq * t)        # l_hip_pitch
                    action[10] = -0.30 * np.sin(2.0 * np.pi * freq * t)      # l_knee
                    action[14] = -0.25 * np.sin(2.0 * np.pi * freq * t)      # r_hip_pitch
                    action[15] = 0.30 * np.sin(2.0 * np.pi * freq * t)       # r_knee

                obs, reward, terminated, truncated, step_info = env.step(action)
                ep_reward += reward
                steps += 1

                pos, rot = env.get_pelvis_pose()
                vx = step_info["forward_vel"]
                vy = env.data.qvel[1]
                l_force = step_info.get("l_force", 0.0)
                r_force = step_info.get("r_force", 0.0)
                sim_time = step * dt

                if l_force > 2.0 and r_force > 2.0:
                    stance_lbl = "DOUBLE"
                elif l_force > 2.0:
                    stance_lbl = "LEFT STANCE"
                elif r_force > 2.0:
                    stance_lbl = "RIGHT STANCE"
                else:
                    stance_lbl = "AIR"

                switches = step_info.get("total_switches", 0)

                history["time"].append(sim_time)
                history["pos_x"].append(pos[0])
                history["pos_y"].append(pos[1])
                history["pos_z"].append(pos[2])
                history["vel_x"].append(vx)
                history["vel_y"].append(vy)
                history["l_force"].append(l_force)
                history["r_force"].append(r_force)
                history["rewards"].append(reward)

                if step % 25 == 0 or terminated or step == args.max_steps - 1:
                    print(f"{step:<6d} {sim_time:<10.2f} {vx:<+10.3f} {pos[0]:<10.2f} {pos[1]:<+10.2f} {pos[2]:<12.3f} {stance_lbl:<14} {switches}")

                # Offscreen Recording
                if args.record:
                    telemetry_dict = {
                        "episode": ep,
                        "step": step,
                        "sim_time": sim_time,
                        "vx": vx,
                        "dist_x": pos[0],
                        "drift_y": pos[1],
                        "height_z": pos[2],
                        "stance": stance_lbl,
                        "switches": switches,
                    }
                    if args.camera == "multiview":
                        renderer_main.update_scene(env.data, camera=cam_main)
                        img1 = renderer_main.render()
                        renderer_side.update_scene(env.data, camera=cam_side)
                        img2 = renderer_side.render()
                        combined = np.hstack([img1, img2])
                        if args.overlay:
                            combined = draw_telemetry_overlay(combined, telemetry_dict)
                        record_frames.append(combined)
                    else:
                        renderer_main.update_scene(env.data, camera=cam_main)
                        img = renderer_main.render()
                        if args.overlay:
                            img = draw_telemetry_overlay(img, telemetry_dict)
                        record_frames.append(img)

                # GUI Viewer Sync
                if gui_viewer is not None:
                    gui_viewer.sync()
                    elapsed = time.time() - step_start
                    desired_step_walltime = dt / max(args.speed, 0.05)
                    sleep_time = desired_step_walltime - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                if terminated:
                    print(f"\n[!] Episode {ep} ended early at step {step} (robot tipped or fell).")
                    break

            duration = time.time() - start_time
            print("-" * 80)
            print(f"Episode {ep} Summary:")
            print(f"  - Total Reward:       {ep_reward:.2f}")
            print(f"  - Steps Survived:     {steps}/{args.max_steps} ({steps * dt:.2f} s sim time)")
            print(f"  - Total Distance:     {pos[0]:.2f} m along +X axis")
            print(f"  - Max Lateral Drift:  {np.max(np.abs(history['pos_y'][-steps:])):.3f} m")
            print(f"  - Foot Alternations:  {switches} switches")
            print(f"  - Wall Time:          {duration:.2f} s\n")

            if gui_viewer is not None and args.loop:
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[!] Visualizer stopped by user.")

    # Save Recorded Video / GIF
    if args.record and record_frames:
        out_path = os.path.abspath(args.record)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        print(f"\n[+] Processing {len(record_frames)} frames for video export ...")

        if out_path.endswith(".gif"):
            from PIL import Image
            pil_frames = [Image.fromarray(f) for f in record_frames]
            pil_frames[0].save(
                out_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(1000.0 / args.fps),
                loop=0,
                optimize=True,
            )
            print(f"[+] Animated GIF successfully saved to: {out_path}")
        else:
            try:
                import cv2
                frame_h, frame_w, _ = record_frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, args.fps, (frame_w, frame_h))
                for f in record_frames:
                    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                writer.release()
                print(f"[+] MP4 video successfully saved to: {out_path}")
            except Exception as e:
                print(f"[!] Failed to write MP4 with cv2 ({e}), saving as GIF instead...")
                gif_path = os.path.splitext(out_path)[0] + ".gif"
                from PIL import Image
                pil_frames = [Image.fromarray(f) for f in record_frames]
                pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=int(1000.0 / args.fps), loop=0)
                print(f"[+] Fallback GIF saved to: {gif_path}")

    # Generate Performance Diagnostics Plot
    if args.plot:
        generate_performance_plots(history, args.plot)

    if gui_viewer is not None:
        gui_viewer.close()
    env.close()
    print("[+] Done.")


if __name__ == "__main__":
    run_visualization()
