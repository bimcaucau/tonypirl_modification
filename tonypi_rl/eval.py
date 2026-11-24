#!/usr/bin/env python3
"""
Evaluation and Visualization Script for Trained TonyPi Policy.

Supports interactive MuJoCo 3D viewer, live telemetry display,
and optional video/frame recording.
"""

import os
import sys
import time
import argparse
import numpy as np

# Ensure tonypi_rl is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tonypi_rl.tonypi_env import TonyPiEnv


def main():
    default_cfg = os.path.join(current_dir, "configs", "v0_straight_walk.yaml")
    parser = argparse.ArgumentParser(description="Evaluate or Visualize TonyPi Policy in MuJoCo")
    parser.add_argument("--config", type=str, default=default_cfg, help="Path to YAML config")
    parser.add_argument("--model", type=str, default=None, help="Path to trained PPO .zip model")
    parser.add_argument("--policy", type=str, default="trained",
                        choices=["trained", "zero", "random", "sine"],
                        help="Policy to execute if model not loaded")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--gui", action="store_true", help="Launch interactive MuJoCo passive viewer")
    parser.add_argument("--render_image", type=str, default=None, help="Save final screenshot to path")
    parser.add_argument("--verbose", action="store_true", help="Print live joint and sensor telemetry")
    args = parser.parse_args()

    # Load YAML config if present
    env_kwargs = {}
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        if str(cfg.get("environment", {}).get("env_class", "")).lower() == "nav":
            print("[!] Error: Navigation evaluation is handled by visualize_nav.py; eval.py is for TonyPiEnv / 18-DoF walk only.")
            sys.exit(1)
        env_kwargs.update(cfg.get("environment", {}))
        env_kwargs.update(cfg.get("rewards", {}))
        env_kwargs.pop("max_episode_steps", None)

    # Create Environment
    render_mode = "human" if args.gui else None
    env_kwargs["render_mode"] = render_mode
    env = TonyPiEnv(**env_kwargs)

    # Load Model if provided
    model = None
    if args.model and os.path.exists(args.model):
        from stable_baselines3 import PPO
        print(f"--> Loading trained PPO model: {args.model}")
        model = PPO.load(args.model, env=env)
    elif args.model:
        print(f"[!] Warning: Model path {args.model} not found. Falling back to {args.policy} policy.")

    print("=" * 65)
    print("  TonyPi Evaluation & Live Telemetry")
    print("=" * 65)
    print(f"Policy:        {args.model if model else args.policy}")
    print(f"Action Mode:   {env.action_mode} (scale = {env.action_scale} * pi)")
    print(f"Episodes:      {args.episodes}")
    print(f"Max Steps/Ep:  {args.max_steps}")
    print("=" * 65)

    total_rewards = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=42 + ep)
        ep_reward = 0.0
        steps = 0

        print(f"\n--- Episode {ep + 1}/{args.episodes} Started ---")

        for step in range(args.max_steps):
            if model is not None:
                action, _states = model.predict(obs, deterministic=True)
            elif args.policy == "zero":
                action = np.zeros(18, dtype=np.float32)
            elif args.policy == "random":
                action = env.action_space.sample()
            elif args.policy == "sine":
                # Gentle sinusoidal swing on leg pitch joints for demonstration
                t = step * 0.02
                action = np.zeros(18, dtype=np.float32)
                action[9] = 0.3 * np.sin(2.0 * np.pi * 1.5 * t)       # l_hip_pitch
                action[10] = -0.3 * np.sin(2.0 * np.pi * 1.5 * t)      # l_knee
                action[14] = -0.3 * np.sin(2.0 * np.pi * 1.5 * t)      # r_hip_pitch
                action[15] = 0.3 * np.sin(2.0 * np.pi * 1.5 * t)       # r_knee

            obs, reward, terminated, truncated, step_info = env.step(action)
            ep_reward += reward
            steps += 1

            if args.verbose or step % 50 == 0:
                pos, rot = env.get_pelvis_pose()
                sensors = env.get_sensor_readings()
                gyro = sensors["imu_gyro"]
                print(
                    f"Ep {ep+1:2d} Step {step:3d} | "
                    f"Z={pos[2]:.3f}m | "
                    f"Vx={step_info["forward_vel"]:+.3f}m/s | "
                    f"Rew={reward:+.2f} | "
                    f"Gyro=({gyro[0]:+.2f},{gyro[1]:+.2f},{gyro[2]:+.2f})"
                )

            if args.gui:
                time.sleep(0.01)

            if terminated:
                print(f"[!] Episode terminated early at step {step+1} (robot fell or tilted).")
                break

        total_rewards.append(ep_reward)
        print(f"Episode {ep + 1} Ended: Total Reward = {ep_reward:.2f} across {steps} steps.")

    # Save render image if requested
    if args.render_image:
        import mujoco
        renderer = mujoco.Renderer(env.model, height=720, width=960)
        renderer.update_scene(env.data, camera="tracking")
        img = renderer.render()
        from PIL import Image
        Image.fromarray(img).save(args.render_image)
        print(f"Saved render screenshot to: {args.render_image}")

    env.close()
    print("\n" + "=" * 65)
    print(f"Evaluation Complete! Mean Reward: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
