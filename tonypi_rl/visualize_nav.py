#!/usr/bin/env python3
"""Inspect high-level nav scene and evaluate trained navigation policies.

Shows the arena (robot, green goal, red boxes), the head-camera RGB,
the 64x64 occupancy mask used by the policy, ToF / sonar, and a privileged
top-down A* overlay (train-only, not in the policy obs).

Usage Examples:
  # 1. Preview initial scene & sensors:
  PYTHONPATH=. python tonypi_rl/visualize_nav.py --render-image tonypi_rl/nav_preview.png
  PYTHONPATH=. python tonypi_rl/visualize_nav.py --gui

  # 2. List available checkpoints:
  PYTHONPATH=. python tonypi_rl/visualize_nav.py --list-checkpoints

  # 3. Evaluate trained policy (auto-detects latest nav model or specify path):
  PYTHONPATH=. python tonypi_rl/visualize_nav.py --model latest --gui
  PYTHONPATH=. python tonypi_rl/visualize_nav.py --model tonypi_rl/models/tonypi_v1_nav_expert/best_model.zip --gui

  # 4. Roll out privileged A* expert:
  PYTHONPATH=. python tonypi_rl/visualize_nav.py --layout demo --expert --render-image tonypi_rl/nav_expert_preview.png

  # 5. Record rollout to MP4 video or animated GIF:
  PYTHONPATH=. python tonypi_rl/visualize_nav.py --expert --record expert_nav.mp4
  PYTHONPATH=. python tonypi_rl/visualize_nav.py --model latest --record policy_nav.gif --camera multiview
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tonypi_rl.tonypi_nav_env import TonyPiNavEnv, ACTION_NAMES, TOF_NAMES, TOF_CUTOFF, SONAR_CUTOFF


def _upsample(mask: np.ndarray, scale: int = None) -> np.ndarray:
    if scale is None:
        scale = max(1, 256 // int(mask.shape[0]))
    return np.kron(mask, np.ones((scale, scale), dtype=np.float32))


def _draw_topdown(ax, env: TonyPiNavEnv, trail=None, path=None):
    from matplotlib.patches import Rectangle, FancyArrow

    ax.set_facecolor("#1a1d24")
    grid = None if env._expert is None else env._expert.grid
    if grid is not None:
        extent = [
            grid.origin_x,
            grid.origin_x + grid.w * grid.res,
            grid.origin_y,
            grid.origin_y + grid.h * grid.res,
        ]
        ax.imshow(
            grid.occ.astype(np.float32),
            origin="lower",
            cmap="Greys",
            extent=extent,
            vmin=0.0,
            vmax=1.0,
            alpha=0.85,
            interpolation="nearest",
            zorder=1,
        )
    for cx, cy, hx, hy in env._obstacle_rects():
        ax.add_patch(
            Rectangle(
                (cx - hx, cy - hy), 2 * hx, 2 * hy,
                facecolor="#dc4632", edgecolor="#ff8a80",
                linewidth=0.6, alpha=0.85, zorder=3,
            )
        )
    if path is not None and len(path) >= 2:
        ax.plot(path[:, 0], path[:, 1], color="#4fc3f7", lw=1.8, zorder=4, label="A*")
    if trail is not None and len(trail) >= 2:
        ax.plot(trail[:, 0], trail[:, 1], color="#ffb74d", lw=1.3, ls="--", zorder=5, label="executed")
    gx, gy = float(env.goal_xy[0]), float(env.goal_xy[1])
    ax.plot(gx, gy, marker="*", color="#69f0ae", markersize=14, zorder=6, label="goal")
    x = float(env.data.qpos[0])
    y = float(env.data.qpos[1])
    yaw = env._pelvis_yaw()
    ax.plot(0.0, 0.0, marker="o", color="#b0bec5", markersize=5, zorder=6)
    ax.plot(x, y, marker="o", color="#66bb6a", markersize=7, zorder=7)
    ax.add_patch(
        FancyArrow(
            x, y, 0.16 * np.cos(yaw), 0.16 * np.sin(yaw),
            width=0.02, head_width=0.07, head_length=0.08,
            color="#66bb6a", length_includes_head=True, zorder=7,
        )
    )
    xs = [0.0, gx, x]
    ys = [0.0, gy, y]
    if path is not None and len(path):
        xs.extend(path[:, 0].tolist())
        ys.extend(path[:, 1].tolist())
    if trail is not None and len(trail):
        xs.extend(trail[:, 0].tolist())
        ys.extend(trail[:, 1].tolist())
    pad = 0.55
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.set_title("World XY  (inflated occ + path)", color="white")
    ax.tick_params(colors="white", labelsize=8)
    ax.set_xlabel("x (m)", color="white")
    ax.set_ylabel("y (m)", color="white")
    for spine in ax.spines.values():
        spine.set_color("#444")
    leg = ax.legend(loc="upper right", fontsize=7, framealpha=0.4)
    if leg is not None:
        for t in leg.get_texts():
            t.set_color("white")


def compose_preview(env: TonyPiNavEnv, obs, info, path: str, trail=None, plan_xy=None, policy_tag: str = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mujoco

    if env.track_renderer is None:
        env.track_renderer = mujoco.Renderer(env.model, height=480, width=640)
    env.track_renderer.update_scene(env.data, camera="arena")
    arena = env.track_renderer.render()
    if not hasattr(env, "_head_preview") or env._head_preview is None:
        env._head_preview = mujoco.Renderer(env.model, height=256, width=256)
    env._head_preview.update_scene(env.data, camera="head_camera")
    head = env._head_preview.render()
    occ = obs["occupancy"][0]
    vec = obs["vec"]
    tof = info["tof"]
    sonar = info["sonar"]

    occ_img = _upsample(occ)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor="#111318")
    geo_val = info.get("geodesic")
    if geo_val is None or geo_val < 0 or not np.isfinite(geo_val):
        geo_val = env.geodesic_distance()
    geo_str = f"{geo_val:.2f}" if (geo_val is not None and np.isfinite(geo_val) and geo_val >= 0) else "n/a"
    tof_str = f"{tof[0]:.2f}" if len(tof) > 0 else "n/a"
    fig.suptitle(
        f"TonyPiNav  goal=({info['goal_x']:.2f}, {info['goal_y']:.2f}) m   "
        f"dist={info['dist']:.2f} m   ToF={tof_str} m   "
        f"geo={geo_str}",
        color="white", fontsize=12,
    )

    ax = axes[0, 0]
    ax.imshow(arena)
    ax.set_title("Arena (tracking)", color="white")
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(head)
    ax.set_title("Head camera RGB", color="white")
    ax.axis("off")

    ax = axes[0, 2]
    ax.imshow(occ_img, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"Occupancy mask {occ.shape[0]}x{occ.shape[1]}  (obs_* geoms)", color="white")
    ax.axis("off")

    ax = axes[1, 0]
    ax.set_facecolor("#1a1d24")
    labels = [f"tof_{i} (VL53L0X)" if len(TOF_NAMES) > 1 else "tof (VL53L0X)" for i in range(len(TOF_NAMES))] + ["sonar"]
    values = list(tof) + [sonar]
    cuts = [1.2] * len(TOF_NAMES) + [5.0]
    colors = ["#dc4632"] * len(TOF_NAMES) + ["#328ce6"]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color="white")
    ax.set_xlabel("range (m)", color="white")
    ax.tick_params(colors="white")
    for i, (v, c) in enumerate(zip(values, cuts)):
        ax.axvline(c, color="#555", linewidth=0.6, linestyle="--")
        ax.text(v + 0.03, i, f"{v:.2f}", va="center", color="white", fontsize=8)
    gx = info.get("gx")
    gy = info.get("gy")
    dist = info.get("dist")
    heading = info.get("heading")
    if any(v is None or not np.isfinite(v) for v in (gx, gy, dist, heading)):
        gx, gy, dist, heading = env._goal_body_frame()
    ax.set_title(
        f"goal body x={gx:+.2f} y={gy:+.2f}\n"
        f"dist={dist:.2f}m  heading={np.degrees(heading):+.1f}°",
        color="white", fontsize=10,
    )
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.set_xlim(0, 5.2)

    ax = axes[1, 1]
    if plan_xy is None:
        plan_xy = env.expert_path()
    _draw_topdown(ax, env, trail=trail, path=plan_xy)

    ax = axes[1, 2]
    ax.set_facecolor("#1a1d24")
    ax.axis("off")
    expert_id = info.get("expert_action")
    if expert_id is None or expert_id < 0:
        expert_id = int(env.expert_action())
    else:
        expert_id = int(expert_id)
    expert_name = ACTION_NAMES[expert_id] if 0 <= expert_id < len(ACTION_NAMES) else "?"
    geo_line = f"{geo_val:.3f} m" if (geo_val is not None and np.isfinite(geo_val) and geo_val >= 0) else "n/a"
    last_act = int(env.prev_action)
    last_act_name = ACTION_NAMES[last_act] if 0 <= last_act < len(ACTION_NAMES) else "?"
    lines = []
    if policy_tag:
        lines.append(f"policy        {policy_tag}")
        lines.append(f"last action   {last_act}  {last_act_name}")
    lines.extend([
        f"layout        {info.get('layout_kind', 'n/a')}",
        f"expert a*     {expert_id}  {expert_name}",
        f"geodesic      {geo_line}",
        f"unreachable   {info.get('expert_unreachable', False)}",
        f"arrived       {info.get('arrived', False)}",
        f"collision     {info.get('collision', False)}",
        f"n obstacles   {info.get('n_obstacles', len(env._active_obs))}",
        f"progress      {info.get('progress', float('nan'))}",
    ])
    if trail is not None:
        lines.append(f"trail steps   {max(0, len(trail) - 1)}")
    ax.text(
        0.04, 0.96, "\n".join(lines),
        transform=ax.transAxes, va="top", ha="left",
        color="white", fontsize=11, family="monospace",
    )
    title = f"Policy & Expert ({policy_tag})" if policy_tag else "Expert (privileged)"
    ax.set_title(title, color="white")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[+] wrote {path}")


def print_obs(obs, info):
    vec = obs["vec"]
    occ = obs["occupancy"][0]
    gx = info.get("gx", float("nan"))
    gy = info.get("gy", float("nan"))
    dist = info.get("dist", float("nan"))
    heading = info.get("heading", float("nan"))
    print("-" * 64)
    print(f"layout         requested={info.get('layout', 'n/a')}  kind={info.get('layout_kind', 'n/a')}")
    print(f"goal world     ({info['goal_x']:+.3f}, {info['goal_y']:+.3f}) m")
    print(f"goal body      x={gx:+.3f}  y={gy:+.3f}  dist={dist:.3f}  heading={np.degrees(heading):+.1f} deg")
    tof_str = f"{info['tof'][0]:.2f}" if len(info.get('tof', [])) > 0 else "n/a"
    print(f"ToF  (m)       {tof_str}   (single VL53L0X, cutoff 1.2)")
    print(f"sonar (m)      {info['sonar']:.2f}")
    print(f"occupancy      shape={obs['occupancy'].shape}  occupied={float(occ.mean()):.3f}  max={float(occ.max()):.0f}")
    print(f"vec dim        {vec.shape[0]}   action names={ACTION_NAMES}")
    ea = info.get("expert_action")
    if ea is not None:
        ea_int = int(ea)
        ea_name = ACTION_NAMES[ea_int] if 0 <= ea_int < len(ACTION_NAMES) else "?"
        geo_str = f"{info['geodesic']:.3f}" if info.get('geodesic', -1) >= 0 else "n/a"
        print(f"expert a*      {ea} {ea_name}   geo={geo_str}  unreachable={info.get('expert_unreachable')}")
    print("-" * 64)


def draw_nav_telemetry_overlay(frame: np.ndarray, telemetry: dict) -> np.ndarray:
    """Draw a clean semi-transparent HUD overlay on video frame for navigation."""
    try:
        import cv2
    except ImportError:
        return frame

    h, w, _ = frame.shape
    img = frame.copy()

    overlay = img.copy()
    banner_h = min(56, h)
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (18, 22, 28), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    step = telemetry.get("step", 0)
    act_name = telemetry.get("action_name", "?")
    act_id = telemetry.get("action", 0)
    act_str = f"Step: {step:03d} | Action: {act_id} ({act_name})"
    pol = telemetry.get("policy", "")
    if pol:
        act_str = f"[{pol}] {act_str}"

    dist = telemetry.get("dist", 0.0)
    geo = telemetry.get("geodesic", -1.0)
    geo_str = f"{geo:.2f}m" if geo >= 0 else "n/a"
    tof = telemetry.get("tof", 0.0)
    sonar = telemetry.get("sonar", 0.0)
    l_kind = telemetry.get("layout_kind", "")
    layout_str = f" | [{l_kind}]" if l_kind else ""
    sensor_str = f"Dist: {dist:.2f}m | Geo: {geo_str} | ToF: {tof:.2f}m | Sonar: {sonar:.2f}m{layout_str}"

    coll = telemetry.get("collision", False)
    arrived = telemetry.get("arrived", False)
    status_str = "COLLISION!" if coll else ("ARRIVED" if arrived else "NAVIGATING")
    status_color = (0, 0, 255) if coll else ((80, 220, 100) if arrived else (220, 225, 230))

    cv2.putText(img, act_str, (16, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 225, 230), 1, cv2.LINE_AA)
    cv2.putText(img, sensor_str, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (100, 220, 140), 1, cv2.LINE_AA)
    if w >= 400:
        cv2.putText(img, status_str, (w - 170, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 2, cv2.LINE_AA)

    return img


def save_video(record_frames: list[np.ndarray], out_path: str, fps: int = 15):
    """Save recorded frames to MP4 video or animated GIF."""
    if not record_frames:
        return
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"\n[+] Processing {len(record_frames)} frames for video export ...")

    if out_path.endswith(".gif"):
        from PIL import Image
        pil_frames = [Image.fromarray(f) for f in record_frames]
        pil_frames[0].save(
            out_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000.0 / max(1, fps)),
            loop=0,
            optimize=True,
        )
        print(f"[+] Animated GIF successfully saved to: {out_path}")
    else:
        try:
            import cv2
            frame_h, frame_w, _ = record_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_w, frame_h))
            for f in record_frames:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"[+] MP4 video successfully saved to: {out_path}")
        except Exception as e:
            print(f"[!] Failed to write MP4 with cv2 ({e}), saving as GIF instead...")
            gif_path = os.path.splitext(out_path)[0] + ".gif"
            from PIL import Image
            pil_frames = [Image.fromarray(f) for f in record_frames]
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(1000.0 / max(1, fps)),
                loop=0,
            )
            print(f"[+] Fallback GIF saved to: {gif_path}")


class FrameRecorder:
    def __init__(self, env: TonyPiNavEnv, record_path: str, camera: str = "arena",
                 resolution: tuple[int, int] = (960, 720), overlay: bool = True, fps: int = 15):
        import mujoco
        self.env = env
        self.record_path = record_path
        self.camera = camera
        self.width, self.height = resolution
        self.overlay = overlay
        self.fps = fps
        self.frames = []

        if self.camera == "multiview":
            half_w = self.width // 2
            self.arena_renderer = mujoco.Renderer(env.model, height=self.height, width=half_w)
            self.head_renderer = mujoco.Renderer(env.model, height=self.height, width=half_w)
        elif self.camera == "head_camera":
            self.head_renderer = mujoco.Renderer(env.model, height=self.height, width=self.width)
            self.arena_renderer = None
        else:
            self.arena_renderer = mujoco.Renderer(env.model, height=self.height, width=self.width)
            self.head_renderer = None

    def capture(self, step: int, action: int, info: dict, policy_name: str = ""):
        if self.camera == "multiview":
            self.arena_renderer.update_scene(self.env.data, camera="arena")
            img1 = self.arena_renderer.render()
            self.head_renderer.update_scene(self.env.data, camera="head_camera")
            img2 = self.head_renderer.render()
            frame = np.hstack([img1, img2])
        elif self.camera == "head_camera":
            self.head_renderer.update_scene(self.env.data, camera="head_camera")
            frame = self.head_renderer.render()
        else:
            self.arena_renderer.update_scene(self.env.data, camera="arena")
            frame = self.arena_renderer.render()

        if self.overlay:
            tof_val = info.get("tof", [0.0])[0] if len(info.get("tof", [])) > 0 else 0.0
            telemetry = {
                "step": step,
                "action": action,
                "action_name": ACTION_NAMES[action] if 0 <= action < len(ACTION_NAMES) else "?",
                "goal_x": info.get("goal_x", float(self.env.goal_xy[0])),
                "goal_y": info.get("goal_y", float(self.env.goal_xy[1])),
                "dist": info.get("dist", 0.0),
                "tof": tof_val,
                "sonar": info.get("sonar", 0.0),
                "geodesic": info.get("geodesic", -1.0),
                "layout_kind": info.get("layout_kind", getattr(self.env, "_layout_kind", "")),
                "collision": info.get("collision", False),
                "arrived": info.get("arrived", False),
                "policy": policy_name,
            }
            frame = draw_nav_telemetry_overlay(frame, telemetry)

        self.frames.append(frame)

    def close(self):
        if getattr(self, "arena_renderer", None) is not None:
            self.arena_renderer.close()
            self.arena_renderer = None
        if getattr(self, "head_renderer", None) is not None:
            self.head_renderer.close()
            self.head_renderer = None

    def save(self):
        if self.record_path and self.frames:
            save_video(self.frames, self.record_path, fps=self.fps)


def run_expert(env: TonyPiNavEnv, gui: bool = False, recorder: FrameRecorder = None):
    trail = [(float(env.data.qpos[0]), float(env.data.qpos[1]))]
    total_r = 0.0
    n_coll = 0
    steps = 0
    info = {}
    obs = None
    if recorder is not None:
        init_info = {
            "goal_x": float(env.goal_xy[0]),
            "goal_y": float(env.goal_xy[1]),
            "dist": float(np.hypot(env.goal_xy[0] - env.data.qpos[0], env.goal_xy[1] - env.data.qpos[1])),
            "tof": [env._read_range(n, TOF_CUTOFF) for n in TOF_NAMES],
            "sonar": env._read_range("sonar", SONAR_CUTOFF),
            "geodesic": env.geodesic_distance(),
        }
        recorder.capture(0, int(env.prev_action), init_info, policy_name="A* Expert")
    try:
        while True:
            a = env.expert_action()
            obs, rew, term, trunc, info = env.step(a)
            total_r += float(rew)
            steps += 1
            n_coll += int(bool(info.get("collision")))
            trail.append((float(env.data.qpos[0]), float(env.data.qpos[1])))
            if recorder is not None:
                recorder.capture(steps, a, info, policy_name="A* Expert")
            if gui and env.viewer is not None:
                env.viewer.sync()
                import time
                time.sleep(0.04)
            if term or trunc:
                break
    except KeyboardInterrupt:
        print("\n[!] expert rollout interrupted")
    trail_xy = np.asarray(trail, dtype=np.float64)
    arrived = bool(info.get("arrived", False))
    print(
        f"expert  arrived={arrived}  steps={steps}  return={total_r:+.2f}  "
        f"collisions={n_coll}  dist={info.get('dist', float('nan')):.3f}  "
        f"geo={info.get('geodesic', float('nan'))}"
    )
    return obs, info, trail_xy, dict(arrived=arrived, steps=steps, ret=total_r, collisions=n_coll)


def inspect_checkpoint(zip_path: str) -> dict:
    info = {
        "path": zip_path,
        "is_nav": False,
        "size_mb": 0.0,
        "timesteps": None,
        "type": "Unknown",
    }
    if not os.path.exists(zip_path):
        return info
    info["size_mb"] = os.path.getsize(zip_path) / (1024 * 1024)
    try:
        import zipfile
        import json
        with zipfile.ZipFile(zip_path, "r") as z:
            if "data" in z.namelist():
                d = json.loads(z.read("data").decode("utf-8"))
                obs_type = str(d.get("observation_space", {}).get(":type:", ""))
                act_type = str(d.get("action_space", {}).get(":type:", ""))
                info["timesteps"] = d.get("num_timesteps")
                if "Dict" in obs_type or "dict" in obs_type.lower():
                    info["is_nav"] = True
                    info["type"] = "Nav (MultiInputPolicy)"
                elif "Box" in obs_type and "Box" in act_type:
                    info["type"] = "Walk (18-DOF Joint Control)"
                else:
                    info["type"] = obs_type or "Model"
    except Exception:
        pass
    return info


def list_available_checkpoints(models_dir: str):
    """List all available checkpoints and training runs found on disk."""
    import glob
    if not os.path.exists(models_dir):
        print(f"[!] No models directory found at: {models_dir}")
        return

    print("=" * 78)
    print(f"  Available Checkpoints & Models in: {models_dir}")
    print("=" * 78)

    top_zips = sorted(glob.glob(os.path.join(models_dir, "*.zip")))
    if top_zips:
        print("\n📁 Root Checkpoints:")
        for zp in top_zips:
            meta = inspect_checkpoint(zp)
            ts_str = f" @ {meta['timesteps']:,} steps" if meta.get("timesteps") else ""
            prefix = "★" if "best" in os.path.basename(zp) else "•"
            print(f"   {prefix} {os.path.basename(zp):<30} ({meta['size_mb']:.2f} MB){ts_str}  [{meta['type']}]")

    subdirs = sorted([d for d in glob.glob(os.path.join(models_dir, "*")) if os.path.isdir(d)])
    if not subdirs and not top_zips:
        print(f"[!] No checkpoints found in: {models_dir}")
        return

    for run_dir in subdirs:
        run_name = os.path.basename(run_dir)
        zips = sorted(glob.glob(os.path.join(run_dir, "*.zip")))
        if not zips:
            continue
        print(f"\n📂 Run: {run_name}")
        for zp in zips:
            meta = inspect_checkpoint(zp)
            ts_str = f" @ {meta['timesteps']:,} steps" if meta.get("timesteps") else ""
            prefix = "★" if "best" in os.path.basename(zp) else "•"
            print(f"   {prefix} {os.path.basename(zp):<30} ({meta['size_mb']:.2f} MB){ts_str}  [{meta['type']}]")

    print("\n" + "=" * 78)
    print("Usage with visualize_nav.py:")
    print("  python tonypi_rl/visualize_nav.py --model latest --gui")
    print("  python tonypi_rl/visualize_nav.py --model <path_or_run_name> --gui")
    print("Usage with visualize_robot.py (for 18-DOF walking policies):")
    print("  python tonypi_rl/visualize_robot.py --model <path_to_walk_model> --gui")
    print("=" * 78)


def find_latest_model(models_dir: str, nav_only: bool = True) -> str | None:
    """Find the most recent trained nav checkpoint or best_model in models_dir."""
    import glob
    if not os.path.exists(models_dir):
        return None
    all_zips = sorted(
        glob.glob(os.path.join(models_dir, "**", "*.zip"), recursive=True),
        key=os.path.getmtime,
        reverse=True,
    )
    if not all_zips:
        return None

    if nav_only:
        for zp in all_zips:
            meta = inspect_checkpoint(zp)
            if meta.get("is_nav"):
                return zp

    return all_zips[0]


def resolve_model_path(model_arg: str, models_base: str) -> str | None:
    import glob
    if not model_arg:
        return None
    if model_arg == "latest":
        auto = find_latest_model(models_base, nav_only=True)
        if auto:
            print(f"[+] Auto-detected latest navigation checkpoint: {auto}")
            return auto
        auto_any = find_latest_model(models_base, nav_only=False)
        if auto_any:
            print(f"[+] Auto-detected latest checkpoint: {auto_any}")
            return auto_any
        print(f"[!] No checkpoints found in {models_base}")
        return None

    if os.path.exists(model_arg):
        return os.path.abspath(model_arg)

    candidates = [
        os.path.join(models_base, model_arg),
        os.path.join(models_base, model_arg, "best_model.zip"),
        os.path.join(models_base, model_arg, "tonypi_final_model.zip"),
        os.path.join(parent_dir, model_arg),
        os.path.join(current_dir, model_arg),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)

    matches = glob.glob(os.path.join(models_base, "**", model_arg), recursive=True)
    if matches:
        return os.path.abspath(matches[0])
    return None


def load_nav_policy(model_path: str, env: TonyPiNavEnv):
    from stable_baselines3 import PPO
    try:
        from tonypi_rl.nav_features import NavFeaturesExtractor
    except ImportError:
        pass

    meta = inspect_checkpoint(model_path)
    if meta.get("type") == "Walk (18-DOF Joint Control)":
        print(f"[!] Warning: '{os.path.basename(model_path)}' is an 18-DOF joint walking policy (Box(63) -> Box(18)),")
        print(f"    not a navigation policy (Dict -> Discrete(7)).")
        print(f"    To visualize walking policies, run: python tonypi_rl/visualize_robot.py --model {model_path} --gui")
        raise ValueError(f"Model '{model_path}' is incompatible with TonyPiNavEnv.")

    print(f"--> Loading trained Nav PPO policy: {model_path}")
    try:
        model = PPO.load(model_path, env=env)
    except Exception as e:
        try:
            from tonypi_rl.guided_ppo import GuidedPPO
            model = GuidedPPO.load(model_path, env=env)
        except Exception:
            raise e
    return model


def run_policy(env: TonyPiNavEnv, model, gui: bool = False, deterministic: bool = True,
               recorder: FrameRecorder = None, policy_name: str = ""):
    trail = [(float(env.data.qpos[0]), float(env.data.qpos[1]))]
    total_r = 0.0
    n_coll = 0
    steps = 0
    info = {}
    obs = env._get_obs()
    if recorder is not None:
        init_info = {
            "goal_x": float(env.goal_xy[0]),
            "goal_y": float(env.goal_xy[1]),
            "dist": float(np.hypot(env.goal_xy[0] - env.data.qpos[0], env.goal_xy[1] - env.data.qpos[1])),
            "tof": [env._read_range(n, TOF_CUTOFF) for n in TOF_NAMES],
            "sonar": env._read_range("sonar", SONAR_CUTOFF),
            "geodesic": env.geodesic_distance(),
        }
        recorder.capture(0, int(env.prev_action), init_info, policy_name=policy_name)
    try:
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            action = int(action)
            obs, rew, term, trunc, info = env.step(action)
            total_r += float(rew)
            steps += 1
            n_coll += int(bool(info.get("collision")))
            trail.append((float(env.data.qpos[0]), float(env.data.qpos[1])))
            if recorder is not None:
                recorder.capture(steps, action, info, policy_name=policy_name)
            if gui and env.viewer is not None:
                env.viewer.sync()
                import time
                time.sleep(0.04)
            if term or trunc:
                break
    except KeyboardInterrupt:
        print("\n[!] policy rollout interrupted")
    trail_xy = np.asarray(trail, dtype=np.float64)
    arrived = bool(info.get("arrived", False))
    print(
        f"policy  arrived={arrived}  steps={steps}  return={total_r:+.2f}  "
        f"collisions={n_coll}  dist={info.get('dist', float('nan')):.3f}  "
        f"geo={info.get('geodesic', float('nan'))}"
    )
    return obs, info, trail_xy, dict(arrived=arrived, steps=steps, ret=total_r, collisions=n_coll)


def main():
    parser = argparse.ArgumentParser(description="Visualize TonyPi high-level nav scene and policies")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained nav policy .zip model, or 'latest' to auto-detect")
    parser.add_argument("--list-checkpoints", action="store_true",
                        help="List all saved models/checkpoints and exit")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic policy actions (default: True)")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false",
                        help="Sample stochastic policy actions")
    parser.add_argument("--gui", action="store_true", help="Open interactive MuJoCo viewer")
    parser.add_argument("--render-image", type=str, default="tonypi_rl/nav_preview.png",
                        help="Save perception preview PNG")
    parser.add_argument("--record", type=str, default=None,
                        help="Save rollout video to MP4 or GIF (e.g. nav.mp4, rollout.gif)")
    parser.add_argument("--camera", type=str, default="arena",
                        choices=["arena", "head_camera", "multiview"],
                        help="Camera perspective for video recording (default: arena)")
    parser.add_argument("--fps", type=int, default=15,
                        help="Frame rate for recorded video / GIF (default: 15)")
    parser.add_argument("--overlay", action="store_true", default=True,
                        help="Render telemetry HUD overlay on recorded video frames")
    parser.add_argument("--no-overlay", dest="overlay", action="store_false",
                        help="Disable telemetry HUD overlay on recorded video")
    parser.add_argument("--resolution", type=str, default="960x720",
                        help="Video render resolution WIDTHxHEIGHT (default: 960x720)")
    parser.add_argument("--random", "--random-spawn", dest="random_spawn", action="store_true",
                        help="Randomize both target and obstacle spawn positions on reset")
    parser.add_argument("--random-yaw", action="store_true",
                        help="Randomize initial robot heading/yaw on reset")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to visualize / record (default: 1)")
    parser.add_argument("--layout", type=str, default="demo",
                        choices=["demo", "empty", "random", "scatter", "wall", "u_trap", "gate", "mixed", "curriculum"],
                        help="Obstacle layout mode (default: demo)")
    parser.add_argument("--goal-mode", type=str, default="fixed", choices=["fixed", "random"])
    parser.add_argument("--n-obstacles", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override maximum episode steps")
    parser.add_argument("--demo-actions", action="store_true",
                        help="Play a few go_forward / turn commands after reset")
    parser.add_argument("--expert", action="store_true",
                        help="Roll out the privileged A* primitive expert")
    parser.add_argument("--no-image", action="store_true", help="Skip writing the PNG")
    args = parser.parse_args()

    models_base = os.path.join(parent_dir, "tonypi_rl", "models")
    if not os.path.exists(models_base):
        models_base = os.path.join(current_dir, "models")

    if args.list_checkpoints:
        list_available_checkpoints(models_base)
        return

    try:
        width, height = map(int, args.resolution.split("x"))
    except ValueError:
        width, height = 960, 720

    render_mode = "human" if args.gui else "rgb_array"
    default_max_steps = 160
    max_steps = args.max_steps if args.max_steps is not None else default_max_steps

    layout = args.layout
    goal_mode = args.goal_mode
    if args.random_spawn:
        if layout in ("demo", "fixed"):
            layout = "mixed"
        goal_mode = "random"
    elif layout in ("random", "scatter", "mixed", "curriculum") and parser.get_default("goal_mode") == args.goal_mode:
        goal_mode = "random"

    env = TonyPiNavEnv(
        render_mode=render_mode,
        layout=layout,
        goal_mode=goal_mode,
        random_spawn=args.random_spawn,
        random_robot_yaw=args.random_yaw,
        n_obstacles=args.n_obstacles,
        max_episode_steps=max_steps,
        use_geodesic_reward=True,
    )

    recorder = None
    if args.record:
        recorder = FrameRecorder(
            env=env,
            record_path=args.record,
            camera=args.camera,
            resolution=(width, height),
            overlay=args.overlay,
            fps=args.fps,
        )

    trail = None
    policy_tag = None
    last_obs = None
    last_info = None
    last_plan_xy = None

    if args.model:
        model_path = resolve_model_path(args.model, models_base)
        if model_path is None:
            print(f"[!] Could not resolve model '{args.model}' in {models_base}")
            if recorder is not None:
                recorder.close()
            env.close()
            sys.exit(1)
        model = load_nav_policy(model_path, env)
        policy_tag = os.path.basename(model_path)
        all_stats = []
        for ep in range(max(1, args.episodes)):
            reset_seed = args.seed + ep
            reset_opts = {}
            if args.random_yaw:
                reset_opts["random_yaw"] = True
            if args.random_spawn:
                reset_opts["random_spawn"] = True
            obs, info = env.reset(seed=reset_seed, options=reset_opts)
            plan_xy = env.expert_path()
            if args.episodes > 1:
                print(f"\n--- Episode {ep + 1}/{args.episodes} (seed={reset_seed}) ---")
            else:
                print("=" * 64)
                print("  TonyPiNav scene preview")
                print("=" * 64)
                print_obs(obs, info)
            obs, info, trail, stats = run_policy(
                env, model, gui=args.gui, deterministic=args.deterministic,
                recorder=recorder, policy_name=policy_tag,
            )
            all_stats.append(stats)
            last_obs, last_info, last_plan_xy = obs, info, plan_xy
            if not stats["arrived"]:
                print("[!] policy did not arrive at goal")
        if len(all_stats) > 1:
            arr_count = sum(1 for s in all_stats if s["arrived"])
            print("\n" + "=" * 64)
            print(f"Summary over {len(all_stats)} episodes:")
            print(f"  Success rate: {arr_count}/{len(all_stats)} ({arr_count/len(all_stats)*100:.1f}%)")
            print(f"  Mean return: {np.mean([s['ret'] for s in all_stats]):+.2f}")
            print(f"  Mean steps: {np.mean([s['steps'] for s in all_stats]):.1f}")
            print(f"  Mean collisions: {np.mean([s['collisions'] for s in all_stats]):.1f}")
            print("=" * 64)
    elif args.expert:
        policy_tag = "A* Expert"
        all_stats = []
        for ep in range(max(1, args.episodes)):
            reset_seed = args.seed + ep
            reset_opts = {}
            if args.random_yaw:
                reset_opts["random_yaw"] = True
            if args.random_spawn:
                reset_opts["random_spawn"] = True
            obs, info = env.reset(seed=reset_seed, options=reset_opts)
            plan_xy = env.expert_path()
            if args.episodes > 1:
                print(f"\n--- Episode {ep + 1}/{args.episodes} (seed={reset_seed}) ---")
            else:
                print("=" * 64)
                print("  TonyPiNav scene preview")
                print("=" * 64)
                print_obs(obs, info)
            obs, info, trail, stats = run_expert(env, gui=args.gui, recorder=recorder)
            all_stats.append(stats)
            last_obs, last_info, last_plan_xy = obs, info, plan_xy
            if not stats["arrived"]:
                print("[!] expert did not arrive (see collisions / unreachable)")
        if len(all_stats) > 1:
            arr_count = sum(1 for s in all_stats if s["arrived"])
            print("\n" + "=" * 64)
            print(f"Summary over {len(all_stats)} episodes:")
            print(f"  Success rate: {arr_count}/{len(all_stats)} ({arr_count/len(all_stats)*100:.1f}%)")
            print(f"  Mean return: {np.mean([s['ret'] for s in all_stats]):+.2f}")
            print(f"  Mean steps: {np.mean([s['steps'] for s in all_stats]):.1f}")
            print(f"  Mean collisions: {np.mean([s['collisions'] for s in all_stats]):.1f}")
            print("=" * 64)
    elif args.demo_actions:
        reset_opts = {}
        if args.random_yaw:
            reset_opts["random_yaw"] = True
        if args.random_spawn:
            reset_opts["random_spawn"] = True
        obs, info = env.reset(seed=args.seed, options=reset_opts)
        plan_xy = env.expert_path()
        print("=" * 64)
        print("  TonyPiNav scene preview")
        print("=" * 64)
        print_obs(obs, info)
        policy_tag = "Demo Actions"
        trail_pts = [(float(env.data.qpos[0]), float(env.data.qpos[1]))]
        if recorder is not None:
            recorder.capture(0, 0, info, policy_name="Demo")
        for a in [1, 1, 6, 1, 5]:
            obs, rew, term, trunc, info = env.step(a)
            trail_pts.append((float(env.data.qpos[0]), float(env.data.qpos[1])))
            tof0 = info['tof'][0] if len(info.get('tof', [])) > 0 else float('nan')
            print(f"  action={ACTION_NAMES[a]:16s}  rew={rew:+.2f}  dist={info['dist']:.3f}  "
                  f"coll={info['collision']}  tof0={tof0:.2f}")
            if recorder is not None:
                recorder.capture(len(trail_pts) - 1, a, info, policy_name="Demo")
            if term or trunc:
                break
        trail = np.asarray(trail_pts, dtype=np.float64)
        last_obs, last_info, last_plan_xy = obs, info, plan_xy
    else:
        reset_opts = {}
        if args.random_yaw:
            reset_opts["random_yaw"] = True
        if args.random_spawn:
            reset_opts["random_spawn"] = True
        obs, info = env.reset(seed=args.seed, options=reset_opts)
        plan_xy = env.expert_path()
        print("=" * 64)
        print("  TonyPiNav scene preview")
        print("=" * 64)
        print_obs(obs, info)
        last_obs, last_info, last_plan_xy = obs, info, plan_xy

    if recorder is not None:
        recorder.save()
        recorder.close()

    if not args.no_image and args.render_image:
        if env.track_renderer is None:
            import mujoco
            env.track_renderer = mujoco.Renderer(env.model, height=480, width=640)
        compose_preview(env, last_obs, last_info, args.render_image, trail=trail, plan_xy=last_plan_xy, policy_tag=policy_tag)

    if args.gui and not (args.expert or args.model):
        print("Viewer open. Close the window or Ctrl+C to exit.")
        try:
            while env.viewer is not None and env.viewer.is_running():
                env.viewer.sync()
                import time
                time.sleep(0.03)
        except KeyboardInterrupt:
            pass
    elif args.gui and (args.expert or args.model):
        print("Rollout complete. Close the window or Ctrl+C to exit.")
        try:
            while env.viewer is not None and env.viewer.is_running():
                env.viewer.sync()
                import time
                time.sleep(0.03)
        except KeyboardInterrupt:
            pass

    if hasattr(env, "_head_preview") and env._head_preview is not None:
        env._head_preview.close()
        env._head_preview = None
    env.close()


if __name__ == "__main__":
    main()
