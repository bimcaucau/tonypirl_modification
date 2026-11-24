#!/usr/bin/env python3
"""
TonyPi Bipedal Locomotion PPO Training Script (Config-Driven).

Loads all training parameters, curriculum settings, environment rewards,
and PPO hyperparameters from a YAML configuration file.
"""

import importlib.util
import inspect
import os
import sys

# Must be set before any mujoco import. Spawn workers inherit this.
os.environ.setdefault("MUJOCO_GL", "egl")

import yaml
import shutil
import argparse
from datetime import datetime

import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

# This file's directory is the package, even when the folder is named
# `tonypi_rl_expert` next to an in-flight `tonypi_rl` on the cluster.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)


def _resolve_under_repo(path: str, repo_root: str) -> str:
    if not path:
        return os.path.abspath(repo_root)
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(repo_root, path))


def _bind_tonypi_rl(package_dir: str) -> str:
    """Force `import tonypi_rl` to *this* tree.

    Cluster layout that used to break expert training:
      HN/tonypi_rl/            stock PPO already running (old TonyPiNavEnv)
      HN/tonypi_rl_expert/     this train.py + expert env
    PYTHONPATH=HN made spawn workers import the in-flight package, which
    rejects `use_geodesic_reward`. Bind by file location, not folder name.
    """
    package_dir = os.path.abspath(package_dir)
    init_py = os.path.join(package_dir, "__init__.py")
    if not os.path.isfile(init_py):
        raise FileNotFoundError(f"tonypi_rl __init__.py not found in {package_dir}")

    existing = sys.modules.get("tonypi_rl")
    if existing is not None:
        existing_dir = os.path.abspath(os.path.dirname(getattr(existing, "__file__", "") or ""))
        if existing_dir == package_dir:
            return package_dir
        for key in [k for k in list(sys.modules) if k == "tonypi_rl" or k.startswith("tonypi_rl.")]:
            del sys.modules[key]

    spec = importlib.util.spec_from_file_location(
        "tonypi_rl",
        init_py,
        submodule_search_locations=[package_dir],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load tonypi_rl from {init_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tonypi_rl"] = module
    spec.loader.exec_module(module)
    return package_dir


def _bootstrap_repo(repo_root: str, package_dir: str = None):
    """Spawn workers do not inherit the parent sys.path insert."""
    repo_root = os.path.abspath(repo_root)
    package_dir = os.path.abspath(package_dir or current_dir)
    if repo_root in sys.path:
        sys.path.remove(repo_root)
    sys.path.insert(0, repo_root)
    py_path = os.environ.get("PYTHONPATH", "")
    if repo_root not in py_path.split(os.pathsep):
        os.environ["PYTHONPATH"] = repo_root + (os.pathsep + py_path if py_path else "")
    _bind_tonypi_rl(package_dir)


_bind_tonypi_rl(current_dir)

from tonypi_rl.tonypi_env import TonyPiEnv

NAV_ENV_KEYS = {
    "xml_path", "layout", "goal_mode", "random_spawn", "random_robot_yaw",
    "n_obstacles", "max_episode_steps",
    "primitive_substeps", "arrive_radius", "progress_weight", "arrive_bonus",
    "time_penalty", "collision_penalty",
    "reverse_penalty", "oscillation_penalty", "heading_align_weight",
    "ram_penalty", "collision_streak_limit",
    "use_geodesic_reward", "expert_robot_radius", "expert_grid_res",
    "expert_heading_weight", "geodesic_progress_clip",
}

NAV_REQUIRED_FILES = [
    os.path.join(current_dir, "tonypi_nav_env.py"),
    os.path.join(current_dir, "perception.py"),
    os.path.join(current_dir, "nav_features.py"),
    os.path.join(parent_dir, "tonypi_mujoco", "scene_nav.xml"),
]

NAV_EXPERT_FILES = [
    os.path.join(current_dir, "nav_planner.py"),
    os.path.join(current_dir, "nav_expert.py"),
    os.path.join(current_dir, "guided_ppo.py"),
]


def _require_nav_files(extra=()):
    missing = [p for p in list(NAV_REQUIRED_FILES) + list(extra) if not os.path.isfile(p)]
    if missing:
        listing = "\n  ".join(missing)
        raise FileNotFoundError(
            "Nav training files are missing on this machine. Sync them from the "
            "dev tree, then rerun.\n  " + listing
        )


def _assert_nav_ctor(cls, kwargs):
    """Fail with the loaded file path if spawn grabbed the in-flight stock env."""
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return
    unknown = [k for k in kwargs if k not in params]
    if not unknown:
        return
    src = inspect.getfile(cls)
    raise TypeError(
        f"TonyPiNavEnv rejected {unknown}. Loaded from:\n  {src}\n"
        f"This train.py lives in:\n  {current_dir}\n"
        "Spawn imported a sibling/in-flight `tonypi_rl` (the stock run), not "
        "this tree. Re-sync this directory and start workers from THIS train.py."
    )


def make_env(rank: int, seed: int = 0, env_class: str = "walk", repo_root: str = None,
             package_dir: str = None, **env_kwargs):
    """Utility function for multiprocess vectorized environments."""
    if repo_root is None:
        repo_root = parent_dir
    if package_dir is None:
        package_dir = current_dir

    # Default-arg capture: spawn must not look up `_bootstrap_repo` on a
    # sibling `tonypi_rl.train` (that copy is 1-arg and still training).
    def _init(_boot=_bootstrap_repo, _assert=_assert_nav_ctor):
        os.environ.setdefault("MUJOCO_GL", "egl")
        _boot(repo_root, package_dir)
        if env_class == "nav":
            from tonypi_rl.tonypi_nav_env import TonyPiNavEnv
            _assert(TonyPiNavEnv, env_kwargs)
            env = TonyPiNavEnv(**env_kwargs)
        else:
            from tonypi_rl.tonypi_env import TonyPiEnv
            env = TonyPiEnv(**env_kwargs)
        env = Monitor(env)
        if env_class == "nav":
            # Belt-and-suspenders: Dummy/Subproc env_method uses get_wrapper_attr.
            raw = env.unwrapped
            env.expert_action = raw.expert_action
            env.geodesic_distance = raw.geodesic_distance
            env.expert_path = raw.expert_path
        env.reset(seed=seed + rank)
        return env
    return _init


def load_config(config_path: str) -> tuple[dict, str]:
    """Load and parse YAML configuration file, returning parsed dict and resolved path."""
    resolved_path = config_path
    if not os.path.exists(resolved_path):
        candidates = [
            os.path.join(current_dir, config_path),
            os.path.join(parent_dir, config_path),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                resolved_path = cand
                break
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(resolved_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg, resolved_path


def main():
    default_cfg = os.path.join(current_dir, "configs", "v0_straight_walk.yaml")

    parser = argparse.ArgumentParser(description="Train TonyPi Humanoid Locomotion from YAML Config")
    parser.add_argument("--config", type=str, default=default_cfg,
                        help=f"Path to YAML config file (default: {default_cfg})")
    # Optional CLI overrides
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--n_envs", type=int, default=None, help="Override number of parallel envs")
    parser.add_argument("--device", type=str, default=None, help="Override device (auto, cpu, cuda)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .zip to resume")
    parser.add_argument("--exp_name", type=str, default=None, help="Override experiment run name")
    args = parser.parse_args()

    # Load YAML configuration
    cfg, resolved_cfg_path = load_config(args.config)

    # Extract sections
    exp_cfg = cfg.get("experiment", {})
    env_cfg = cfg.get("environment", {})
    rew_cfg = cfg.get("rewards", {})
    ppo_cfg = cfg.get("ppo", {})
    log_cfg = cfg.get("logging", {})
    curriculum_cfg = cfg.get("curriculum", {})

    # Apply overrides
    exp_name = args.exp_name or exp_cfg.get("name", "tonypi_v0")
    total_timesteps = args.timesteps or ppo_cfg.get("timesteps", 500000)
    n_envs = args.n_envs or ppo_cfg.get("n_envs", 4)
    device = args.device or exp_cfg.get("device", "auto")
    seed = exp_cfg.get("seed", 42)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{exp_name}_{timestamp}"

    # Setup directories
    base_log_dir = _resolve_under_repo(log_cfg.get("log_dir", "tonypi_rl/logs"), parent_dir)
    base_model_dir = _resolve_under_repo(log_cfg.get("model_dir", "tonypi_rl/models"), parent_dir)
    run_log_dir = os.path.join(base_log_dir, run_name)
    run_model_dir = os.path.join(base_model_dir, run_name)

    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(run_model_dir, exist_ok=True)

    # Copy config file to run log directory for reproducibility
    shutil.copyfile(resolved_cfg_path, os.path.join(run_log_dir, "config.yaml"))

    env_class = str(env_cfg.get("env_class", "walk")).lower()
    expert_cfg = cfg.get("expert") or {}
    expert_enabled = env_class == "nav" and bool(expert_cfg.get("enabled", False))
    if env_class == "nav":
        os.environ.setdefault("MUJOCO_GL", "egl")
        _require_nav_files(NAV_EXPERT_FILES if expert_enabled else ())

    print("=" * 72)
    if env_class == "nav":
        print("  TonyPi high-level navigation PPO (action groups + occupancy)")
    else:
        print("  Hiwonder TonyPi 18-DOF Bipedal RL Training (Config-Driven)")
    print("=" * 72)
    print(f"Config File:         {resolved_cfg_path}")
    print(f"Package dir:         {current_dir}")
    print(f"Env class:           {env_class}")
    if env_class == "nav":
        from tonypi_rl.tonypi_nav_env import TonyPiNavEnv
        print(f"Nav env file:        {inspect.getfile(TonyPiNavEnv)}")
        _assert_nav_ctor(
            TonyPiNavEnv,
            {k: env_cfg[k] for k in env_cfg if k in NAV_ENV_KEYS},
        )
    print(f"Curriculum Stage:    {curriculum_cfg.get('stage', 'v0')}")
    print(f"Run Name:            {run_name}")
    print(f"Total Timesteps:     {total_timesteps:,}")
    print(f"Parallel Envs:       {n_envs}")
    if env_class == "nav":
        from tonypi_rl.perception import OCCUPANCY_SIZE
        print(f"Layout/goal:         {env_cfg.get('layout')} / {env_cfg.get('goal_mode')}")
        print(f"Obstacles:           {env_cfg.get('n_obstacles')}")
        print(f"Policy:              MultiInputPolicy (CNN {OCCUPANCY_SIZE}x{OCCUPANCY_SIZE} + MLP vec)")
        if expert_enabled:
            print(f"Expert:              mix={expert_cfg.get('mix_prob')} "
                  f"bc_coef={expert_cfg.get('bc_coef')} "
                  f"warmup={int(expert_cfg.get('bc_warmup_steps', 0)):,} "
                  f"geodesic={env_cfg.get('use_geodesic_reward', False)}")
        else:
            print("Expert:              off (stock PPO)")
    else:
        print(f"Action Mode:         {env_cfg.get('action_mode')} (scale = {env_cfg.get('action_scale')} * pi)")
        print(f"Include Lin Vel:     {env_cfg.get('include_lin_vel', False)} (False = real hardware parity)")
        print(f"Foot Contact Reward: Enabled (alternating foot switch bonus = {rew_cfg.get('foot_switch_reward', 2.0)})")
        print(f"Straight Line Pen:   Yaw = {rew_cfg.get('heading_yaw_weight', 1.2)}, Pos = {rew_cfg.get('straight_line_pos_weight', 2.0)}")
    print(f"Device:              {device}")
    print(f"Logs:                {run_log_dir}")
    print(f"Models:              {run_model_dir}")
    print("=" * 72)

    env_kwargs = {}
    env_kwargs.update(env_cfg)
    env_kwargs.update(rew_cfg)
    env_kwargs.pop("env_class", None)
    if env_class == "nav":
        env_kwargs = {k: v for k, v in env_kwargs.items() if k in NAV_ENV_KEYS}
        if expert_enabled and "heading_weight" in expert_cfg and "expert_heading_weight" not in env_kwargs:
            env_kwargs["expert_heading_weight"] = expert_cfg["heading_weight"]
    else:
        env_kwargs.pop("max_episode_steps", None)

    # Resolve relative xml_path
    if "xml_path" in env_kwargs and env_kwargs["xml_path"]:
        env_kwargs["xml_path"] = _resolve_under_repo(env_kwargs["xml_path"], parent_dir)

    env_fns = [
        make_env(
            i, seed=seed, env_class=env_class,
            repo_root=parent_dir, package_dir=current_dir, **env_kwargs,
        )
        for i in range(n_envs)
    ]
    # spawn: each worker imports mujoco/EGL itself. fork after a parent EGL context dies.
    if n_envs > 1:
        start_method = "spawn" if env_class == "nav" else None
        train_env = SubprocVecEnv(env_fns, start_method=start_method)
        print(f"Vec env:             SubprocVecEnv x{n_envs} (start={start_method or 'default'})")
    else:
        train_env = DummyVecEnv(env_fns)
        print("Vec env:             DummyVecEnv x1")

    eval_env = DummyVecEnv([
        make_env(
            999, seed=seed + 100, env_class=env_class,
            repo_root=parent_dir, package_dir=current_dir, **env_kwargs,
        )
    ])

    # Setup Callbacks
    save_freq = log_cfg.get("save_freq", 25000)
    eval_freq = log_cfg.get("eval_freq", 10000)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq // n_envs, 1000),
        save_path=run_model_dir,
        name_prefix="tonypi_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_model_dir,
        log_path=run_log_dir,
        eval_freq=max(eval_freq // n_envs, 1000),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    if env_class == "nav":
        from tonypi_rl.nav_features import NavFeaturesExtractor
        policy_name = str(ppo_cfg.get("policy", "MultiInputPolicy"))
        policy_kwargs = dict(
            features_extractor_class=NavFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.ReLU,
        )
    else:
        policy_name = "MlpPolicy"
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.Tanh,
        )

    Algo = PPO
    algo_kwargs = {}
    if expert_enabled:
        from tonypi_rl.guided_ppo import GuidedPPO, bc_warmup
        Algo = GuidedPPO
        algo_kwargs = dict(
            bc_coef=float(expert_cfg.get("bc_coef", 0.15)),
            bc_anneal_steps=int(expert_cfg.get("bc_anneal_steps", 2_000_000)),
            mix_prob=float(expert_cfg.get("mix_prob", 0.3)),
            mix_anneal_steps=int(expert_cfg.get("mix_anneal_steps", 1_000_000)),
        )

    # Resume or Create PPO Model
    target_kl = ppo_cfg.get("target_kl", None)
    if target_kl is not None:
        target_kl = float(target_kl)

    resume_path = args.resume
    if resume_path and os.path.exists(resume_path):
        print(f"--> Resuming training from checkpoint: {resume_path}")
        model = Algo.load(
            resume_path,
            env=train_env,
            device=device,
            tensorboard_log=run_log_dir,
        )
        if expert_enabled and isinstance(model, GuidedPPO):
            model.mix_prob0 = float(expert_cfg.get("mix_prob", getattr(model, "mix_prob0", 0.3)))
            model.mix_anneal_steps = int(expert_cfg.get("mix_anneal_steps", getattr(model, "mix_anneal_steps", 8_000_000)))
            model.bc_coef0 = float(expert_cfg.get("bc_coef", getattr(model, "bc_coef0", 0.15)))
            model.bc_anneal_steps = int(expert_cfg.get("bc_anneal_steps", getattr(model, "bc_anneal_steps", 10_000_000)))
        if target_kl is not None:
            model.target_kl = target_kl
        if "n_epochs" in ppo_cfg:
            model.n_epochs = int(ppo_cfg["n_epochs"])
        if "ent_coef" in ppo_cfg:
            model.ent_coef = float(ppo_cfg["ent_coef"])
        if "learning_rate" in ppo_cfg:
            model.learning_rate = float(ppo_cfg["learning_rate"])
    else:
        print(f"--> Initializing fresh {Algo.__name__} policy ({policy_name})...")
        model = Algo(
            policy=policy_name,
            env=train_env,
            learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
            n_steps=int(ppo_cfg.get("n_steps", 2048)),
            batch_size=int(ppo_cfg.get("batch_size", 64)),
            n_epochs=int(ppo_cfg.get("n_epochs", 10)),
            gamma=float(ppo_cfg.get("gamma", 0.99)),
            gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
            clip_range=float(ppo_cfg.get("clip_range", 0.2)),
            ent_coef=float(ppo_cfg.get("ent_coef", 0.005)),
            target_kl=target_kl,
            policy_kwargs=policy_kwargs,
            tensorboard_log=run_log_dir,
            device=device,
            verbose=1,
            **algo_kwargs,
        )

    if expert_enabled and not (resume_path and os.path.exists(resume_path)):
        warmup_steps = int(expert_cfg.get("bc_warmup_steps", 0))
        if warmup_steps > 0:
            from tonypi_rl.guided_ppo import bc_warmup
            bc_warmup(
                model,
                train_env,
                n_transitions=warmup_steps,
                n_epochs=int(expert_cfg.get("bc_epochs", 8)),
                batch_size=int(expert_cfg.get("bc_batch_size", 1024)),
            )

    # Start Training
    resuming = bool(resume_path and os.path.exists(resume_path))
    if resuming:
        already = int(model.num_timesteps)
        remaining = int(total_timesteps) - already
        if remaining <= 0:
            print(f"[!] checkpoint already at {already:,} >= target {total_timesteps:,}; nothing to train")
            remaining = 0
        learn_kwargs = dict(
            total_timesteps=remaining,
            reset_num_timesteps=False,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )
    else:
        learn_kwargs = dict(
            total_timesteps=total_timesteps,
            reset_num_timesteps=True,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )

    print(f"--> Commencing {exp_name} training loop for {total_timesteps:,} steps...")
    if expert_enabled and isinstance(model, GuidedPPO):
        curr_mix = model._mix_prob()
        curr_bc = model._bc_coef()
        mix_ann = getattr(model, "mix_anneal_steps", 0)
        bc_ann = getattr(model, "bc_anneal_steps", 0)
        if resuming:
            print(f"    GuidedPPO clock:     num_timesteps={already:,}  target={total_timesteps:,}  remaining={remaining:,}")
            print(f"    GuidedPPO schedule:  mix_prob={curr_mix:.4f} (0.3 → 0 at mix_anneal={mix_ann:,})")
            print(f"                         bc_coef={curr_bc:.4f} (0.15 → 0 at bc_anneal={bc_ann:,})")
            print(f"                         reset_num_timesteps=False")
        else:
            print(f"    GuidedPPO schedule:  mix_prob={curr_mix:.4f} (anneal={mix_ann:,}), "
                  f"bc_coef={curr_bc:.4f} (anneal={bc_ann:,})")
    try:
        if resuming and remaining <= 0:
            print("[*] Skipping model.learn() since remaining timesteps <= 0")
        else:
            model.learn(**learn_kwargs)
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user.")
    except (ValueError, Exception) as e:
        emergency_path = os.path.join(run_model_dir, "tonypi_emergency_nan.zip")
        try:
            model.save(emergency_path)
            print(f"\n[!] Emergency save: model checkpoint saved to {emergency_path} due to error: {e}")
        except Exception as save_err:
            print(f"\n[!] Emergency save failed: {save_err}")
        raise e

    # Save final model
    final_path = os.path.join(run_model_dir, "tonypi_final_model.zip")
    model.save(final_path)
    print(f"--> Training finished! Final model saved to: {final_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
