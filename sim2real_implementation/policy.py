"""Policy loader and interface for TonyPi navigation.

Loads the GuidedPPO / PPO discrete policy from a zip checkpoint without requiring
Mujoco or the full simulation environment.
"""
from __future__ import annotations

import json
import logging
import os
import re
import zipfile
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Ensure NavFeaturesExtractor and GuidedPPO are registered/importable for SB3 deserialization
from tonypi_rl.nav_features import NavFeaturesExtractor  # noqa: F401
from tonypi_rl.guided_ppo import GuidedPPO
from stable_baselines3 import PPO

logger = logging.getLogger(__name__)


class DummyNavEnv(gym.Env):
    """Minimal Gymnasium environment replicating TonyPiNavEnv observation and action spaces."""

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Dict({
            "vec": spaces.Box(-np.inf, np.inf, (18,), np.float32),
            "occupancy": spaces.Box(0.0, 1.0, (1, 64, 64), np.float32),
        })

    def reset(self, **kw):
        return {
            "vec": np.zeros(18, dtype=np.float32),
            "occupancy": np.zeros((1, 64, 64), dtype=np.float32),
        }, {}

    def step(self, a):
        raise RuntimeError("DummyNavEnv is for space definition and checkpoint loading only.")


def inspect_checkpoint(zip_path: str) -> dict[str, Any]:
    """Inspect checkpoint data without full load, verifying space contract.

    Refuses any checkpoint that is not:
    - Dict observation space with:
        - occupancy: (1, 64, 64)
        - vec: (18,)
    - Discrete(7) action space
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Checkpoint not found: {zip_path}")

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    info = {
        "path": zip_path,
        "size_mb": size_mb,
        "is_nav": False,
        "timesteps": None,
        "type": "Unknown",
        "valid": False,
        "reason": "",
    }

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            if "data" not in z.namelist():
                info["reason"] = "No 'data' entry found in zip archive"
                return info

            raw_data = z.read("data").decode("utf-8")
            d = json.loads(raw_data)
            info["timesteps"] = d.get("num_timesteps")

            obs_data = d.get("observation_space", {})
            act_data = d.get("action_space", {})

            obs_type = str(obs_data.get(":type:", ""))
            act_type = str(act_data.get(":type:", ""))

            if "Dict" not in obs_type and "dict" not in obs_type.lower():
                if "Box" in obs_type and "Box" in act_type:
                    info["type"] = "Walk (18-DOF Joint Control)"
                else:
                    info["type"] = obs_type or "Non-Dict Model"
                info["reason"] = f"Expected Dict observation space, got {obs_type}"
                return info

            # Check action space
            if "Discrete" not in act_type and "discrete" not in act_type.lower():
                info["reason"] = f"Expected Discrete action space, got {act_type}"
                return info

            act_n = str(act_data.get("n", ""))
            if act_n != "7":
                info["reason"] = f"Expected Discrete(7) action space, got Discrete({act_n})"
                return info

            # Check sub-spaces
            spaces_val = obs_data.get("spaces", "")
            spaces_str = json.dumps(spaces_val) if isinstance(spaces_val, dict) else str(spaces_val)

            if "vec" not in spaces_str or "occupancy" not in spaces_str:
                info["reason"] = f"observation_space missing 'vec' or 'occupancy' keys: {spaces_str}"
                return info

            # Expected shapes: vec is (18,), occupancy is (1, 64, 64)
            if "(18,)" not in spaces_str:
                info["reason"] = f"Expected vec shape (18,), got: {spaces_str}"
                return info

            if "(1, 64, 64)" not in spaces_str and "(1,64,64)" not in spaces_str:
                info["reason"] = f"Expected occupancy shape (1, 64, 64), got: {spaces_str}"
                return info

            info["is_nav"] = True
            info["valid"] = True
            info["type"] = "Nav (Dict 64x64+18 -> Discrete(7))"
            return info

    except Exception as e:
        logger.debug("Fast JSON metadata inspection failed (%s); attempting fallback load to inspect spaces.", e)
        # Fallback: attempt SB3 load into DummyNavEnv and inspect spaces directly
        try:
            env = DummyNavEnv()
            try:
                m = GuidedPPO.load(zip_path, env=env, device="cpu", print_system_info=False)
            except Exception:
                m = PPO.load(zip_path, env=env, device="cpu", print_system_info=False)

            obs_sp = m.observation_space
            act_sp = m.action_space

            if not isinstance(obs_sp, spaces.Dict):
                info["reason"] = f"Fallback load: Expected Dict observation space, got {type(obs_sp)}"
                return info

            if "vec" not in obs_sp.spaces or "occupancy" not in obs_sp.spaces:
                info["reason"] = f"Fallback load: Missing 'vec' or 'occupancy' in spaces: {list(obs_sp.spaces.keys())}"
                return info

            if obs_sp.spaces["vec"].shape != (18,):
                info["reason"] = f"Fallback load: Expected vec shape (18,), got {obs_sp.spaces['vec'].shape}"
                return info

            if obs_sp.spaces["occupancy"].shape != (1, 64, 64):
                info["reason"] = f"Fallback load: Expected occupancy shape (1, 64, 64), got {obs_sp.spaces['occupancy'].shape}"
                return info

            if not isinstance(act_sp, spaces.Discrete) or act_sp.n != 7:
                info["reason"] = f"Fallback load: Expected Discrete(7) action space, got {act_sp}"
                return info

            info["is_nav"] = True
            info["valid"] = True
            info["type"] = "Nav (Dict 64x64+18 -> Discrete(7)) [via load fallback]"
            info["timesteps"] = getattr(m, "num_timesteps", None)
            return info

        except Exception as e_fallback:
            info["reason"] = f"Inspection and fallback load failed: {e_fallback}"
            return info


def load_nav_policy(path: str, device: str = "cpu"):
    """Load SB3 navigation policy from zip checkpoint after validating contract."""
    meta = inspect_checkpoint(path)
    if not meta["valid"]:
        raise ValueError(
            f"Refusing to load checkpoint '{path}': {meta['reason']} (type: {meta['type']})"
        )

    env = DummyNavEnv()
    try:
        model = GuidedPPO.load(path, env=env, device=device, print_system_info=False)
        logger.info("Loaded checkpoint using GuidedPPO: %s (steps: %s)", path, meta["timesteps"])
        return model
    except Exception as e1:
        try:
            model = PPO.load(path, env=env, device=device, print_system_info=False)
            logger.info("Loaded checkpoint using PPO: %s (steps: %s)", path, meta["timesteps"])
            return model
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load checkpoint '{path}' with GuidedPPO ({e1}) and PPO ({e2})"
            ) from e2


def act(model, obs: dict, deterministic: bool = True) -> int:
    """Run model predict and return discrete action integer [0..6]."""
    a, _ = model.predict(obs, deterministic=deterministic)
    return int(np.asarray(a).reshape(-1)[0])
