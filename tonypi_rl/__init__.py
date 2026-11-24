"""TonyPi Reinforcement Learning package."""
from gymnasium.envs.registration import register

register(
    id="TonyPi-v0",
    entry_point="tonypi_rl.tonypi_env:TonyPiEnv",
    max_episode_steps=1000,
)

register(
    id="TonyPiNav-v0",
    entry_point="tonypi_rl.tonypi_nav_env:TonyPiNavEnv",
    max_episode_steps=160,
)
