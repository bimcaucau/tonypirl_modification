"""Small CNN+MLP features extractor for TonyPiNav Dict observations.

NatureCNN (SB3 default) dies on 64x64: its 8x8-s4 / 4x4-s2 / 3x3 stack
shrinks the map to 0. This one is sized for occupancy (1, 64, 64) + vec (18,).
"""
from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class NavFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        occ_space = observation_space["occupancy"]
        n_vec = int(observation_space["vec"].shape[0])
        n_in = int(occ_space.shape[0])

        self.cnn = nn.Sequential(
            nn.Conv2d(n_in, 8, kernel_size=3, stride=2, padding=1),   # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),     # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),    # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),    # 8 -> 4
            nn.ReLU(),
            nn.Flatten(),   # 32*4*4 = 512
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *occ_space.shape)
            n_cnn = int(self.cnn(dummy).shape[1])

        self.vec_net = nn.Sequential(
            nn.Linear(n_vec, 64),
            nn.ReLU(),
        )
        self.merge = nn.Sequential(
            nn.Linear(n_cnn + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations) -> torch.Tensor:
        cnn_out = self.cnn(observations["occupancy"].float())
        vec_out = self.vec_net(observations["vec"].float())
        return self.merge(torch.cat([cnn_out, vec_out], dim=1))
