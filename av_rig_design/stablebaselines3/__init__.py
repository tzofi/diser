import os

import numpy as np

from stablebaselines3.a2c import A2C
from stablebaselines3.common.utils import get_system_info
from stablebaselines3.ddpg import DDPG
from stablebaselines3.dqn import DQN
from stablebaselines3.her.her_replay_buffer import HerReplayBuffer
from stablebaselines3.ppo import PPO
from stablebaselines3.sac import SAC
from stablebaselines3.td3 import TD3

# Small monkey patch so gym 0.21 is compatible with numpy >= 1.24
# TODO: remove when upgrading to gym 0.26
np.bool = bool  # type: ignore[attr-defined]

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()


def HER(*args, **kwargs):
    raise ImportError(
        "Since Stable Baselines 2.1.0, `HER` is now a replay buffer class `HerReplayBuffer`.\n "
        "Please check the documentation for more information: https://stable-baselines3.readthedocs.io/"
    )


__all__ = [
    "A2C",
    "DDPG",
    "DQN",
    "PPO",
    "SAC",
    "TD3",
    "HerReplayBuffer",
    "get_system_info",
]
