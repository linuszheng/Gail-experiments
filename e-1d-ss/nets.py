

from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Type, cast

import gym
import numpy as np
import torch
from gym import spaces
from stable_baselines3.common import preprocessing
from torch import nn
from imitation.rewards.reward_nets import RewardNet
from imitation.util import networks, util


torch.set_printoptions(precision=3)




class MyRewardNet(RewardNet):
    """MLP that takes as input the state, action, next state and done flag.

    These inputs are flattened and then concatenated to one another. Each input
    can enabled or disabled by the `use_*` constructor keyword arguments.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        features_to_use,
        la_indices,
        **kwargs,
    ):

        super().__init__(observation_space, action_space)
        self.features_to_use = features_to_use
        self.la_indices = la_indices
        combined_size = len(features_to_use)*2 + len(la_indices)*2
        

        full_build_mlp_kwargs: Dict[str, Any] = {
            "hid_sizes": (32, 32),
            # "hid_sizes": (16, 16),
            **kwargs,
            # we do not want the values below to be overridden
            "in_size": combined_size,
            "out_size": 1,
            "squeeze_output": True,
        }

        self.mlp = networks.build_mlp(**full_build_mlp_kwargs)

    def forward(self, state, action, next_state, done):
        cur_obs_and_la = torch.flatten(state[:,self.features_to_use+self.la_indices], 1)
        next_obs_and_la = torch.flatten(next_state[:,self.features_to_use+self.la_indices], 1)
        inputs = torch.cat((cur_obs_and_la, next_obs_and_la), axis=1)
        outputs = self.mlp(inputs)
        assert outputs.shape == state.shape[:1]
        return outputs