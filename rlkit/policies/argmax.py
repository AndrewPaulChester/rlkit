"""
Torch argmax policy
"""
import numpy as np
from torch import nn
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.policies.base import Policy


class ArgmaxDiscretePolicy(nn.Module, Policy):
    def __init__(self, qf):
        super().__init__()
        self.qf = qf

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = ptu.from_numpy(obs).float()
        q_values = self.qf(obs).squeeze(0)
        # try:
        #     obs = ptu.from_numpy(obs).float()
        #     q_values = self.qf(obs).squeeze(0)
        # except TypeError as e:
        #     action_obs = _flatten_tuple(_convert_to_torch(obs))
        #     q_values = self.qf(action_obs).squeeze(0)
        q_values_np = ptu.get_numpy(q_values)
        return (q_values_np.argmax(), False), {}
