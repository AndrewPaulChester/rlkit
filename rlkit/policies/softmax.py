"""
Torch argmax policy
"""
import numpy as np
from scipy.special import softmax
from torch import nn

import rlkit.torch.pytorch_util as ptu
from rlkit.policies.base import Policy


class SoftmaxDiscretePolicy(nn.Module, Policy):
    def __init__(self, qf, temperature=1):
        super().__init__()
        self.qf = qf
        self.temperature = temperature

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = ptu.from_numpy(obs).float()
        q_values = self.qf(obs).squeeze(0)
        q_values_np = ptu.get_numpy(q_values)
        probabilities = softmax(q_values_np / self.temperature)
        action = np.random.choice(np.arange(0, len(probabilities)), p=probabilities)
        print(f"chose action {action} with probability {probabilities[action]}")
        return ((action, False), {})
