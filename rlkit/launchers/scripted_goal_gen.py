import gym
from torch import nn as nn
import json

from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import (
    EpsilonGreedy,
    AnnealedEpsilonGreedy,
)
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.torch.dqn.dqn import DQNTrainer
from rlkit.torch.conv_networks import CNN
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.samplers.rollout_functions import hierarchical_rollout
from rlkit.policies.base import Policy

from gym_taxi.utils.config import FIXED_GRID_SIZE, DISCRETE_ENVIRONMENT_STATES

SYMBOLIC_ACTION_COUNT = 4 * (FIXED_GRID_SIZE * FIXED_GRID_SIZE + 1)

from gym_agent.learn_plan_policy import LearnPlanPolicy


class ScriptedPolicy(nn.Module, Policy):
    def __init__(self, qf):
        super().__init__()
        self.qf = qf
        self.json_requested = True

    def get_action(self, obs):
        obs = json.loads(obs)

        action = {
            "delivered": [p["pid"] for p in obs["passenger"]],
            "empty": False,
            "passenger": None,
            "location": None,
        }
        return (action, False), {"subgoal": 1}


def experiment(variant):
    setup_logger("name-of-experiment", variant=variant)
    ptu.set_gpu_mode(True)

    expl_env = gym.make(variant["env_name"])
    eval_env = gym.make(variant["env_name"])
    obs_dim = expl_env.observation_space.image.shape[1]
    channels = expl_env.observation_space.image.shape[0]
    action_dim = SYMBOLIC_ACTION_COUNT
    symbolic_action_space = gym.spaces.Discrete(SYMBOLIC_ACTION_COUNT)
    symb_env = gym.make(variant["env_name"])
    symb_env.action_space = symbolic_action_space

    qf = CNN(
        input_width=obs_dim,
        input_height=obs_dim,
        input_channels=channels,
        output_size=action_dim,
        kernel_sizes=[8, 4],
        n_channels=[16, 32],
        strides=[4, 2],
        paddings=[0, 0],
        hidden_sizes=[256],
    )
    target_qf = CNN(
        input_width=obs_dim,
        input_height=obs_dim,
        input_channels=channels,
        output_size=action_dim,
        kernel_sizes=[8, 4],
        n_channels=[16, 32],
        strides=[4, 2],
        paddings=[0, 0],
        hidden_sizes=[256],
    )
    qf_criterion = nn.MSELoss()

    eval_learner_policy = ScriptedPolicy(qf)
    expl_learner_policy = ScriptedPolicy(qf)

    eval_policy = LearnPlanPolicy(eval_learner_policy)
    expl_policy = LearnPlanPolicy(expl_learner_policy)
    eval_path_collector = MdpPathCollector(
        eval_env, eval_policy, rollout=hierarchical_rollout, render=True
    )
    expl_path_collector = MdpPathCollector(
        expl_env, expl_policy, rollout=hierarchical_rollout, render=True
    )
    trainer = DQNTrainer(
        qf=qf,
        target_qf=target_qf,
        qf_criterion=qf_criterion,
        **variant["trainer_kwargs"]
    )
    replay_buffer = EnvReplayBuffer(variant["replay_buffer_size"], symb_env)
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant["algorithm_kwargs"]
    )
    algorithm.to(ptu.device)
    algorithm.train()
