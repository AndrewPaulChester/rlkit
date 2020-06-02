import gym
from torch import nn as nn
import gym_craft.lottery

from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import (
    EpsilonGreedy,
    AnnealedEpsilonGreedy,
    LinearEpsilonGreedy,
)
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.policies.softmax import SoftmaxDiscretePolicy
from rlkit.torch.dqn.dqn import DQNTrainer
from rlkit.launchers import common
from rlkit.torch.networks import Mlp
from rlkit.torch.conv_networks import CNN
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from a2c_ppo_acktr.envs import TransposeImage

from gym_taxi.utils.spaces import Json
from gym_taxi.utils.wrappers import BoxWrapper


def experiment(variant):
    setup_logger("name-of-experiment", variant=variant)
    ptu.set_gpu_mode(True)

    expl_env = gym.make(variant["env_name"])
    eval_env = gym.make(variant["env_name"])

    # OLD - Taxi image env
    # if isinstance(expl_env.observation_space, Json):
    #     expl_env = BoxWrapper(expl_env)
    #     eval_env = BoxWrapper(eval_env)
    #     # obs_shape = expl_env.observation_space.image.shape

    # obs_shape = expl_env.observation_space.shape
    # if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:  # convert WxHxC into CxWxH
    #     expl_env = TransposeImage(expl_env, op=[2, 0, 1])
    #     eval_env = TransposeImage(eval_env, op=[2, 0, 1])

    # obs_shape = expl_env.observation_space.shape
    # channels, obs_width, obs_height = obs_shape
    # action_dim = eval_env.action_space.n

    # qf = CNN(
    #     input_width=obs_width,
    #     input_height=obs_height,
    #     input_channels=channels,
    #     output_size=action_dim,
    #     kernel_sizes=[8, 4],
    #     n_channels=[16, 32],
    #     strides=[4, 2],
    #     paddings=[0, 0],
    #     hidden_sizes=[256],
    # )
    # target_qf = CNN(
    #     input_width=obs_width,
    #     input_height=obs_height,
    #     input_channels=channels,
    #     output_size=action_dim,
    #     kernel_sizes=[8, 4],
    #     n_channels=[16, 32],
    #     strides=[4, 2],
    #     paddings=[0, 0],
    #     hidden_sizes=[256],
    # )

    (
        obs_shape,
        obs_space,
        action_space,
        n,
        mlp,
        channels,
        fc_input,
    ) = common.get_spaces(expl_env)

    qf = Mlp(
        input_size=n,
        output_size=action_space.n,
        hidden_sizes=[256, 256],
        init_w=variant["init_w"],
        b_init_value=variant["b_init_value"],
    )
    target_qf = Mlp(
        input_size=n,
        output_size=action_space.n,
        hidden_sizes=[256, 256],
        init_w=variant["init_w"],
        b_init_value=variant["b_init_value"],
    )

    qf_criterion = nn.MSELoss()

    if variant["softmax"]:
        eval_policy = SoftmaxDiscretePolicy(qf, variant["temperature"])
    else:
        eval_policy = ArgmaxDiscretePolicy(qf)

    expl_policy = PolicyWrappedWithExplorationStrategy(
        LinearEpsilonGreedy(action_space, anneal_schedule=variant["anneal_schedule"]),
        eval_policy,
    )
    eval_path_collector = MdpPathCollector(
        eval_env, eval_policy, render=variant["render"]
    )
    expl_path_collector = MdpPathCollector(
        expl_env, expl_policy, render=variant["render"]
    )
    trainer = DQNTrainer(
        qf=qf,
        target_qf=target_qf,
        qf_criterion=qf_criterion,
        **variant["trainer_kwargs"]
    )
    replay_buffer = EnvReplayBuffer(variant["replay_buffer_size"], expl_env)
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
