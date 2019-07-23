import gym
from torch import nn as nn


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


def experiment(variant):
    setup_logger("name-of-experiment", variant=variant)
    ptu.set_gpu_mode(True)

    expl_env = gym.make(variant["env_name"])
    eval_env = gym.make(variant["env_name"])
    obs_dim = expl_env.observation_space.shape[1]
    channels = expl_env.observation_space.shape[0]
    action_dim = eval_env.action_space.n

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
    eval_policy = ArgmaxDiscretePolicy(qf)
    expl_policy = PolicyWrappedWithExplorationStrategy(
        AnnealedEpsilonGreedy(
            expl_env.action_space, anneal_rate=variant["anneal_rate"]
        ),
        eval_policy,
    )
    eval_path_collector = MdpPathCollector(eval_env, eval_policy)
    expl_path_collector = MdpPathCollector(expl_env, expl_policy)
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
