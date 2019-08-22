import gym
from torch import nn as nn
import os

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
from rlkit.samplers.data_collector import MdpStepCollector, MdpPathCollector

from a2c_ppo_acktr.envs import TransposeImage, make_vec_envs
from a2c_ppo_acktr.model import CNNBase, create_output_distribution
from a2c_ppo_acktr import utils
from a2c_ppo_acktr.wrappers import (
    WrappedPolicy,
    A2CTrainer,
    RolloutStepCollector,
    TorchIkostrikovRLAlgorithm,
)


def experiment(variant):
    setup_logger("name-of-experiment", variant=variant)
    ptu.set_gpu_mode(True)
    # missing torch seeds + torch.set_num_threads
    log_dir = os.path.expanduser(variant["log_dir"])
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    # expl_env = gym.make(variant["env_name"])
    expl_envs = make_vec_envs(
        variant["env_name"],
        variant["seed"],
        variant["num_processes"],
        variant["gamma"],
        variant["log_dir"],  # probably change this?
        ptu.device,
        False,
        1,
    )
    # eval_env = gym.make(variant["env_name"])
    eval_envs = make_vec_envs(
        variant["env_name"],
        variant["seed"],
        variant["num_processes"],
        variant["gamma"],
        variant["log_dir"],
        ptu.device,
        False,
        1,
    )
    obs_shape = expl_envs.observation_space.shape
    # if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:  # convert WxHxC into CxWxH
    #     expl_env = TransposeImage(expl_env, op=[2, 0, 1])
    #     eval_env = TransposeImage(eval_env, op=[2, 0, 1])
    # obs_shape = expl_env.observation_space.shape

    channels, obs_width, obs_height = obs_shape
    action_space = expl_envs.action_space

    base_kwargs = {"num_inputs": channels, "recurrent": variant["recurrent_policy"]}
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
    base = CNNBase(**base_kwargs)

    dist = create_output_distribution(action_space, base.output_size)

    eval_policy = WrappedPolicy(
        obs_shape, action_space, base=base, deterministic=True, dist=dist
    )
    expl_policy = WrappedPolicy(
        obs_shape, action_space, base=base, deterministic=False, dist=dist
    )

    # qf_criterion = nn.MSELoss()
    # eval_policy = ArgmaxDiscretePolicy(qf)
    # expl_policy = PolicyWrappedWithExplorationStrategy(
    #     AnnealedEpsilonGreedy(
    #         expl_env.action_space, anneal_rate=variant["anneal_rate"]
    #     ),
    #     eval_policy,
    # )

    eval_path_collector = RolloutStepCollector(
        eval_envs,
        eval_policy,
        ptu.device,
        max_num_epoch_paths_saved=variant["algorithm_kwargs"][
            "num_eval_steps_per_epoch"
        ],
        num_processes=variant["num_processes"],
    )
    expl_path_collector = RolloutStepCollector(
        expl_envs,
        expl_policy,
        ptu.device,
        max_num_epoch_paths_saved=variant["num_steps"],
        num_processes=variant["num_processes"],
    )

    trainer = A2CTrainer(actor_critic=expl_policy, **variant["trainer_kwargs"])

    replay_buffer = EnvReplayBuffer(variant["replay_buffer_size"], expl_envs)

    algorithm = TorchIkostrikovRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_envs,
        evaluation_env=eval_envs,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant["algorithm_kwargs"],
        # batch_size,
        # max_path_length,
        # num_epochs,
        # num_eval_steps_per_epoch,
        # num_expl_steps_per_train_loop,
        # num_trains_per_train_loop,
        # num_train_loops_per_epoch=1,
        # min_num_steps_before_training=0,
    )

    algorithm.to(ptu.device)

    algorithm.train()

