import gym
from torch import nn as nn
import os
import numpy as np


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
from rlkit.launchers import common
from rlkit.samplers.data_collector import MdpStepCollector, MdpPathCollector

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import TransposeImage, make_vec_envs
from a2c_ppo_acktr.model import CNNBase, create_output_distribution
from a2c_ppo_acktr.wrappers import (
    WrappedPolicy,
    PPOTrainer,
    RolloutStepCollector,
    HierarchicalStepCollector,
    TorchIkostrikovRLAlgorithm,
)
from a2c_ppo_acktr import distributions

from gym_agent.learn_plan_policy import LearnPlanPolicy


def experiment(variant):
    common.initialise(variant)

    expl_envs, eval_envs = common.create_environments(variant)

    (
        obs_shape,
        obs_space,
        action_space,
        n,
        mlp,
        channels,
        fc_input,
    ) = common.get_spaces(expl_envs)

    # CHANGE TO ORDINAL ACTION SPACE
    action_space = gym.spaces.Box(-np.inf, np.inf, (8,))
    expl_envs.action_space = action_space
    eval_envs.action_space = action_space

    base = common.create_networks(variant, n, mlp, channels, fc_input)

    bernoulli_dist = distributions.Bernoulli(base.output_size, 4)
    passenger_dist = distributions.Categorical(base.output_size, 5)
    delivered_dist = distributions.Categorical(base.output_size, 5)
    continuous_dist = distributions.DiagGaussian(base.output_size, 2)
    dist = distributions.DistributionGeneratorTuple(
        (bernoulli_dist, continuous_dist, passenger_dist, delivered_dist)
    )

    eval_policy = LearnPlanPolicy(
        WrappedPolicy(
            obs_shape,
            action_space,
            ptu.device,
            base=base,
            deterministic=True,
            dist=dist,
            num_processes=variant["num_processes"],
            obs_space=obs_space,
        ),
        num_processes=variant["num_processes"],
        vectorised=True,
        json_to_screen=expl_envs.observation_space.converter,
    )
    expl_policy = LearnPlanPolicy(
        WrappedPolicy(
            obs_shape,
            action_space,
            ptu.device,
            base=base,
            deterministic=False,
            dist=dist,
            num_processes=variant["num_processes"],
            obs_space=obs_space,
        ),
        num_processes=variant["num_processes"],
        vectorised=True,
        json_to_screen=expl_envs.observation_space.converter,
    )

    eval_path_collector = HierarchicalStepCollector(
        eval_envs,
        eval_policy,
        ptu.device,
        max_num_epoch_paths_saved=variant["algorithm_kwargs"][
            "num_eval_steps_per_epoch"
        ],
        num_processes=variant["num_processes"],
        render=variant["render"],
        gamma=1,
        no_plan_penalty=True,
    )
    expl_path_collector = HierarchicalStepCollector(
        expl_envs,
        expl_policy,
        ptu.device,
        max_num_epoch_paths_saved=variant["num_steps"],
        num_processes=variant["num_processes"],
        render=variant["render"],
        gamma=variant["trainer_kwargs"]["gamma"],
        no_plan_penalty=variant.get("no_plan_penalty", False),
    )
    # added: created rollout(5,1,(4,84,84),Discrete(6),1), reset env and added obs to rollout[step]

    trainer = PPOTrainer(actor_critic=expl_policy.learner, **variant["trainer_kwargs"])
    # missing: by this point, rollout back in sync.
    replay_buffer = EnvReplayBuffer(variant["replay_buffer_size"], expl_envs)
    # added: replay buffer is new
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
    # missing: device back in sync
    algorithm.train()

