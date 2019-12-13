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

from a2c_ppo_acktr.wrappers.policies import WrappedPolicy, MultiPolicy
from a2c_ppo_acktr.wrappers.trainers import PPOTrainer, MultiTrainer
from a2c_ppo_acktr.wrappers.data_collectors import (
    RolloutStepCollector,
    HierarchicalStepCollector,
    ThreeTierStepCollector,
)
from a2c_ppo_acktr.wrappers.algorithms import TorchIkostrikovRLAlgorithm
from a2c_ppo_acktr import distributions

from a2c_ppo_acktr import distributions

from gym_agent.learn_plan_policy import LearnPlanPolicy
from gym_agent.controller import CraftController
from gym_agent.planner import ENHSPPlanner


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

    # # CHANGE TO ORDINAL ACTION SPACE
    # action_space = gym.spaces.Box(-np.inf, np.inf, (8,))
    # expl_envs.action_space = action_space
    # eval_envs.action_space = action_space
    ANCILLARY_GOAL_SIZE = 7
    SYMBOLIC_ACTION_SIZE = 12

    base = common.create_networks(variant, n, mlp, channels, fc_input)
    control_base = common.create_networks(
        variant, n, mlp, channels, fc_input + SYMBOLIC_ACTION_SIZE
    )  # for uvfa goal representation

    bernoulli_dist = distributions.Bernoulli(base.output_size, 3)
    item_dist = distributions.Categorical(base.output_size, 6)
    quantity_dist = distributions.Categorical(base.output_size, 5)
    move_dist = distributions.Categorical(base.output_size, 4)
    clear_dist = distributions.Categorical(base.output_size, 4)
    dist = distributions.DistributionGeneratorTuple(
        (bernoulli_dist, item_dist, quantity_dist, move_dist, clear_dist)
    )

    control_dist = distributions.Categorical(base.output_size, 14)

    eval_learner = WrappedPolicy(
        obs_shape,
        action_space,
        ptu.device,
        base=base,
        deterministic=True,
        dist=dist,
        num_processes=variant["num_processes"],
        obs_space=obs_space,
    )

    planner = ENHSPPlanner()

    # multihead
    # eval_controller = CraftController(
    #     MultiPolicy(
    #         obs_shape,
    #         action_space,
    #         ptu.device,
    #         18,
    #         base=base,
    #         deterministic=True,
    #         num_processes=variant["num_processes"],
    #         obs_space=obs_space,
    #     )
    # )

    # expl_controller = CraftController(
    #     MultiPolicy(
    #         obs_shape,
    #         action_space,
    #         ptu.device,
    #         18,
    #         base=base,
    #         deterministic=False,
    #         num_processes=variant["num_processes"],
    #         obs_space=obs_space,
    #     )
    # )

    # uvfa
    eval_controller = CraftController(
        WrappedPolicy(
            obs_shape,
            action_space,
            ptu.device,
            base=control_base,
            dist=control_dist,
            deterministic=True,
            num_processes=variant["num_processes"],
            obs_space=obs_space,
            symbolic_action_size=SYMBOLIC_ACTION_SIZE,
        )
    )

    expl_controller = CraftController(
        WrappedPolicy(
            obs_shape,
            action_space,
            ptu.device,
            base=control_base,
            dist=control_dist,
            deterministic=False,
            num_processes=variant["num_processes"],
            obs_space=obs_space,
            symbolic_action_size=SYMBOLIC_ACTION_SIZE,
        )
    )
    function_env = gym.make(variant["env_name"])

    eval_policy = LearnPlanPolicy(
        eval_learner,
        planner,
        eval_controller,
        num_processes=variant["num_processes"],
        vectorised=True,
        env=function_env,
    )

    expl_learner = WrappedPolicy(
        obs_shape,
        action_space,
        ptu.device,
        base=base,
        deterministic=False,
        dist=dist,
        num_processes=variant["num_processes"],
        obs_space=obs_space,
    )

    expl_policy = LearnPlanPolicy(
        expl_learner,
        planner,
        expl_controller,
        num_processes=variant["num_processes"],
        vectorised=True,
        env=function_env,
    )

    eval_path_collector = ThreeTierStepCollector(
        eval_envs,
        eval_policy,
        ptu.device,
        ANCILLARY_GOAL_SIZE,
        SYMBOLIC_ACTION_SIZE,
        max_num_epoch_paths_saved=variant["algorithm_kwargs"][
            "num_eval_steps_per_epoch"
        ],
        num_processes=variant["num_processes"],
        render=variant["render"],
        gamma=1,
        no_plan_penalty=True,
    )
    expl_path_collector = ThreeTierStepCollector(
        expl_envs,
        expl_policy,
        ptu.device,
        ANCILLARY_GOAL_SIZE,
        SYMBOLIC_ACTION_SIZE,
        max_num_epoch_paths_saved=variant["num_steps"],
        num_processes=variant["num_processes"],
        render=variant["render"],
        gamma=variant["trainer_kwargs"]["gamma"],
        no_plan_penalty=variant.get("no_plan_penalty", False),
    )
    # added: created rollout(5,1,(4,84,84),Discrete(6),1), reset env and added obs to rollout[step]

    learn_trainer = PPOTrainer(
        actor_critic=expl_policy.learner, **variant["trainer_kwargs"]
    )
    control_trainer = PPOTrainer(
        actor_critic=expl_policy.controller.policy, **variant["trainer_kwargs"]
    )
    trainer = MultiTrainer([control_trainer, learn_trainer])
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

    algorithm.train()

