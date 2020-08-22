import gym
from torch import nn as nn
import os
import pickle
import numpy as np


from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import (
    EpsilonGreedy,
    AnnealedEpsilonGreedy,
    LinearEpsilonGreedy,
)
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.policies.softmax import SoftmaxDiscretePolicy
from rlkit.torch.dqn.dqn import DQNTrainer
from rlkit.torch.dqn.double_dqn import DoubleDQNTrainer
from rlkit.torch.conv_networks import CNN
from rlkit.torch.networks import Mlp
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.data_management.env_replay_buffer import PlanReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.launchers import common
from rlkit.samplers.rollout_functions import intermediate_rollout
from rlkit.samplers.data_collector import MdpStepCollector, MdpPathCollector

from rlkit.samplers.data_collector.path_collector import IntermediatePathCollector

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import TransposeImage, make_vec_envs
from a2c_ppo_acktr.model import CNNBase, create_output_distribution

from a2c_ppo_acktr.wrappers.policies import WrappedPolicy, MultiPolicy
from a2c_ppo_acktr.wrappers.trainers import PPOTrainer, MultiTrainer, DummyTrainer
from a2c_ppo_acktr.wrappers.data_collectors import (
    RolloutStepCollector,
    HierarchicalStepCollector,
    ThreeTierStepCollector,
)
from a2c_ppo_acktr.wrappers.algorithms import TorchIkostrikovRLAlgorithm
from a2c_ppo_acktr import distributions

from gym_agent.learn_plan_policy import LearnPlanPolicy
from gym_agent.controller import CraftController, PretrainedController
from gym_agent.planner import ENHSPPlanner, DummyHierarchicalPlanner


def experiment(variant):
    # common.initialise(variant)

    setup_logger("name-of-experiment", variant=variant)
    ptu.set_gpu_mode(True)

    expl_env = gym.make(variant["env_name"], seed=5)
    eval_env = gym.make(variant["env_name"], seed=5)

    if variant["action_space"] == "planner":
        ancillary_goal_size = 16
        planner = ENHSPPlanner()
    elif variant["action_space"] == "skills":
        ancillary_goal_size = 14
        planner = DummyHierarchicalPlanner()

    ANCILLARY_GOAL_SIZE = ancillary_goal_size
    SYMBOLIC_ACTION_SIZE = (
        12
    )  # Size of embedding (ufva/multihead) for goal space direction to controller
    GRID_SIZE = 31

    action_dim = ANCILLARY_GOAL_SIZE
    symbolic_action_space = gym.spaces.Discrete(ANCILLARY_GOAL_SIZE)
    symb_env = gym.make(variant["env_name"])
    symb_env.action_space = symbolic_action_space

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
        output_size=action_dim,
        hidden_sizes=[256, 256],
        init_w=variant["init_w"],
        b_init_value=variant["b_init_value"],
    )
    target_qf = Mlp(
        input_size=n,
        output_size=action_dim,
        hidden_sizes=[256, 256],
        init_w=variant["init_w"],
        b_init_value=variant["b_init_value"],
    )

    # collect
    filepath = "/home/achester/anaconda3/envs/goal-gen/.guild/runs/e77c75eed02e4b38a0a308789fbfcbd8/data/params.pkl"  # collect
    with (open(filepath, "rb")) as openfile:
        while True:
            try:
                policies = pickle.load(openfile)
            except EOFError:
                break

    loaded_collect_policy = policies["exploration/policy"]
    loaded_collect_policy.rnn_hxs = loaded_collect_policy.rnn_hxs[0].unsqueeze(0)
    eval_collect = CraftController(loaded_collect_policy, n=GRID_SIZE)
    expl_collect = CraftController(loaded_collect_policy, n=GRID_SIZE)

    # other
    # filepath = "/home/achester/anaconda3/envs/goal-gen/.guild/runs/cf5c31afe0724acd8f6398d77a80443e/data/params.pkl"  # other (RC 28)
    filepath = "/home/achester/anaconda3/envs/goal-gen/.guild/runs/4989f4bcbadb4ac58c3668c068d63225/data/params.pkl"  # other (RC 55)
    # filepath = "/home/achester/Documents/misc/craft-model/params.pkl"
    with (open(filepath, "rb")) as openfile:
        while True:
            try:
                policies = pickle.load(openfile)
            except EOFError:
                break

    loaded_other_policy = policies["exploration/policy"]
    loaded_other_policy.rnn_hxs = loaded_other_policy.rnn_hxs[0].unsqueeze(0)
    eval_other = CraftController(loaded_other_policy, n=GRID_SIZE)
    expl_other = CraftController(loaded_other_policy, n=GRID_SIZE)

    eval_controller = PretrainedController([eval_collect, eval_other])
    expl_controller = PretrainedController([expl_collect, expl_other])

    function_env = gym.make(variant["env_name"])

    qf_criterion = nn.MSELoss()
    if variant["softmax"]:
        expl_learner = SoftmaxDiscretePolicy(qf, variant["temperature"])
        eval_learner = SoftmaxDiscretePolicy(qf, variant["temperature"])
    else:
        expl_learner = ArgmaxDiscretePolicy(qf)
        eval_learner = PolicyWrappedWithExplorationStrategy(
            EpsilonGreedy(symbolic_action_space, 0.05), ArgmaxDiscretePolicy(qf)
        )

    expl_learner = PolicyWrappedWithExplorationStrategy(
        LinearEpsilonGreedy(
            symbolic_action_space, anneal_schedule=variant["anneal_schedule"]
        ),
        expl_learner,
    )

    eval_policy = LearnPlanPolicy(
        eval_learner,
        planner,
        eval_controller,
        num_processes=1,
        vectorised=False,
        env=function_env,
    )

    expl_policy = LearnPlanPolicy(
        expl_learner,
        planner,
        expl_controller,
        num_processes=1,
        vectorised=False,
        env=function_env,
    )

    eval_path_collector = IntermediatePathCollector(
        eval_env,
        eval_policy,
        rollout=intermediate_rollout,
        gamma=1,
        render=variant["render"],
        naive_discounting=variant["trainer_kwargs"]["naive_discounting"],
        experience_interval=variant["experience_interval"],
    )
    expl_path_collector = IntermediatePathCollector(
        expl_env,
        expl_policy,
        rollout=intermediate_rollout,
        gamma=variant["trainer_kwargs"]["discount"],
        render=variant["render"],
        naive_discounting=variant["trainer_kwargs"]["naive_discounting"],
        experience_interval=variant["experience_interval"],
    )

    if variant["double_dqn"]:
        trainer = DoubleDQNTrainer(
            qf=qf,
            target_qf=target_qf,
            qf_criterion=qf_criterion,
            **variant["trainer_kwargs"]
        )
    else:
        trainer = DQNTrainer(
            qf=qf,
            target_qf=target_qf,
            qf_criterion=qf_criterion,
            **variant["trainer_kwargs"]
        )
    replay_buffer = PlanReplayBuffer(variant["replay_buffer_size"], symb_env)

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
