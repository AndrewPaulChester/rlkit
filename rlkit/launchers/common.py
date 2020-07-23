import os
import gym

from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu

from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr import utils
from a2c_ppo_acktr.model import CNNBase, MLPBase
from a2c_ppo_acktr import distributions

from gym_taxi.utils.spaces import Json


def initialise(variant):
    setup_logger("name-of-experiment", variant=variant)
    ptu.set_gpu_mode(True)
    log_dir = os.path.expanduser(variant["log_dir"])
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)


def create_environments(variant):
    expl_envs = make_vec_envs(
        variant["env_name"],
        variant["seed"],
        variant["num_processes"],
        variant["gamma"],
        variant["log_dir"],  # probably change this?
        ptu.device,
        False,
        1,
        pytorch=False,
    )
    # eval_env = gym.make(variant["env_name"])
    eval_envs = make_vec_envs(
        variant["env_name"],
        variant["seed"],
        variant["num_processes"],
        1,
        variant["log_dir"],
        ptu.device,
        True,  # allow_early resets
        1,
        pytorch=False,
    )
    return expl_envs, eval_envs


def get_spaces(expl_envs):
    n = None
    mlp = False
    channels = None
    fc_input = None

    if isinstance(expl_envs.observation_space, Json):
        obs_space = expl_envs.observation_space.image
        n = expl_envs.observation_space.grid_size
    else:
        obs_space = expl_envs.observation_space

    if isinstance(obs_space, gym.spaces.Tuple):
        obs_shape = obs_space[0].shape
        channels, obs_width, obs_height = obs_shape
        fc_input = obs_space[1].shape[0]
    elif len(obs_space.shape) == 1:
        obs_shape = obs_space.shape
        n = obs_shape[0]
        fc_input = 0
        mlp = True
    else:
        obs_shape = obs_space.shape
        channels, obs_width, obs_height = obs_shape
        fc_input = 0

    action_space = expl_envs.action_space

    return (obs_shape, obs_space, action_space, n, mlp, channels, fc_input)


def create_networks(variant, n, mlp, channels, fc_input, conv=None, ikostrikov=True):
    if mlp:
        if fc_input:
            base = create_MLP(n + fc_input, ikostrikov)
        else:
            base = create_MLP(n, ikostrikov)
    else:
        base_kwargs = {
            "num_inputs": channels,
            "recurrent": variant["recurrent_policy"],
            "fc_size": fc_input,
            "conv": conv,
        }
        base = create_CNN(base_kwargs, ikostrikov)
    return base


def create_MLP(n, ikostrikov=True):
    if ikostrikov:
        return MLPBase(n)


def create_CNN(base_kwargs, ikostrikov=True):
    if ikostrikov:
        return CNNBase(**base_kwargs)
    raise NotImplementedError("No current support for autogenerated CNNs for RLKit")


def create_symbolic_action_distributions(action_space, base_output_size):
    if action_space == "full":
        bernoulli_dist = distributions.Bernoulli(base_output_size, 2)
        item_dist = distributions.Categorical(base_output_size, 6)
        quantity_dist = distributions.Categorical(base_output_size, 5)
        move_dist = distributions.Categorical(base_output_size, 4)
        # clear_dist = distributions.Categorical(base_output_size, 4)
        dist = distributions.DistributionGeneratorTuple(
            (bernoulli_dist, item_dist, quantity_dist, move_dist)
        )
    elif action_space == "move-only":
        bernoulli_dist = distributions.Bernoulli(base_output_size, 1)
        move_dist = distributions.Categorical(base_output_size, 4)
        dist = distributions.DistributionGeneratorTuple((bernoulli_dist, move_dist))
    elif action_space == "move-continuous":
        bernoulli_dist = distributions.Bernoulli(base_output_size, 1)
        move_dist = distributions.DiagGaussian(base_output_size, 2)
        dist = distributions.DistributionGeneratorTuple((bernoulli_dist, move_dist))
    elif action_space == "move-uniform":
        bernoulli_dist = distributions.Bernoulli(base_output_size, 1)
        move_x = distributions.Categorical(base_output_size, 9)
        move_y = distributions.Categorical(base_output_size, 9)
        dist = distributions.DistributionGeneratorTuple(
            (bernoulli_dist, move_x, move_y)
        )
    elif action_space == "rooms":
        action_dist = distributions.Categorical(base_output_size, 3)
        move_x = distributions.Categorical(base_output_size, 5)
        move_y = distributions.Categorical(base_output_size, 5)
        item_dist = distributions.Categorical(base_output_size, 6)
        quantity_dist = distributions.Categorical(base_output_size, 5)
        dist = distributions.DistributionGeneratorTuple(
            (action_dist, move_x, move_y, item_dist, quantity_dist)
        )
    return dist
