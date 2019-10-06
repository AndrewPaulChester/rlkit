import os
import gym

from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu

from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr import utils
from a2c_ppo_acktr.model import CNNBase, MLPBase

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
        variant["gamma"],
        variant["log_dir"],
        ptu.device,
        False,
        1,
        pytorch=False,
    )
    return expl_envs, eval_envs


def get_spaces(expl_envs):
    if isinstance(expl_envs.observation_space, Json):
        obs_space = expl_envs.observation_space.image
    else:
        obs_space = expl_envs.observation_space
    # if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:  # convert WxHxC into CxWxH
    #     expl_env = TransposeImage(expl_env, op=[2, 0, 1])
    #     eval_env = TransposeImage(eval_env, op=[2, 0, 1])
    # obs_shape = expl_env.observation_space.shape
    mlp = False
    n = None
    channels = None
    fc_input = None

    if isinstance(obs_space, gym.spaces.Tuple):
        obs_shape = obs_space[0].shape
        channels, obs_width, obs_height = obs_shape
        fc_input = obs_space[1].shape[0]
    elif len(obs_space.shape) == 1:
        obs_shape = obs_space.shape
        n = obs_shape[0]
        mlp = True
    else:
        obs_shape = obs_space.shape
        channels, obs_width, obs_height = obs_shape
        fc_input = 0

    action_space = expl_envs.action_space

    return (obs_shape, obs_space, action_space, n, mlp, channels, fc_input)


def create_networks(variant, n, mlp, channels, fc_input):
    if mlp:
        base = MLPBase(n)
    else:
        base_kwargs = {
            "num_inputs": channels,
            "recurrent": variant["recurrent_policy"],
            "fc_size": fc_input,
        }
        base = CNNBase(**base_kwargs)
    return base
