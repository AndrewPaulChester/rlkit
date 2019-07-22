import numpy as np

import math
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.core.trainer import Trainer
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector, MdpPathCollector


class StubTrainer(Trainer):
    def train(self, data):
        pass


class StubReplayBuffer(ReplayBuffer):
    def add_sample(
        self, observation, action, reward, next_observation, terminal, **kwargs
    ):
        pass

    def terminate_episode(self):
        pass

    def num_steps_can_sample(self, **kwargs):
        pass

    def random_batch(self, batch_size):
        pass


class LogPathCollector(MdpPathCollector):
    def __init__(
        self,
        env,
        policy,
        max_num_epoch_paths_saved=None,
        render=False,
        render_kwargs=None,
        num_processes=1,
    ):
        super().__init__(env, policy)
        self.actions = [[] for _ in range(num_processes)]
        self.explored = [[] for _ in range(num_processes)]
        self.rewards = [[] for _ in range(num_processes)]

    def add_step(self, actions, action_log_probs, rewards, done):

        actions = actions.cpu().squeeze(1).numpy()
        rewards = rewards.cpu().squeeze(1).numpy()
        explored = action_log_probs.cpu().squeeze(1).numpy() < math.log(0.5)
        paths = []

        for i in range(len(actions)):
            self.actions[i].append(actions[i])
            self.rewards[i].append(rewards[i])
            self.explored[i].append(explored[i])

            if done[i]:
                acts = np.array(self.actions[i])
                if len(acts.shape) == 1:
                    acts = np.expand_dims(acts, 1)
                paths.append(
                    dict(
                        observations={},
                        actions=acts,
                        explored=np.array(self.explored[i]).reshape(-1, 1),
                        rewards=np.array(self.rewards[i]).reshape(-1, 1),
                        next_observations={},
                        terminals={},
                        agent_infos={},
                        env_infos={},
                    )
                )
                self.actions[i] = []
                self.explored[i] = []
                self.rewards[i] = []

        if paths:
            self._epoch_paths.extend(paths)
            self._num_paths_total += len(paths)
            self._num_steps_total += sum([len(p) for p in paths])


class LogRLAlgorithm(BaseRLAlgorithm):
    def __init__(self, exploration_env=None, evaluation_env=None, num_processes=1):
        trainer = StubTrainer()
        exploration_data_collector = LogPathCollector(
            None, None, num_processes=num_processes
        )
        evaluation_data_collector = LogPathCollector(
            None, None, num_processes=num_processes
        )
        replay_buffer = StubReplayBuffer()
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )

    def _train(self,):
        pass

    def training_mode(self, mode):
        pass

