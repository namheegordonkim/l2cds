import random
from abc import abstractmethod
from collections import deque
from typing import List

import gym
import numpy as np
import torch.cuda
from baselines.common.vec_env import VecEnv
from sklearn.preprocessing import StandardScaler
from torch import nn

from models import NNet
from utils.containers import ModelDict, EnvsContainer, DataDict
from utils.datasets import ExperienceTuple, DatasetSARS, TupleSARS
from utils.keys import ModelKey, DataKey
from utils.utils import device


class SampleCollector:
    """
    Generic sample collector defining collect_samples() operation
    """

    def __init__(self, data_collection_amount: int):
        self.data_collection_amount = data_collection_amount

    @abstractmethod
    def collect_data_dict(self, model_container: ModelDict, envs_container: EnvsContainer) -> DataDict:
        raise NotImplementedError

    def collect_experience_tuple_array(self, model_dict: ModelDict, envs_container: EnvsContainer) -> \
            List[ExperienceTuple]:

        actor: NNet = model_dict.get(ModelKey.actor)
        states = envs_container.states
        _, state_dim = states.shape
        state_scaler = model_dict.get(ModelKey.state_scaler)

        # use sample() for choosing actions to allow generous exploration
        states_tensor = torch.as_tensor(envs_container.states).reshape(-1, state_dim).float().to(device)
        states_scaled_tensor = state_scaler.forward(states_tensor)
        actions, log_probs, _, _ = actor.sample(states_scaled_tensor)
        actions = actions.cpu().detach().numpy()
        log_probs = log_probs.cpu().detach().numpy()
        next_states, rewards, dones, _ = envs_container.envs.step(actions)

        envs_container.states = next_states

        tups = []
        for state, action, log_prob, reward, done, next_state in zip(states, actions, log_probs, rewards,
                                                                     dones, next_states):
            tup = TupleSARS(state, action, log_prob, reward, next_state, done)
            tups.append(tup)
        return tups

    def collect_experience_tuple_matrix(self, model_dict: ModelDict, envs_container: EnvsContainer):
        """
        Return an NxT matrix, each cell containing an experience tuple
        """

        filled = 0
        tup_arrays = []
        while filled < self.data_collection_amount:
            tup_array = np.asarray(self.collect_experience_tuple_array(model_dict, envs_container), dtype=np.object)
            tup_arrays.append(tup_array)
            filled += len(tup_array)
        tup_matrix = np.stack(tup_arrays, axis=1)
        return tup_matrix


class SampleCollectorCumulativeRewards(SampleCollector):

    def __init__(self, data_collection_amount: int, reset_every_collection: bool):
        super().__init__(data_collection_amount)
        self.reset_every_collection = reset_every_collection

    def collect_data_dict(self, model_dict: ModelDict, envs_container: EnvsContainer) -> DataDict:
        if self.reset_every_collection:
            envs_container.states = envs_container.envs.reset()
        tup_matrix = self.collect_experience_tuple_matrix(model_dict, envs_container)
        state_dim = envs_container.env.observation_space.shape[0]
        action_dim = envs_container.env.action_space.shape[0]
        n, t = tup_matrix.shape
        states_matrix = np.zeros((n, t, state_dim), dtype=np.float)
        next_states_matrix = np.zeros((n, t, state_dim), dtype=np.float)
        actions_matrix = np.zeros((n, t, action_dim), dtype=np.float)
        rewards_matrix = np.zeros((n, t), dtype=np.float)
        dones_matrix = np.zeros((n, t), dtype=np.bool)
        # phases_matrix = np.zeros((n, t), dtype=np.int)
        log_probs_matrix = np.zeros((n, t), dtype=np.float)
        value_predictions_matrix = np.zeros((n, t), dtype=np.float)

        critic = model_dict.get(ModelKey.critic)
        state_scaler = model_dict.get(ModelKey.state_scaler)

        for i in range(n):
            for j in range(t):
                tup = tup_matrix[i, j]
                states_matrix[i, j] = tup.state
                actions_matrix[i, j] = tup.action
                next_states_matrix[i, j] = tup.next_state
                rewards_matrix[i, j] = tup.reward
                dones_matrix[i, j] = tup.done
                # phases_matrix[i, j] = tup.phase
                log_probs_matrix[i, j] = tup.log_prob
                state_tensor = torch.as_tensor(tup.state.reshape(1, -1)).float()
                state_scaled_tensor = state_scaler.forward(state_tensor)
                value_prediction, _ = critic.forward(state_scaled_tensor)
                value_predictions_matrix[i, j] = value_prediction.cpu().detach().numpy().squeeze()

        cumulative_rewards_matrix = compute_cumulative_rewards_matrix(value_predictions_matrix, next_states_matrix,
                                                                      rewards_matrix, dones_matrix,
                                                                      critic, state_scaler)
        # scale advantage
        advantages_matrix = cumulative_rewards_matrix - value_predictions_matrix
        advantages_reshaped = advantages_matrix.reshape(-1, 1)
        advantage_scaler = StandardScaler()
        advantages_reshaped_scaled = advantage_scaler.fit_transform(advantages_reshaped)

        data_dict = DataDict(n * t)
        data_dict.set(DataKey.states, states_matrix.reshape(-1, state_dim))
        data_dict.set(DataKey.actions, actions_matrix.reshape(-1, action_dim))
        data_dict.set(DataKey.next_states, next_states_matrix.reshape(-1, state_dim))
        data_dict.set(DataKey.log_probs, log_probs_matrix.reshape(-1, ))
        data_dict.set(DataKey.rewards, rewards_matrix.reshape(-1, ))
        data_dict.set(DataKey.dones, dones_matrix.reshape(-1, ))
        data_dict.set(DataKey.advantages, advantages_reshaped_scaled.reshape(-1, ))
        data_dict.set(DataKey.cumulative_rewards, cumulative_rewards_matrix.reshape(-1, ))

        return data_dict


class SampleCollectorActionReplay(SampleCollector):

    def __init__(self, buffer_length, sample_size):
        super().__init__(0)
        self.buffer = deque(maxlen=buffer_length)
        self.sample_size = sample_size

    def sample_from_buffer(self):
        n_samples = np.min([len(self.buffer), self.sample_size])
        idx = np.random.choice(range(n_samples), n_samples, replace=False)
        buffer_as_list = list(self.buffer)
        return [buffer_as_list[i] for i in idx]

    def collect_data_dict(self, model_dict: ModelDict, envs_container: EnvsContainer,
                          use_action_replay=True) -> DataDict:
        tuple_array = self.collect_experience_tuple_array(model_dict, envs_container)
        tuple_list = np.stack(tuple_array).reshape(-1, ).tolist()
        if use_action_replay:
            tups_from_buffer = self.sample_from_buffer()
            for tup in tuple_list:
                self.buffer.append(tup)
            tuple_list.extend(tups_from_buffer)
        dataset = DatasetSARS.from_tuple_list(tuple_list)
        data_dict = DataDict(len(tuple_array))
        data_dict.set(DataKey.states, dataset.states)
        data_dict.set(DataKey.actions, dataset.actions)
        data_dict.set(DataKey.next_states, dataset.next_states)
        data_dict.set(DataKey.log_probs, dataset.log_probs)
        data_dict.set(DataKey.rewards, dataset.rewards)
        data_dict.set(DataKey.dones, dataset.dones)
        # data_dict.set(DataKey.phases, dataset.phases)
        return data_dict


def collect_sample(env: gym.Env, state: np.ndarray, done: np.ndarray, actor_model: nn.Module, noise=-2.0,
                   random_seed=1):
    # random seed is used to make sure different thread generate different trajectories
    random.seed(random_seed)
    torch.manual_seed(random_seed + 1)
    np.random.seed(random_seed + 2)
    actor_model.set_noise(noise)

    # if there's any done environments, we should reset
    if done:
        state = env.reset().astype(np.float32)

    # generate actions and observes
    mu, _ = actor_model(torch.as_tensor(state).unsqueeze(0))
    action = mu.squeeze().cpu().detach().numpy()
    next_state, reward, done, info = env.step(action)

    return state, action, reward, next_state, done


def collect_sars_samples(envs: VecEnv, states: np.ndarray, actor_model: nn.Module, get_log_probs=False):
    """
    Run sampling (not stepping) across all environments for one timestep
    """
    actor_model = actor_model.to(device)

    # generate actions and observes
    actions, log_probs, _, _ = actor_model.sample(torch.as_tensor(states).float().to(device))
    actions = actions.cpu().detach().numpy()
    log_probs = log_probs.cpu().detach().numpy()
    next_states, rewards, dones, _ = envs.step(actions)

    torch.cuda.empty_cache()

    if get_log_probs:
        return states, actions, log_probs, rewards, next_states, dones
    else:
        return states, actions, rewards, next_states, dones


def collect_samples_until_horizon(envs: VecEnv, actor_model: nn.Module, sampling_func, horizon: int,
                                  get_log_probs=False):
    actor_model.to(device)

    tups = []

    states = envs.reset()
    for _ in range(horizon):
        tup = sampling_func(envs, states, actor_model, get_log_probs=get_log_probs)
        tups.append(tup)

    torch.cuda.empty_cache()

    actor_model.cpu()

    return tups


def compute_cumulative_rewards_matrix(value_predictions_matrix, next_states_matrix, rewards_matrix, dones_matrix,
                                      critic, state_scaler):
    assert (np.array_equal(rewards_matrix.shape, dones_matrix.shape))
    n, t = rewards_matrix.shape
    cumulative_rewards_matrix = np.zeros((n, t), dtype=np.float)
    for i in range(n):
        start = 0
        for j in range(t):
            next_state = next_states_matrix[i, j]
            done = dones_matrix[i, j]
            cumulative_reward = 0
            if done:
                cumulative_reward = 0

            elif j >= t - 1:
                next_state_tensor = torch.as_tensor(next_state).float().reshape(1, -1)
                next_state_scaled_tensor = state_scaler.forward(next_state_tensor)
                cumulative_reward_tensor, _ = critic.forward(next_state_scaled_tensor)
                cumulative_reward = cumulative_reward_tensor.cpu().detach().numpy().squeeze()

            if done or j >= t - 1:
                rewards_array = rewards_matrix[i, start:j + 1]

                cumulative_rewards_array = compute_cumulative_rewards_array(cumulative_reward, rewards_array, 0.99)
                cumulative_rewards_matrix[i, start:j + 1] = cumulative_rewards_array
                start = j + 1

    return cumulative_rewards_matrix


def compute_cumulative_rewards_array(cumulative_reward_prediction, rewards_array, gamma):
    """
    rewards_array is a 1D array of floats, sorted in ascending order of timestamp
    """
    t, = rewards_array.shape
    cumulative_rewards_array = np.zeros_like(rewards_array)
    cumulative_rewards_array[-1] = gamma * cumulative_reward_prediction + rewards_array[-1]
    for i in np.arange(t - 2, -1, -1):
        cumulative_rewards_array[i] = gamma * cumulative_rewards_array[i + 1] + rewards_array[i]
    return cumulative_rewards_array


def compute_gae_array(next_value_prediction, value_predictions_array, rewards_array, gamma, tau):
    """
    rewards_array is a 1D array of floats, sorted in ascending order of timestamp
    GAE: https://arxiv.org/pdf/1506.02438.pdf
    """
    t, = rewards_array.shape
    gae_array = np.zeros_like(rewards_array)
    gae_array[-1] = get_gae_delta(rewards_array[-1], value_predictions_array[-1], next_value_prediction, gamma)
    for i in np.arange(t - 2, -1, -1):
        delta = get_gae_delta(rewards_array[i], value_predictions_array[i], value_predictions_array[i + 1], gamma)
        gae_array[i] = delta + (gamma * tau * gae_array[i + 1])
    return gae_array


def get_gae_delta(reward, value, next_value, gamma):
    return -value + reward + gamma * next_value
