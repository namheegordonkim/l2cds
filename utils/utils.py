import argparse
import os
from abc import abstractmethod

import numpy as np
import pygame
import torch
import torch.multiprocessing as mp
import yaml
from pygame import OPENGL, GL_DEPTH_SIZE
from stable_baselines3.common.vec_env import SubprocVecEnv
from tqdm import tqdm

from utils.radam import RAdam

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class TrafficLight:
    """used by chief to allow workers to run or not"""

    def __init__(self, val=True):
        self.val = mp.Value("b", False)
        self.lock = mp.Lock()
        self.explore = mp.Value("b", True)

    def get(self):
        with self.lock:
            return self.val.value

    def switch(self):
        with self.lock:
            self.val.value = (not self.val.value)


class Counter:
    """enable the chief to access worker's total number of updates"""

    def __init__(self, val=True):
        self.val = mp.Value("i", 0)
        self.lock = mp.Lock()

    def get(self):
        # used by chief
        with self.lock:
            return self.val.value

    def increment(self):
        # used by workers
        with self.lock:
            self.val.value += 1

    def reset(self):
        # used by chief
        with self.lock:
            self.val.value = 0


def collect_random_state_trajectory(env, n_states):
    states = []
    state = env.reset()
    action_dim = env.action_space.shape[0]

    for _ in tqdm(range(n_states)):
        states.append(state)
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if done:
            state = env.reset()

    return np.stack(states)


def compute_total_reward_from_start(env, action_getter, horizon=3000) -> float:
    state = env.reset()
    total_reward = 0
    for _ in range(horizon):
        action = action_getter.get_action(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break

    return total_reward


# def compute_cumulative_reward(rewards: np.ndarray, gamma) -> np.ndarray:
#     cumulative_rewards =


def add_experiment_args(parser):
    parser.add_argument("--yml_path", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--save_prefix", type=str, required=True)


def make_and_parse_args():
    parser = argparse.ArgumentParser()
    add_experiment_args(parser)
    parser.add_argument("--dummy", action='store_true')
    args = parser.parse_args()
    print(args)
    return args


def get_config_dict_from_yml(yml_path, name):
    with open(yml_path, 'r') as f:
        param_dict = yaml.load(f, yaml.SafeLoader)[name]
    return param_dict


class EnvFactory:

    @abstractmethod
    def make_env(self):
        raise NotImplementedError


def get_envs(factory: EnvFactory):
    num_envs = len(os.sched_getaffinity(0))

    env = factory.make_env()

    def make_env():
        def _thunk():
            env = factory.make_env()
            return env

        return _thunk

    envs = [make_env() for _ in range(num_envs)]
    envs = SubprocVecEnv(envs)
    return env, envs


def make_tensor_from_list(lst):
    stacked = np.stack(lst, axis=0).astype(np.float32)
    tensor = torch.as_tensor(stacked).to(device)
    if len(tensor.shape) == 1:
        tensor = tensor.reshape(-1, 1)
    return tensor


def concatenate_tensor_containers(containers):
    # must assume that all tensor containers are of the same class

    # use dict to append all tensors into lists
    result_dict = dict()
    for container in containers:
        for attribute, value in container.__dict__.items():
            if attribute in result_dict.keys():
                result_dict[attribute].append(value)
            else:
                result_dict[attribute] = [value]

    # concatenate the lists of tensors
    for attribute, value in result_dict.items():
        if all([v is None for v in value]):
            result_dict[attribute] = None
        else:
            result_dict[attribute] = torch.cat(value)

    # finally turn into tensor container
    container = type(containers[0])(**result_dict)
    return container


class SimpleSimulator:
    """
    Wrapper implementing get_next_state() for coding convenience
    """

    def __init__(self, env):
        self.env = env

    def set_state(self, state):
        raise NotImplementedError

    def get_next_state(self, state, action):
        self.set_state(state)
        next_state, _, _, _ = self.env.step(action)
        return next_state

    def get_next_states(self, states, actions):
        n, _ = states.shape
        next_states = np.zeros_like(states)
        for i in range(n):
            next_states[i, :] = self.get_next_state(states[i, :], actions[i, :])
        return next_states


class OptimizerFactory:

    def __init__(self, model, lr):
        self.model = model
        self.lr = lr

    @abstractmethod
    def make(self):
        raise NotImplementedError


class OptimizerFactoryRAdam(OptimizerFactory):

    def __init__(self, model, lr):
        super().__init__(model, lr)

    def make(self):
        return RAdam(params=self.model.parameters(), lr=self.lr)


class OptimizerFactoryDummy(OptimizerFactory):

    def __init__(self, optimizer):
        super().__init__(None, None)
        self.optimizer = optimizer

    def make(self):
        return self.optimizer


def throw_box(env, throw_from):
    x_displacement = 5
    x_displacement = x_displacement * throw_from
    y_displacement = 3
    walker_x, walker_y = env.robot_skeleton.q[0:2]
    box_x = walker_x + x_displacement
    box_y = walker_y + y_displacement
    env.throw_box_from(box_x, box_y)


def wait_for_keys(env):
    # hang until further key input
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                throw_from = np.random.choice([-1, 1])
                throw_box(env, throw_from)
            if event.key == pygame.K_r:
                env.reset()
            if event.key == pygame.K_RIGHT:
                throw_box(env, -1)
            if event.key == pygame.K_LEFT:
                throw_box(env, 1)


def initialize_pygame():
    pygame.init()
    pygame.key.set_repeat(33)
    size = width, height = 640, 480
    center = width / 2, height / 2
    screen = pygame.display.set_mode(size)
    pygame.display.gl_set_attribute(GL_DEPTH_SIZE, 24)
    pygame.display.set_mode(size, OPENGL)

    return center, screen


def reflect_control_vector(v):
    l = int(len(v) / 2)
    ret = np.copy(v)
    ret[:l] = v[-l:]
    ret[-l:] = v[:l]
    return ret


def interpolate_arrays(arr1: np.ndarray, arr2: np.ndarray, n_results=10):
    """
    Given two rank-1 np.ndarrays, compute np.ndarrays of same interval to fit a linear space between the two.
    Returned array is end-exclusive, meaning arr1 is in the returned array but arr2 is not.
    """
    d, = arr1.shape

    # preallocate
    interpolated = np.zeros([n_results, d])

    # populate
    beta = (arr2 - arr1) / n_results
    for i in range(n_results):
        interpolated[i, :] = arr1 + i * beta

    return interpolated


