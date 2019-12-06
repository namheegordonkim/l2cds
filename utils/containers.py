import numpy as np
import torch

from utils.keys import ModelKey, TensorKey, DataKey


class CommonDict:
    def __init__(self):
        self.dict = dict()

    def as_list(self):
        return list(self.dict.values())


class ModelDict(CommonDict):
    def get(self, key: ModelKey):
        return self.dict[key]

    def set(self, key: ModelKey, value):
        self.dict[key] = value


class TensorDict(CommonDict):
    def get(self, key: TensorKey) -> torch.Tensor:
        return self.dict[key]

    def set(self, key: TensorKey, value: torch.Tensor):
        self.dict[key] = value


class DataDict(CommonDict):
    def __init__(self, n_examples):
        super().__init__()
        self.n_examples = n_examples

    def get(self, key: DataKey) -> np.ndarray:
        return self.dict[key]

    def set(self, key: DataKey, value: np.ndarray):
        self.dict[key] = value


class EnvsContainer:

    def __init__(self, env, envs, states):
        self.env = env
        self.envs = envs
        self.states = states
