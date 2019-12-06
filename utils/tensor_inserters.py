from abc import abstractmethod
from typing import Callable, List

import numpy as np
import torch
from torch import nn as nn

from utils.containers import TensorDict, DataDict, ModelDict
from utils.keys import DataKey, TensorKey, ModelKey
from utils.utils import device


class TensorInserter:
    """
    Implementation of the inserter pattern using dict based data containers.
    """

    @abstractmethod
    def insert_tensors(self, tensor_dict: TensorDict, data_dict: DataDict, model_dict: ModelDict, batch_idx: np.ndarray) \
            -> TensorDict:
        pass


class TensorInserterGenerate(TensorInserter):

    def __init__(self, generate_lambda: Callable, target_tensor_key: TensorKey):
        self.generate_lambda = generate_lambda
        self.target_tensor_key = target_tensor_key

    def insert_tensors(self, tensor_dict: TensorDict, data_dict: DataDict, model_dict: ModelDict,
                       batch_idx: np.ndarray) -> TensorDict:
        tensor = self.generate_lambda().to(device)
        tensor_dict.set(self.target_tensor_key, tensor)
        return tensor_dict


class TensorInserterTensorize(TensorInserter):

    def __init__(self, data_key: DataKey, tensor_key: TensorKey, dtype: torch.dtype):
        self.data_key = data_key
        self.tensor_key = tensor_key
        self.dtype = dtype

    def insert_tensors(self, tensor_dict: TensorDict, data_dict: DataDict, model_dict: ModelDict,
                       batch_idx: np.ndarray) -> TensorDict:
        data = self.get_data(data_dict, model_dict, batch_idx)
        tensor = torch.as_tensor(data, dtype=self.dtype).to(device).reshape(len(batch_idx), -1)
        tensor_dict.set(self.tensor_key, tensor)
        return tensor_dict

    def get_data(self, data_dict: DataDict, model_dict: ModelDict, batch_idx: np.ndarray):
        data = data_dict.get(self.data_key)
        return data[batch_idx]


class TensorInserterTensorizeScaled(TensorInserterTensorize):
    def __init__(self, data_key: DataKey, scaler_key: ModelKey, tensor_key: TensorKey, dtype: torch.dtype):
        super().__init__(data_key, tensor_key, dtype)
        self.scaler_key = scaler_key

    def get_data(self, data_dict: DataDict, model_dict: ModelDict, batch_idx: np.ndarray):
        data = data_dict.get(self.data_key)[batch_idx]
        tensor = torch.as_tensor(data, dtype=self.dtype).to(device).reshape(len(batch_idx), -1)
        scaler = model_dict.get(self.scaler_key)
        scaled = scaler.forward(tensor)
        return scaled


class TensorInserterLambda(TensorInserter):

    def __init__(self, generate_lambda: Callable[[], torch.Tensor], target_tensor_key: TensorKey):
        self.generate_lambda = generate_lambda
        self.target_tensor_key = target_tensor_key

    def insert_tensors(self, tensor_dict: TensorDict, data_dict: DataDict, model_dict: ModelDict,
                       batch_idx: np.ndarray) -> TensorDict:
        tensor = self.generate_lambda()
        tensor_dict.set(self.target_tensor_key, tensor)
        return tensor_dict


class TensorInserterTensorizeTransformed(TensorInserterTensorize):

    def __init__(self, data_key: DataKey, transform_lambda: Callable[[np.ndarray], np.ndarray], tensor_key: TensorKey,
                 dtype: torch.dtype):
        super().__init__(data_key, tensor_key, dtype)
        self.transform_lambda = transform_lambda

    def get_data(self, data_dict: DataDict, model_dict: ModelDict, batch_idx: np.ndarray):
        data = super().get_data(data_dict, model_dict, batch_idx)
        return self.transform_lambda(data)


class TensorInserterForward(TensorInserter):

    def __init__(self, source_tensor_key: TensorKey, model_key: ModelKey, target_tensor_key: TensorKey,
                 noise_scale: float = 0.):
        self.model_key = model_key
        self.source_tensor_key = source_tensor_key
        self.target_tensor_key = target_tensor_key
        self.noise_scale = noise_scale

    def insert_tensors(self, tensor_dict: TensorDict, data_dict: DataDict, model_dict: ModelDict,
                       batch_idx: np.ndarray) -> TensorDict:
        model = model_dict.get(self.model_key)
        source_tensor = self.get_model_input(tensor_dict)
        if self.noise_scale > 1e-9:
            source_tensor = self.add_noise(source_tensor)
        target_tensor, _ = model.forward(source_tensor)
        tensor_dict.set(self.target_tensor_key, target_tensor)
        return tensor_dict

    def get_model_input(self, tensor_dict) -> torch.Tensor:
        return tensor_dict.get(self.source_tensor_key)

    def add_noise(self, tensor) -> torch.Tensor:
        n, d = tensor.shape
        noise = torch.as_tensor(np.random.normal(0, self.noise_scale, (n, d))).float().to(device)
        tensor = tensor + noise
        return tensor


class TensorInserterUniTransform(TensorInserter):

    def __init__(self, source_tensor_key: TensorKey, transform_lambda: Callable[[torch.Tensor], torch.Tensor],
                 target_tensor_key: TensorKey):
        self.source_tensor_key = source_tensor_key
        self.transform_lambda = transform_lambda
        self.target_tensor_key = target_tensor_key

    def insert_tensors(self, tensor_dict: TensorDict, data_dict: DataDict, model_dict: ModelDict,
                       batch_idx: np.ndarray) -> TensorDict:
        source_tensor = tensor_dict.get(self.source_tensor_key)
        transformed = self.transform_lambda(source_tensor).to(device)
        tensor_dict.set(self.target_tensor_key, transformed)
        return tensor_dict


class TensorInserterApplyModel(TensorInserter):

    def __init__(self, source_tensor_keys: List[TensorKey], model_key: ModelKey,
                 transform_lambda: Callable[[nn.Module, List[torch.Tensor]], List[torch.Tensor]],
                 target_tensor_keys: List[TensorKey]):
        self.transform_lambda = transform_lambda
        self.source_tensor_keys = source_tensor_keys
        self.target_tensor_keys = target_tensor_keys
        self.model_key = model_key

    def insert_tensors(self, tensor_dict: TensorDict, data_dict: DataDict, model_dict: ModelDict,
                       batch_idx: np.ndarray) -> TensorDict:
        model = model_dict.get(self.model_key)
        source_tensors = [tensor_dict.get(key) for key in self.source_tensor_keys]
        result_tensor_list = self.transform_lambda(model, source_tensors)
        for target_tensor_key, result_tensor in zip(self.target_tensor_keys, result_tensor_list):
            tensor_dict.set(target_tensor_key, result_tensor)
        return tensor_dict


class TensorInserterGetModelAttribute(TensorInserter):

    def __init__(self, model_key: ModelKey, attribute_lambda: Callable[[nn.Module], torch.Tensor],
                 target_tensor_key: TensorKey):
        self.model_key = model_key
        self.attribute_lambda = attribute_lambda
        self.target_tensor_key = target_tensor_key

    def insert_tensors(self, tensor_dict: TensorDict, data_dict: DataDict, model_dict: ModelDict,
                       batch_idx: np.ndarray) -> TensorDict:
        model = model_dict.get(self.model_key)
        attribute = self.attribute_lambda(model)
        tensor_dict.set(self.target_tensor_key, attribute)
        return tensor_dict


class TensorInserterListTransform(TensorInserter):

    def __init__(self, source_tensor_keys: List[TensorKey],
                 transform_lambda: Callable[[List[torch.Tensor]], torch.Tensor], target_tensor_key: TensorKey):
        self.source_tensor_keys = source_tensor_keys
        self.transform_lambda = transform_lambda
        self.target_tensor_key = target_tensor_key

    def insert_tensors(self, tensor_dict: TensorDict, data_dict: DataDict, model_dict: ModelDict,
                       batch_idx: np.ndarray) -> TensorDict:
        source_tensors = [tensor_dict.get(key) for key in self.source_tensor_keys]
        transformed = self.transform_lambda(source_tensors)
        tensor_dict.set(self.target_tensor_key, transformed)
        return tensor_dict


class TensorInserterConcatenateForward(TensorInserterForward):

    def __init__(self, source_tensor_keys: List[TensorKey], model_key: ModelKey, target_tensor_key: TensorKey,
                 noise_scale: float = 0.):
        super().__init__(None, model_key, target_tensor_key, noise_scale)
        self.source_tensor_keys = source_tensor_keys

    def get_model_input(self, tensor_dict) -> torch.Tensor:
        tensors = [tensor_dict.get(key) for key in self.source_tensor_keys]
        concatenated = torch.cat(tensors, dim=1)
        return concatenated


class TensorInserterSum(TensorInserter):
    """
    Add the tensors specified by the tensor keys
    """

    def __init__(self, source_tensor_keys: List[TensorKey], target_tensor_key: TensorKey):
        self.source_tensor_keys = source_tensor_keys
        self.target_tensor_key = target_tensor_key

    def insert_tensors(self, tensor_dict: TensorDict, data_dict: DataDict, model_dict: ModelDict,
                       batch_idx: np.ndarray) -> TensorDict:
        tensors = torch.stack([tensor_dict.get(tensor_key) for tensor_key in self.source_tensor_keys])
        summed = torch.sum(tensors, dim=0)
        tensor_dict.set(self.target_tensor_key, summed)
        return tensor_dict


class TensorInserterSample(TensorInserterForward):

    def __init__(self, source_tensor_key: TensorKey, model_key: ModelKey,
                 samples_tensor_key: TensorKey, log_probs_tensor_key: TensorKey):
        super().__init__(source_tensor_key, model_key, None)
        self.samples_tensor_key = samples_tensor_key
        self.log_probs_tensor_key = log_probs_tensor_key

    def insert_tensors(self, tensor_dict: TensorDict, data_dict: DataDict, model_dict: ModelDict,
                       batch_idx: np.ndarray) -> TensorDict:
        model = model_dict.get(self.model_key)
        source_tensor = self.get_model_input(tensor_dict)
        if self.noise_scale > 1e-9:
            source_tensor = self.add_noise(source_tensor)
        samples, log_probs, _, _ = model.sample(source_tensor)
        tensor_dict.set(self.samples_tensor_key, samples)
        tensor_dict.set(self.log_probs_tensor_key, log_probs)
        return tensor_dict


class TensorInserterSeq(TensorInserter):

    def __init__(self, inserters: List[TensorInserter]):
        self.inserters = inserters

    def insert_tensors(self, tensor_dict: TensorDict, data_dict: DataDict, model_dict: ModelDict,
                       batch_idx: np.ndarray) -> TensorDict:
        for inserter in self.inserters:
            tensor_dict = inserter.insert_tensors(tensor_dict, data_dict, model_dict, batch_idx)
        return tensor_dict
