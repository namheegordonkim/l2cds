from abc import abstractmethod
from typing import Callable, List

import numpy as np
import torch
import torch.nn as nn

from utils.containers import TensorDict
from utils.keys import TensorKey
from utils.utils import device


def pairwise_distances(x, y=None):
    """
    Input: x is a Nxd matrix
           y is an optional Mxd matrix
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist


class LossCalculator2:

    def __init__(self, weight=1.):
        self.weight = weight

    @abstractmethod
    def get_loss(self, tensor_dicts: List[TensorDict]):
        raise NotImplementedError


class LossCalculatorInputTarget(LossCalculator2):

    def __init__(self, input_tensor_key: TensorKey, target_tensor_key: TensorKey, loss_function: nn.Module,
                 weight: float):
        super().__init__(weight)
        self.input_tensor_key = input_tensor_key
        self.target_tensor_key = target_tensor_key
        self.loss_function = loss_function

    def get_loss(self, tensor_dicts: List[TensorDict]):
        loss = torch.scalar_tensor(0).float().to(device)
        for tensor_dict in tensor_dicts:
            input_tensor = tensor_dict.get(self.input_tensor_key)
            target_tensor = tensor_dict.get(self.target_tensor_key)
            loss += self.weight * self.loss_function.forward(input_tensor, target_tensor)
        return loss


class LossCalculatorApply(LossCalculator2):

    def __init__(self, input_tensor_key: TensorKey, transform_lambda: Callable[[torch.Tensor], torch.Tensor],
                 weight: float):
        super().__init__(weight)
        self.input_tensor_key = input_tensor_key
        self.transform_lambda = transform_lambda

    def get_loss(self, tensor_dicts: List[TensorDict]):
        loss = torch.scalar_tensor(0).float().to(device)
        for tensor_dict in tensor_dicts:
            tensor = tensor_dict.get(self.input_tensor_key)
            loss += self.weight * self.transform_lambda(tensor)
        return loss


class LossCalculatorL2Mean(LossCalculator2):

    def __init__(self, input_tensor_key: TensorKey, weight: float):
        super().__init__(weight)
        self.input_tensor_key = input_tensor_key

    def get_loss(self, tensor_dicts: TensorDict):
        return self.weight * torch.mean(tensor_dicts.get(self.input_tensor_key) ** 2)


class LossCalculatorNearestNeighborL2(LossCalculator2):

    def __init__(self, feature_tensor_key: TensorKey, origins_tensor_key: TensorKey, weight: float):
        super().__init__(weight)
        self.feature_tensor_key = feature_tensor_key
        self.origins_tensor_key = origins_tensor_key

    def get_loss(self, tensor_dicts: List[TensorDict]):
        loss = 0
        for i in range(len(tensor_dicts)):
            for j in np.arange(i + 1, len(tensor_dicts)):
                source_tensor_dict = tensor_dicts[i]
                target_tensor_dict = tensor_dicts[j]
                source_features = source_tensor_dict.get(self.feature_tensor_key)
                target_features = target_tensor_dict.get(self.feature_tensor_key)
                distance_matrix = pairwise_distances(source_features, target_features)

                min_distances, _ = torch.min(distance_matrix, dim=1)
                loss += torch.mean(min_distances)

                min_distances, _ = torch.min(distance_matrix, dim=0)
                loss += torch.mean(min_distances)

        return self.weight * loss


class LossCalculatorSum(LossCalculator2):
    def __init__(self, loss_calculators: List[LossCalculator2]):
        super().__init__()
        self.loss_calculators = loss_calculators

    def get_loss(self, tensor_dicts: List[TensorDict]):
        loss = torch.scalar_tensor(0).float().to(device)
        for l in self.loss_calculators:
            loss += l.get_loss(tensor_dicts)
        return torch.sum(loss)
