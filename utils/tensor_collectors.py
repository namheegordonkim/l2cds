from abc import abstractmethod
from typing import List

import numpy as np
import torch

from utils.containers import DataDict, ModelDict, TensorDict
from utils.keys import TensorKey
from utils.tensor_inserters import TensorInserter
from utils.utils import device


class TensorListGetter:

    @abstractmethod
    def get_tensor_dict_list(self, tensor_inserter: TensorInserter, data_dicts: List[DataDict],
                             model_dicts: List[ModelDict], batch_idx) -> List[TensorDict]:
        raise NotImplementedError


class TensorListGetterOneToOne(TensorListGetter):

    def get_tensor_dict_list(self, tensor_inserter: TensorInserter, data_dicts: List[DataDict],
                             model_dicts: List[ModelDict], batch_idx) -> List[TensorDict]:
        tensor_dicts = [
            tensor_inserter.insert_tensors(TensorDict(), data_dict, model_dict, batch_idx)
            for data_dict, model_dict in zip(data_dicts, model_dicts)]
        # add origin labels
        for i, tensor_dict in enumerate(tensor_dicts):
            origins_tensor = torch.full([len(batch_idx)], i).long().to(device)
            tensor_dict.set(TensorKey.origins_tensor, origins_tensor)
        return tensor_dicts


class TensorCollector:
    def __init__(self, tensor_inserter: TensorInserter, tensor_list_getter: TensorListGetter):
        self.tensor_inserter = tensor_inserter
        self.tensor_list_getter = tensor_list_getter

    def get_tensor_dicts(self, data_dicts: List[DataDict], model_dicts: List[ModelDict],
                        batch_idx: np.ndarray) -> List[TensorDict]:
        tensor_dicts = self.tensor_list_getter.get_tensor_dict_list(self.tensor_inserter, data_dicts,
                                                                    model_dicts, batch_idx)
        return tensor_dicts
