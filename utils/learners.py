from typing import List

import numpy as np
import torch
from tqdm import tqdm

from utils.containers import DataDict, ModelDict
from utils.loss_calculators import LossCalculator2
from utils.tensor_collectors import TensorCollector
from utils.utils import device


class HomogeneousLearner:
    """
    Takes one dataset and one model dict to perform one learning iteration
    """

    def __init__(self, loss_calculator: LossCalculator2, tensor_collector: TensorCollector, optimizer: torch.optim,
                 n_epochs):
        self.optimizer = optimizer
        self.tensor_collector = tensor_collector
        self.loss_calculator = loss_calculator
        self.n_epochs = n_epochs

    def train_one_episode(self, data_dict: DataDict, model_dict: ModelDict, batch_size=64):

        # shuffle data dicts
        random_idx = np.random.choice(range(data_dict.n_examples), data_dict.n_examples, replace=False)
        for k, v in data_dict.dict.items():
            data_dict.set(k, v[random_idx])

        # cast model to CUDA or CPU
        for model in model_dict.as_list():
            model.to(device)

        epoch_loss = 0
        for epoch in tqdm(range(self.n_epochs)):
            epoch_loss += self.train_one_epoch(data_dict, model_dict, batch_size)
        episode_loss = epoch_loss / self.n_epochs

        torch.cuda.empty_cache()
        return episode_loss

    def train_one_epoch(self, data_dict: DataDict, model_dict: ModelDict, batch_size=64):

        n_examples = data_dict.n_examples
        n_batches = int(n_examples / batch_size)
        batch_idxs = np.array_split(np.random.choice(np.arange(n_examples), n_examples, replace=False),
                                    n_batches)
        epoch_loss = torch.scalar_tensor(0)

        for model in model_dict.as_list():
            model.to(device)

        for batch_idx in tqdm(batch_idxs):
            tensor_dicts = self.tensor_collector.get_tensor_dict([data_dict], [model_dict], batch_idx)
            tensor_dict = tensor_dicts[0]
            loss = torch.scalar_tensor(0).float().to(device)

            loss += self.loss_calculator.get_loss(tensor_dict)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss / n_batches

        torch.cuda.empty_cache()

        return epoch_loss.cpu().detach().squeeze()


class HeterogeneousLearner:
    """
    Takes multiple datasets and multiple model conatiners to perform one learning iteration
    """

    def __init__(self, loss_calculator: LossCalculator2, tensor_collector: TensorCollector, optimizer: torch.optim,
                 n_epochs):
        self.optimizer = optimizer
        self.tensor_collector = tensor_collector
        self.loss_calculator = loss_calculator
        self.n_epochs = n_epochs

    def train_one_episode(self, data_dicts: List[DataDict], model_dicts: List[ModelDict],
                          batch_size=64):

        # shuffle data dicts
        for data_dict in data_dicts:
            random_idx = np.random.choice(range(data_dict.n_examples), data_dict.n_examples, replace=False)
            for k, v in data_dict.dict.items():
                data_dict.set(k, v[random_idx])

        # cast model to CUDA or CPU
        for model_dict in model_dicts:
            for model in model_dict.as_list():
                model.to(device)

        epoch_loss = 0
        for epoch in tqdm(range(self.n_epochs)):
            epoch_loss += self.train_one_epoch(data_dicts, model_dicts, batch_size)
        episode_loss = epoch_loss / self.n_epochs

        torch.cuda.empty_cache()
        return episode_loss

    def train_one_epoch(self, data_dicts: List[DataDict], model_dicts: List[ModelDict],
                        batch_size=64):

        n_examples = data_dicts[0].n_examples
        n_batches = int(n_examples / batch_size)
        batch_idxs = np.array_split(np.random.choice(np.arange(n_examples), n_examples, replace=False),
                                    n_batches)
        epoch_loss = torch.scalar_tensor(0).to(device)

        for batch_idx in batch_idxs:
            tensor_dicts = self.tensor_collector.get_tensor_dicts(data_dicts, model_dicts, batch_idx)
            loss = self.loss_calculator.get_loss(tensor_dicts).to(device)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss / n_batches

        torch.cuda.empty_cache()

        return epoch_loss.cpu().detach().squeeze()


class HeterogeneousSequentialLearner:
    def __init__(self, loss_calculators: List[LossCalculator2], tensor_collectors: List[TensorCollector], optimizers,
                 n_epochs_list):
        self.n_epochs_list = n_epochs_list
        self.optimizers = optimizers
        self.tensor_collectors = tensor_collectors
        self.loss_calculators = loss_calculators

    def train_one_episode(self, data_dicts: List[DataDict], model_dicts: List[ModelDict],
                          batch_size=64):

        # shuffle data dicts
        for data_dict in data_dicts:
            random_idx = np.random.choice(range(data_dict.n_examples), data_dict.n_examples, replace=False)
            for k, v in data_dict.dict.items():
                data_dict.set(k, v[random_idx])

        # cast model to CUDA or CPU
        for model_dict in model_dicts:
            for model in model_dict.as_list():
                model.to(device)

        episode_losses = []
        for tensor_collector, loss_calculator, optimizer, n_epochs in \
                zip(self.tensor_collectors, self.loss_calculators, self.optimizers, self.n_epochs_list):
            episode_loss = 0
            for epoch in tqdm(range(n_epochs)):
                episode_loss += self.train_one_epoch(data_dicts, model_dicts, tensor_collector, loss_calculator,
                                                     optimizer, batch_size)
            episode_loss /= n_epochs

            episode_losses.append(episode_loss)

        return episode_losses

    def train_one_epoch(self, data_dicts: List[DataDict], model_dicts: List[ModelDict],
                        tensor_collector: TensorCollector, loss_calculator: LossCalculator2, optimizer, batch_size):

        n_examples = data_dicts[0].n_examples
        n_batches = int(n_examples / batch_size)
        batch_idxs = np.array_split(np.random.choice(np.arange(n_examples), n_examples, replace=False),
                                    n_batches)
        epoch_loss = torch.scalar_tensor(0)

        for batch_idx in tqdm(batch_idxs):
            tensor_dict = tensor_collector.get_tensor_dict(data_dicts, model_dicts, batch_idx)
            loss = loss_calculator.get_loss(tensor_dict)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss / n_batches

        torch.cuda.empty_cache()

        return epoch_loss.cpu().detach().squeeze()
