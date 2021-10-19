"""
Learn the correspondence between two L-shaped distributions via a shared latent space
"""
from typing import List

import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import MSELoss

# To avoid manipulating PYTHONPATH, add leading dots
from ..utils.containers import DataDict, ModelDict
from ..utils.keys import DataKey, ModelKey, TensorKey
from ..utils.loss_calculators import LossCalculatorSum, LossCalculatorInputTarget, LossCalculatorNearestNeighborL2
from ..models import NNet, ScalerWrapper
from ..utils.learners import HeterogeneousLearner
from ..utils.radam import RAdam
import numpy as np

from ..utils.tensor_collectors import TensorListGetterOneToOne, TensorCollector
from ..utils.tensor_inserters import TensorInserterSeq, TensorInserterTensorizeScaled, TensorInserterForward


def main():
    data_dict0: DataDict = torch.load("./data/shapes0.pkl", map_location="cpu")
    data_dict1: DataDict = torch.load("./data/shapes1.pkl", map_location="cpu")
    data_dicts = [data_dict0, data_dict1]

    model_dicts: List[ModelDict] = []

    identity = nn.Identity()
    tanh = nn.Tanh()
    encoded_state_dim = 2
    learnable_parameters = []
    for data_dict in data_dicts:
        _, state_dim = data_dict.get(DataKey.states).shape
        states = data_dict.get(DataKey.states)
        state_encoder = NNet(state_dim, encoded_state_dim, identity, hidden_dims=[256, 256])
        state_decoder = NNet(encoded_state_dim, state_dim, identity, hidden_dims=[256, 256])
        state_scaler_ = StandardScaler()
        state_scaler_.fit(states)
        state_scaler = ScalerWrapper(state_scaler_)

        learnable_parameters.extend(list(state_encoder.parameters()))
        learnable_parameters.extend(list(state_decoder.parameters()))

        model_dict = ModelDict()
        model_dict.set(ModelKey.state_scaler, state_scaler)
        model_dict.set(ModelKey.state_encoder, state_encoder)
        model_dict.set(ModelKey.state_decoder, state_decoder)
        model_dicts.append(model_dict)

    tensor_list_getter = TensorListGetterOneToOne()
    tensor_collector = TensorCollector(TensorInserterSeq([
        TensorInserterTensorizeScaled(DataKey.states, ModelKey.state_scaler, TensorKey.states_tensor, torch.float),
        TensorInserterForward(TensorKey.states_tensor, ModelKey.state_encoder,
                              TensorKey.encoded_states_tensor),
        TensorInserterForward(TensorKey.encoded_states_tensor, ModelKey.state_decoder,
                              TensorKey.decoded_states_tensor),
    ]), tensor_list_getter)

    mse_loss = MSELoss()
    loss_calculator = LossCalculatorSum([
        LossCalculatorInputTarget(TensorKey.decoded_states_tensor, TensorKey.states_tensor, mse_loss, 1.),
        LossCalculatorNearestNeighborL2(TensorKey.encoded_states_tensor, TensorKey.origins_tensor, 1.),
    ])

    optimizer = RAdam(params=learnable_parameters, lr=3e-4)

    optim = HeterogeneousLearner(loss_calculator, tensor_collector, optimizer, 1000)
    for episode in range(1000):
        loss = optim.train_one_episode(data_dicts, model_dicts, batch_size=5000)
        print("Episode {:d}\tLoss: {:f}".format(episode, loss))

        model_path = "./torch_model/shapes_model_dicts_{:07d}_{:02d}.pkl".format(episode, seed)
        torch.save(model_dicts, model_path)
        print("Saved models to {:s}".format(model_path))


if __name__ == "__main__":
    seed = 2
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    main()
