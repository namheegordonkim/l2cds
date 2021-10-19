import numpy as np
import torch
import torch.nn as nn
import torch.optim
from sklearn.preprocessing import StandardScaler

from utils.containers import ModelDict
from utils.keys import DataKey, ModelKey, TensorKey
from utils.loss_calculators import LossCalculatorInputTarget, \
    LossCalculatorNearestNeighborL2, LossCalculatorSum
from models import NNet, ScalerWrapper
from utils.learners import HeterogeneousLearner
from utils.radam import RAdam
from utils.tensor_collectors import TensorCollector, TensorListGetterOneToOne
from utils.tensor_inserters import TensorInserterSeq, TensorInserterTensorizeScaled, TensorInserterForward, \
    TensorInserterSum


def do(seed):
    torch.random.manual_seed(seed)
    np.random.seed(seed)

    dataset0 = torch.load("./data/wedges0.pkl")
    dataset1 = torch.load("./data/wedges1.pkl")
    data_dicts = [dataset0, dataset1]
    encoded_state_dim = 2

    identity = nn.Identity()
    softsign = nn.Softsign()

    parameters = []
    encoded_velocity_predictor = NNet(encoded_state_dim, encoded_state_dim, softsign, hidden_dims=[256, 256])
    state_dict = encoded_velocity_predictor.state_dict()
    for param_key in list(state_dict.keys())[:-2]:
        state_dict[param_key] *= 0
    encoded_velocity_predictor.load_state_dict(state_dict)

    parameters.extend(list(encoded_velocity_predictor.parameters()))

    model_dicts = []
    for data_dict in data_dicts:
        _, state_dim = data_dict.get(DataKey.states).shape
        state_encoder = NNet(state_dim, encoded_state_dim, identity, hidden_dims=[256, 256])
        state_decoder = NNet(encoded_state_dim, state_dim, identity, hidden_dims=[256, 256])

        state_scaler_ = StandardScaler()
        state_scaler_.fit(data_dict.get(DataKey.states))
        state_scaler = ScalerWrapper(state_scaler_)

        model_dict = ModelDict()
        model_dict.set(ModelKey.state_encoder, state_encoder)
        model_dict.set(ModelKey.state_decoder, state_decoder)
        model_dict.set(ModelKey.encoded_velocity_predictor, encoded_velocity_predictor)
        model_dict.set(ModelKey.state_scaler, state_scaler)

        # model_dicts.append(model_container)
        model_dicts.append(model_dict)

        parameters.extend(list(state_encoder.parameters()))
        parameters.extend(list(state_decoder.parameters()))

    tensor_collector = TensorCollector(TensorInserterSeq([
        TensorInserterTensorizeScaled(DataKey.states, ModelKey.state_scaler, TensorKey.states_tensor, torch.float),
        TensorInserterTensorizeScaled(DataKey.next_states, ModelKey.state_scaler, TensorKey.next_states_tensor,
                                      torch.float),
        TensorInserterForward(TensorKey.states_tensor, ModelKey.state_encoder, TensorKey.encoded_states_tensor),
        TensorInserterForward(TensorKey.next_states_tensor, ModelKey.state_encoder,
                              TensorKey.encoded_next_states_tensor),
        TensorInserterForward(TensorKey.encoded_states_tensor, ModelKey.encoded_velocity_predictor,
                              TensorKey.encoded_velocity_predictions_tensor),
        TensorInserterSum([TensorKey.encoded_states_tensor, TensorKey.encoded_velocity_predictions_tensor],
                          TensorKey.encoded_next_state_predictions_tensor),
        TensorInserterForward(TensorKey.encoded_states_tensor, ModelKey.state_decoder,
                              TensorKey.decoded_states_tensor)
    ]), TensorListGetterOneToOne())

    mse_loss = nn.MSELoss()
    loss_calculator = LossCalculatorSum([
        LossCalculatorInputTarget(TensorKey.decoded_states_tensor, TensorKey.states_tensor, mse_loss, 1.),
        LossCalculatorInputTarget(TensorKey.encoded_next_state_predictions_tensor, TensorKey.encoded_next_states_tensor,
                                  mse_loss, 1e4),
        LossCalculatorNearestNeighborL2(TensorKey.encoded_states_tensor, TensorKey.origins_tensor, 1.)
    ])

    optimizer = RAdam(params=parameters, lr=3e-4)

    optim = HeterogeneousLearner(loss_calculator, tensor_collector, optimizer, n_epochs=100)
    for episode in range(10):
        loss = optim.train_one_episode(data_dicts, model_dicts, batch_size=5000)
        print("Episode {:d}\tLoss: {:.10f}".format(episode, loss))

        model_dict = dict()
        model_dict['wedges0'] = model_dicts[0]
        model_dict['wedges1'] = model_dicts[1]
        model_path = "./torch_model/wedges_model_containers_{:07d}_{:02d}.pkl".format(episode,
                                                                                      seed)
        torch.save(model_dict, model_path)
        print("Saved models to {:s}".format(model_path))


def main():
    for seed in np.arange(0, 10):
        do(seed)
    # do(1)


if __name__ == "__main__":
    main()
