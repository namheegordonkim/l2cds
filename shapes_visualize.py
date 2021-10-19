import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from utils.containers import TensorDict
from utils.keys import DataKey, ModelKey, TensorKey
from utils.loss_calculators import LossCalculatorNearestNeighborL2

parser = argparse.ArgumentParser()
parser.add_argument("--model_dicts_path", type=str, required=True)

args = parser.parse_args()


def visualize_with_paths(data_dict_paths, model_dicts_path):
    data_dicts = [torch.load(dataset_path, map_location="cpu") for dataset_path in data_dict_paths]
    data_dict1, data_dict2 = data_dicts

    points1 = data_dict1.get(DataKey.states)
    points2 = data_dict2.get(DataKey.states)

    n, d = points1.shape
    sample_idx = np.random.choice(range(n), 1000, replace=False)
    points1 = points1[sample_idx]
    points2 = points2[sample_idx]

    model_dicts = torch.load(model_dicts_path, map_location="cpu")
    model_dict1, model_dict2 = model_dicts
    points1_decoded, points1_decoded_into_points2, points1_encoded = convert(model_dict1, model_dict2,
                                                                             points1)
    points2_decoded, points2_decoded_into_points1, points2_encoded = convert(model_dict2, model_dict1,
                                                                             points2)

    state_scaler1 = model_dict1.get(ModelKey.state_scaler)
    state_scaler2 = model_dict2.get(ModelKey.state_scaler)

    nn_loss_calculator = LossCalculatorNearestNeighborL2(TensorKey.encoded_states_tensor, TensorKey.origins_tensor, 1.)
    mse_loss = nn.MSELoss()

    tensor_dict1 = TensorDict()
    points1_tensor = torch.as_tensor(points1).float()
    points1_scaled_tensor = state_scaler1.forward(points1_tensor)
    points1_decoded_tensor = torch.as_tensor(points1_decoded).float()
    points1_decoded_scaled_tensor = state_scaler1.forward(points1_decoded_tensor)
    tensor_dict1.set(TensorKey.encoded_states_tensor, torch.as_tensor(points1_encoded).float())
    tensor_dict1.set(TensorKey.origins_tensor, torch.zeros(n))
    tensor_dict1.set(TensorKey.states_tensor, points1_scaled_tensor)

    tensor_dict2 = TensorDict()
    points2_tensor = torch.as_tensor(points2).float()
    points2_scaled_tensor = state_scaler2.forward(points2_tensor)
    tensor_dict2.set(TensorKey.encoded_states_tensor, torch.as_tensor(points2_encoded).float())
    tensor_dict2.set(TensorKey.origins_tensor, torch.ones(n))
    tensor_dict2.set(TensorKey.states_tensor, points2_scaled_tensor)

    print(mse_loss.forward(points1_decoded_scaled_tensor, points1_scaled_tensor))

    nn_loss = nn_loss_calculator.get_loss([tensor_dict1, tensor_dict2])
    print(nn_loss)

    plt.figure()
    plt.xlim(-2, 6)
    plt.ylim(-4, 4)
    plt.scatter(points1[:, 0], points1[:, 1], alpha=0.5)
    plt.scatter(points2[:, 0], points2[:, 1], alpha=0.5)

    diffs = points1_decoded_into_points2 - points1
    n, _ = points1.shape
    for i in range(n):
        eps = np.random.random()
        if eps < 0.10:
            plt.arrow(points1[i, 0], points1[i, 1], diffs[i, 0], diffs[i, 1], alpha=0.2, width=0.05, length_includes_head=True)

    plt.figure()
    plt.scatter(points1_encoded[:, 0], points1_encoded[:, 1], alpha=0.5)
    plt.scatter(points2_encoded[:, 0], points2_encoded[:, 1], alpha=0.5, c="C1")
    plt.show()


def convert(model_dict1, model_dict2, points1):
    points1_scaler = model_dict1.get(ModelKey.state_scaler)
    points1_encoder = model_dict1.get(ModelKey.state_encoder)
    points1_decoder = model_dict1.get(ModelKey.state_decoder)

    points2_scaler = model_dict2.get(ModelKey.state_scaler)
    points2_decoder = model_dict2.get(ModelKey.state_decoder)

    points1_tensor = torch.as_tensor(points1).float()
    points1_scaled_tensor = points1_scaler.forward(points1_tensor).cpu()
    points1_encoded_tensor, _ = points1_encoder.forward(points1_scaled_tensor)
    points1_encoded = points1_encoded_tensor.detach().cpu().numpy()

    points1_decoded_tensor, _ = points1_decoder.forward(points1_encoded_tensor)
    points1_decoded_tensor = points1_scaler.reverse(points1_decoded_tensor)
    points1_decoded = points1_decoded_tensor.detach().cpu().numpy()

    points1_decoded_into_points2_tensor, _ = points2_decoder.forward(points1_encoded_tensor)
    points1_decoded_into_points2_tensor = points2_scaler.reverse(points1_decoded_into_points2_tensor)
    points1_decoded_into_points2 = points1_decoded_into_points2_tensor.detach().cpu().numpy()

    return points1_decoded, points1_decoded_into_points2, points1_encoded


def main():
    data_dict_paths = ["./data/shapes0.pkl", "./data/shapes1.pkl"]
    visualize_with_paths(data_dict_paths, args.model_dicts_path)


if __name__ == "__main__":
    np.random.seed(0)
    main()
