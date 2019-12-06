import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.containers import ModelDict
from utils.keys import ModelKey, DataKey


def encode(model_dict, states):
    states_scaled = model_dict.get(ModelKey.state_scaler).scaler.transform(states)
    states_tensor = torch.as_tensor(states_scaled).float()
    encoded_states_tensor, _ = model_dict.get(ModelKey.state_encoder)(states_tensor)
    encoded_states = encoded_states_tensor.detach().numpy()
    return encoded_states


def predict_velocity(model_dict, states):
    encoded_states = encode(model_dict, states)
    encoded_states_tensor = torch.as_tensor(encoded_states).float()
    encoded_velocities_predictions_tensor, _ = model_dict.get(ModelKey.encoded_velocity_predictor)(encoded_states_tensor)
    encoded_velocities_predictions = encoded_velocities_predictions_tensor.detach().numpy()
    return encoded_velocities_predictions


def convert(model_dict1, model_dict2, points1):
    points1_scaler = model_dict1.get(ModelKey.state_scaler).scaler
    points2_scaler = model_dict2.get(ModelKey.state_scaler).scaler
    points1_tensor = torch.as_tensor(points1_scaler.transform(points1)).float()
    points1_encoded_tensor, _ = model_dict1.get(ModelKey.state_encoder)(points1_tensor)
    points1_encoded = points1_encoded_tensor.detach().cpu().numpy()
    points1_decoded_tensor, _ = model_dict1.get(ModelKey.state_decoder)(points1_encoded_tensor)
    points1_decoded = points1_decoded_tensor.detach().cpu().numpy()
    points1_decoded = points1_scaler.inverse_transform(points1_decoded)
    points1_decoded_into_points2_tensor, _ = model_dict2.get(ModelKey.state_decoder)(points1_encoded_tensor)
    points1_decoded_into_points2 = points1_decoded_into_points2_tensor.detach().cpu().numpy()
    points1_decoded_into_points2 = points2_scaler.inverse_transform(points1_decoded_into_points2)
    return points1_decoded, points1_decoded_into_points2, points1_encoded


def main():
    vis_idx = np.random.choice(range(10000), 500, replace=False)
    data_dict0 = torch.load("./data/wedges0.pkl")
    data_dict1 = torch.load("./data/wedges1.pkl")

    states0 = data_dict0.get(DataKey.states)[vis_idx]
    states1 = data_dict1.get(DataKey.states)[vis_idx]
    next_states0 = data_dict0.get(DataKey.next_states)[vis_idx]
    next_states1 = data_dict1.get(DataKey.next_states)[vis_idx]

    # n, d = states0.shape
    # sample_idx = np.random.choice(range(n), 500, replace=False)
    # states0 = states0[sample_idx]
    # states1 = states1[sample_idx]

    model_dict = torch.load(args.model_dicts_path, map_location="cpu")
    model_dict0: ModelDict = model_dict['wedges0']
    model_dict1: ModelDict = model_dict['wedges1']

    states0_decoded, states0_decoded_into_states1, states0_encoded = convert(model_dict0, model_dict1,
                                                                             states0)
    states1_decoded, states1_decoded_into_states0, states1_encoded = convert(model_dict1, model_dict0,
                                                                             states1)

    # plt.figure()
    # states0_scaled = model_dict0.get(ModelKey.state_scaler).scaler.transform(states0)
    # plt.scatter(states0_scaled[:, 0], states0_scaled[:, 1], alpha=0.5)
    # plt.show()

    plt.figure()
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.scatter(states0[:, 0], states0[:, 1], alpha=0.5)
    plt.scatter(states1[:, 0], states1[:, 1], alpha=0.5)
    for state0, next_state0 in zip(states0, next_states0):
        velocity0 = next_state0 - state0
        plt.arrow(state0[0], state0[1], velocity0[0], velocity0[1], alpha=0.2, width=0.005, length_includes_head=True, color='b')
    for state1, next_state1 in zip(states1, next_states1):
        velocity1 = next_state1 - state1
        plt.arrow(state1[0], state1[1], velocity1[0], velocity1[1], alpha=0.2, width=0.005, length_includes_head=True, color='orange')

    plt.scatter(states0_decoded[:, 0], states0_decoded[:, 1], alpha=0.5)
    # plt.scatter(states0_decoded_into_states1[:, 0], states0_decoded_into_states1[:, 1], alpha=0.5, c='C3')
    plt.scatter(states1_decoded[:, 0], states1_decoded[:, 1], alpha=0.5)
    # plt.scatter(states1_decoded_into_states0[:, 0], states1_decoded_into_states0[:, 1], alpha=0.5, c='C2')
    #
    # diffs = states0_decoded_into_states1 - states0
    # # diffs = states0_decoded - states0
    # for state0, diff in zip(states0, diffs):
    #     eps = np.random.random()
    #     if eps < 0.1:
    #         plt.arrow(state0[0], state0[1], diff[0], diff[1], alpha=0.2, width=0.0005, length_includes_head=True)
    #
    # diffs = states1_decoded - states1
    # for state1, diff in zip(states1, diffs):
    #     eps = np.random.random()
    #     if eps < 0.1:
    #         plt.arrow(state1[0], state1[1], diff[0], diff[1], alpha=0.2, width=0.000005, length_includes_head=True)

    plt.show()
    #
    model_dict0.get(ModelKey.encoded_velocity_predictor).eval()
    model_dict1.get(ModelKey.encoded_velocity_predictor).eval()
    plt.figure()
    plt.scatter(states0_encoded[:, 0], states0_encoded[:, 1], alpha=0.2)
    plt.scatter(states1_encoded[:, 0], states1_encoded[:, 1], alpha=0.2)

    encoded_velocities0 = predict_velocity(model_dict0, states0)
    for state0, velocity0 in zip(states0_encoded, encoded_velocities0):
        plt.arrow(state0[0], state0[1], velocity0[0], velocity0[1], alpha=0.2, width=0.001, length_includes_head=True, color='b')
    encoded_velocities1 = predict_velocity(model_dict1, states1)
    for state1, velocity1 in zip(states1_encoded, encoded_velocities1):
        plt.arrow(state1[0], state1[1], velocity1[0], velocity1[1], alpha=0.2, width=0.001, length_includes_head=True, color='orange')

    plt.show()

    # visualize forward latent dynamics field
    # encoded_states = np.random.uniform((-1, -1), (1, 1), (10000, 2))
    # encoded_next_state_predictor = model_dict0.encoded_next_state_predictor
    # encoded_next_states, _ = encoded_next_state_predictor(torch.as_tensor(encoded_states).float())
    # encoded_next_states = encoded_next_states.detach().numpy()
    # plt.figure()
    # plt.scatter(encoded_states[:, 0], encoded_states[:, 1], alpha=0.5)
    # plt.scatter(encoded_next_states[:, 0], encoded_next_states[:, 1], alpha=0.5)
    # diffs = encoded_next_states - encoded_states
    # for encoded_state, diff in zip(encoded_states, diffs):
    #     plt.arrow(encoded_state[0], encoded_state[1], diff[0], diff[1], alpha=0.02, width=0.005,
    #               length_includes_head=True)
    # plt.show()

    # encoder = model_dict0.state_encoder
    # scaler = model_dict0.state_scaler
    # encoded_states, _ = encoder(torch.as_tensor(scaler.transform(states0)).float())
    # encoded_states = encoded_states.detach().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dicts_path", type=str, required=True)

    args = parser.parse_args()

    np.random.seed(0)
    main()
