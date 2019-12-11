import argparse
import time
from collections import deque
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from torch.distributions import Normal

from utils.containers import ModelDict, DataDict
from utils.factories import DartEnvFactory
from utils.keys import ModelKey, DataKey


def main():
    model_dicts_path = config_dict["model_dicts_path"]
    data_dict_path = config_dict["data_dict_path"]

    model_dicts: List[ModelDict] = list(torch.load(model_dicts_path, map_location="cpu").values())
    data_dict: DataDict = torch.load(data_dict_path, map_location="cpu")

    encoded_states = data_dict.get(DataKey.encoded_states)
    compressor = PCA(n_components=8)
    compressor.fit(encoded_states)

    plt.figure()

    if args.config_name == "ostrich2d_to_walker2d":
        ostrich2d_model_dict, walker2d_model_dict = model_dicts
        ostrich2d_state_scaler = ostrich2d_model_dict.get(ModelKey.state_scaler)
        ostrich2d_state_decoder = ostrich2d_model_dict.get(ModelKey.state_decoder)
        walker2d_state_scaler = walker2d_model_dict.get(ModelKey.state_scaler)
        walker2d_state_decoder = walker2d_model_dict.get(ModelKey.state_decoder)

        encoded_velocity_predictor = ostrich2d_model_dict.get(ModelKey.encoded_velocity_predictor)

        n_states, _ = encoded_states.shape

        factory = DartEnvFactory("./custom_dart_assets/ground.skel", 0, 0, False, False)
        env = factory.make_env()

        env.dart_world.add_skeleton("./custom_dart_assets/walker2d_reg.sdf")
        q = env.dart_world.skeletons[-1].q
        q[0] = -1
        walker2d_ndofs = env.dart_world.skeletons[-1].ndofs
        env.dart_world.skeletons[-1].set_positions(q)

        env.dart_world.add_skeleton("./custom_dart_assets/ostrich2d.sdf")
        q = env.dart_world.skeletons[-1].q
        q[0] = 1
        ostrich2d_ndofs = env.dart_world.skeletons[-1].ndofs
        env.dart_world.skeletons[-1].set_positions(q)
        encoded_state_trajectory = deque(maxlen=50)

        start_state_idx = np.random.choice(np.arange(n_states), 1)
        encoded_state = encoded_states[start_state_idx]
        while True:
            encoded_state_for_viz = compressor.transform(encoded_state.reshape(1, -1))
            encoded_state_trajectory.append(encoded_state_for_viz)
            encoded_state_trajectory_np = np.concatenate(encoded_state_trajectory, axis=0)
            plt.cla()
            # plt.xlim(-10, 10)
            # plt.ylim(-2.5, 2.5)
            plt.plot(encoded_state_trajectory_np[:, 0], encoded_state_trajectory_np[:, 1])
            plt.scatter(encoded_state_trajectory_np[-1, 0], encoded_state_trajectory_np[-1, 1])

            plt.draw()
            env.render()

            encoded_state_tensor = torch.as_tensor(encoded_state).float().reshape(1, -1)
            decoded_state_tensor, _ = walker2d_state_decoder.forward(encoded_state_tensor)
            state_tensor = walker2d_state_scaler.reverse(decoded_state_tensor)
            state = state_tensor.detach().numpy().reshape(-1)
            env.dart_world.skeletons[1].set_positions(np.concatenate([[-1], state[:walker2d_ndofs - 1]]))

            decoded_state_tensor, _ = ostrich2d_state_decoder.forward(encoded_state_tensor)
            state_tensor = ostrich2d_state_scaler.reverse(decoded_state_tensor)
            state = state_tensor.detach().numpy().reshape(-1)

            env.dart_world.skeletons[2].set_positions(np.concatenate([[1], state[:ostrich2d_ndofs - 1]]))

            encoded_velocity_input = encoded_state_tensor
            encoded_velocity_tensor, _ = encoded_velocity_predictor.forward(encoded_velocity_input)
            normal = Normal(0, 1e-2)
            noise = normal.sample(encoded_velocity_tensor.shape)
            encoded_state_tensor += (encoded_velocity_tensor + noise) * 0.5

            time.sleep(0.05)
            plt.pause(0.001)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", required=True, type=str)

    args = parser.parse_args()
    with open("./configs/visualize_latent_states.yml", "r") as f:
        config_dict = yaml.safe_load(f)[args.config_name]
    main()
