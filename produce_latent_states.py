import argparse
from typing import List

import torch
import numpy as np
import yaml

from utils.containers import ModelDict, DataDict
from utils.keys import ModelKey, DataKey


def main():
    model_dicts_path = config_dict["model_dicts_path"]
    data_dict_paths = config_dict["data_dict_paths"]

    model_dicts: List[ModelDict] = list(torch.load(model_dicts_path, map_location="cpu").values())
    data_dicts = [torch.load(path, map_location="cpu") for path in data_dict_paths]

    encoded_states_list = []
    for model_dict, data_dict in zip(model_dicts, data_dicts):
        states = data_dict.get(DataKey.states)
        state_scaler = model_dict.get(ModelKey.state_scaler)
        state_encoder = model_dict.get(ModelKey.state_encoder)

        states_tensor = torch.as_tensor(states).float()
        states_scaled_tensor = state_scaler.forward(states_tensor)
        encoded_states_tensor, _ = state_encoder.forward(states_scaled_tensor)
        encoded_states = encoded_states_tensor.detach().numpy()
        encoded_states_list.append(encoded_states)

    encoded_states = np.concatenate(encoded_states_list, axis=0)
    n_states, _ = encoded_states.shape
    data_dict = DataDict(n_states)
    data_dict.set(DataKey.encoded_states, encoded_states)
    torch.save(data_dict, "{:s}_encoded_states.pkl".format(args.output_prefix))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", required=True, type=str)
    parser.add_argument("--output_prefix", required=True, type=str)

    args = parser.parse_args()

    with open("./configs/produce_latent_states.yml", "r") as f:
        config_dict = yaml.safe_load(f)[args.config_name]
    main()
