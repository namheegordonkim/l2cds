import argparse
from typing import List, Dict

import numpy as np
import torch
import yaml
from scipy.spatial import distance_matrix

from utils.containers import DataDict, ModelDict
from utils.keys import ModelKey, DataKey
from utils.walker2d_evaluate import correspond


def get_symmetric_nn_distance(source, target):
    source_vs_target = distance_matrix(source, target)
    mean_distance_from_source = np.mean(np.min(source_vs_target, axis=0))
    mean_distance_from_target = np.mean(np.min(source_vs_target, axis=1))
    nn_distances_symmetric = (mean_distance_from_source + mean_distance_from_target) / 2
    return nn_distances_symmetric


def get_symmetric_nn_distances(source, targets):
    distances = []
    for target in targets:
        distance = get_symmetric_nn_distance(source, target)
        distances.append(distance)
    return distances


def get_target(source_states, source_model_dict, target_model_dict):
    source_state_scaler = source_model_dict.get(ModelKey.state_scaler)
    source_state_encoder = source_model_dict.get(ModelKey.state_encoder)

    target_state_scaler = target_model_dict.get(ModelKey.state_scaler)
    target_state_decoder = target_model_dict.get(ModelKey.state_decoder)
    corresponded_states = correspond(source_states, source_state_scaler, target_state_scaler, source_state_encoder,
                                     target_state_decoder)
    return corresponded_states


def get_source_latent_targets(source_states, source_model_dicts, target_model_dicts):
    results = []
    for source_model_dict, target_model_dict in zip(source_model_dicts, target_model_dicts):
        source_state_scaler = source_model_dict.get(ModelKey.state_scaler)
        source_state_encoder = source_model_dict.get(ModelKey.state_encoder)
        target_state_scaler = target_model_dict.get(ModelKey.state_scaler)
        target_state_decoder = target_model_dict.get(ModelKey.state_decoder)
        corresponded_states = correspond(source_states, source_state_scaler, target_state_scaler, source_state_encoder,
                                         target_state_decoder)
        results.append(corresponded_states)
    return results


def get_source_latent_target_latent_sources(source_states, source_model_dicts, target_model_dicts):
    results = []
    for source_model_dict, target_model_dict in zip(source_model_dicts, target_model_dicts):
        source_state_scaler = source_model_dict.get(ModelKey.state_scaler)
        source_state_encoder = source_model_dict.get(ModelKey.state_encoder)
        source_state_decoder = source_model_dict.get(ModelKey.state_decoder)

        target_state_scaler = target_model_dict.get(ModelKey.state_scaler)
        target_state_encoder = target_model_dict.get(ModelKey.state_encoder)
        target_state_decoder = target_model_dict.get(ModelKey.state_decoder)

        corresponded_target_states = correspond(source_states, source_state_scaler, target_state_scaler,
                                                source_state_encoder,
                                                target_state_decoder)

        corresponded_source_states = correspond(corresponded_target_states, target_state_scaler, source_state_scaler,
                                                target_state_encoder,
                                                source_state_decoder)
        results.append(corresponded_source_states)
    return results


def main():
    dfs = []
    for experiment_name, config_dict in config_object.items():
        print("Experiment: {:s}".format(experiment_name))

        # unpack config stuff
        dataset_paths = config_dict['dataset_paths']
        alpha_models_paths = config_dict['alpha_models_paths']

        data_dicts: List[DataDict] = [torch.load(path) for path in dataset_paths]
        alpha_model_dict_pairs: List[Dict[ModelDict]] = [torch.load(path, map_location="cpu") for path in
                                                         alpha_models_paths]
        character_names = ["O", "W"]
        # plt.figure()
        for source_idx in range(2):
            target_idx = 1 - source_idx
            # target_idx = source_idx
            source_character_name = character_names[source_idx]
            target_character_name = character_names[target_idx]
            print("Source character: {:s}".format(source_character_name))
            print("Target character: {:s}".format(target_character_name))
            source_data_dict = data_dicts[source_idx]
            target_data_dict = data_dicts[target_idx]
            source_model_dicts = [list(d.values())[source_idx] for d in alpha_model_dict_pairs]
            target_model_dicts = [list(d.values())[target_idx] for d in alpha_model_dict_pairs]

            source_states = source_data_dict.get(DataKey.states)
            corresponded_states_list = get_source_latent_targets(source_states, source_model_dicts,
                                                                 source_model_dicts)

            sls_distances = get_symmetric_nn_distances(source_states, corresponded_states_list)
            print(
                "{:s}L{:s}: {:.2f} ({:.2f})".format(source_character_name, source_character_name,
                                                    np.mean(sls_distances),
                                                    np.std(sls_distances)))

            source_states = source_data_dict.get(DataKey.states)
            corresponded_states_list = get_source_latent_target_latent_sources(source_states, source_model_dicts,
                                                                               target_model_dicts)

            sltls_distances = get_symmetric_nn_distances(source_states, corresponded_states_list)
            print("{:s}L{:s}L{:s}: {:.2f} ({:.2f})".format(source_character_name, target_character_name,
                                                           source_character_name, np.mean(sltls_distances),
                                                           np.std(sltls_distances)))

            source_states = source_data_dict.get(DataKey.states)
            target_states = target_data_dict.get(DataKey.states)
            corresponded_states_list = get_source_latent_targets(target_states, target_model_dicts,
                                                                 source_model_dicts)
            tls_distances = get_symmetric_nn_distances(source_states, corresponded_states_list)
            print("{:s}L{:s}: {:.2f} ({:.2f})".format(target_character_name, source_character_name,
                                                      np.mean(tls_distances), np.std(tls_distances)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    with open("./configs/correspondence_evaluate.yml", "r") as f:
        config_object = yaml.safe_load(f)

    main()
