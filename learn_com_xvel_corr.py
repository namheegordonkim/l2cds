from typing import List

from scipy import stats
import matplotlib.pyplot as plt
import torch
import numpy as np
import statsmodels.api as sm

from utils.containers import DataDict, ModelDict
from utils.keys import ModelKey, DataKey
from utils.walker2d_evaluate import correspond


def main():
    np.random.seed(0)

    model_dicts_path = "./good_results/ostrich2d_to_walker2d_alpha_model_dicts_0000084_00.pkl"
    model_dicts = torch.load(model_dicts_path, map_location="cpu")
    data_dict_paths = ["./data/ostrich2d_0001000.pkl", "./data/walker2d_reg_0001000.pkl"]
    data_dicts: List[DataDict] = [torch.load(p, map_location="cpu") for p in data_dict_paths]

    # ostrich to walker
    source_data_dict = data_dicts[0]

    source_model_dict: ModelDict = list(model_dicts.values())[0]
    source_state_scaler = source_model_dict.get(ModelKey.state_scaler)
    source_state_encoder = source_model_dict.get(ModelKey.state_encoder)

    target_model_dict: ModelDict = list(model_dicts.values())[1]
    target_state_scaler = target_model_dict.get(ModelKey.state_scaler)
    target_state_decoder = target_model_dict.get(ModelKey.state_decoder)

    source_states = source_data_dict.get(DataKey.states)
    _, n_source_features = source_states.shape
    n_perturbations = 10
    source_states_with_perturbed_com_xvel = []
    # add perturbations to root x-velocity
    for state in source_states:
        new_states = np.tile(state, [n_perturbations, 1])
        perturbation_amounts = np.random.uniform(-3, 3, n_perturbations)
        new_states[:, 10] += perturbation_amounts
        source_states_with_perturbed_com_xvel.append(new_states)
    source_states_with_perturbed_com_xvel = np.concatenate(source_states_with_perturbed_com_xvel, axis=0)

    estimated_target_states = correspond(source_states_with_perturbed_com_xvel, source_state_scaler, target_state_scaler, source_state_encoder,
                                         target_state_decoder)

    source_com_xvel = source_states_with_perturbed_com_xvel[:, 10]
    estimated_target_com_xvel = estimated_target_states[:, 8]
    model = sm.OLS(estimated_target_com_xvel, source_com_xvel)
    results = model.fit()

    slope, intercept, rvalue, pvalue, stderr = stats.linregress(source_com_xvel, estimated_target_com_xvel)

    plt.figure()
    plt.title("OLW, correlation between corresponding root x-velocities")
    plt.scatter(source_com_xvel, estimated_target_com_xvel, marker="^", alpha=0.7, linewidths=0.01, label="original data")
    plt.plot(source_com_xvel, results.params[0] * source_com_xvel, 'r', label="$\\beta={:.3f},R^2={:.2f}, p={:.3f}$".format(
        results.params[0], results.rsquared, results.pvalues[0]))
    plt.plot(source_com_xvel, intercept + slope * source_com_xvel, 'orange',
             label="$\\beta_0={:.3f}, \\beta_0={:.3f}, R^2={:.3f}, p={:.3f}$".format(
                 slope, intercept, rvalue**2, pvalue))
    plt.xlabel("ostrich2d's root x-velocity")
    plt.ylabel("walker2d's root x-velocity (corresponded)")
    plt.legend()
    plt.show()

    # walker to ostrich
    source_data_dict = data_dicts[1]

    source_model_dict: ModelDict = list(model_dicts.values())[1]
    source_state_scaler = source_model_dict.get(ModelKey.state_scaler)
    source_state_encoder = source_model_dict.get(ModelKey.state_encoder)

    target_model_dict: ModelDict = list(model_dicts.values())[0]
    target_state_scaler = target_model_dict.get(ModelKey.state_scaler)
    target_state_decoder = target_model_dict.get(ModelKey.state_decoder)

    source_states = source_data_dict.get(DataKey.states)
    _, n_source_features = source_states.shape
    n_perturbations = 10
    source_states_with_perturbed_com_xvel = []
    # add perturbations to root x-velocity
    for state in source_states:
        new_states = np.tile(state, [n_perturbations, 1])
        perturbation_amounts = np.random.uniform(-3, 3, n_perturbations)
        new_states[:, 8] += perturbation_amounts
        source_states_with_perturbed_com_xvel.append(new_states)
    source_states_with_perturbed_com_xvel = np.concatenate(source_states_with_perturbed_com_xvel, axis=0)

    estimated_target_states = correspond(source_states_with_perturbed_com_xvel, source_state_scaler,
                                         target_state_scaler, source_state_encoder,
                                         target_state_decoder)

    source_com_xvel = source_states_with_perturbed_com_xvel[:, 8]
    estimated_target_com_xvel = estimated_target_states[:, 10]

    model = sm.OLS(estimated_target_com_xvel, source_com_xvel)
    results = model.fit()

    slope, intercept, rvalue, pvalue, stderr = stats.linregress(source_com_xvel, estimated_target_com_xvel)

    plt.figure()
    plt.title("WLO, correlation between corresponding root x-velocities")
    plt.scatter(source_com_xvel, estimated_target_com_xvel, marker="^", alpha=0.7, linewidths=0.01,
                label="original data")
    plt.plot(source_com_xvel, results.params[0] * source_com_xvel, 'r',
             label="$\\beta={:.3f},R^2={:.2f}, p={:.3f}$".format(
                 results.params[0], results.rsquared, results.pvalues[0]))
    plt.plot(source_com_xvel, intercept + slope * source_com_xvel, 'orange',
             label="$\\beta_0={:.3f}, \\beta_0={:.3f}, R^2={:.3f}, p={:.3f}$".format(
                 slope, intercept, rvalue ** 2, pvalue))
    plt.xlabel("walker2d's root x-velocity")
    plt.ylabel("ostrich2d's root x-velocity (corresponded)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
