import argparse
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from torch import nn
from tqdm import tqdm

from utils.containers import ModelDict, DataDict
from utils.keys import ModelKey, TensorKey, DataKey
from utils.loss_calculators import LossCalculatorSum, LossCalculatorInputTarget, \
    LossCalculatorNearestNeighborL2, LossCalculatorApply
from models import NNet, ScalerWrapper
from utils.learners import HeterogeneousLearner
from utils.radam import RAdam
from utils.tensor_collectors import TensorCollector, TensorListGetterOneToOne
from utils.tensor_inserters import TensorInserterApplyModel, TensorInserterForward, TensorInserterSeq, \
    TensorInserterTensorizeScaled, TensorInserterTensorizeTransformed, TensorInserterUniTransform, TensorInserterSum
from utils.walker2d_evaluate import RewardGetterDummy


def apply_forward_dynamics(noise_scale):
    return TensorInserterForward(TensorKey.encoded_states_tensor, ModelKey.encoded_velocity_predictor,
                                 TensorKey.encoded_velocity_predictions_tensor, noise_scale=noise_scale)


def learn_with_new_model_dicts(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    latent_state_dim = config_dict['latent_state_dim']
    dataset_paths = config_dict['dataset_paths']

    nn_weight = config_dict["nn_weight"]
    ae_weight = config_dict["ae_weight"]
    fd_weight = config_dict["fd_weight"]
    pv_weight = config_dict["pv_weight"]
    noise_scale = config_dict["noise_scale"]

    identity = nn.Identity()
    tanh = nn.Tanh()

    # load datasets
    data_dicts = []
    for path in dataset_paths:
        data_dict: DataDict = torch.load(path)
        data_dicts.append(data_dict)

    encoded_velocity_predictor = NNet(latent_state_dim, latent_state_dim, tanh,
                                      hidden_dims=[256, 256])

    trainable_parameters = []
    trainable_parameters.extend(list(encoded_velocity_predictor.parameters()))

    model_dicts = []
    for i, data_dict in enumerate(data_dicts):
        states = data_dict.get(DataKey.states)
        actions = data_dict.get(DataKey.actions)
        _, state_dim = states.shape
        _, action_dim = actions.shape

        state_encoder = NNet(state_dim, latent_state_dim, identity, hidden_dims=[256, 256])
        state_decoder = NNet(latent_state_dim, state_dim, identity, hidden_dims=[256, 256])
        state_scaler_ = StandardScaler()
        state_scaler_.fit(states)
        state_scaler = ScalerWrapper(state_scaler_)

        model_dict = ModelDict()
        model_dict.set(ModelKey.state_encoder, state_encoder)
        model_dict.set(ModelKey.state_decoder, state_decoder)
        model_dict.set(ModelKey.encoded_velocity_predictor, encoded_velocity_predictor)
        model_dict.set(ModelKey.state_scaler, state_scaler)
        model_dicts.append(model_dict)

        trainable_parameters.extend(list(state_encoder.parameters()))
        trainable_parameters.extend(list(state_decoder.parameters()))

    tensor_collector = TensorCollector(TensorInserterSeq([
        # put data into tensors
        TensorInserterTensorizeScaled(DataKey.states, ModelKey.state_scaler, TensorKey.states_tensor, torch.float),
        TensorInserterTensorizeScaled(DataKey.next_states, ModelKey.state_scaler, TensorKey.next_states_tensor,
                                      torch.float),
        TensorInserterTensorizeTransformed(DataKey.phases, (lambda x: x / 20.), TensorKey.phases_tensor, torch.float),

        # generate forwarded tensors
        TensorInserterForward(TensorKey.states_tensor, ModelKey.state_encoder,
                              TensorKey.encoded_states_tensor),
        TensorInserterForward(TensorKey.next_states_tensor, ModelKey.state_encoder,
                              TensorKey.encoded_next_states_tensor),
        TensorInserterForward(TensorKey.encoded_states_tensor, ModelKey.state_decoder,
                              TensorKey.decoded_states_tensor, noise_scale=noise_scale),
        generate_noise_for_encoded_states(noise_scale),
        apply_forward_dynamics(noise_scale),
        # break encoded states into position and velocity
        TensorInserterUniTransform(TensorKey.encoded_states_tensor,
                                   lambda tensor: tensor[:, :int(tensor.shape[1] / 2)],
                                   TensorKey.encoded_positions_tensor),
        TensorInserterUniTransform(TensorKey.encoded_states_tensor,
                                   lambda tensor: tensor[:, int(tensor.shape[1] / 2):],
                                   TensorKey.encoded_velocities_tensor),
        TensorInserterSum([TensorKey.encoded_positions_tensor, TensorKey.encoded_velocities_tensor],
                          TensorKey.encoded_next_position_predictions_tensor),
        TensorInserterUniTransform(TensorKey.encoded_next_states_tensor,
                                   lambda tensor: tensor[:, :int(tensor.shape[1] / 2)],
                                   TensorKey.encoded_next_positions_tensor),
        # apply forward dynamics prediction
        TensorInserterSum([TensorKey.encoded_states_tensor, TensorKey.encoded_velocity_predictions_tensor],
                          TensorKey.encoded_next_state_predictions_tensor),
    ]), TensorListGetterOneToOne())

    mse_loss = nn.MSELoss()
    loss_calculator = LossCalculatorSum([
        # state cycle consistency
        LossCalculatorInputTarget(TensorKey.decoded_states_tensor, TensorKey.states_tensor, mse_loss, ae_weight),
        # forward dynamics
        LossCalculatorInputTarget(TensorKey.encoded_next_position_predictions_tensor,
                                  TensorKey.encoded_next_positions_tensor, mse_loss, fd_weight),
        LossCalculatorInputTarget(TensorKey.encoded_next_state_predictions_tensor,
                                  TensorKey.encoded_next_states_tensor,
                                  mse_loss, pv_weight),
        # bipartite nearest neighbor in encoded state space
        LossCalculatorNearestNeighborL2(TensorKey.encoded_states_tensor, TensorKey.origins_tensor, nn_weight),
        # make velocity norms close to one
        LossCalculatorApply(TensorKey.encoded_velocity_predictions_tensor,
                            lambda tensor: torch.mean((torch.norm(tensor, 2, dim=1) - 1)) ** 2, 1.),
    ])

    optimizer = RAdam(params=trainable_parameters, lr=3e-4)

    reward_getter = RewardGetterDummy()

    num_episodes = config_dict['num_episodes']
    num_epochs = config_dict['num_epochs']
    save_every = config_dict['save_every']
    batch_size = config_dict['batch_size']
    print_every = save_every

    optim = HeterogeneousLearner(loss_calculator, tensor_collector, optimizer, num_epochs)

    print("Run training")

    losses = []
    rewards = []
    # first choose a dataset, and then train one episode
    for episode in tqdm(range(num_episodes)):
        # full batch loss calculation
        episode_loss = optim.train_one_episode(data_dicts, model_dicts, batch_size)
        rewards_mean, rewards_std = reward_getter.get_reward_mean_and_std()

        if episode % save_every == 0:
            save_models(dataset_paths, model_dicts, losses, rewards, episode, seed)
            fig_path = "{:s}_enc_{:07d}_{:02d}.png".format(args.output_prefix, episode, seed)
            save_encoded_states_figure(fig_path, data_dicts[0], data_dicts[1], model_dicts[0], model_dicts[1])

        if episode % print_every == 0:
            print("Episode {:d}\tLoss: {:f}".format(episode, episode_loss))
            print("Episode {:d}\tReward Mean: {:f}\tStd: {:f}".format(episode, rewards_mean, rewards_std))

        losses.append(episode_loss)
        rewards.append(rewards_mean)


def generate_noise_for_encoded_states(noise_scale):
    return TensorInserterUniTransform(TensorKey.encoded_states_tensor,
                                      lambda tensor: torch.as_tensor(
                                          np.random.normal(0, noise_scale, tensor.shape)).float(),
                                      TensorKey.noise_tensor)


def save_encoded_states_figure(fig_path, data_dict_source, data_dict_target, model_dict_source, model_dict_target):
    states_source = data_dict_source.get(DataKey.states)
    states_target = data_dict_target.get(DataKey.states)
    state_scaler_source: ScalerWrapper = model_dict_source.get(ModelKey.state_scaler)
    state_encoder_source: nn.Module = model_dict_source.get(ModelKey.state_encoder)
    state_scaler_target: ScalerWrapper = model_dict_target.get(ModelKey.state_scaler)
    state_encoder_target: nn.Module = model_dict_target.get(ModelKey.state_encoder)

    states_source_tensor = torch.as_tensor(states_source).float()
    states_source_scaled_tensor = state_scaler_source.forward(states_source_tensor)
    encoded_states_source_tensor, _ = state_encoder_source.forward(states_source_scaled_tensor)
    encoded_states_source = encoded_states_source_tensor.cpu().detach().numpy()

    states_target_tensor = torch.as_tensor(states_target).float()
    states_target_scaled_tensor = state_scaler_target.forward(states_target_tensor)
    encoded_states_target_tensor, _ = state_encoder_target.forward(states_target_scaled_tensor)
    encoded_states_target = encoded_states_target_tensor.cpu().detach().numpy()

    plt.figure()
    plt.scatter(encoded_states_source[:, 0], encoded_states_source[:, 1], alpha=0.3)
    plt.scatter(encoded_states_target[:, 0], encoded_states_target[:, 1], alpha=0.3)
    plt.savefig(fig_path)
    plt.close()
    print("Saved encoded states figure to {:s}".format(fig_path))


def save_models(dataset_paths: List[str], model_dicts: List[ModelDict], losses, rewards, episode, seed):
    output_prefix = args.output_prefix
    model_containers_dict = dict(zip(dataset_paths, model_dicts))
    model_containers_output_path = "{:s}_alpha_model_dicts_{:07d}_{:02d}.pkl".format(output_prefix, episode, seed)
    torch.save(model_containers_dict, model_containers_output_path)
    print("Saved model containers to {:s}".format(model_containers_output_path))

    loss_fig_output_path = "{:s}_losses_{:07d}_{:02d}.png".format(output_prefix, episode, seed)
    plt.figure()
    plt.plot(losses)
    plt.savefig(loss_fig_output_path)
    print("Saved losses figure to {:s}".format(loss_fig_output_path))
    plt.close()

    reward_fig_output_path = "{:s}_rewards_{:07d}_{:02d}.png".format(output_prefix, episode, seed)
    plt.figure()
    plt.plot(rewards)
    plt.savefig(reward_fig_output_path)
    print("Saved rewards figure to {:s}".format(reward_fig_output_path))
    plt.close()


def main():
    learn_with_new_model_dicts(seed)


if __name__ == "__main__":
    """
    Train a latent policy based on the datasets specified in the YAML config file.
    Save these objects:
     - a model container for each dataset containing encoder and decoder networks
     - an actor network for the latent policy
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, required=True)
    parser.add_argument("--seed", type=int)

    args = parser.parse_args()

    with open("./configs/learn_correspondence.yml", "r") as f:
        config_dict = yaml.load(f, yaml.SafeLoader)[args.config_name]

    seed = args.seed
    if seed is None:
        seed = 0

    main()
