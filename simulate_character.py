import argparse

import numpy as np
import torch
import yaml
from tqdm import tqdm

from characters.pendulum2d import Pendulum2DSolver
from models import DummyNet
from utils.containers import DataDict, ModelDict
from utils.datasets import TupleSARSP, DatasetSARSP
from utils.factories import get_env_factory
from utils.keys import ModelKey, DataKey
from utils.rl_common import ActionGetterFromState
from utils.utils import reflect_control_vector


def main():
    config_name = args.config_name
    with open('./configs/simulate_character.yml', 'r') as f:
        config_dict = yaml.safe_load(f)[config_name]
    noise_scale = config_dict["noise_scale"]
    reset_every = config_dict["reset_every"]
    num_examples = config_dict["num_examples"]
    push_scale = config_dict["push_scale"]

    factory = get_env_factory(config_dict, config_name, True, False)

    env = factory.make_env()
    env.seed()

    if config_name == "pendulum2d":
        state_scaler = DummyNet()
        actor_model = Pendulum2DSolver(env)
        model_dict = ModelDict()
        model_dict.set(ModelKey.state_scaler, state_scaler)
        model_dict.set(ModelKey.actor, actor_model)
    else:
        model_dict = torch.load(config_dict['expert_path'], map_location='cpu')

    state_scaler = model_dict.get(ModelKey.state_scaler)
    actor_model = model_dict.get(ModelKey.actor)

    action_getter = ActionGetterFromState(state_scaler, actor_model)

    perturb_every = 61

    # collect state-action pairs
    tups = []
    print('Collect state-action pairs')
    state = env.reset()
    done = False
    burn_in_counter = 40  # discard the first few ill-defined states
    pbar = tqdm(total=num_examples)
    example_counter = 0
    trajectory_counter = 0
    while example_counter < num_examples:

        if done or trajectory_counter >= reset_every:
            state = env.reset()
            burn_in_counter = 40
            trajectory_counter = 0  # start trajectory over

        phase = env.phase  # current phase
        next_phase = (phase + 1) % 20
        log_prob = 0
        action = action_getter.get_action(state)

        if trajectory_counter > 0 and trajectory_counter % perturb_every == 0:
            push_strength = np.random.normal(-push_scale, push_scale)
            push_strength = np.clip(push_strength, -push_scale, push_scale)
            dq = env.robot_skeleton.dq
            dq[0] += push_strength
            env.robot_skeleton.set_velocities(dq)

        noise = np.random.normal(0, noise_scale)
        next_state, reward, done, _ = env.step(action + noise)

        state_ = state.copy()
        action_ = action.copy()
        next_state_ = next_state.copy()

        if config_dict["mirror"]:
            if phase >= 10:
                state_ = env.state_getter._mirror_state(state_)
                action_ = reflect_control_vector(action_)
            if next_phase >= 10:
                next_state_ = env.state_getter._mirror_state(next_state_)

        if burn_in_counter <= 0:
            tups.append(TupleSARSP(state_[:-1], phase, action_, log_prob, reward, next_state_[:-1], done))
            example_counter += 1
            trajectory_counter += 1
            pbar.update(1)

        burn_in_counter = max(burn_in_counter - 1, 0)
        state = next_state

    pbar.close()

    print('Turn state-action pairs into a dataset')
    dataset = DatasetSARSP.from_tuple_list(tups)

    data_dict = DataDict(len(tups))
    data_dict.set(DataKey.states, dataset.states)
    data_dict.set(DataKey.actions, dataset.actions)
    data_dict.set(DataKey.next_states, dataset.next_states)
    data_dict.set(DataKey.rewards, dataset.rewards)
    data_dict.set(DataKey.dones, dataset.dones)
    data_dict.set(DataKey.phases, dataset.phases)
    data_dict.set(DataKey.log_probs, dataset.log_probs)

    print('Save dataset as a pickle')
    save_filename = '{:s}_{:07d}.pkl'.format(args.output_prefix, num_examples)
    torch.save(data_dict, save_filename)
    print('Saved to {:s}'.format(save_filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_prefix", required=True, type=str)
    parser.add_argument("--config_name", required=True, type=str)

    args = parser.parse_args()

    main()
