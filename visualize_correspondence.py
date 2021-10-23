import argparse
import time
from collections import deque

import numpy as np
import pygame
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from torch import nn

from characters.pendulum2d import Pendulum2DSolver
from models import DummyNet
from utils.containers import ModelDict
from utils.factories import Walker2DPhaseFactory, DartEnvFactoryOstrich2D, Pendulum2DFactory
from utils.keys import ModelKey
from utils.rl_common import ActionGetterFromState
from utils.utils import initialize_pygame, wait_for_keys
from utils.visualize_utils import PhaseSpaceDiagram, RED, WHITE
from utils.walker2d_evaluate import correspond


def expert_act(input_state, input_phase, input_state_scaler, expert_actor):
    state_with_phase = np.concatenate([input_state, np.asarray([input_phase])]).reshape(1, -1)
    state_with_phase_tensor = torch.as_tensor(state_with_phase).float()
    state_with_phase_scaled_tensor = input_state_scaler.forward(state_with_phase_tensor)
    action_tensor, _ = expert_actor.forward(state_with_phase_scaled_tensor)
    action = action_tensor.cpu().detach().numpy().reshape(-1, )
    return action


def render_one_way_correspondence(factory, model_dicts, expert_dict):
    center, screen = initialize_pygame()

    # For pendulum phase visualization
    diagram = PhaseSpaceDiagram(center)
    walker_to_pendulum_buffer = deque(maxlen=30)
    diagram.add_buffer(walker_to_pendulum_buffer, RED)

    env = factory.make_env()
    if "to_walker2d" in args.config_name:
        env.dart_world.add_skeleton("./custom_dart_assets/walker2d_reg.sdf")
    if "to_ostrich2d" in args.config_name:
        env.dart_world.add_skeleton("./custom_dart_assets/ostrich2d.sdf")
    if "to_pendulum2d" in args.config_name:
        env.dart_world.add_skeleton("./custom_dart_assets/pendulum2d.sdf")

    if "to_pendulum2d" in args.config_name:
        pass
    if "pendulum2d_to" in args.config_name:
        model_dict_source: ModelDict = list(model_dicts.values())[1]
        model_dict_target: ModelDict = list(model_dicts.values())[0]
    elif "walker2d_to" in args.config_name:
        model_dict_source: ModelDict = list(model_dicts.values())[1]
        model_dict_target: ModelDict = list(model_dicts.values())[0]
    else:
        model_dict_source: ModelDict = list(model_dicts.values())[0]
        model_dict_target: ModelDict = list(model_dicts.values())[1]

    state_scaler_source: StandardScaler = model_dict_source.get(ModelKey.state_scaler)
    state_encoder_source: nn.Module = model_dict_source.get(ModelKey.state_encoder)

    state_scaler_target: StandardScaler = model_dict_target.get(ModelKey.state_scaler)
    state_decoder_target: nn.Module = model_dict_target.get(ModelKey.state_decoder)

    expert_state_scaler = expert_dict.get(ModelKey.state_scaler)
    if "pendulum2d_to" in args.config_name:
        expert_actor = Pendulum2DSolver(env)
        expert_state_scaler = DummyNet()
    else:
        expert_actor = expert_dict.get(ModelKey.actor)
    action_getter = ActionGetterFromState(expert_state_scaler, expert_actor)

    state_source = env.reset()

    for i in range(100000):
        screen.fill(WHITE)
        # draw everything
        env.render()

        # draw pendulum phase
        diagram.render(screen)
        pygame.display.flip()

        state_source_ = state_source.copy()[:-1]
        if env.phase >= 10:
            state_source_ = env.state_getter._mirror_state(state_source_)

        state_source_to_target = correspond(state_source_, state_scaler_source, state_scaler_target,
                                            state_encoder_source, state_decoder_target)

        action_source = action_getter.get_action(state_source)
        q = env.dart_world.skeletons[1].q
        ndofs_target = env.dart_world.skeletons[-1].ndofs
        env.dart_world.skeletons[-1].set_positions(np.concatenate([[q[0]], state_source_to_target[:ndofs_target - 1]]))
        env.dart_world.skeletons[-1].set_velocities(np.zeros(ndofs_target))
        diagram.push(state_source_to_target, walker_to_pendulum_buffer)

        state_source, _, _, _ = env.step(action_source)

        wait_for_keys(env)
        time.sleep(0.05)


def render_one_way_correspondence_with_multiple_models(factory, list_of_model_dict_pairs, expert_dict):
    center, screen = initialize_pygame()
    diagram = PhaseSpaceDiagram(center)
    walker_to_pendulum_buffer = deque(maxlen=30)
    diagram.add_buffer(walker_to_pendulum_buffer, RED)

    env = factory.make_env()
    for i in range(len(list_of_model_dict_pairs)):
        if "to_walker2d" in args.config_name:
            env.dart_world.add_skeleton("./custom_dart_assets/walker2d_reg.sdf")
        if "to_ostrich2d" in args.config_name:
            env.dart_world.add_skeleton("./custom_dart_assets/ostrich2d.sdf")
        if "to_pendulum2d" in args.config_name:
            env.dart_world.add_skeleton("./custom_dart_assets/pendulum2d.sdf")

    expert_state_scaler = expert_dict.get(ModelKey.state_scaler)
    expert_actor = expert_dict.get(ModelKey.actor)
    action_getter = ActionGetterFromState(expert_state_scaler, expert_actor)

    true_state = env.reset()
    for i in range(100000):
        screen.fill(WHITE)
        # draw everything
        env.render()
        diagram.render(screen)
        pygame.display.flip()

        true_state_ = true_state.copy()[:-1]
        if env.phase >= 10:
            true_state_ = env.state_getter._mirror_state(true_state_)

        action_source = action_getter.get_action(true_state)

        # compute correspondence per model
        for i, model_dict_pair in enumerate(list_of_model_dict_pairs):
            if "to_walker2d" in args.config_name:
                model_dict_source, model_dict_target = model_dict_pair
            else:
                model_dict_target, model_dict_source = model_dict_pair

            # unpack model dicts
            state_scaler_source: nn.Module = model_dict_source.get(ModelKey.state_scaler)
            state_encoder_source: nn.Module = model_dict_source.get(ModelKey.state_encoder)

            state_scaler_target: nn.Module = model_dict_target.get(ModelKey.state_scaler)
            state_decoder_target: nn.Module = model_dict_target.get(ModelKey.state_decoder)

            state_source_to_target = correspond(true_state_, state_scaler_source, state_scaler_target,
                                                state_encoder_source, state_decoder_target)

            # render corresponded states
            skeleton_idx = 2 + i
            q_source = env.dart_world.skeletons[1].q
            x_target = q_source[0] - 0.8 * len(list_of_model_dict_pairs) + i * 2
            ndofs_target = env.dart_world.skeletons[skeleton_idx].ndofs
            env.dart_world.skeletons[skeleton_idx].set_positions(
                np.concatenate([[x_target], state_source_to_target[:ndofs_target - 1]]))
            env.dart_world.skeletons[skeleton_idx].set_velocities(np.zeros(ndofs_target))

        diagram.push(state_source_to_target, walker_to_pendulum_buffer)

        true_state, _, _, _ = env.step(action_source)

        wait_for_keys(env)
        time.sleep(0.05)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--create_box", action="store_true")

    args = parser.parse_args()

    with open('./configs/visualize_correspondence.yml', 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)[args.config_name]

    if "walker2d_to_" in args.config_name:
        factory = Walker2DPhaseFactory("./custom_dart_assets/walker2d_reg.skel", 1, None, False, True, args.create_box,
                                       True)
    elif "ostrich2d_to" in args.config_name:
        factory = DartEnvFactoryOstrich2D("./custom_dart_assets/ostrich2d.skel", 25, 20, False, args.create_box,
                                          True, True)
    else:
        factory = Pendulum2DFactory("./custom_dart_assets/pendulum2d.skel", "./artifacts/pendulum2d_generated.skel",
                                    1.0, False, True)
    expert_dict = config_dict['expert_dict_path']
    expert_dict = torch.load(expert_dict, map_location='cpu')
    if "model_dicts_paths" in config_dict.keys():
        model_dicts_paths = config_dict['model_dicts_paths']
        all_model_dicts = [torch.load(m, map_location='cpu') for m in model_dicts_paths]
        list_of_model_dict_pairs = [tuple(d.values()) for d in all_model_dicts]
        render_one_way_correspondence_with_multiple_models(factory, list_of_model_dict_pairs, expert_dict)

    if 'model_dicts_path' in config_dict.keys():
        model_dicts_path = config_dict['model_dicts_path']
        model_dicts = torch.load(model_dicts_path, map_location='cpu')
        render_one_way_correspondence(factory, model_dicts, expert_dict)
