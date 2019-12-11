import argparse
import time

import numpy as np
import pygame
import torch
import yaml

from utils.factories import get_env_factory
from utils.keys import ModelKey, DataKey
from utils.utils import wait_for_keys
from utils.walker2d_evaluate import expert_act


def visualize_actor(model_dict):
    factory = get_env_factory(config_dict, args.config_name, False, args.create_box)
    env = factory.make_env()

    state_scaler = model_dict.get(ModelKey.state_scaler)
    expert_actor = model_dict.get(ModelKey.actor)

    pygame.init()
    pygame.key.set_repeat(33)
    size = width, height = 640, 480
    screen = pygame.display.set_mode(size)
    clock = pygame.time.Clock()

    state = env.reset()
    for i in range(10000):
        env.render()
        x, y = env.robot_skeleton.q[:2]

        phase = state[-1]
        state = state[:-1]
        action = expert_act(state, phase, state_scaler, expert_actor)
        state, reward, done, _ = env.step(action)

        wait_for_keys(env)
        time.sleep(0.05)

    env.close()


def visualize_dataset(data_dict):
    clock = pygame.time.Clock()
    factory = get_env_factory(config_dict, args.config_name, False, False)
    env = factory.make_env()

    states = data_dict.get(DataKey.states)
    actions = data_dict.get(DataKey.actions)
    phases = data_dict.get(DataKey.phases)
    env.reset()
    for i, (state, action, phase) in enumerate(zip(states, actions, phases)):
        if args.config_name == "pendulum2d":
            q, dq = state
            env.robot_skeleton.q = [q]
            env.robot_skeleton.dq = [dq]
        else:
            qpos = np.concatenate([np.zeros(1), state[:env.robot_skeleton.ndofs - 1]])
            env.robot_skeleton.set_positions(qpos)
        env.render()
        time.sleep(0.05)


def main():
    if args.model_dict_path is not None:
        model_dict = torch.load(args.model_dict_path, map_location="cpu")
        visualize_actor(model_dict)

    if args.data_dict_path is not None:
        data_dict = torch.load(args.data_dict_path)
        visualize_dataset(data_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str)
    parser.add_argument("--model_dict_path", type=str)
    parser.add_argument("--data_dict_path", type=str)
    parser.add_argument("--create_box", action="store_true")

    args = parser.parse_args()

    with open("./configs/visualize_character.yml", "r") as f:
        config_dict = yaml.safe_load(f)[args.config_name]

    main()
