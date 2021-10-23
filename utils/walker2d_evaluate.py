import torch
import numpy as np

from utils.rl_common import RewardGetter
from utils.utils import device


class RewardGetterDummy(RewardGetter):

    def get_cumulative_reward(self):
        return 0

    def get_reward_mean_and_std(self):
        return 0, 0


class CanonicalActorEvaluator(RewardGetter):

    def __init__(self, env, input_state_scaler, output_state_scaler, input_state_encoder, output_state_decoder,
                 output_state_encoder, canonical_actor, input_action_decoder):
        self.env = env
        self.input_state_scaler = input_state_scaler
        self.output_state_scaler = output_state_scaler
        self.output_state_encoder = output_state_encoder
        self.canonical_actor = canonical_actor
        self.input_action_decoder = input_action_decoder
        self.output_state_decoder = output_state_decoder
        self.input_state_encoder = input_state_encoder

    def get_cumulative_reward(self):
        return evaluate_canonical_actor(self.env, self.input_state_scaler, self.output_state_scaler,
                                        self.input_state_encoder, self.output_state_decoder, self.output_state_encoder,
                                        self.canonical_actor, self.input_action_decoder)

    def get_reward_mean_and_std(self):
        rewards = []
        for i in range(10):
            total_reward = self.get_cumulative_reward()
            rewards.append(total_reward)
        mean = np.mean(rewards)
        std = np.std(rewards)
        return mean, std


class AutoencoderEvaluator(RewardGetter):

    def __init__(self, env, state_scaler, state_encoder, state_decoder, expert_state_scaler, expert_actor):
        self.expert_state_scaler = expert_state_scaler
        self.state_scaler = state_scaler
        self.state_encoder = state_encoder
        self.state_decoder = state_decoder
        self.expert_actor = expert_actor
        self.env = env

    def get_cumulative_reward(self):
        return evaluate_actor(self.env, self.state_scaler, self.state_scaler, self.state_encoder, self.state_decoder,
                              self.expert_state_scaler, self.expert_actor)

    def get_reward_mean_and_std(self):
        rewards = []
        for i in range(10):
            total_reward = self.get_cumulative_reward()
            rewards.append(total_reward)
        mean = np.mean(rewards)
        std = np.std(rewards)
        return mean, std


class CorrespondenceEvaluator(RewardGetter):
    def __init__(self, env, input_state_scaler, output_state_scaler, input_state_encoder, output_state_encoder,
                 input_state_decoder, output_state_decoder,
                 expert_state_scaler, expert_actor):
        self.env = env

        self.input_state_scaler = input_state_scaler
        self.output_state_scaler = output_state_scaler

        self.input_state_encoder = input_state_encoder
        self.input_state_decoder = input_state_decoder

        self.output_state_encoder = output_state_encoder
        self.output_state_decoder = output_state_decoder

        self.expert_state_scaler = expert_state_scaler
        self.expert_actor = expert_actor

    def get_cumulative_reward(self):
        return evaluated_reconstructed_actor(self.env, self.input_state_scaler, self.output_state_scaler,
                                             self.input_state_encoder, self.output_state_encoder,
                                             self.input_state_decoder, self.output_state_decoder,
                                             self.expert_state_scaler, self.expert_actor)

    def get_reward_mean_and_std(self):
        rewards = []
        for i in range(10):
            total_reward = self.get_cumulative_reward()
            rewards.append(total_reward)
        mean = np.mean(rewards)
        std = np.std(rewards)
        return mean, std


def correspond(input_state, input_state_scaler, output_state_scaler, input_state_encoder, output_state_decoder):
    # input_state_tensor = torch.as_tensor(input_state).reshape(1, -1).float().to(device)
    input_state_tensor = torch.as_tensor(input_state).float().cpu()
    input_state_scaled_tensor = input_state_scaler.forward(input_state_tensor).cpu()
    encoded_state_tensor, _ = input_state_encoder.forward(input_state_scaled_tensor)
    output_state_scaled_tensor, _ = output_state_decoder.forward(encoded_state_tensor)
    output_state_tensor = output_state_scaler.reverse(output_state_scaled_tensor)
    output_state = output_state_tensor.cpu().detach().numpy().reshape(-1, )
    return output_state


def canonical_act(input_state, input_phase, input_state_scaler, input_state_encoder, canonical_actor, action_decoder):
    input_state_tensor = torch.as_tensor(input_state).reshape(1, -1).float().to(device)
    input_state_scaled_tensor = input_state_scaler.forward(input_state_tensor)
    encoded_state_tensor, _ = input_state_encoder.forward(input_state_scaled_tensor)
    actor_input = torch.cat([encoded_state_tensor, torch.as_tensor([[input_phase / 20.]]).float().to(device)], dim=1)
    encoded_action_tensor, _ = canonical_actor.forward(actor_input)
    decoded_action_tensor, _ = action_decoder.forward(encoded_action_tensor)
    decoded_action = decoded_action_tensor.cpu().detach().numpy().reshape(-1, )
    return decoded_action


def expert_act(input_state, input_phase, input_state_scaler, expert_actor):
    state_with_phase = np.concatenate([input_state, np.asarray([input_phase])]).reshape(1, -1)
    state_with_phase_tensor = torch.as_tensor(state_with_phase).float()
    state_with_phase_scaled_tensor = input_state_scaler.forward(state_with_phase_tensor).cpu()
    action_tensor, _ = expert_actor.forward(state_with_phase_scaled_tensor)
    action = action_tensor.cpu().detach().numpy().reshape(-1, )
    return action


def evaluate_canonical_actor(env, input_state_scaler, output_state_scaler, input_state_encoder, output_state_decoder,
                             output_state_encoder, canonical_actor, input_action_decoder):
    state = env.reset()
    state = state[:-1]
    total_reward = 0
    for i in range(1000):
        phase = env.phase

        corresponded_state = correspond(state, input_state_scaler, output_state_scaler, input_state_encoder,
                                        output_state_decoder)
        action = canonical_act(corresponded_state, phase, output_state_scaler, output_state_encoder, canonical_actor,
                               input_action_decoder)

        state, reward, done, _ = env.step(action)
        state = state[:-1]
        total_reward += reward
        if done:
            break

    return total_reward


def evaluate_actor(env, input_state_scaler, output_state_scaler, state_encoder, state_decoder, expert_state_scaler,
                   expert_actor):
    state = env.reset()
    state = state[:-1]
    total_reward = 0
    for i in range(1000):
        phase = env.phase

        corresponded_state = correspond(state, input_state_scaler, output_state_scaler, state_encoder,
                                        state_decoder)
        action = expert_act(corresponded_state, phase, expert_state_scaler, expert_actor)

        state, reward, done, _ = env.step(action)
        state = state[:-1]
        total_reward += reward
        if done:
            break

    return total_reward


def evaluated_reconstructed_actor(env, input_state_scaler, output_state_scaler, input_state_encoder,
                                  output_state_encoder, input_state_decoder, output_state_decoder,
                                  expert_state_scaler, expert_actor):
    state = env.reset()
    state = state[:-1]
    total_reward = 0
    for i in range(1000):
        phase = env.phase

        output_state = correspond(state, input_state_scaler, output_state_scaler, input_state_encoder,
                                  output_state_decoder)
        reconstructed_state = correspond(output_state, output_state_scaler, input_state_scaler, output_state_encoder,
                                         input_state_decoder)
        action = expert_act(reconstructed_state, phase, expert_state_scaler, expert_actor)

        state, reward, done, _ = env.step(action)
        state = state[:-1]
        total_reward += reward
        if done:
            break

    return total_reward
