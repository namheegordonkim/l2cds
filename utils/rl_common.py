from abc import abstractmethod
from typing import List

import gym
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from tqdm import tqdm

from utils.containers import ModelDict, EnvsContainer, DataDict
from utils.loss_calculators import LossCalculator2
from utils.sample_collectors import SampleCollector
from utils.tensor_collectors import TensorCollector
from utils.utils import device, compute_total_reward_from_start

plt.ioff()


class ModelUpdater:

    @abstractmethod
    def update(self, loss):
        raise NotImplementedError


class ModelUpdaterSeq:

    def __init__(self, model_updaters: List[ModelUpdater]):
        self.model_updaters = model_updaters

    def update(self, loss):
        for model_updater in self.model_updaters:
            model_updater.update(loss)


class ModelUpdaterOptimizer(ModelUpdater):

    def __init__(self, optimizers):
        self.optimizers = optimizers

    def update(self, loss):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in self.optimizers:
            optimizer.step()


class ModelUpdaterTargetUpdate(ModelUpdater):

    def __init__(self, source_model: nn.Module, target_model: nn.Module, tau=0.005):
        self.tau = tau
        self.source_model = source_model
        self.target_model = target_model

    def update(self, loss):
        for target_param, param in zip(self.target_model.parameters(),
                                       self.source_model.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)


class RewardGetter:

    @abstractmethod
    def get_cumulative_reward(self):
        raise NotImplementedError


class PostProcessor:

    @abstractmethod
    def process(self):
        raise NotImplementedError


def initialize_scaler(env: gym.Env, actor_model: nn.Module, num_iter=10) -> StandardScaler:
    state = env.reset_for_normalization()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    scaler = StandardScaler()
    states = np.zeros([num_iter, state_dim], dtype=np.float32)
    print("Collect states for scaler")
    for i in tqdm(range(num_iter)):
        states[i, :] = state
        state_tensor = torch.as_tensor(state).reshape(1, -1).float()
        mu, log_std = actor_model(state_tensor)
        eps = torch.randn(mu.size())
        action = mu + log_std.exp() * eps
        env_action = action.cpu().data.numpy().reshape(action_dim, )
        state, reward, done, _ = env.step(env_action)
        if done:
            state = env.reset_for_normalization()
    print("Fit scaler")
    scaler.fit(X=states, y=None)

    return scaler


class RLLearner:

    def __init__(self, sample_collector: SampleCollector, model_dict: ModelDict,
                 envs_container: EnvsContainer, reward_getter: RewardGetter, n_episodes, n_epochs, save_every,
                 batch_size, output_prefix):
        self.output_prefix = output_prefix
        self.n_episodes = n_episodes
        self.n_epochs = n_epochs
        self.save_every = save_every
        self.batch_size = batch_size
        self.reward_getter = reward_getter
        self.sample_collector = sample_collector
        self.model_dict = model_dict
        self.envs_container = envs_container
        self.t = 0

    def run(self):
        reward_means = []
        reward_stds = []
        for episode in tqdm(range(self.n_episodes)):

            self.train_one_episode()

            if episode % self.save_every == 0:
                # save model
                model_save_path = "{:s}_model_dict_{:07d}.pkl".format(self.output_prefix, episode)
                torch.save(self.model_dict, model_save_path)
                reward_mean, reward_std = self.reward_getter.get_cumulative_reward()
                reward_means.append(reward_mean)
                reward_stds.append(reward_std)
                print("Saved models to {}".format(model_save_path))

                # save loss/reward figure
                losses_fig_save_path = "{:s}_losses_{:07d}.png".format(self.output_prefix, episode)
                plt.figure()
                plt.title("Episode {}".format(episode))
                plt.xlabel("Episodes")
                plt.ylabel("Average Reward")
                plt.plot(reward_means)
                mean_arr = np.asarray(reward_means)
                std_arr = np.asarray(reward_stds)
                plt.fill_between(np.arange(0, self.t + 1), mean_arr - std_arr, mean_arr + std_arr, alpha=0.5)
                plt.savefig(losses_fig_save_path)
                plt.close()
                print("Saved figure to {}".format(losses_fig_save_path))

                # save animation
                # gif_fig_save_path = "{:s}_movie_{:07d}.gif".format(self.output_prefix, episode)
                # frames = self.frame_getter.get_frames()
                # imageio.mimwrite(gif_fig_save_path, frames, 'GIF-PIL', fps=10, quantizer=0)
                # print("Saved GIF animation to {}".format(gif_fig_save_path))

                print("Episode {:d}\tReward:{:f}".format(episode, reward_mean))
                self.t += 1

    @abstractmethod
    def train_one_episode(self):
        raise NotImplementedError


class SimpleRLLearner(RLLearner):

    def __init__(self, sample_collector: SampleCollector, model_dict: ModelDict, envs_container: EnvsContainer,
                 reward_getter: RewardGetter, tensor_collector: TensorCollector, loss_calculator: LossCalculator2,
                 optimizer, n_episodes, n_epochs, save_every, batch_size, output_prefix):
        super().__init__(sample_collector, model_dict, envs_container, reward_getter, n_episodes, n_epochs, save_every,
                         batch_size, output_prefix)
        self.tensor_collector = tensor_collector
        self.loss_calculator = loss_calculator
        self.optimizer = optimizer

    def train_one_episode(self):
        # for learning, use GPU
        for model in self.model_dict.as_list():
            model.to(device)

        data_dict = self.sample_collector.collect_data_dict(self.model_dict, self.envs_container)

        for epoch in tqdm(range(self.n_epochs)):
            self.train_one_epoch(data_dict, self.batch_size)

    def train_one_epoch(self, data_dict: DataDict, batch_size=64):
        n_examples = data_dict.n_examples
        all_idx = np.random.choice(range(n_examples), n_examples, replace=False)
        n_batches = int(n_examples / batch_size)
        batch_idxs = np.array_split(all_idx, n_batches)
        for batch_idx in batch_idxs:
            tensor_dict = self.tensor_collector.get_tensor_dict([data_dict], [self.model_dict], batch_idx)
            loss = self.loss_calculator.get_loss(tensor_dict)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


class SimpleSeqRLLearner(RLLearner):
    """
    RL steps involving sequential tensor collection, e.g. SAC
    """

    def __init__(self, sample_collector: SampleCollector, model_dict: ModelDict,
                 envs_container: EnvsContainer, reward_getter: RewardGetter,
                 tensor_collectors: List[TensorCollector],
                 loss_calculators: List[LossCalculator2], model_updaters: List[ModelUpdater],
                 post_processor: PostProcessor,
                 n_episodes, n_epochs_list, update_frequencies_list, save_every, batch_size, output_prefix):
        super().__init__(sample_collector, model_dict, envs_container, reward_getter, n_episodes, None, save_every,
                         batch_size, output_prefix)
        self.model_updaters = model_updaters
        self.update_frequencies_list = update_frequencies_list
        self.n_epochs_list = n_epochs_list
        self.reward_getter = reward_getter
        self.tensor_collectors = tensor_collectors
        self.loss_calculators = loss_calculators
        self.post_processor = post_processor
        self.episode = 0

    def train_one_episode(self):
        # for learning, use GPU
        for model in self.model_dict.as_list():
            model.to(device)

        data_dict = self.sample_collector.collect_data_dict(self.model_dict, self.envs_container)

        for tensor_collector, loss_calculator, model_updater, n_epochs, update_frequency \
                in zip(self.tensor_collectors, self.loss_calculators, self.model_updaters, self.n_epochs_list,
                       self.update_frequencies_list):

            if self.episode % update_frequency == 0:
                for epoch in tqdm(range(n_epochs)):
                    self.train_one_epoch(data_dict, tensor_collector, loss_calculator, model_updater, self.batch_size)

        self.post_processor.process()
        self.episode += 1

    def train_one_epoch(self, data_dict: DataDict, tensor_collector: TensorCollector, loss_calculator: LossCalculator2,
                        model_updater: ModelUpdater, batch_size):
        n_examples = data_dict.n_examples
        all_idx = np.random.choice(range(n_examples), n_examples, replace=False)
        n_batches = int(n_examples / batch_size)
        n_batches = max(n_batches, 1)
        batch_idxs = np.array_split(all_idx, n_batches)
        for batch_idx in batch_idxs:
            tensor_dicts = tensor_collector.get_tensor_dicts([data_dict], [self.model_dict], batch_idx)
            loss = loss_calculator.get_loss(tensor_dicts)
            model_updater.update(loss)


class ActionGetter:
    """
    Deal with action per timestep
    """
    @abstractmethod
    def get_action(self, state):
        raise NotImplementedError


class ActionGetterFromState(ActionGetter):
    def __init__(self, state_scaler, actor):
        self.state_scaler = state_scaler
        self.actor = actor

    def get_action(self, state):
        state_tensor = torch.as_tensor(state).reshape(1, -1).float().cpu()
        state_scaled_tensor = self.state_scaler.forward(state_tensor).cpu()
        action_tensor, _ = self.actor.forward(state_scaled_tensor)
        action = action_tensor.cpu().detach().numpy().reshape(-1, )
        return action


class ActionGetterFromStatePhase(ActionGetter):
    def __init__(self, state_scaler, actor):
        self.state_scaler = state_scaler
        self.actor = actor

    def get_action(self, state):
        phase_tensor = torch.as_tensor([state[-1] / 20]).reshape(1, -1).float().to(device)
        state = state[:-1]
        state_tensor = torch.as_tensor(state).reshape(1, -1).float().to(device)
        state_scaled_tensor = self.state_scaler.forward(state_tensor)
        actor_input_tensor = torch.cat([state_scaled_tensor, phase_tensor], dim=1)
        action_tensor, _ = self.actor.forward(actor_input_tensor)
        action = action_tensor.cpu().detach().numpy().reshape(-1, )
        return action


class ActionGetterFromEncodedState(ActionGetter):

    def __init__(self, state_scaler, state_encoder, actor):
        self.state_scaler = state_scaler
        self.state_encoder = state_encoder
        self.actor = actor

    def get_action(self, state):
        phase_tensor = torch.as_tensor([state[-1] / 20.]).reshape(1, -1).float().to(device)
        state = state[:-1]
        state_tensor = torch.as_tensor(state).reshape(1, -1).float().to(device)
        state_scaled_tensor = self.state_scaler.forward(state_tensor)
        encoded_state_tensor, _ = self.state_encoder.forward(state_scaled_tensor)
        encoded_state_phase_tensor = torch.cat([encoded_state_tensor, phase_tensor], dim=1)
        action_tensor, _ = self.actor.forward(encoded_state_phase_tensor)
        action = action_tensor.cpu().detach().numpy().reshape(-1, )
        return action


class RewardGetterSimple(RewardGetter):

    def __init__(self, env, action_getter: ActionGetter, horizon: int):
        self.action_getter = action_getter
        self.env = env
        self.horizon = horizon

    def get_cumulative_reward(self):
        total_reward = compute_total_reward_from_start(self.env, self.action_getter, self.horizon)
        return total_reward


class RewardGetterSequential(RewardGetterSimple):

    def __init__(self, env, action_getter: ActionGetter, horizon: int, n_resets: int):
        super().__init__(env, action_getter, horizon)
        self.n_resets = n_resets

    def get_cumulative_reward(self):
        rewards = []
        for _ in tqdm(range(self.n_resets)):
            total_reward = super().get_cumulative_reward()
            rewards.append(total_reward)
        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards)
        return reward_mean, reward_std


class PostProcessorDummy(PostProcessor):

    def process(self):
        pass


class PostProcessorSeq(PostProcessor):

    def __init__(self, post_processors: List[PostProcessor]):
        self.post_processors = post_processors

    def process(self):
        for post_processor in self.post_processors:
            post_processor.process()


class PostProcessorLinearAnnealLogStd(PostProcessor):

    def __init__(self, model: nn.Module, anneal_rate: float, min_logstd: float):
        self.model = model
        self.anneal_rate = anneal_rate
        self.min_logstd = min_logstd

    def process(self):
        self.model.log_std -= self.anneal_rate
        min_logstd = torch.ones_like(self.model.log_std) * self.min_logstd
        self.model.log_std = torch.max(self.model.log_std, min_logstd)


class PostProcessorHardCopyModelWeights(PostProcessor):

    def __init__(self, target_model: nn.Module, source_model: nn.Module, update_every: int, tau=0.005):
        self.target_model = target_model
        self.source_model = source_model
        self.tau = tau
        self.update_every = update_every
        self.t = 0

    def process(self):
        if self.t % self.update_every == 0:
            for target_param, param in zip(self.target_model.parameters(),
                                           self.source_model.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        self.t += 1


def get_ppo_surrogate_tensor(new_log_probs_tensor, old_log_probs_tensor, advantages_tensor):
    ratio = torch.exp(new_log_probs_tensor - old_log_probs_tensor)
    surr1 = ratio * advantages_tensor
    surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages_tensor
    return torch.min(surr1, surr2)
