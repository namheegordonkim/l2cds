from abc import abstractmethod

import torch
from torch import nn
from torch.distributions import Normal

from utils.keys import DataKey, ModelKey, TensorKey
from utils.loss_calculators import LossCalculatorInputTarget, LossCalculatorSum, LossCalculatorApply
from utils.rl_common import get_ppo_surrogate_tensor
from utils.tensor_inserters import TensorInserterSeq, TensorInserterTensorizeScaled, TensorInserterTensorize, \
    TensorInserterForward, TensorInserterListTransform, TensorInserterApplyModel


class PipelineAssembler:
    """
    Similar to strategy pattern, returns tensor inserters and loss calculators
    """

    @abstractmethod
    def assemble(self):
        raise NotImplementedError


class PipelineAssemblerPPO(PipelineAssembler):
    def assemble(self):
        critic_tensor_inserter = TensorInserterSeq([
            TensorInserterTensorizeScaled(DataKey.states, ModelKey.state_scaler, TensorKey.states_tensor, torch.float),
            TensorInserterTensorize(DataKey.cumulative_rewards, TensorKey.cumulative_rewards_tensor, torch.float),
            TensorInserterForward(TensorKey.states_tensor, ModelKey.critic,
                                  TensorKey.cumulative_reward_predictions_tensor),
        ])
        mse_loss = nn.MSELoss()
        critic_loss_calculator = LossCalculatorInputTarget(TensorKey.cumulative_reward_predictions_tensor,
                                                           TensorKey.cumulative_rewards_tensor,
                                                           mse_loss, 1.0)
        actor_tensor_inserter = TensorInserterSeq([
            # TensorInserterTensorize(DataKey.advantages, TensorKey.advantages_tensor, torch.float),
            TensorInserterTensorizeScaled(DataKey.states, ModelKey.state_scaler, TensorKey.states_tensor, torch.float),
            TensorInserterTensorize(DataKey.actions, TensorKey.actions_tensor, torch.float),
            TensorInserterTensorize(DataKey.cumulative_rewards, TensorKey.cumulative_rewards_tensor, torch.float),
            TensorInserterForward(TensorKey.states_tensor, ModelKey.critic,
                                  TensorKey.cumulative_reward_predictions_tensor),
            TensorInserterListTransform(
                [TensorKey.cumulative_rewards_tensor, TensorKey.cumulative_reward_predictions_tensor],
                lambda l: l[0] - l[1], TensorKey.advantages_tensor),
            TensorInserterTensorize(DataKey.log_probs, TensorKey.log_probs_tensor, torch.float),
            TensorInserterApplyModel([TensorKey.states_tensor, TensorKey.actions_tensor], ModelKey.actor,
                                     lambda model, tensors: [model.get_log_prob(tensors[0], tensors[1])],
                                     [TensorKey.new_log_probs_tensor]),
            TensorInserterListTransform(
                [TensorKey.new_log_probs_tensor, TensorKey.log_probs_tensor, TensorKey.advantages_tensor],
                lambda l: get_ppo_surrogate_tensor(l[0], l[1], l[2]),
                TensorKey.ppo_surrogates_tensor)
        ])
        actor_loss_calculator = LossCalculatorSum([
            LossCalculatorApply(TensorKey.ppo_surrogates_tensor, lambda x: -torch.mean(x), 1.,),
            # LossCalculatorApply(TensorKey.actions_tensor, lambda tensor: torch.mean(tensor ** 2), 1e-1),
        ])

        return actor_loss_calculator, actor_tensor_inserter, critic_loss_calculator, critic_tensor_inserter


class PipelineAssemblerTD3(PipelineAssembler):
    """
    actor: one-handed neural network
    critic: two-handed neural network
    """
    def assemble(self):
        critic_tensor_inserter = TensorInserterSeq([
            TensorInserterTensorizeScaled(DataKey.states, ModelKey.state_scaler, TensorKey.states_tensor, torch.float),
            TensorInserterTensorize(DataKey.actions, TensorKey.actions_tensor, torch.float),
            TensorInserterTensorize(DataKey.rewards, TensorKey.rewards_tensor, torch.float),
            TensorInserterTensorize(DataKey.dones, TensorKey.dones_tensor, torch.float),
            TensorInserterTensorizeScaled(DataKey.next_states, ModelKey.state_scaler, TensorKey.next_states_tensor,
                                          torch.float),
            TensorInserterListTransform([TensorKey.actions_tensor],
                                        lambda tensors: Normal(0, 0.2).sample(tensors[0].shape),
                                        TensorKey.noise_tensor),
            TensorInserterListTransform([TensorKey.noise_tensor],
                                        lambda tensors: torch.clamp(tensors[0], -0.5, 0.5),
                                        TensorKey.noise_tensor),
            TensorInserterApplyModel([TensorKey.next_states_tensor],
                                     ModelKey.target_actor,
                                     lambda model, tensors: [model.forward(tensors[0])[0]],
                                     [TensorKey.next_actions_tensor]),
            TensorInserterListTransform([TensorKey.next_actions_tensor, TensorKey.noise_tensor],
                                        lambda tensors: tensors[0] + tensors[1],
                                        TensorKey.next_actions_tensor),
            TensorInserterApplyModel([TensorKey.next_states_tensor, TensorKey.next_actions_tensor],
                                     ModelKey.target_critic,
                                     lambda model, tensors: model.forward(tensors[0], tensors[1]),
                                     [TensorKey.target_q1_tensor, TensorKey.target_q2_tensor]),
            TensorInserterListTransform([TensorKey.target_q1_tensor, TensorKey.target_q2_tensor],
                                        lambda tensors: torch.min(tensors[0], tensors[1]),
                                        TensorKey.target_q_tensor),
            TensorInserterListTransform([TensorKey.target_q_tensor, TensorKey.rewards_tensor, TensorKey.dones_tensor],
                                        lambda tensors: get_target_q(tensors[0], tensors[1], tensors[2], 0.99),
                                        TensorKey.target_q_tensor),
            TensorInserterApplyModel([TensorKey.states_tensor, TensorKey.actions_tensor], ModelKey.critic,
                                     lambda model, tensors: model.forward(tensors[0], tensors[1]),
                                     [TensorKey.current_q1_tensor, TensorKey.current_q2_tensor])
        ])
        mse_loss = nn.MSELoss()

        critic_loss_calculator = LossCalculatorSum([
            LossCalculatorInputTarget(TensorKey.current_q1_tensor, TensorKey.target_q_tensor, mse_loss, 1.),
            LossCalculatorInputTarget(TensorKey.current_q2_tensor, TensorKey.target_q_tensor, mse_loss, 1.)
        ])

        actor_tensor_inserter = TensorInserterSeq([
            TensorInserterTensorizeScaled(DataKey.states, ModelKey.state_scaler, TensorKey.states_tensor, torch.float),
            TensorInserterForward(TensorKey.states_tensor, ModelKey.actor, TensorKey.actions_tensor),
            TensorInserterApplyModel([TensorKey.states_tensor, TensorKey.actions_tensor], ModelKey.critic,
                                     lambda model, tensors: [model.forward(tensors[0], tensors[1])[0]],
                                     [TensorKey.current_q1_tensor])
        ])

        actor_loss_calculator = LossCalculatorSum([
            LossCalculatorApply(TensorKey.current_q1_tensor, lambda tensor: -torch.mean(tensor), 1.),
            LossCalculatorApply(TensorKey.actions_tensor, lambda tensor: torch.mean(tensor ** 2), 1e-2),
        ])

        return actor_loss_calculator, actor_tensor_inserter, critic_loss_calculator, critic_tensor_inserter


def get_target_q(target_critic_min_output_tensor, rewards_tensor, dones_tensor, discount) -> torch.Tensor:
    return rewards_tensor + ((1 - dones_tensor) * discount * target_critic_min_output_tensor).detach()
