from enum import Enum


class ModelKey(Enum):
    state_encoder = 0
    state_decoder = 1
    latent_actor = 2
    encoded_velocity_predictor = 3
    action_decoder = 4
    state_scaler = 5
    actor = 6
    critic = 7
    advantage_scaler = 8
    target_critic = 9
    target_actor = 10
    expert_state_scaler = 11
    position_encoder = 12
    position_decoder = 13
    velocity_encoder = 14
    velocity_decoder = 15
    encoded_acceleration_predictor = 16


class TensorKey(Enum):
    null = -1
    states_tensor = 0
    encoded_states_tensor = 1
    encoded_actions_tensor = 2
    decoded_states_tensor = 3
    encoded_next_states_tensor = 4
    decoded_next_states_tensor = 5
    actions_tensor = 6
    phases_tensor = 7
    next_states_tensor = 8
    rewards_tensor = 9
    log_probs_tensor = 10
    dones_tensor = 11
    encoded_next_state_predictions_tensor = 12
    encoded_velocity_predictions_tensor = 13
    origins_tensor = 14
    decoded_actions_tensor = 15
    cumulative_reward_predictions_tensor = 16
    cumulative_rewards_tensor = 17
    advantages_tensor = 18
    new_log_probs_tensor = 19
    ppo_surrogates_tensor = 20
    actor_log_std_tensor = 21
    next_actions_tensor = 22
    target_q_tensor = 23
    current_q1_tensor = 24
    current_q2_tensor = 25
    target_q1_tensor = 26
    target_q2_tensor = 27
    noise_tensor = 28
    tmp_tensor = 29
    target_actions_tensor = 30
    positions_tensor = 31
    velocities_tensor = 32
    encoded_positions_tensor = 33
    encoded_velocities_tensor = 34
    encoded_accelerations_tensor = 35
    decoded_positions_tensor = 36
    decoded_velocities_tensor = 37
    encoded_acceleration_predictions_tensor = 38
    encoded_next_velocities_tensor = 39
    encoded_next_positions_tensor = 40
    next_positions_tensor = 41
    next_velocities_tensor = 42
    encoded_next_position_predictions_tensor = 43
    encoded_next_velocity_predictions_tensor = 44


class DataKey(Enum):
    states = 0
    actions = 1
    rewards = 2
    log_probs = 3
    next_states = 4
    dones = 5
    phases = 6
    cumulative_rewards = 7
    advantages = 8
    encoded_states = 9


class TensorInserterKey(Enum):
    actor_tensor_inserter = 0
    critic_tensor_inserter = 1
    entropy_tensor_inserter = 2


class LossCalculatorKey(Enum):
    actor_loss_calculator = 0
    critic_loss_calculator = 1
    entropy_loss_calculator = 2
