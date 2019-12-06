from typing import List

import numpy as np


class ExperienceTuple:

    def __init__(self):
        pass


class TupleSARS(ExperienceTuple):
    def __init__(self, state, action, log_prob, reward, next_state, done):
        super().__init__()
        self.state = state
        self.action = action
        self.log_prob = log_prob
        self.reward = reward
        self.next_state = next_state
        self.done = done


class TupleSARSS(TupleSARS):
    def __init__(self, state, subproblem_state, subproblem_action, subproblem_next_state, action, log_prob, reward,
                 next_state, done):
        super().__init__(state, action, log_prob, reward, next_state, done)
        self.subproblem_state = subproblem_state
        self.subproblem_action = subproblem_action
        self.subproblem_next_state = subproblem_next_state


class TupleSARSP(TupleSARS):
    def __init__(self, state, phase, action, log_prob, reward, next_state, done):
        super().__init__(state, action, log_prob, reward, next_state, done)
        self.phase = phase


class StateActionPair:
    def __init__(self, state, action):
        self.state = state
        self.action = action


class Dataset:

    def __init__(self, n_examples, state_dim, action_dim):
        self.n_examples = n_examples
        self.state_dim = state_dim
        self.action_dim = action_dim


class StateDataset(Dataset):

    def __init__(self, states):
        n_examples, state_dim = states.shape
        super().__init__(n_examples, state_dim, 0)
        self.states = states


class StatePhaseDataset(Dataset):

    def __init__(self, states, phases):
        n, state_dim = states.shape
        super().__init__(n, state_dim, 0)
        self.states = states
        self.phases = phases


class StatePhaseActionDataset(Dataset):
    def __init__(self, states, phases, actions, state_dim, action_dim):
        n, _ = states.shape
        super().__init__(n)
        self.phases = phases
        self.states = states
        self.actions = actions
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_examples, _ = self.states.shape

    def shuffle(self):
        idx = np.random.choice(range(self.n_examples), self.n_examples)
        self.phases = self.phases[idx]
        self.states = self.states[idx]
        self.actions = self.actions[idx]

    @staticmethod
    def from_tuple_list(tups: List[TupleSARSP]):
        # peek to get dimensions
        n = len(tups)
        tup = tups[0]
        # assume state and action are all numpy arrays of 1 dimension
        state_dim, = tup.state.shape
        action_dim, = tup.action.shape
        states = np.zeros([n, state_dim])
        actions = np.zeros([n, action_dim])
        phases = np.zeros([n, ])
        for i, tup in enumerate(tups):
            states[i] = tup.state
            actions[i] = tup.action
            phases[i] = tup.phase
        return StatePhaseActionDataset(states, phases, actions, state_dim, action_dim)


class DatasetSARS(Dataset):

    def __init__(self, states, actions, log_probs, rewards, dones, next_states):
        n_examples, state_dim = states.shape
        _, action_dim = actions.shape
        super().__init__(n_examples, state_dim, action_dim)
        self.states = states
        self.actions = actions
        self.log_probs = log_probs
        self.rewards = rewards
        self.dones = dones
        self.next_states = next_states

    @staticmethod
    def from_tuple_list(tups: List[TupleSARSP]):
        n = len(tups)
        state_dim, = tups[0].state.shape
        action_dim, = tups[0].action.shape
        states = np.zeros([n, state_dim])
        actions = np.zeros([n, action_dim])
        log_probs = np.zeros([n, ])
        rewards = np.zeros([n, ])
        dones = np.zeros([n, ])
        next_states = np.zeros([n, state_dim])

        for i, tup in enumerate(tups):
            states[i] = tup.state
            actions[i] = tup.action
            log_probs[i] = tup.log_prob
            rewards[i] = tup.reward
            dones[i] = tup.done
            next_states[i] = tup.next_state

        return DatasetSARS(states, actions, log_probs, rewards, dones, next_states)


class DatasetSARSP(DatasetSARS):

    def __init__(self, states, phases, actions, log_probs, rewards, dones, next_states):
        super().__init__(states, actions, log_probs, rewards, dones, next_states)
        self.phases = phases

    @staticmethod
    def from_tuple_list(tups: List[TupleSARSP]):
        n = len(tups)
        state_dim, = tups[0].state.shape
        action_dim, = tups[0].action.shape
        states = np.zeros([n, state_dim])
        actions = np.zeros([n, action_dim])
        phases = np.zeros([n, ])
        log_probs = np.zeros([n, ])
        rewards = np.zeros([n, ])
        dones = np.zeros([n, ])
        next_states = np.zeros([n, state_dim])

        for i, tup in enumerate(tups):
            states[i] = tup.state
            actions[i] = tup.action
            phases[i] = tup.phase
            log_probs[i] = tup.log_prob
            rewards[i] = tup.reward
            dones[i] = tup.done
            next_states[i] = tup.next_state

        return DatasetSARSP(states, phases, actions, log_probs, rewards, dones, next_states)