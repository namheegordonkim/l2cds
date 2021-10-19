import numpy as np
import torch

from utils.containers import DataDict
from utils.keys import DataKey


def main():
    states0 = []
    actions0 = []
    next_states0 = []
    states1 = []
    actions1 = []
    next_states1 = []

    n_steps = 10
    n_trajectories = 1000
    starting_states0 = np.linspace((0, 0), (1, 0), n_trajectories)
    starting_states1 = np.linspace((0, 0), (0, 1), n_trajectories)
    goal_state = np.array([1., 1.])

    for starting_state0, starting_state1 in zip(starting_states0, starting_states1):
        state0 = starting_state0
        state1 = starting_state1
        vel0 = (goal_state - starting_state0) / n_steps
        vel1 = (goal_state - starting_state1) / n_steps
        for t in range(n_steps):
            next_state0 = state0 + vel0
            next_state1 = state1 + vel1

            states0.append(state0)
            next_states0.append(next_state0)
            states1.append(state1)
            next_states1.append(next_state1)
            actions0.append(np.asarray([0., 0.]))
            actions1.append(np.asarray([0., 0.]))

            state0 = next_state0
            state1 = next_state1

    n_examples = n_steps * n_trajectories
    random_idx0 = np.random.choice(range(n_examples), n_examples, replace=False)
    states0 = np.stack(states0)[random_idx0]
    next_states0 = np.stack(next_states0)[random_idx0]
    actions0 = np.stack(actions0)[random_idx0]

    random_idx1 = np.random.choice(range(n_examples), n_examples, replace=False)
    states1 = np.stack(states1)[random_idx1]
    next_states1 = np.stack(next_states1)[random_idx1]
    actions1 = np.stack(actions1)[random_idx1]

    dataset0 = DataDict(n_examples)
    dataset0.set(DataKey.states, states0)
    dataset0.set(DataKey.next_states, next_states0)
    dataset0.set(DataKey.actions, actions0)

    dataset1 = DataDict(n_examples)
    dataset1.set(DataKey.states, states1)
    dataset1.set(DataKey.next_states, next_states1)
    dataset1.set(DataKey.actions, actions1)

    torch.save(dataset0, "./data/wedges0.pkl")
    torch.save(dataset1, "./data/wedges1.pkl")


if __name__ == "__main__":
    main()
