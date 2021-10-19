import numpy as np
import matplotlib.pyplot as plt
import torch

from utils.containers import DataDict
from utils.keys import DataKey


def main():
    samples1 = np.random.uniform((0, 0), (1, 1), (10000, 2))
    samples2 = np.random.uniform((1, 1), (2, 2), (10000, 2))
    samples3 = np.random.uniform((1, 0), (2, 1), (10000, 2))
    samples = np.concatenate([samples1, samples2, samples3], axis=0)
    samples[:, 0] += 1
    samples[:, 1] += 1

    angle = np.pi / 2
    rotation_matrix = np.asarray([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    data_dict = DataDict(30000)
    data_dict.set(DataKey.states, samples)
    torch.save(data_dict, "./data/shapes0.pkl")

    rotated_samples = samples @ rotation_matrix
    data_dict = DataDict(30000)
    data_dict.set(DataKey.states, rotated_samples)
    torch.save(data_dict, "./data/shapes1.pkl")

    plt.figure()
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.scatter(rotated_samples[:, 0], rotated_samples[:, 1])
    plt.show()


if __name__ == "__main__":
    main()
