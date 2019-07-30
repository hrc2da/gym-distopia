import json
import numpy as np
from sympy.utilities.iterables import multiset_permutations
from random import shuffle

NUM_DISTRICTS = 8
BLOCKS_PER_DISTRICT = 2


def load_data(path):
    data = []
    with open(path) as infile:
        data = np.array(json.load(infile))
    return data


def augment_assignments(assignments, index_permutations):
    # this makes a 6 GB file, I've included a zipped version of it in the data dir
    augmented_data = np.zeros(
        (
            len(assignments) * len(index_permutations),
            NUM_DISTRICTS,
            BLOCKS_PER_DISTRICT,
            2,
        )
    )
    # this could be vectorized perhaps
    i = 0
    for n, assignment in enumerate(assignments):
        print("Augmenting data point {}".format(n))
        for perm in index_permutations:
            augmented_data[i] = assignment[perm]
            i += 1
    np.save("data/augmented_trimmed_3.npy", augmented_data)


def limited_augment_assignments(assignments, index_permutations, num_augments):
    augmented_data = np.zeros(
        (
            len(assignments) * min(num_augments, len(index_permutations)),
            NUM_DISTRICTS,
            BLOCKS_PER_DISTRICT,
            2,
        )
    )
    # this could be vectorized perhaps
    i = 0
    for n, assignment in enumerate(assignments):
        print("Augmenting data point {}".format(n))
        shuffle(index_permutations)
        current_perm = index_permutations[: min(num_augments, len(index_permutations))]
        for perm in current_perm:
            augmented_data[i] = assignment[perm]
            i += 1
    np.save("data/limited_augmented_data.npy", augmented_data)


data = load_data("trimmed3.json")
n_samples, _ = data.shape
data = data.reshape(n_samples, NUM_DISTRICTS, BLOCKS_PER_DISTRICT, 2)
indices = list(range(8))
index_perms = np.array(list(multiset_permutations(indices)))
limited_augment_assignments(data, index_perms, 100)
