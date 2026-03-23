#!/usr/bin/env python3

import numpy as np

# binary vector for testing, lenght 36
testing_binary_vector = [
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    1,
    0,
    1,
    0,
    1,
    1,
    0,
    1,
    0,
    1,
    0,
    0,
    0,
    0,
    1,
]
# real vector for testing, lenght 36
testing_real_vector = np.array(36 * [1 / 36])


def mutate_binary(vector, mutation_rate=0.1):
    """
    Mutate a binary vector by flipping bits with a given mutation rate.

    Args:
        vector (list): The binary vector to mutate.
        mutation_rate (float): The probability of mutating each bit.

    Returns:
        list: The mutated binary vector.
    """

    mutated_vector = vector.copy()
    mutation_mask = np.random.rand(len(mutated_vector)) < mutation_rate
    mutated_vector = np.array(mutated_vector)  # Convert to numpy array for vectorized operations
    mutated_vector[mutation_mask] = 1 - mutated_vector[mutation_mask]  # Flip the bits where mask is True
    return mutated_vector


def mutate_real(vector, mutation_rate=0.1, eta_m=5):
    """
    Mutate a real-valued vector by using polynomial mutation. Use range constraints to make sure the
    weights are in the range [0, 1]. The mutation is done by adding a small perturbation to the vector.

    Args:
        vector (list): The real-valued vector to mutate.
        mutation_rate (float): The probability of mutating each element.
        eta_m (int): Distribution index for the mutation operator.

    Returns:
        list: The mutated real-valued vector.
    """

    mutated_vector = vector.copy()
    mutation_mask = np.random.rand(len(mutated_vector)) < mutation_rate
    mutation_range = 1 - 0  # Upperbound x_U (1) - Lowerbound x_L (0)

    # perturbation
    delta_q = np.empty_like(mutated_vector)
    delta_q_mask = np.random.rand(np.sum(mutation_mask).astype(int))
    delta_q[mutation_mask] = np.where(
        # if r_i < 0.5, do the first operation, else do the second
        delta_q_mask < 0.5,
        # 2 * r_i ^ (1 / (eta_m + 1)) - 1
        (2 * delta_q_mask ** (1 / (eta_m + 1))) - 1,
        # 1 - (2 * (1 - r_i)) ^ (1 / (eta_m + 1)))
        1 - (2 * (1 - delta_q_mask)) ** (1 / (eta_m + 1)),
    )

    # Apply the mutation
    mutated_vector[mutation_mask] += delta_q[mutation_mask] * mutation_range

    # Just to make sure, the weights need to be in the range [0, 1]
    mutated_vector = np.clip(mutated_vector, 0, 1)

    # Repair step, make sure the vector is not all zeroes
    normalizer = np.sum(mutated_vector)
    if normalizer == 0:
        # if the vector is all zeroes, set it to average (1 / len(vector))
        mutated_vector = np.ones_like(mutated_vector) / len(mutated_vector)
    # if not, simply renormalize, so that the weights sum to 1
    else:
        mutated_vector = mutated_vector / normalizer

    return mutated_vector


def swap(vector, mutation_rate=0.05):
    """
    Swap pairs of bits in a binary vector at random positions with a given swap rate.

    Args:
        vector (list): The binary vector to swap bits in.
        swap_rate (float): The probability of swapping each pair of bits.

    Returns:
        list: The binary vector with swapped bits.
    """

    mutated_vector = vector.copy()
    n_swaps = np.random.binomial(len(mutated_vector) // 2, mutation_rate)
    swap_indices = np.random.choice(len(mutated_vector), size=(n_swaps, 2), replace=False)
    # print("Swap indices:", swap_indices)
    for i, j in swap_indices:
        mutated_vector[i], mutated_vector[j] = mutated_vector[j], mutated_vector[i]
    return mutated_vector


def crossover_onepoint(parent1, parent2):
    """
    Perform one-point crossover between two binary vectors.

    Args:
        parent1 (list): The first parent binary vector.
        parent2 (list): The second parent binary vector.

    Returns:
        tuple: Two offspring binary vectors resulting from the crossover.
    """

    # Ensure both parents are of the same length
    if len(parent1) != len(parent2):
        raise ValueError("Parents must be of the same length for crossover.")

    # Randomly select a crossover point
    crossover_point = np.random.randint(1, len(parent1) - 1)
    offspring1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    offspring2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

    return offspring1, offspring2


def crossover_uniform(parent1, parent2):
    """
    Perform uniform crossover between two binary vectors.

    Args:
        parent1 (list): The first parent binary vector.
        parent2 (list): The second parent binary vector.

    Returns:
        tuple: Two offspring binary vectors resulting from the crossover.
    """

    # Ensure both parents are of the same length
    if len(parent1) != len(parent2):
        raise ValueError("Parents must be of the same length for crossover.")

    # Randomly select bits from each parent
    mask = np.random.rand(len(parent1)) < 0.5
    offspring1 = np.where(mask, parent1, parent2)
    offspring2 = np.where(mask, parent2, parent1)

    return offspring1, offspring2
