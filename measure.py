#!/usr/bin/env python3

from functools import partial
from itertools import product
import json
import pickle
import sys

import numpy as np
from tqdm import tqdm
from ea import mutate_binary, mutate_real, crossover_uniform, crossover_onepoint, swap
from nsga import NSGA_binary, NSGA_real
import time

import logging

# logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")


# Evolutionary operators values
# Maybe try for combining multiple mutations or crossovers
mutation_rates_binary = [0.01, 1 / 36, 0.1]
mutate_functions_binary = [mutate_binary]
mutation_rates_real = [0.01, 1 / 36, 0.1]
crossover_rates = [0.5, 0.7, 0.9]
crossover_functions = [crossover_uniform]

# NSGA values
population_sizes = [50, 100, 200, 500]

# Real NSGA options
eta_ms = [5, 15, 25]
# disable_thresholds = [0.0001, 0.001, 0.01, 1 / 36, 0.05, 0.1]
disable_thresholds = [0.001]

# other
runs = 10
budget = 25000 # Total number of evaluations, generations will be calculated accordingly as budget / population_size

# Frozen mutation and crossover rates (and eta_m for real variant) for budget experiments
def run_nsga_binary_budget():
    for population_size, mutation_rate, mutate_function, crossover_rate, crossover_function in product(
        population_sizes, [1/36], mutate_functions_binary, [0.9], crossover_functions
    ):
        generations = budget // population_size
        logging.info(
            f"Running NSGA_binary with population_size={population_size}, mutation_rate={mutation_rate}, "
            f"{mutate_function.__name__}, {crossover_function.__name__}, crossover_rate={crossover_rate}"
            f" for {generations} generations."
        )
        stats = {
            "population_size": population_size,
            "mutation_rate": mutation_rate,
            "mutate_function": mutate_function.__name__,
            "crossover_function": crossover_function.__name__,
            "crossover_rate": crossover_rate,
            # "times": [],
            # "lowest_eer_fitnesses": [],
            # "lowest_params_fitnesses": [],
            # "hypervolumes": [],
        }
        # gens = {
        #     50: stats,
        #     100: stats,
        #     150: stats,
        #     200: stats,
        #     250: stats,
        #     300: stats,
        #     350: stats,
        #     400: stats,
        #     450: stats,
        #     500: stats,
        # }
        measurements = {}
        for run in range(runs):
            nsga = NSGA_binary(population_size)
            nsga._mutate_function = mutate_function
            nsga._crossover_function = crossover_function
            nsga._mutation_rate = mutation_rate
            nsga._crossover_rate = crossover_rate
            measurements[run] = stats
            # Run and measure each 50 generations
            # total_time = 0
            for i in tqdm(range(1, generations + 1)):  # Run the same instance, save results for each generation
                # t_start = time.time()
                nsga.run(1)
                # t_end = time.time()
                # t = t_end - t_start

                best_ndf = nsga.fronts[0]
                lowest_eer_fitness = best_ndf.fitness[0]
                lowest_params_fitness = best_ndf.fitness[-1]

                measurements[run][i] = {
                    "lowest_eer_fitness": lowest_eer_fitness.tolist(),
                    "lowest_params_fitness": lowest_params_fitness.tolist(),
                    "hypervolume": best_ndf.hypervolume(),
                }
                # gens[i * 50]["times"].append(t + total_time)
                # total_time += t
                # gens[i * 50]["lowest_eer_fitnesses"].append(lowest_eer_fitness.tolist())
                # gens[i * 50]["lowest_params_fitnesses"].append(lowest_params_fitness.tolist())
                # gens[i * 50]["hypervolumes"].append(best_ndf.hypervolume())

        # Save the results to a JSON file
        with open(
            f"results/budget_binary_{population_size}.json",
            "w",
        ) as f:
            json.dump(measurements, f, indent=4)


# Frozen mutation and crossover rates (and eta_m for real variant) for budget experiments
def run_nsga_real_budget():
    for population_size, mutation_rate, crossover_rate, crossover_function, eta_m, disable_threshold in product(
        population_sizes, [1/36], [0.9], crossover_functions, [5], [0.001]
    ):
        generations = budget // population_size
        logging.info(
            f"Running NSGA_real with population_size={population_size}, mutation_rate={mutation_rate}, "
            f"{crossover_function.__name__}, crossover_rate={crossover_rate}, "
            f"eta_m={eta_m}, disable_threshold={disable_threshold} for {generations} generations."
        )
        stats = {
            "population_size": population_size,
            "mutation_rate": mutation_rate,
            "crossover_function": crossover_function.__name__,
            "crossover_rate": crossover_rate,
            "eta_m": eta_m,
            "disable_threshold": disable_threshold,
            # "times": [],
            # "lowest_eer_fitnesses": [],
            # "lowest_params_fitnesses": [],
            # "hypervolumes": [],
        }

        measurements = {}
        for run in range(runs):
            nsga = NSGA_real(population_size)
            nsga._mutate_function = partial(mutate_real, eta_m=eta_m)
            nsga._crossover_function = crossover_function
            nsga._mutation_rate = mutation_rate
            nsga._crossover_rate = crossover_rate
            nsga.threshold = disable_threshold
            measurements[run] = stats
            # Run and measure each 50 generations
            # total_time = 0
            for i in tqdm(range(1, generations + 1)):  # Run the same instance, save results for each generation
                # t_start = time.time()
                nsga.run(1)
                # t_end = time.time()
                # t = t_end - t_start

                best_ndf = nsga.fronts[0]
                lowest_eer_fitness = best_ndf.fitness[0]
                lowest_params_fitness = best_ndf.fitness[-1]

                measurements[run][i] = {
                    "lowest_eer_fitness": lowest_eer_fitness.tolist(),
                    "lowest_params_fitness": lowest_params_fitness.tolist(),
                    "hypervolume": best_ndf.hypervolume(),
                }
                # gens[i * 50]["times"].append(t + total_time)
                # total_time += t
                # gens[i * 50]["lowest_eer_fitnesses"].append(lowest_eer_fitness.tolist())
                # gens[i * 50]["lowest_params_fitnesses"].append(lowest_params_fitness.tolist())
                # gens[i * 50]["hypervolumes"].append(best_ndf.hypervolume())

        # Save the results to a JSON file
        with open(
            f"results/budget_real_{population_size}.json",
            "w",
        ) as f:
            json.dump(measurements, f, indent=4)


# Frozen population and generation (budget) for parameter experiments
def run_nsga_binary_parameters():
    for population_size, mutation_rate, mutate_function, crossover_rate, crossover_function in product(
        [100], mutation_rates_binary, mutate_functions_binary, crossover_rates, crossover_functions
    ):
        generations = budget // population_size
        logging.info(
            f"Running NSGA_binary with population_size={population_size}, mutation_rate={mutation_rate}, "
            f"{mutate_function.__name__}, {crossover_function.__name__}, crossover_rate={crossover_rate}"
            f" for {generations} generations."
        )
        stats = {
            "population_size": population_size,
            "mutation_rate": mutation_rate,
            "mutate_function": mutate_function.__name__,
            "crossover_function": crossover_function.__name__,
            "crossover_rate": crossover_rate,
        }
        measurements = {}
        for run in range(runs):
            nsga = NSGA_binary(population_size)
            nsga._mutate_function = mutate_function
            nsga._crossover_function = crossover_function
            nsga._mutation_rate = mutation_rate
            nsga._crossover_rate = crossover_rate
            measurements[run] = stats
            # Run and measure each generation
            # total_time = 0
            for generation in tqdm(range(1, generations + 1)):  # Run the same instance, save results for each generation
                # t_start = time.time()
                nsga.run(1)
                # t_end = time.time()
                # t = t_end - t_start

                best_ndf = nsga.fronts[0]
                lowest_eer_fitness = best_ndf.fitness[0]
                lowest_params_fitness = best_ndf.fitness[-1]

                measurements[run][generation] = {
                    "lowest_eer_fitness": lowest_eer_fitness.tolist(),
                    "lowest_params_fitness": lowest_params_fitness.tolist(),
                    "hypervolume": best_ndf.hypervolume(),
                }
                # gens[i * 50]["times"].append(t + total_time)
                # total_time += t
                # gens[i * 50]["lowest_eer_fitnesses"].append(lowest_eer_fitness.tolist())
                # gens[i * 50]["lowest_params_fitnesses"].append(lowest_params_fitness.tolist())
                # gens[i * 50]["hypervolumes"].append(best_ndf.hypervolume())

        # Save the results to a JSON file
        with open(
            f"results/parameters_binary_mutation{mutation_rate}_crossover{crossover_rate}.json",
            "w",
        ) as f:
            json.dump(measurements, f, indent=4)


# Frozen population and generation (budget) for parameter experiments
def run_nsga_real_parameters():
    for population_size, mutation_rate, mutate_function, crossover_rate, crossover_function, eta_m, disable_threshold in product(
        [100], mutation_rates_binary, mutate_functions_binary, crossover_rates, crossover_functions, eta_ms, disable_thresholds
    ):
        generations = budget // population_size
        logging.info(
            f"Running NSGA_real with population_size={population_size}, mutation_rate={mutation_rate}, "
            f"{mutate_function.__name__}, {crossover_function.__name__}, crossover_rate={crossover_rate}, "
            f"eta_m={eta_m}, disable_threshold={disable_threshold} "
            f"for {generations} generations."
        )
        stats = {
            "population_size": population_size,
            "mutation_rate": mutation_rate,
            "mutate_function": mutate_function.__name__,
            "crossover_function": crossover_function.__name__,
            "crossover_rate": crossover_rate,
            "eta_m": eta_m,
            "disable_threshold": disable_threshold,
        }
        measurements = {}
        for run in range(runs):
            nsga = NSGA_real(population_size)
            nsga._mutate_function = partial(mutate_real, eta_m=eta_m)
            nsga._crossover_function = crossover_function
            nsga._mutation_rate = mutation_rate
            nsga._crossover_rate = crossover_rate
            nsga.threshold = disable_threshold
            measurements[run] = stats
            # Run and measure each generation
            # total_time = 0
            for generation in tqdm(range(1, generations + 1)):  # Run the same instance, save results for each generation
                # t_start = time.time()
                nsga.run(1)
                # t_end = time.time()
                # t = t_end - t_start

                best_ndf = nsga.fronts[0]
                lowest_eer_fitness = best_ndf.fitness[0]
                lowest_params_fitness = best_ndf.fitness[-1]

                measurements[run][generation] = {
                    "lowest_eer_fitness": lowest_eer_fitness.tolist(),
                    "lowest_params_fitness": lowest_params_fitness.tolist(),
                    "hypervolume": best_ndf.hypervolume(),
                }
                # gens[i * 50]["times"].append(t + total_time)
                # total_time += t
                # gens[i * 50]["lowest_eer_fitnesses"].append(lowest_eer_fitness.tolist())
                # gens[i * 50]["lowest_params_fitnesses"].append(lowest_params_fitness.tolist())
                # gens[i * 50]["hypervolumes"].append(best_ndf.hypervolume())

        # Save the results to a JSON file
        with open(
            f"results/parameters_real_mutation{mutation_rate}_crossover{crossover_rate}_eta{eta_m}_threshold{disable_threshold}.json",
            "w",
        ) as f:
            json.dump(measurements, f, indent=4)


# Frozen population as well as parameters for validation and convergence experiments
def run_nsga_binary_convergence():
    population_size = 100
    generations = 1000

    mutation_rate = 1/36
    crossover_rate = 0.7

    logging.info(
        f"Running NSGA_binary convergence with mutation_rate={mutation_rate}, crossover_rate={crossover_rate}, "
        f"population_size={population_size} for {generations} generations."
    )

    stats = {
        "population_size": population_size,
        "mutation_rate": mutation_rate,
        "mutate_function": mutate_binary.__name__,
        "crossover_function": crossover_uniform.__name__,
        "crossover_rate": crossover_rate,
    }

    measurements = {}
    for run in range(runs):
        nsga = NSGA_binary(population_size)
        nsga._mutate_function = mutate_binary
        nsga._crossover_function = crossover_uniform
        nsga._mutation_rate = mutation_rate
        nsga._crossover_rate = crossover_rate
        measurements[run] = stats
        # Run and measure each generation
        for i in tqdm(range(1, generations + 1)):  # Run the same instance, save results for each generation
            t_start = time.perf_counter()
            nsga.run(1)
            t_end = time.perf_counter()
            t = t_end - t_start

            best_ndf = nsga.fronts[0]
            lowest_eer_fitness = best_ndf.fitness[0]
            lowest_params_fitness = best_ndf.fitness[-1]

            measurements[run][i] = {
                "lowest_eer_fitness": lowest_eer_fitness.tolist(),
                "lowest_params_fitness": lowest_params_fitness.tolist(),
                "hypervolume": best_ndf.hypervolume(),
                "generation_time": t,
            }
        
        # Save the final state of NSGA
        pickle.dump(nsga, open(f"results/nsga_binary_convergence_run{run}.pkl", "wb"))

    # Save the results to a JSON file
    with open(
        f"results/convergence_binary.json",
        "w",
    ) as f:
        json.dump(measurements, f, indent=4)


# Frozen population as well as parameters for validation and convergence experiments
def run_nsga_real_convergence():
    population_size = 100
    generations = 1000

    mutation_rate = 0.01
    crossover_rate = 0.5
    eta_m = 15
    disable_threshold = 0.001

    logging.info(
        f"Running NSGA_real convergence with mutation_rate={mutation_rate}, crossover_rate={crossover_rate}, "
        f"eta_m={eta_m}, disable_threshold={disable_threshold}, population_size={population_size} for {generations} generations."
    )

    stats = {
        "population_size": population_size,
        "mutation_rate": mutation_rate,
        "mutate_function": mutate_real.__name__,
        "crossover_function": crossover_uniform.__name__,
        "crossover_rate": crossover_rate,
        "eta_m": eta_m,
        "disable_threshold": disable_threshold,
    }

    measurements = {}
    for run in range(runs):
        nsga = NSGA_real(population_size)
        nsga._mutate_function = partial(mutate_real, eta_m=eta_m)
        nsga._crossover_function = crossover_uniform
        nsga._mutation_rate = mutation_rate
        nsga._crossover_rate = crossover_rate
        nsga.threshold = disable_threshold
        measurements[run] = stats

        # Run and measure each generation
        for i in tqdm(range(1, generations + 1)):  # Run the same instance, save results for each generation
            t_start = time.perf_counter()
            nsga.run(1)
            t_end = time.perf_counter()
            t = t_end - t_start

            best_ndf = nsga.fronts[0]
            lowest_eer_fitness = best_ndf.fitness[0]
            lowest_params_fitness = best_ndf.fitness[-1]

            measurements[run][i] = {
                "lowest_eer_fitness": lowest_eer_fitness.tolist(),
                "lowest_params_fitness": lowest_params_fitness.tolist(),
                "hypervolume": best_ndf.hypervolume(),
                "generation_time": t,
            }
        
        # Save the final state of NSGA
        pickle.dump(nsga, open(f"results/nsga_real_convergence_run{run}.pkl", "wb"))

    # Save the results to a JSON file
    with open(
        f"results/convergence_real.json",
        "w",
    ) as f:
        json.dump(measurements, f, indent=4)


if __name__ == "__main__":
    # Read mutation rate from the first argument
    if len(sys.argv) < 3:
        print(f"Usage: python measure.py <real|binary> <budget|parameters|convergence>")
        sys.exit(1)

    mode = sys.argv[1]
    experiment_type = sys.argv[2]

    if mode == "binary" and experiment_type == "budget":
        run_nsga_binary_budget()
    elif mode == "real" and experiment_type == "budget":
        run_nsga_real_budget()
    elif mode == "binary" and experiment_type == "parameters":
        run_nsga_binary_parameters()
    elif mode == "real" and experiment_type == "parameters":
        run_nsga_real_parameters()
    elif mode == "binary" and experiment_type == "convergence":
        run_nsga_binary_convergence()
    elif mode == "real" and experiment_type == "convergence":
        run_nsga_real_convergence()
    else:
        print(f"Unknown mode: {mode} or experiment type: {experiment_type}")
        sys.exit(1)
