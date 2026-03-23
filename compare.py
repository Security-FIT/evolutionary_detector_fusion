#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nsga import NSGA_real, NSGA_binary
import pickle

if __name__ == "__main__":
    plt.figure(figsize=(6, 4))

    manual_eers = [3.95, 3.35, 3.44]
    manual_params = [3.528682e09, 6.214198e09, 1.856189e10]
    plt.scatter(
        manual_eers,
        manual_params,
        c="tab:orange",
        marker="s",
        label="Manual Fusion",
    )

    lr_fusions = json.load(open("logistic_regression_fusion.json", "r"))
    lr_fusions = pd.DataFrame(lr_fusions)
    plt.scatter(
        lr_fusions["eer"] * 100,
        lr_fusions["num_params"],
        c="tab:blue",
        marker="^",
        label="Logistic Regression",
    )

    individuals = []
    for run in range(10):
        with open(f"./results/nsga_binary_convergence_run{run}.pkl", "rb") as f:
            nsga_binary: NSGA_binary = pickle.load(f)
        individuals.append(nsga_binary.fronts[0].individuals)
    
    nsga_binary_all = NSGA_binary(population_size=1000)
    nsga_binary_all.population = np.concatenate(individuals)
    # No generation run, just sort the combined population
    nsga_binary_all.run(generations=0)

    plt.scatter(
        nsga_binary_all.fronts[0].fitness[:, 0] * 100,
        nsga_binary_all.fronts[0].fitness[:, 1],
        color="tab:purple",
        label=f"NSGA-II (binary-coded)",
        marker="d",
    )

    individuals = []
    for run in range(10):
        with open(f"./results/nsga_real_convergence_run{run}.pkl", "rb") as f:
            nsga_real: NSGA_real = pickle.load(f)
        individuals.append(nsga_real.fronts[0].individuals)

    nsga_real_all = NSGA_real(population_size=1000)
    nsga_real_all.population = np.concatenate(individuals)
    # No generation run, just sort the combined population
    nsga_real_all.run(generations=0)

    plt.scatter(
        nsga_real_all.fronts[0].fitness[:, 0] * 100,
        nsga_real_all.fronts[0].fitness[:, 1],
        color="tab:red",
        label=f"NSGA-II (real-valued)",
        marker="p",
    )

    plt.legend()
    plt.xlabel("EER [%]")
    plt.ylabel("Number of Parameters")
    plt.grid()
    # plt.show()
    plt.savefig("figures/compare_color.pdf")
