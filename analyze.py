#!/usr/bin/env python3
from itertools import product
import json
from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def analyze_budget_results(population_size: int, variant: Literal["binary", "real"] = "binary"):
    # Load the results from the JSON file
    with open(f"results/budget_{variant}_{population_size}.json", "r") as f:
        measurements = json.load(f)

    # Prepare a list to hold all data for the DataFrame
    all_data = []

    # Process each run's measurements
    for run, stats in measurements.items():
        hypervolumes = []
        for key, value in stats.items():
            if key in [
                "mutate_function",
                "crossover_function",
                "crossover_rate",
                "mutation_rate",
                "eta_m",
                "disable_threshold",
                "population_size",
            ]:
                continue  # Skip metadata entries
            hypervolumes.append(value["hypervolume"])
        data_entry = {
            "run": int(run),
            "hypervolumes": hypervolumes,
        }
        all_data.append(data_entry)

    # Create a DataFrame from the collected data
    df = pd.DataFrame(all_data)

    return df


def draw_computational_budget_plot():
    variants = ["real", "binary"]
    population_sizes = [50, 100, 200, 500]

    # Increase default font sizes and other text sizes for readability
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    fig = plt.figure(figsize=(7.5, 5))

    # Use different linestyles for variants so they are easy to distinguish
    linestyle_map = {"real": "-", "binary": "--"}

    for variant in variants:
        for population_size in population_sizes:
            df = analyze_budget_results(population_size, variant)

            # Average the hypervolumes across runs for each generation and show std
            avg_hypervolumes = df["hypervolumes"].apply(pd.Series).mean()
            std_hypervolumes = df["hypervolumes"].apply(pd.Series).std()
            # print(f"{variant} pop={population_size} std: {std_hypervolumes}")

            # x-axis: generation index (1-based) times population size
            x = np.array([(i + 1) * population_size for i in range(len(avg_hypervolumes))])

            label = f"{variant} pop={population_size}"
            plt.plot(
                x,
                avg_hypervolumes,
                label=label,
                linestyle=linestyle_map.get(variant, "-"),
                linewidth=2.5,
            )
            plt.fill_between(
                x, (avg_hypervolumes - std_hypervolumes), (avg_hypervolumes + std_hypervolumes), alpha=0.15
            )

    plt.xlabel("Number of Fitness Evaluations (generations × population size)")
    plt.xlim((0, 25000))
    plt.ylabel("Hypervolume")
    # plt.title("Average Hypervolume per Evaluations")
    plt.grid(True)
    plt.legend(title="Variant / Population")

    out_path = f"figures/budget_both.pdf"
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved hypervolume plot to: {out_path}")
    plt.show()


def get_convergence_generation(hypervolumes: list, window: int = 30, threshold: float = 1e-5) -> int | None:
    """Return the first generation at which hypervolume changes < threshold over last `window` generations.

    Returns None if the algorithm never converges within the recorded generations.
    """
    for gen in range(window, len(hypervolumes)):
        window_hv = hypervolumes[gen - window : gen]
        if max(window_hv) - min(window_hv) < threshold:
            return gen  # Return the generation index where convergence is detected
    return None


def analyze_parameter_effects(
    variant: Literal["binary", "real"],
    mutation_rate: float,
    crossover_rate: float,
    eta_m: int | None,
    disable_threshold: float | None,
):
    # Load the results from the JSON file
    file_name = f"results/parameters_{variant}_mutation{mutation_rate}_crossover{crossover_rate}"
    file_name += f"_eta{eta_m}" if eta_m is not None and variant == "real" else ""
    file_name += (
        f"_threshold{disable_threshold}" if disable_threshold is not None and variant == "real" else ""
    )
    file_name += ".json"

    with open(file_name, "r") as f:
        measurements = json.load(f)

    # Prepare a list to hold all data for the DataFrame
    all_data = []

    # Process each run's measurements
    for run, stats in measurements.items():
        hypervolumes = []
        for key, value in stats.items():
            if key in [
                "mutate_function",
                "crossover_function",
                "crossover_rate",
                "mutation_rate",
                "eta_m",
                "disable_threshold",
                "population_size",
            ]:
                continue  # Skip metadata entries
            hypervolumes.append(value["hypervolume"])
        data_entry = {
            "run": int(run),
            "hypervolumes": hypervolumes,
        }
        all_data.append(data_entry)

    # Create a DataFrame from the collected data
    df = pd.DataFrame(all_data)

    return df


def draw_parameter_effects_plot(variant: Literal["binary", "real"]):
    mutation_rates = [0.01, 1 / 36, 0.1]
    crossover_rates = [0.5, 0.7, 0.9]
    eta_ms = [5, 15, 25] if variant == "real" else [None]
    disable_thresholds = [0.001] if variant == "real" else [None]

    results = []

    for mutation_rate, crossover_rate, eta_m, disable_threshold in product(
        mutation_rates, crossover_rates, eta_ms, disable_thresholds
    ):
        try:
            df = analyze_parameter_effects(variant, mutation_rate, crossover_rate, eta_m, disable_threshold)
        except FileNotFoundError:
            continue  # No results yet

        # Compute convergence generation for each run
        convergence_gens = []
        for _, row in df.iterrows():
            conv_gen = get_convergence_generation(row["hypervolumes"])
            if conv_gen is not None:
                convergence_gens.append(conv_gen)

        # Mean and std of convergence generations (only for runs that converged)
        if convergence_gens:
            mean_conv_gen = np.mean(convergence_gens)
            std_conv_gen = np.std(convergence_gens)
        else:
            mean_conv_gen = np.nan
            std_conv_gen = np.nan

        # Over the 10 runs
        avg_hypervolumes = df["hypervolumes"].apply(pd.Series).mean()
        std_hypervolumes = df["hypervolumes"].apply(pd.Series).std()

        # final_avg_hypervolume = avg_hypervolumes.iloc[int(mean_conv_gen) - 1] if not np.isnan(mean_conv_gen) else avg_hypervolumes.iloc[-1]
        final_avg_hypervolume = avg_hypervolumes.iloc[-1]
        final_std_hypervolume = std_hypervolumes.iloc[-1]

        results.append(
            {
                "mutation_rate": mutation_rate,
                "crossover_rate": crossover_rate,
                "eta_m": eta_m,
                "disable_threshold": disable_threshold,
                "final_avg_hypervolume": final_avg_hypervolume,
                "final_std_hypervolume": final_std_hypervolume,
                "mean_convergence_gen": mean_conv_gen,
                "std_convergence_gen": std_conv_gen,
            }
        )

    results_df = pd.DataFrame(results)

    # --- 3. Plotting Logic ---
    sns.set_context("paper", font_scale=1.2)

    if variant == "binary":
        # --- BINARY VARIANT: Single Heatmap ---
        fig = plt.figure(figsize=(8, 6))

        # Pivot the data for the heatmap (Matrix form)
        heatmap_data = results_df.pivot(
            index="crossover_rate", columns="mutation_rate", values="final_avg_hypervolume"
        )
        conv_data = results_df.pivot(
            index="crossover_rate", columns="mutation_rate", values="mean_convergence_gen"
        )
        std_conv_data = results_df.pivot(
            index="crossover_rate", columns="mutation_rate", values="std_convergence_gen"
        )

        # Build custom annotations: HV on first line, convergence on second line in brackets
        annot = heatmap_data.copy().astype(object)
        for r in heatmap_data.index:
            for c in heatmap_data.columns:
                hv = heatmap_data.loc[r, c]
                m_conv = conv_data.loc[r, c]
                if pd.isna(hv):
                    annot.loc[r, c] = ""
                elif pd.isna(m_conv):
                    # annot.loc[r, c] = f"{hv:.4f}\n(N/A)"
                    annot.loc[r, c] = f"{hv:.4f}"
                else:
                    # annot.loc[r, c] = f"{hv:.4f}\n({m_conv:.0f} gens.)"
                    annot.loc[r, c] = f"{hv:.4f}"

        # Draw heatmap with custom annotations (larger annotation font)
        ax = sns.heatmap(
            heatmap_data,
            annot=annot,
            fmt="",
            annot_kws={"fontsize": 14},
            cmap="viridis",
            vmin=results_df["final_avg_hypervolume"].min(),
            vmax=results_df["final_avg_hypervolume"].max(),
            cbar_kws={"label": "Mean Hypervolume"},
        )

        # Rename 1/36 column label
        ax.set_xticklabels([f"1/36" if "0.027" in t.get_text() else t.get_text() for t in ax.get_xticklabels()])

        # Set axis labels
        ax.set_xlabel("Mutation rate")
        ax.set_ylabel("Crossover rate")

        # Invert Y axis so higher crossover rates are at the top (optional, but standard)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()
        fig.savefig("figures/parameter_sensitivity_binary.pdf")

    elif variant == "real":
        # --- REAL VARIANT: Faceted Heatmaps (1 row, 3 columns) ---
        unique_etas = sorted(results_df["eta_m"].unique())
        num_etas = len(unique_etas)

        # Create subplots
        fig, axes = plt.subplots(1, num_etas, figsize=(6 * num_etas, 5), sharey=True)
        if num_etas == 1:
            axes = [axes]  # Handle edge case if only 1 eta exists

        # Determine global min/max for consistent color scaling across all plots
        vmin_hv = results_df["final_avg_hypervolume"].min()
        vmax_hv = results_df["final_avg_hypervolume"].max()
        vmin_conv = results_df["mean_convergence_gen"].min()
        vmax_conv = results_df["mean_convergence_gen"].max()

        for i, eta in enumerate(unique_etas):
            ax = axes[i]

            # Filter data for this specific Eta
            subset = results_df[results_df["eta_m"] == eta]
            heatmap_data = subset.pivot(
                index="crossover_rate", columns="mutation_rate", values="final_avg_hypervolume"
            )
            conv_data = subset.pivot(
                index="crossover_rate", columns="mutation_rate", values="mean_convergence_gen"
            )
            std_conv_data = subset.pivot(
                index="crossover_rate", columns="mutation_rate", values="std_convergence_gen"
            )

            # Build custom annotations: HV on first line, convergence on second line in brackets
            annot = heatmap_data.copy().astype(object)
            for r in heatmap_data.index:
                for c in heatmap_data.columns:
                    hv = heatmap_data.loc[r, c]
                    m_conv = conv_data.loc[r, c]
                    if pd.isna(hv):
                        annot.loc[r, c] = ""
                    elif pd.isna(m_conv):
                        # annot.loc[r, c] = f"{hv:.4f}\n(N/A)"
                        annot.loc[r, c] = f"{hv:.4f}"
                    else:
                        # annot.loc[r, c] = f"{hv:.4f}\n({m_conv:.0f} gens.)"
                        annot.loc[r, c] = f"{hv:.4f}"

            # Draw heatmap without an individual colorbar so subplots keep equal size
            sns.heatmap(
                heatmap_data,
                ax=ax,
                annot=annot,
                fmt="",
                annot_kws={"fontsize": 14},
                cmap="viridis",
                vmin=vmin_hv,
                vmax=vmax_hv,
                cbar=False,
            )

            # Rename 1/36 column label
            ax.set_xticklabels([f"1/36" if "0.027" in t.get_text() else t.get_text() for t in ax.get_xticklabels()])

            # Set axis labels (only set y-label on first column)
            ax.set_xlabel("Mutation rate")
            if i == 0:
                ax.set_ylabel("Crossover rate")
            else:
                ax.set_ylabel("")

            ax.set_title(rf"$\eta_m = {eta}$")

            ax.invert_yaxis()

        # Create a dedicated axis for a single shared colorbar to the right of subplots
        # Coordinates are [left, bottom, width, height] in figure fraction.
        # Make the colorbar narrower and move it slightly left so it's closer
        # to the last heatmap.
        cax = fig.add_axes([0.9, 0.13, 0.01, 0.72])  # type: ignore
        mappable = axes[-1].collections[0]
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label("Mean Hypervolume")

        # plt.suptitle("Parameter Sensitivity (Real-Valued Variant)", y=0.95)
        plt.tight_layout(rect=[0, 0, 0.9, 1.0])
        plt.show()
        fig.savefig("figures/parameter_sensitivity_real.pdf")


def analyze_convergence_results(variant: Literal["binary", "real"] = "real"):
    # Load the results from the JSON file
    with open(f"results/convergence_{variant}.json", "r") as f:
        measurements = json.load(f)

    # Prepare a list to hold all data for the DataFrame
    all_data = []

    # Process each run's measurements
    for run, stats in measurements.items():
        hypervolumes = []
        times = []
        for key, value in stats.items():
            if key in [
                "mutate_function",
                "crossover_function",
                "crossover_rate",
                "mutation_rate",
                "eta_m",
                "disable_threshold",
                "population_size",
            ]:
                continue  # Skip metadata entries
            hypervolumes.append(value["hypervolume"])
            times.append(value["generation_time"])
        data_entry = {
            "run": int(run),
            "hypervolumes": hypervolumes,
            "times": times,
        }
        all_data.append(data_entry)

    # Create a DataFrame from the collected data
    df = pd.DataFrame(all_data)

    return df


def draw_convergence_plot():
    variants = ["binary", "real"]

    # Increase default font sizes and other text sizes for readability
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    fig = plt.figure(figsize=(6, 3))

    for variant in variants:
        df = analyze_convergence_results(variant)

        # Compute convergence generation for each run
        convergence_gens = []
        for _, row in df.iterrows():
            conv_gen = get_convergence_generation(row["hypervolumes"])
            if conv_gen is not None:
                convergence_gens.append(conv_gen)

        # Mean and std (NaN if no run converged)
        mean_conv_gen = np.mean(convergence_gens) if convergence_gens else np.nan
        std_conv_gen = np.std(convergence_gens) if convergence_gens else np.nan

        print(f"{variant}: mean convergence generation = {mean_conv_gen} (std={std_conv_gen}, n={len(convergence_gens)})")

        # Average the hypervolumes across runs for each generation and show std
        avg_hypervolumes = df["hypervolumes"].apply(pd.Series).mean()
        std_hypervolumes = df["hypervolumes"].apply(pd.Series).std()

        # x-axis: generation index (1-based)
        x = np.array([i + 1 for i in range(len(avg_hypervolumes))])

        label = "Binary-coded NSGA-II" if variant == "binary" else "Real-valued NSGA-II"
        # Plot individual runs (light lines) then the mean (highlighted)
        # for _, row in df.iterrows():
        #     hv = pd.Series(row["hypervolumes"])
        #     plt.plot(x, hv, color="gray", alpha=0.35, linewidth=1)

        plt.plot(
            x,
            avg_hypervolumes,
            label=label,
            linewidth=2.5,
        )
        plt.fill_between(
            x, (avg_hypervolumes - std_hypervolumes), (avg_hypervolumes + std_hypervolumes), alpha=0.15
        )

    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    # plt.title("Average Hypervolume per Generation")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.xlim((0, 1000))

    out_path = f"figures/hypervolumes_convergence.pdf"
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved hypervolume convergence plot to: {out_path}")
    plt.show()


def calculate_convergence_time():
    variants = ["binary", "real"]

    for variant in variants:
        df = analyze_convergence_results(variant)

        # Compute convergence generation for each run
        convergence_times = []
        for _, row in df.iterrows():
            conv_gen = get_convergence_generation(row["hypervolumes"])
            if conv_gen is not None:
                conv_time = sum(row["times"][:conv_gen])
                convergence_times.append(conv_time)
        
        print(f"{variant}: mean convergence time = {np.mean(convergence_times):.2f}s (std={np.std(convergence_times):.2f}s, n={len(convergence_times)})")
        print("full times:", convergence_times)

if __name__ == "__main__":
    # draw_computational_budget_plot()
    draw_parameter_effects_plot("binary")
    draw_parameter_effects_plot("real")
    # draw_convergence_plot()
    # calculate_convergence_time()
