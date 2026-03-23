#!/usr/bin/env python3

from joblib import Parallel, delayed
import logging
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import List, Tuple

from ea import mutate_binary, mutate_real, swap, crossover_onepoint, crossover_uniform
from load_scores import load_all_scores_cached
from metrics import calculate_EER, get_num_params
from time_method import time_method


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def mutate_placeholder(vector, mutation_rate=0.1):
    """
    Placeholder for a mutation function for initialization. Needs to be replaced with a real mutation function.
    """
    raise NotImplementedError(
        "Mutation function not implemented. The class needs to assign to self._mutation_function"
    )


class NDF:
    """
    Class to represent a non-dominated front (NDF).
    """

    def __init__(self, individuals: np.ndarray, fitness: np.ndarray, rank: int):
        """
        Initialize the NDF with individuals and their fitness.

        Args:
            individuals (np.ndarray): Array of individuals in the NDF.
            fitness (np.ndarray): Fitness values of the individuals.
            rank (int): Rank of the NDF.
        """
        self.individuals = individuals
        self.fitness = fitness
        self.rank = rank  # Rank of the NDF
        self.crowding_distance = np.zeros(len(individuals))

    def __len__(self):
        """
        Get the number of individuals in the NDF.

        Returns:
            int: Number of individuals in the NDF.
        """
        return len(self.individuals)

    def __repr__(self):
        """
        String representation of the NDF.

        Returns:
            str: String representation of the NDF.
        """
        return f"NDF(individuals=\n{self.individuals}\nfitness=\n{self.fitness}\ncrowding_distance=\n{self.crowding_distance}\n)"

    def __iter__(self):
        """
        Iterate over the individuals in the NDF.

        Returns:
            Iterator: Iterator over the individuals in the NDF.
        """
        return iter([self.individuals, self.fitness, self.crowding_distance])

    def sort_by_objectives(self):
        """
        Sort the NDF based on fitness and crowding distance.
        """
        # Sort by fitness (lexicographical order)
        # LEXSORT SORTS BY THE LAST KEY FIRST, WHO THE F THOUGHT THIS WAS A GOOD IDEA?
        sorted_indices = np.lexsort((self.fitness[:, 1], self.fitness[:, 0]))
        self.individuals = self.individuals[sorted_indices]
        self.fitness = self.fitness[sorted_indices]
        self.crowding_distance = self.crowding_distance[sorted_indices]

        # if np.any(self.crowding_distance != 0):  # Only sort more if crowding distance is not zero
        #     return
        # else:
        #     # Sort by crowding distance
        #     sorted_indices = np.argsort(self.crowding_distance)[::-1]
        #     self.individuals = self.individuals[sorted_indices]
        #     self.fitness = self.fitness[sorted_indices]

    def sort_by_crowding_distance(self):
        """
        Sort the NDF based on crowding distance.
        """
        # Sort by crowding distance (descending order)
        sorted_indices = np.argsort(self.crowding_distance)[::-1]
        self.individuals = self.individuals[sorted_indices]
        self.fitness = self.fitness[sorted_indices]
        self.crowding_distance = self.crowding_distance[sorted_indices]

    def calculate_crowding_distance(self):
        # Original implementation
        self.crowding_distance = np.zeros(len(self))
        for i in range(len(self)):
            if i == 0 or i == len(self) - 1:
                self.crowding_distance[i] = float("inf")
            else:  # Probably ok, but not 1000000% sure
                self.crowding_distance[i] = np.sum(  # Do both objectives
                    (self.fitness[i + 1] - self.fitness[i - 1])  # Compute the distance
                    / (self.fitness[-1] - self.fitness[0])  # Normalize (idx -1 is max, idx 0 is min)
                )

    def calculate_crowding_distance_vectorized(self):
        """
        Calculate the crowding distance for the NDF.
        """
        self.sort_by_objectives()  # Sort the NDF before calculating crowding distance, results in O(n log n) complexity
        self.crowding_distance = np.zeros(len(self))

        # Use numpy vectorization for blazing fast performance
        # Sort each objective
        sorted_indices = np.argsort(self.fitness, axis=0)
        
        # Get sorted fitness values
        sorted_fitness = np.take_along_axis(self.fitness, sorted_indices, axis=0)

        # Normalization factors for each objective
        norm = sorted_fitness[-1] - sorted_fitness[0]
        norm[norm == 0] = 1e-9  # Prevent division by zero

        # Calulate crowding distances (excluding boundary points with infinite distance by definition)
        crowding_distance = np.zeros(self.fitness.shape)
        distances = (sorted_fitness[2:] - sorted_fitness[:-2]) / norm
        crowding_distance[1:-1] = distances
        crowding_distance[0] = float("inf")
        crowding_distance[-1] = float("inf")

        # Scatter-add the crowding distances back to the original order
        np.add.at(self.crowding_distance, sorted_indices.ravel(), crowding_distance.ravel())

    def hypervolume(self) -> float:
        """
        Calculate the hypervolume of the NDF. Expects the NDF to be sorted.
        The hypervolume is the volume of the space dominated by the NDF.

        Returns:
            float: Normalized hypervolume of the NDF (range [0, 1]).
        """
        # Reference point for hypervolume calculation
        # 20% EER (worst single model) and 18561894140 params (all models combined)
        max_params = 18561894140
        max_eer = 0.2
        reference_point = np.array((max_eer, max_params))
        hypervolume = 0.0

        sorted_indices = np.lexsort((self.fitness[:, 1], self.fitness[:, 0]))
        sorted_fitness = self.fitness[sorted_indices]

        last_y = reference_point[1]  # Initialize last_y to the reference point y-coordinate
        for i in range(len(sorted_fitness)):
            width = (reference_point[0] - sorted_fitness[i, 0]) / max_eer  # EER, normalize to [0, 1]
            height = (last_y - sorted_fitness[i, 1]) / max_params  # Number of parameters, normalize to [0, 1]
            if width > 0 and height > 0:
                hypervolume += width * height
            last_y = sorted_fitness[i, 1]

        return hypervolume


class NSGA:
    """
    Class to implement the Non-dominated Sorting Genetic Algorithm (NSGA-II).
    """

    def __init__(self, population_size: int):
        """
        Initialize the NSGA-II algorithm with population size and mutation rate.

        Args:
            population_size (int): Size of the population.
        """
        self.population_size = population_size
        self.scores_df = load_all_scores_cached()
        self.model_names = np.array([col for col in self.scores_df.columns if col not in ["file", "label"]])

        self.fronts: List[NDF] = []  # List of non-dominated fronts
        self.mating_pool: np.ndarray = np.array([])  # Mating pool for crossover
        self.hypervolumes: List[float] = []  # List of hypervolumes for each generation
        self.patience = 30  # Patience for early stopping

        # Civilization fitness to hold track of all the fitness values
        # self.civilization_fitness: np.ndarray = np.empty((0, 2))  # EER and number of parameters

        self._mutate_function = mutate_placeholder  # Mutation function
        self._crossover_function = crossover_uniform  # Crossover function
        self._mutation_rate = 1 / 36  # Mutation rate
        self._crossover_rate = 0.7  # Crossover rate

    def _individual_to_names(self, individual: np.ndarray) -> List[str]:
        """
        Convert a binary individual to a list of model names.

        Args:
            individual (np.ndarray): The individual to convert.

        Returns:
            List[str]: List of model names.
        """
        return self.model_names[individual.astype(bool)].tolist()

    def _plot(self, population: bool = True, best_ndf: bool = True, show: bool = True):
        """
        Plot the population fitness.

        Args:
            population (bool):  Whether to plot the population fitness or not.
            best_ndf (bool):    Whether to plot the best NDF or not.
            show (bool):        Whether to show the plot or not.
        """
        # plt.scatter(
        #     self.civilization_fitness[:, 0] * 100,
        #     self.civilization_fitness[:, 1],
        #     c="lightblue",
        #     alpha=0.9,
        #     marker="o",
        #     label="Complete civilization",
        # )
        if population:
            plt.scatter(
                self.fitness[:, 0] * 100,
                self.fitness[:, 1],
                c="tab:blue",
                marker="o",
                alpha=0.9,
                label="Population",
            )
        if best_ndf:
            plt.scatter(
                self.fronts[0].fitness[:, 0] * 100,
                self.fronts[0].fitness[:, 1],
                marker="o",
                color="tab:red",
                label=f"Best NDF {type(self).__name__}",
            )
        if show:
            plt.legend()
            plt.xlabel("EER [%]")
            plt.ylabel("Number of Parameters")
            plt.title("Population Fitness")
            plt.grid()
            plt.show()

    def _evaluate_individual(self, individual: np.ndarray) -> Tuple[float, int]:
        """
        Evaluate the fitness of an individual.

        Args:
            individual (np.ndarray): The individual to evaluate.

        Returns:
            tuple: A tuple containing EER and number of parameters.
        """

        raise NotImplementedError("This method should be implemented in a subclass.")

    # @time_method
    def _evaluate_individuals(self, individuals: np.ndarray) -> np.ndarray:
        """
        Evaluate the fitness of an array of individuals.

        Args:
            individuals (np.ndarray): The individuals to evaluate.

        Returns:
            np.ndarray: An array of fitness values for each individual.
        """
        fitness = Parallel(n_jobs=-1, backend="threading")(
            delayed(self._evaluate_individual)(individual) for individual in individuals
        )
        # fitness = np.array([self._evaluate_individual(ind) for ind in individuals])
        return np.array(fitness)

    @staticmethod
    def _dominates(individual1: np.ndarray, individual2: np.ndarray) -> bool:
        """
        Check if individual1 dominates individual2.

        Args:
            individual1 (np.ndarray): First individual (its fitness).
            individual2 (np.ndarray): Second individual (its fitness).

        Returns:
            bool: True if individual1 dominates individual2, False otherwise.
        """
        # individual1 is as good in all objectives and better in at least one
        return all(individual1 <= individual2) and any(individual1 < individual2)

    # @time_method
    def _non_dominated_sorting(self):
        """
        Perform non-dominated sorting on the population.
        This method uses a manual approach to count dominated individuals.
        Updates the self.fronts property with the non-dominated fronts.
        """

        self.fronts = []  # Reset the fronts
        # Remove duplicate individuals before sorting
        _, unique_indices = np.unique(self.population, axis=0, return_index=True)
        population = self.population[unique_indices].copy()
        fitness = self.fitness[unique_indices].copy()
        # population = self.population.copy()
        # fitness = self.fitness.copy()

        rank = 0  # Rank of the current front
        while len(population) > 0:
            dominated_count = np.zeros(len(population), dtype=int)

            # This is ugly, make it faster
            for i in range(len(population)):
                for j in range(len(population)):
                    if i != j and self._dominates(fitness[i], fitness[j]):
                        dominated_count[j] += 1
                    elif i != j and self._dominates(fitness[j], fitness[i]):
                        dominated_count[i] += 1

            # Initialize the non-dominated front
            ndf = NDF(population[dominated_count == 0], fitness[dominated_count == 0], rank)
            # If there are no non-dominated individuals, break
            if len(ndf) == 0:
                break

            ndf.calculate_crowding_distance()

            # Append the non-dominated front to the list of fronts
            self.fronts.append(ndf)

            # Remove the individuals in the first front from the population
            population = population[dominated_count > 0]
            fitness = fitness[dominated_count > 0]

            # Increment the rank for the next front
            rank += 1

    # @time_method
    def _non_dominated_sorting_fast(self):
        """
        Perform non-dominated sorting on the population using a faster method.
        Implemented according to https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=996017
        Updates the self.fronts property with the non-dominated fronts.
        """
        # Try the fast non-dominated sorting algorithm
        self.fronts = []  # Reset the fronts
        # Remove duplicate individuals before sorting
        _, unique_indices = np.unique(self.population, axis=0, return_index=True)
        population = self.population[unique_indices].copy()
        fitness = self.fitness[unique_indices].copy()

        # For each individual p, store:
        # S_p (dominated_solutions): set of indices of individuals dominated by p
        # n_p (domination_count_fast): number of individuals that dominate p
        dominated_solutions = [set() for _ in range(len(population))]
        domination_count_fast = np.zeros(len(population), dtype=int)
        current_front_indices = []  # Indices of individuals in the current front (starts with F_1)

        # Step 1: Calculate S_p and n_p for each individual
        for p in range(len(population)):
            for q in range(len(population)):
                if p != q:
                    if self._dominates(fitness[p], fitness[q]):
                        # p dominates q
                        dominated_solutions[p].add(q)
                    elif self._dominates(fitness[q], fitness[p]):
                        # q dominates p
                        domination_count_fast[p] += 1

            # If no individuals dominate p, add it to the first front
            if domination_count_fast[p] == 0:
                current_front_indices.append(p)

        ndf = NDF(population[current_front_indices], fitness[current_front_indices], 0)
        ndf.calculate_crowding_distance()
        self.fronts.append(ndf)  # F_1 is found and added to the list of fronts

        # Extract subsequent fronts iteratively
        front_index = 1
        # While the current front is not empty
        while current_front_indices:
            next_front_indices = []  # Indices for the next front

            # For each individual p_index in the current front
            for p_index in current_front_indices:
                # For each individual q_index dominated by p_index
                for q_index in dominated_solutions[p_index]:
                    # Decrement the domination count of q_index
                    domination_count_fast[q_index] -= 1
                    # If the domination count of q_index becomes 0, it belongs to the next front
                    if domination_count_fast[q_index] == 0:
                        next_front_indices.append(q_index)

            if not next_front_indices:
                break

            # Add the found individuals to the list of fronts
            ndf = NDF(population[next_front_indices], fitness[next_front_indices], front_index)
            ndf.calculate_crowding_distance()
            self.fronts.append(ndf)

            # Move to the next front
            front_index += 1
            current_front_indices = next_front_indices

    # @time_method
    def _non_dominated_sorting_fast_vectorized(self):
        """
        Perform non-dominated sorting on the population using a vectorized method.
        Implemented according to https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=996017 and
        vectorized in NumPy for blazing fast performance.
        Updates the self.fronts property with the non-dominated fronts.
        """
        # Try the fast non-dominated sorting algorithm
        self.fronts = []  # Reset the fronts
        # Remove duplicate individuals before sorting
        _, unique_indices = np.unique(self.population, axis=0, return_index=True)
        population = self.population[unique_indices].copy()
        fitness = self.fitness[unique_indices].copy()

        # Create fitness matrix for vectorized comparison
        fitness_matrix_p = fitness[:, np.newaxis, :]  # Shape (N, 1, M)
        fitness_matrix_q = fitness[np.newaxis, :, :]  # Shape (1, N, M)

        dominates = np.all(fitness_matrix_p <= fitness_matrix_q, axis=2) & np.any(
            fitness_matrix_p < fitness_matrix_q, axis=2
        )
        domination_count = np.sum(dominates, axis=0)
        dominated_solutions = [set(np.where(dominates[p, :])[0]) for p in range(len(population))]

        ndf = NDF(population[domination_count == 0], fitness[domination_count == 0], 0)
        ndf.calculate_crowding_distance_vectorized()
        self.fronts.append(ndf)  # F_1 is found and added to the list of fronts

        # Extract subsequent fronts iteratively
        front_index = 1
        current_front_indices = np.where(domination_count == 0)[0].tolist()
        # While the current front is not empty
        while current_front_indices:
            next_front_indices = []  # Indices for the next front

            # For each individual p_index in the current front
            for p_index in current_front_indices:
                # For each individual q_index dominated by p_index
                for q_index in dominated_solutions[p_index]:
                    # Decrement the domination count of q_index
                    domination_count[q_index] -= 1
                    # If the domination count of q_index becomes 0, it belongs to the next front
                    if domination_count[q_index] == 0:
                        next_front_indices.append(q_index)

            if not next_front_indices:
                break

            # Add the found individuals to the list of fronts
            ndf = NDF(population[next_front_indices], fitness[next_front_indices], front_index)
            ndf.calculate_crowding_distance_vectorized()
            self.fronts.append(ndf)

            # Move to the next front
            front_index += 1
            current_front_indices = next_front_indices

    # @time_method
    def _binary_tournament_selection(self):
        """
        Selects parents for creating offsprings using binary tournament selection.
        """
        individuals, ranks, crowding_distances = zip(
            *[(ndf.individuals, np.full(len(ndf), ndf.rank), ndf.crowding_distance) for ndf in self.fronts],
            strict=True,
        )
        individuals = np.concatenate(individuals)
        ranks = np.concatenate(ranks)
        crowding_distances = np.concatenate(crowding_distances)

        mating_pool = []
        indices = np.random.choice(len(individuals), size=(self.population_size, 2))
        for i1, i2 in indices:
            if ranks[i1] < ranks[i2] or (
                ranks[i1] == ranks[i2] and crowding_distances[i1] > crowding_distances[i2]
            ):
                mating_pool.append(individuals[i1])
            else:
                mating_pool.append(individuals[i2])

        self.mating_pool = np.array(mating_pool)

    # @time_method
    def _create_offsprings(self) -> np.ndarray:
        """
        Create offsprings from the mating pool using crossover and mutation.
        """
        offsprings = []
        for i in range(0, len(self.mating_pool), 2):
            parent1 = self.mating_pool[i]
            parent2 = self.mating_pool[i + 1]

            # Crossover
            offspring1, offspring2 = (
                self._crossover_function(parent1, parent2)
                if np.random.rand() < self._crossover_rate
                else (parent1, parent2)
            )

            # Mutation
            offspring1 = self._mutate_function(offspring1, self._mutation_rate)
            offspring2 = self._mutate_function(offspring2, self._mutation_rate)

            # Do not add degenerate individuals (all zeros == no models, does not make sense)
            if any(offspring1):
                offsprings.append(offspring1)
            if any(offspring2):
                offsprings.append(offspring2)

        return np.array(offsprings)

    # @time_method
    def _combine_population(self, offsprings: np.ndarray):
        """
        Combine the current population with the offsprings to create a new population.

        Args:
            offsprings (np.ndarray): The offsprings to combine with the current population.
        """
        offspring_fitness = self._evaluate_individuals(np.array(offsprings))
        # self.civilization_fitness = np.vstack((self.civilization_fitness, offspring_fitness))

        # Create the joined population R_t
        self.population = np.vstack((self.population, offsprings))
        self.fitness = np.vstack((self.fitness, offspring_fitness))

    # @time_method
    def _select_new_population(self):
        """
        Select the new population from the combined population of parents and offsprings.
        """
        new_population = []  # List of individuals for the new population
        new_fitness = []  # List of fitness values for the new population
        for front in self.fronts:
            # Select as many individuals as possible from the best fronts
            if len(new_population) + len(front) < self.population_size:
                new_population.extend(front.individuals)
                new_fitness.extend(front.fitness)
            # If the current front is too large, select individuals based on crowding distance
            else:
                front.sort_by_crowding_distance()
                open_slots = self.population_size - len(new_population)
                new_population.extend(front.individuals[:open_slots])
                new_fitness.extend(front.fitness[:open_slots])
                break
        self.population = np.array(new_population)
        self.fitness = np.array(new_fitness)

    def _early_stopping(
        self, generation_hypervolume: float, threshold: float = 1e-5
    ) -> bool:
        """
        Check if the algorithm should stop early based on hypervolume.
        """
        self.hypervolumes.append(generation_hypervolume)

        if len(self.hypervolumes) < self.patience:
            return False

        # improvement = np.diff(self.hypervolumes[-patience:])
        # avg_improvement = np.mean(improvement[-patience:])
        gain = self.hypervolumes[-1] - self.hypervolumes[-self.patience + 1]

        return gain < threshold

    # @time_method
    def run(self, generations: int):
        """
        Run the NSGA-II algorithm for a specified number of generations.

        Args:
            generations (int): Number of generations to run the algorithm.
        """
        # Initialization (generation 0)
        self.fitness = self._evaluate_individuals(self.population)

        # self.civilization_fitness = np.vstack((self.civilization_fitness, self.fitness))
        self._non_dominated_sorting_fast_vectorized()
        self.hypervolumes.append(self.fronts[0].hypervolume())

        # Main loop
        # for generation in tqdm(range(generations)):
        for generation in range(generations):
            self._select_new_population()  # Select the new population from the combined population of parents and offsprings
            self._binary_tournament_selection()  # Select parents for creating offsprings
            offsprings = self._create_offsprings()  # Create offsprings Q_t
            self._combine_population(offsprings)  # Create the combined population R_t
            self._non_dominated_sorting_fast_vectorized()  # Perform non-dominated sorting on the joint population

            # Early stopping
            # if self._early_stopping(self.fronts[0].hypervolume()):
            #     logger.info(
            #         f"Early stopping triggered, hypervolume did not significantly improve over the last {self.patience} generations."
            #     )
            #     break


class NSGA_binary(NSGA):
    """
    Class to implement the Non-dominated Sorting Genetic Algorithm (NSGA-II) for binary individuals.
    """

    def __init__(self, population_size: int):
        """
        Initialize the NSGA-II algorithm with population size and mutation rate.

        Args:
            population_size (int): Size of the population.
        """
        # logger.info("Initializing binary NSGA-II")
        super().__init__(population_size)
        self._mutate_function = mutate_binary  # Mutation function

        # Initialize the population with random binary individuals
        self.population: np.ndarray = np.random.randint(2, size=(self.population_size, 36))

        # Just for fun, lets add some special individuals to the population
        self.population[0] = np.ones(36)  # All models equally weighted
        # Individuals of each of the single models
        for i in range(1, 37):
            self.population[i] = np.zeros(36)
            self.population[i, i - 1] = 1

        # Initialize the fitness
        self.fitness: np.ndarray = np.array(
            np.ones((len(self.population), 2)) * float("inf")
        )  # EER and number of parameters

    def _evaluate_individual(self, individual: np.ndarray) -> Tuple[float, int]:
        """
        Evaluate the fitness of an individual.

        Args:
            individual (np.ndarray): The individual to evaluate.

        Returns:
            tuple: A tuple containing EER and number of parameters.
        """
        models = self.model_names[individual.astype(bool)]
        if len(models) == 0:
            return float("inf"), int("inf")  # We always need at least one model

        # Use float32 to save memory and faster calculation
        fusion_scores = self.scores_df[models].to_numpy(dtype=np.float32).mean(axis=1)
        labels = self.scores_df["label"].values
        eer = calculate_EER(labels, fusion_scores)

        num_params = sum(get_num_params(model) for model in models)

        return eer, num_params


class NSGA_real(NSGA):
    """
    Class to implement the Non-dominated Sorting Genetic Algorithm (NSGA-II) for real-valued individuals.
    """

    def __init__(self, population_size: int):
        """
        Initialize the NSGA-II algorithm with population size and mutation rate.

        Args:
            population_size (int): Size of the population.
        """
        # logger.info("Initializing real-valued NSGA-II")
        super().__init__(population_size)
        self._mutate_function = mutate_real  # Mutation function

        self.threshold = 0.001  # Threshold for disabling models
        # Initialize the population with random real-valued individuals
        self.population: np.ndarray = np.random.rand(self.population_size, 36)

        # Just for fun, lets add some special individuals to the population
        self.population[0] = np.ones(36) / 36  # All models equally weighted
        # Individuals of each of the single models
        for i in range(1, 37):
            self.population[i] = np.zeros(36)
            self.population[i, i - 1] = 1.0

        # Normalize each individual to sum to 1
        self.population = self.population / np.sum(self.population, axis=1, keepdims=True)
        # Disable models with weights < self.threshold - set to 0
        self.population[self.population < self.threshold] = 0
        # Normalize again to sum to 1
        self.population = self.population / np.sum(self.population, axis=1, keepdims=True)

        # Initialize the fitness
        self.fitness: np.ndarray = np.array(
            np.ones((len(self.population), 2)) * float("inf")
        )  # EER and number of parameters

    def _evaluate_individual(self, individual: np.ndarray) -> Tuple[float, int]:
        """
        Evaluate the fitness of an individual.

        Args:
            individual (np.ndarray): The individual to evaluate (the model weights).

        Returns:
            tuple: A tuple containing EER and number of parameters.
        """
        models = self.model_names[individual.astype(bool)]
        if len(models) == 0:
            return float("inf"), int("inf")

        weights = individual[individual > 0]

        fusion_scores = self.scores_df[models].dot(weights).astype(np.float32)
        labels = self.scores_df["label"].values
        eer = calculate_EER(labels, fusion_scores)
        num_params = sum(get_num_params(model) for model in models)

        return eer, num_params

    # @time_method
    def _combine_population(self, offsprings: np.ndarray):
        """
        Combine the current population with the offsprings to create a new population.

        Args:
            offsprings (np.ndarray): The offsprings to combine with the current population.
        """
        # Set model with weights < self.threshold to 0 (disable low-contributing models)
        offsprings[offsprings < self.threshold] = 0
        # Remove individuals with all zeros (no models)
        offsprings = offsprings[offsprings.sum(axis=1) > 0]
        # Normalize the offsprings to sum to 1
        offsprings = offsprings / np.sum(offsprings, axis=1, keepdims=True)

        offspring_fitness = self._evaluate_individuals(offsprings)
        # self.civilization_fitness = np.vstack((self.civilization_fitness, offspring_fitness))

        # Create the joined population R_t
        self.population = np.vstack((self.population, offsprings))
        self.fitness = np.vstack((self.fitness, offspring_fitness))


if __name__ == "__main__":
    nsga_real = NSGA_real(population_size=100)
    nsga_real.run(generations=500)
    # nsga_real._plot(population=True, best_ndf=True, show=True)
