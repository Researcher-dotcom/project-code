import numpy as np
import random

def pareto_fitness_diversity_selection_simple(population, num_to_select, min_diff_threshold=1e-2, seed=None):
    """
    Select individuals based on sorted fitness (response variable) with diversity enforced on fitness values.

    Args:
        population: list or array of individuals, where last element is fitness.
        num_to_select: number of individuals to select.
        min_diff_threshold: minimum difference in fitness to accept a new individual.
        seed: random seed for reproducibility.

    Returns:
        selected: list of selected individuals (same format as input).
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    population = np.array(population)
    # Sort by fitness ascending (assuming minimization)
    sorted_pop = population[np.argsort(population[:, -1])]

    selected = []
    selected_fitnesses = []

    for candidate in sorted_pop:
        fitness = candidate[-1]
        if not selected:
            selected.append(candidate)
            selected_fitnesses.append(fitness)
        else:
            # Check difference in fitness to all selected
            differences = np.abs(np.array(selected_fitnesses) - fitness)
            if np.all(differences >= min_diff_threshold):
                selected.append(candidate)
                selected_fitnesses.append(fitness)

        if len(selected) >= num_to_select:
            break

    # If not enough selected, fill randomly (without replacement)
    if len(selected) < num_to_select:
        selected_set = {tuple(ind) for ind in selected}
        remaining = [ind for ind in population if tuple(ind) not in selected_set]
        random.shuffle(remaining)
        selected.extend(remaining[:num_to_select - len(selected)])

    return selected[:num_to_select]