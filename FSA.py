import time
import numpy as np


def FSA(population, objective_function, lb, ub, iter_max):
    """
    Flamingo Search Algorithm

    Parameters:
    - objective_function: Function to minimize
    - lb: Lower bound of search space (array or scalar)
    - ub: Upper bound of search space (array or scalar)
    - dim: Dimension of the search space
    - pop_size: Flamingo population size
    - iter_max: Maximum number of iterations
    - mpb: Proportion of migrating flamingos (float between 0 and 1)

    Returns:
    - fg: Optimal fitness value
    - xbest: Optimal solution
    """
    # Initialize the flamingo population randomly within bounds
    pop_size, dim = population.shape[0], population.shape[1]
    fitness = np.array([objective_function(ind) for ind in population])

    # Rank the population and find the current best individual
    best_idx = np.argmin(fitness)
    xbest = population[best_idx]
    fg = fitness[best_idx]
    convergence = np.zeros(iter_max)
    ct = time.time()
    mpb = 0.3
    t = 0
    while t < iter_max:
        R = np.random.rand()
        mpr = int(R * pop_size * (1 - mpb))
        mp0 = int(mpb * pop_size)
        mpt = pop_size - mp0 - mpr

        # Update locations for migrating flamingos (MPb)
        for i in range(mp0):
            for j in range(dim):
                population[i, j] += np.random.normal(0, 1) * (ub[i, j] - lb[i, j]) * 0.01

        # Update locations for random flamingos (MPr)
        for i in range(mp0, mp0 + mpr):
            for j in range(dim):
                population[i, j] += np.random.uniform(-1, 1) * (ub[i, j] - lb[i,j]) * 0.01

        # Update locations for remaining flamingos (MPt)
        for i in range(mp0 + mpr, pop_size):
            for j in range(dim):
                population[i, j] += np.random.normal(0, 1) * (ub[i, j] - lb[i, j]) * 0.01

        # Boundary detection
        population = np.clip(population, lb, ub)

        # Evaluate the fitness of the updated population
        fitness = np.array([objective_function(ind) for ind in population])

        # Update the best individual
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < fg:
            fg = fitness[best_idx]
            xbest = population[best_idx]
        convergence[t] = fg

        t += 1
    ct = time.time() - ct
    return fg, convergence, xbest, ct
