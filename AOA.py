import time
import numpy as np


def AOA(population, objective_function, lb, ub, max_iter):
    population_size, dim = population.shape[0], population.shape[1]
    """
    Addax Optimization Algorithm (AOA) implementation in Python.

    Parameters:
    objective_function: The function to be optimized.
    bounds: A list of tuples defining the lower and upper bounds for each variable.
    max_iter: Maximum number of iterations.
    population_size: Number of addax individuals in the population.

    Returns:
    best_solution: The best solution found.
    best_fitness: The fitness value of the best solution.
    """
    fitness = objective_function(population)
    best_fitness = float('inf')
    index = np.where(fitness == np.min(fitness))[0][0]
    Bestfit = fitness[index]
    best_solution = population[index, :]

    Convergence_curve = np.zeros(max_iter)
    ct = time.time()

    for iteration in range(max_iter):
        for i in range(population_size):
            # Exploration phase
            r1 = np.random.uniform(0, 1)
            r2 = np.random.uniform(0, 1)
            A = 2 * r1 * np.cos(np.pi * r2)
            C = 2 * r1

            if r1 < 0.5:
                X_new = best_solution + A * (population[i] - best_solution)
            else:
                X_new = best_solution - A * (population[i] - best_solution)

            # Exploitation phase
            if r1 < 0.5:
                X_new = X_new + C * np.random.randn(len(lb[i]))
            else:
                X_new = best_solution + C * np.random.randn(len(ub[i]))

            # Boundary handling
            X_new = np.clip(X_new, lb, ub)

            # Evaluate fitness
            fitness = objective_function(X_new)

            # Update best solution
            if fitness[i] < Bestfit:
                best_fitness = fitness[i]
                best_solution = X_new[i]

        Convergence_curve[iteration] = best_fitness

    ct = time.time() - ct
    return best_fitness, Convergence_curve, best_solution, ct
