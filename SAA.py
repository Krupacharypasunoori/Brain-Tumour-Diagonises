import time
import numpy as np


def SAA(population, objective_function, Xmin, Xmax,Itermax):
    """
    Snow Avalanches Algorithm (SAA)

    Parameters:
        objective_function: Callable, the objective function to minimize
        Xmin: float, lower bound of the search space
        Xmax: float, upper bound of the search space
        D: int, dimension of the problem
        NP: int, population size
        Itermax: int, maximum number of iterations
        si: float, selection probability threshold

    Returns:
        XBest: ndarray, the best solution found
        f(XBest): float, the fitness value of the best solution
    """
    # Step 1: Initialize parameters
    Iter = 0

    si = 0.7

    # Step 2: Generate the random initial population
    NP, D = population.shape[0], population.shape[1]

    # Calculate the fitness of each individual
    fitness = np.array([objective_function(ind) for ind in population])

    # Identify the best solution
    XBest = population[np.argmin(fitness)]
    f_XBest = np.min(fitness)

    ct = time.time()
    convergence = np.zeros(Itermax)
    # Main loop
    while Iter < Itermax:

        for i in range(NP):
            # Step 8: Select members
            r1, r2, r3 = np.random.choice([j for j in range(NP) if j != i], 3, replace=False)
            Xr1, Xr2, Xr3 = population[r1], population[r2], population[r3]

            # Step 9-16: Generate new solutions based on conditions
            if np.random.rand() < si:
                Xnew = XBest + np.random.rand(D) * (Xr1 - Xr2)
            elif np.random.rand() < si:
                Xnew = Xr3 + np.random.rand(D) * (Xr1 - Xr2)
            elif np.random.rand() < si:
                Xnew = population[i] + np.random.rand(D) * (Xr1 - Xr2)
            else:
                Xnew = population[i] + np.random.rand(D) * (Xmax - Xmin)

            # Ensure Xnew stays within bounds
            Xnew = np.clip(Xnew, Xmin, Xmax)

            # Step 18-20: Update individual if the new solution is better
            f_Xnew = objective_function(Xnew)
            if f_Xnew[i] < fitness[i]:
                population[i] = Xnew[i]
                fitness[i] = f_Xnew[i]

            # Step 21: Update the global best solution if necessary
            if f_Xnew[i] < f_XBest:
                XBest = Xnew[i]
                f_XBest = f_Xnew[i]
        convergence[Iter] = f_XBest

        Iter += 1

    # Return the best solution found
    ct = time.time() - ct
    return f_XBest, convergence, XBest, ct
