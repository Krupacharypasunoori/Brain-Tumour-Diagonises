import time
import numpy as np


def SCO(S, fobj, lb, ub, T):
    s, dim = S.shape[0], S.shape[1]
    ub = ub * np.ones(dim)
    lb = lb * np.ones(dim)
    Range = ub - lb
    P = 0  # P=0 indicates no fitness improvement, P=1 means fitness is improved

    # Initialize a random candidate solution
    # S = np.zeros(dim)
    # for i in range(dim):
    #     S[i] = lb[i] + np.random.rand() * (ub[i] - lb[i])  # S is the global best Position
    BF = fobj(S)
    POO = 0  # Initial counter to count unsuccessful fitness improvements
    m = 5  # number of unsuccessful attempts to improve the fitness
    alpha = 1000  # number of function evaluations in the First phase
    b = 2.4
    Best_Fitness = np.zeros(T)
    gbest = np.zeros(dim)
    ct = time.time()
    for t in range(T):
        w = np.exp(-(b * t / T) ** b)  # Equation (3) in the paper
        if t > alpha:
            if np.sum(P) == 0:  # Counting the number of unsuccessful fitness improvements
                POO = 1 + POO  # Counter to count unsuccessful fitness improvements

        K = np.random.rand()
        x = np.zeros(dim)
        for j in range(dim):
            EE = w * K * Range[j]
            if t < alpha:
                if np.random.rand() < 0.5:
                    x[j] = S[0, j] + (w * abs(S[0, j]))
                else:  # Equation (2) in the paper
                    x[j] = S[0, j] - (w * abs(S[0, j]))
            else:
                if POO == m:
                    POO = 0  # Reset counter
                    if np.random.rand() < 0.5:
                        x[j] = S[0, j]  + np.random.rand() * Range[j]
                    else:  # Equation (5) in the paper
                        x[j] = S[0, j]  - np.random.rand() * Range[j]
                else:
                    if np.random.rand() < 0.5:
                        x[j] = S[0, j]  + EE
                    else:  # Equation (4) in the paper
                        x[j] = S[0, j] - EE

            # Check if a dimension of the candidate solution goes out of boundaries
            if x[j] > ub[0, j]:
                x[j] = S[0, j]  # Equation (6) in the paper
            if x[j] < lb[0, j]:
                x[j] = S[0, j]

        Fit = np.min(BF)
        Best_Fitness[t] = Fit
        gbest = S
    ct = time.time() - ct
    return Fit, Best_Fitness, gbest, ct
