import time
import numpy as np


# CuttleFish Optimization(CO)
def CO(Positions, fobj, VRmin, VRmax, Max_iter):
    N, dim = Positions.shape[0], Positions.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]

    # Initialize velocities
    velocities = np.zeros((N, dim))

    # Initialize the best position and best fitness
    best_position = np.zeros((dim, 1))
    best_fitness = float('inf')

    Convergence_curve = np.zeros((Max_iter, 1))
    t = 0
    ct = time.time()
    for i in range(Max_iter):
        # Evaluate fitness
        fitness_values = np.array([fobj(pos) for pos in Positions])

        # Update the best position and best fitness
        min_fitness = np.min(fitness_values)
        min_index = np.argmin(fitness_values)
        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_position = Positions[min_index]

        # Update velocities and positions
        for j in range(N):
            r1 = np.random.random(dim)
            r2 = np.random.random(dim)
            velocities[j] = velocities[j] + 0.5 * (best_position - Positions[j]) + 0.5 * (
                    Positions[min_index] - Positions[j])
            Positions[j] = Positions[j] + velocities[j]

            # Clip positions to the search space
            Positions[j] = np.clip(Positions[j], lb, ub)

        Convergence_curve[t] = best_fitness
        t = t + 1
    best_fitness = Convergence_curve[Max_iter - 1][0]
    ct = time.time() - ct

    return best_fitness, Convergence_curve, best_position, ct


