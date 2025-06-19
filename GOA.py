import time
import numpy as np


# Gannet Optimization Algorithm (GOA)
def GOA(positions, fobj, VRmin, VRmax, max_iter):
    num_gannets, dim = positions.shape[0], positions.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]

    best_position = np.zeros((dim, 1))
    best_score = float('inf')

    Convergence_curve = np.zeros((max_iter, 1))
    t = 0
    ct = time.time()
    for t in range(max_iter):
        for i in range(num_gannets):
            score = fobj(positions[i])
            if score < best_score:
                best_score = score
                best_position = positions[i].copy()

            for j in range(dim):
                r1 = np.random.rand()
                r2 = np.random.rand()
                flight_distance = np.exp(t / max_iter)
                plunge_distance = np.exp(t / max_iter)

                positions = positions + r1 * flight_distance * (
                        best_position[j] - positions)
                positions = positions+ r2 * plunge_distance * (
                        lb + np.random.rand() * (ub - lb) - positions)

                # Ensure the positions are within bounds
                positions = np.clip(positions, lb, ub)
        Convergence_curve[t] = best_score
        t = t + 1
    best_score = Convergence_curve[max_iter - 1][0]
    ct = time.time() - ct

    return best_score, Convergence_curve, best_position, ct
