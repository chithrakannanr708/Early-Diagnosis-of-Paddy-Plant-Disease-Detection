import time
import numpy as np


# Migration Algorithm
def MA(population, fobj, VRmin, VRmax, Max_iter):
    N, dim = population.shape[0], population.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]
    lr = 0.1
    migration_ratio = 0.2

    best_fitness = float('inf')
    best_solution = np.zeros((dim, 1))

    Convergence_curve = np.zeros((Max_iter, 1))

    t = 0
    ct = time.time()
    for t in range(Max_iter):
        # Calculate fitness for each individual
        fitness = np.array([fobj(individual) for individual in population])

        # Sort the population based on fitness
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices]

        # Calculate the number of individuals to migrate
        num_migrate = int(migration_ratio * N)

        # Perform migration
        for i in range(num_migrate):
            source_individual = population[i]
            destination_index = np.random.randint(num_migrate, N)
            destination_individual = population[destination_index]

            # Update the destination individual based on the source individual
            population[destination_index] = destination_individual + lr * (source_individual - destination_individual)

            # Boundary checking
            population[destination_index] = np.clip(population[destination_index], lb, ub)

        Convergence_curve[t] = best_fitness
        t = t + 1
    best_fitness = Convergence_curve[Max_iter - 1][0]
    ct = time.time() - ct

    return best_fitness, Convergence_curve, best_solution, ct

