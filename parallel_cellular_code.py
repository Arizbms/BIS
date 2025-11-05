import numpy as np
import random

def f(x):
    return x**2

def PCA(f, n=10, MaxIter=20, lb=-5, ub=5):
    cells = np.random.uniform(lb, ub, n)
    fitness = np.array([f(x) for x in cells])
    best_cell = cells[np.argmin(fitness)]

    for t in range(MaxIter):
        new_cells = cells.copy()
        for i in range(n):
            left = cells[(i - 1) % n]
            right = cells[(i + 1) % n]
            neighbors = [left, cells[i], right]

            p1, p2 = random.sample(neighbors, 2)
            child = (p1 + p2) / 2 + np.random.uniform(-0.1, 0.1)
            child = np.clip(child, lb, ub)

            if f(child) < f(cells[i]):
                new_cells[i] = child

        cells = new_cells.copy()
        fitness = np.array([f(x) for x in cells])
        current_best = cells[np.argmin(fitness)]
        if f(current_best) < f(best_cell):
            best_cell = current_best

    print(f"Best Solution: {best_cell:.8f}")
    print(f"Fitness Value: {f(best_cell):.8f}")

if __name__ == "__main__":
    PCA(f, n=10, MaxIter=30, lb=-5, ub=5)


O/p:
Best Solution: -0.00010070
Fitness Value: 0.00000001
