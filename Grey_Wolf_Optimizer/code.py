import numpy as np

def f(x):
    return np.sum(x**2)

def GWO(f, n=5, MaxIter=20, dim=1, lb=-5, ub=5):
    X = np.random.uniform(lb, ub, (n, dim))
    fitness = np.array([f(x) for x in X])
    idx = np.argsort(fitness)
    Alpha, Beta, Delta = X[idx[0]], X[idx[1]], X[idx[2]]

    for t in range(MaxIter):
        a = 2 - 2 * (t / MaxIter)
        for i in range(n):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1, C1 = 2*a*r1 - a, 2*r2
                D_alpha = abs(C1 * Alpha[j] - X[i][j])
                X1 = Alpha[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2, C2 = 2*a*r1 - a, 2*r2
                D_beta = abs(C2 * Beta[j] - X[i][j])
                X2 = Beta[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3, C3 = 2*a*r1 - a, 2*r2
                D_delta = abs(C3 * Delta[j] - X[i][j])
                X3 = Delta[j] - A3 * D_delta

                X[i][j] = (X1 + X2 + X3) / 3
            X[i] = np.clip(X[i], lb, ub)

        fitness = np.array([f(x) for x in X])
        idx = np.argsort(fitness)
        Alpha, Beta, Delta = X[idx[0]], X[idx[1]], X[idx[2]]

    best_solution = Alpha.flatten()
    best_value = float(f(Alpha))
    print("Best Solution:", best_solution)
    print("Fitness Value:", best_value)
    return best_solution, best_value

if __name__ == "__main__":
    GWO(f, n=5, MaxIter=20, lb=-5, ub=5)




O/p:
Best Solution: [0.00571139]
Fitness Value: 3.261993780884954e-05
