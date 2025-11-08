import random, math, numpy as np

POP_SIZE = 6
CHROMO_LEN = 4
MAX_GEN = 15
Pc = 0.8
Pm = 0.1
kappa = 1.0
eps = 1e-6

def decode(chromosome):
    return int("".join(map(str, chromosome)), 2)

def objective(x):
    return x**2

def fitness(pop):
    raw_scores = [objective(decode(chromo)) for chromo in pop]
    mu, sigma = np.mean(raw_scores), np.std(raw_scores)
    scaled = [max(eps, s - (mu - kappa * sigma)) for s in raw_scores]
    T = np.median(scaled)
    probs = [math.exp(-f / T) for f in scaled]
    total = sum(probs)
    return raw_scores, [p / total for p in probs]

def roulette_wheel(pop, probs):
    r, cumulative = random.random(), 0
    for chromo, p in zip(pop, probs):
        cumulative += p
        if r <= cumulative:
            return chromo.copy()
    return pop[-1].copy()

def crossover(p1, p2):
    if random.random() < Pc:
        point = random.randint(1, CHROMO_LEN - 1)
        return p1[:point] + p2[point:], p2[:point] + p1[point:]
    return p1.copy(), p2.copy()

def mutate(chromo):
    for i in range(CHROMO_LEN):
        if random.random() < Pm:
            chromo[i] = 1 - chromo[i]
    return chromo

def genetic_algorithm():
    population = [[random.randint(0,1) for _ in range(CHROMO_LEN)] for _ in range(POP_SIZE)]
    best_solution, best_fitness = None, float("inf")

    for gen in range(1, MAX_GEN + 1):
        raw, probs = fitness(population)
        for chromo, fit in zip(population, raw):
            if fit < best_fitness:
                best_solution, best_fitness = chromo.copy(), fit
        print(f"Gen {gen}: Best x={decode(best_solution)} f={best_fitness:.2f}")
        new_pop = [best_solution.copy()]
        while len(new_pop) < POP_SIZE:
            p1, p2 = roulette_wheel(population, probs), roulette_wheel(population, probs)
            c1, c2 = crossover(p1, p2)
            new_pop.append(mutate(c1))
            if len(new_pop) < POP_SIZE:
                new_pop.append(mutate(c2))
        population = new_pop

    print(f"\nFinal Best: x={decode(best_solution)} f={best_fitness:.2f}")

genetic_algorithm()


O/p:
Gen 1: Best x=2 f=4.00
Gen 2: Best x=2 f=4.00
Gen 3: Best x=2 f=4.00
Gen 4: Best x=2 f=4.00
Gen 5: Best x=2 f=4.00
Gen 6: Best x=2 f=4.00
Gen 7: Best x=2 f=4.00
Gen 8: Best x=0 f=0.00
Gen 9: Best x=0 f=0.00
Gen 10: Best x=0 f=0.00
Gen 11: Best x=0 f=0.00
Gen 12: Best x=0 f=0.00
Gen 13: Best x=0 f=0.00
Gen 14: Best x=0 f=0.00
Gen 15: Best x=0 f=0.00

Final Best: x=0 f=0.00
