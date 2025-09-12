import random
import math
import numpy as np

POP_SIZE = 6         # number of chromosomes
CHROMO_LEN = 4       # bits (represents 0–15)
MAX_GEN = 15         # generations
Pc = 0.8             # crossover probability
Pm = 0.1             # mutation probability
kappa = 1.0          # scaling factor
eps = 1e-6           # small number for stability

# -------------------------
# HELPERS
# -------------------------
def decode(chromosome):
    """Convert binary list to integer"""
    return int("".join(map(str, chromosome)), 2)

def objective(x):
    """Objective function g(x) = x^2"""
    return x**2

def fitness(pop):
    """Compute selection probabilities using Boltzmann scaling"""
    raw_scores = [objective(decode(chromo)) for chromo in pop]

    # scale scores (avoid negative or zero)
    mu = np.mean(raw_scores)
    sigma = np.std(raw_scores)
    scaled = [max(eps, s - (mu - kappa * sigma)) for s in raw_scores]

    # Boltzmann probabilities
    T = np.median(scaled)
    probs = [math.exp(-f / T) for f in scaled]
    total = sum(probs)
    probs = [p / total for p in probs]

    return raw_scores, probs

def roulette_wheel(pop, probs):
    """Select one chromosome based on probability"""
    r = random.random()
    cumulative = 0
    for chromo, p in zip(pop, probs):
        cumulative += p
        if r <= cumulative:
            return chromo.copy()
    return pop[-1].copy()

def crossover(p1, p2):
    """Single-point crossover"""
    if random.random() < Pc:
        point = random.randint(1, CHROMO_LEN - 1)
        c1 = p1[:point] + p2[point:]
        c2 = p2[:point] + p1[point:]
        return c1, c2
    return p1.copy(), p2.copy()

def mutate(chromo):
    """Bit-flip mutation"""
    for i in range(CHROMO_LEN):
        if random.random() < Pm:
            chromo[i] = 1 - chromo[i]
    return chromo

# MAIN ALGORITHM

def genetic_algorithm():
    # Step 1: Initialize population
    population = [[random.randint(0,1) for _ in range(CHROMO_LEN)] for _ in range(POP_SIZE)]
    
    best_solution = None
    best_fitness = float("inf")   # lower is better
    
    for gen in range(1, MAX_GEN + 1):
        raw, probs = fitness(population)
        
        # track best
        for chromo, fit in zip(population, raw):
            if fit < best_fitness:
                best_solution = chromo.copy()
                best_fitness = fit
        
        print(f"Gen {gen}:")
        for i, chromo in enumerate(population):
            print(f"  Chromo {chromo} -> x={decode(chromo)} Raw={raw[i]} Prob={probs[i]:.3f}")
        print(f"  Best so far: x={decode(best_solution)} with fitness {best_fitness:.2f}\n")
        
        # Step 2: New population
        new_pop = [best_solution.copy()]  # Elitism
        
        while len(new_pop) < POP_SIZE:
            p1 = roulette_wheel(population, probs)
            p2 = roulette_wheel(population, probs)
            
            c1, c2 = crossover(p1, p2)
            new_pop.append(mutate(c1))
            if len(new_pop) < POP_SIZE:
                new_pop.append(mutate(c2))
        
        population = new_pop
    
    print("Final Best Solution:", best_solution, "x=", decode(best_solution), "fitness=", best_fitness)


genetic_algorithm()
