import numpy as np

# Function to minimize: f(x) = x^2 + 3x + 5
def objective_function(x):
    x = x[0]
    return x**2 + 3 * x + 5

# Gene Expression Algorithm
def gene_expression_algorithm(
    func, pop_size=20, mutation_rate=0.1,
    crossover_rate=0.8, generations=100, bounds=(-10, 10)
):
    num_genes = 1  # Only one variable: x

    # Initialize population with random values
    population = np.random.uniform(bounds[0], bounds[1], size=(pop_size, num_genes))

    def evaluate(pop):
        return np.array([func(ind) for ind in pop])

    def select(fitness_vals):
        inv_fitness = 1 / (fitness_vals + 1e-6)
        probs = inv_fitness / np.sum(inv_fitness)
        selected_indices = np.random.choice(np.arange(pop_size), size=pop_size, p=probs)
        return population[selected_indices]

    def crossover(parent1, parent2):
        if np.random.rand() < crossover_rate:
            # Simple averaging crossover
            child = 0.5 * (parent1 + parent2)
            return child.copy(), child.copy()
        return parent1.copy(), parent2.copy()

    def mutate(individual):
        for i in range(num_genes):
            if np.random.rand() < mutation_rate:
                individual[i] += np.random.uniform(-1, 1) * 0.5
        # Keep within bounds
        return np.clip(individual, bounds[0], bounds[1])

    best_solution = None
    best_value = float("inf")

    for gen in range(generations):
        fitness = evaluate(population)
        gen_best_idx = np.argmin(fitness)

        # Track best result
        if fitness[gen_best_idx] < best_value:
            best_value = fitness[gen_best_idx]
            best_solution = population[gen_best_idx]

        # Print progress
        if gen % 10 == 0 or gen == generations - 1:
            print(f"Gen {gen}: x = {best_solution[0]:.4f}, f(x) = {best_value:.4f}")

        # Selection
        selected = select(fitness)

        # Create next population
        next_gen = []
        for i in range(0, pop_size, 2):
            p1 = selected[i]
            p2 = selected[(i + 1) % pop_size]
            c1, c2 = crossover(p1, p2)
            next_gen.append(mutate(c1))
            next_gen.append(mutate(c2))

        population = np.array(next_gen)

    return best_solution[0], best_value


# Run the algorithm
if __name__ == "__main__":
    best_x, best_fx = gene_expression_algorithm(objective_function)
    print("\nBest solution found:")
    print(f"x = {best_x:.4f}, f(x) = {best_fx:.4f}")
