import random
import math

# Distance matrix between cities (example: 5 cities)
dist_matrix = [
    [0, 2, 9, 10, 7],
    [1, 0, 6, 4, 3],
    [15, 7, 0, 8, 3],
    [6, 3, 12, 0, 11],
    [10, 4, 8, 5, 0]
]

num_cities = len(dist_matrix)
num_nests = 6             # number of host nests (population)
pa = 0.25                 # probability to abandon a nest
max_iter = 100            # maximum iterations

# --- Helper Functions ---
def route_length(route):
    """Calculate total distance of a route"""
    return sum(dist_matrix[route[i]][route[(i + 1) % num_cities]] for i in range(num_cities))

def random_route():
    """Generate a random route (a permutation of cities)"""
    route = list(range(num_cities))
    random.shuffle(route)
    return route

def levy_flight(route):
    """Simple 'Levy flight' = swap two random cities"""
    a, b = random.sample(range(num_cities), 2)
    route[a], route[b] = route[b], route[a]
    return route

# --- Initialization ---
nests = [random_route() for _ in range(num_nests)]
fitness = [route_length(r) for r in nests]
best_route = nests[fitness.index(min(fitness))]
best_fitness = min(fitness)

# --- Main Loop ---
for t in range(max_iter):
    for i in range(num_nests):
        new_route = levy_flight(nests[i][:])  # make small random change
        new_fit = route_length(new_route)

        # If new route is better, replace it
        if new_fit < fitness[i]:
            nests[i] = new_route
            fitness[i] = new_fit

    # Abandon some nests (worst ones) with probability pa
    for i in range(num_nests):
        if random.random() < pa:
            nests[i] = random_route()
            fitness[i] = route_length(nests[i])

    # Update best
    current_best = min(fitness)
    if current_best < best_fitness:
        best_fitness = current_best
        best_route = nests[fitness.index(current_best)]

print("Best Route Found:", best_route)
print("Best Distance:", best_fitness)


O/p:
Best Route Found: [3, 1, 0, 2, 4]
Best Distance: 21
