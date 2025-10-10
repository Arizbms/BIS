import numpy as np

# 4 cities coordinates
cities = [(0,0), (1,0), (1,1), (0,1)]

# Calculate distance between two cities
def dist(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# Distance matrix
n = len(cities)
distance = [[dist(cities[i], cities[j]) for j in range(n)] for i in range(n)]

# Parameters
pheromone = [[1 for _ in range(n)] for _ in range(n)]  # initial pheromone
alpha = 1     # pheromone importance
beta = 2      # distance importance
evaporation = 0.5
Q = 10        # pheromone deposit amount
ants = 3
iterations = 5

best_path = None
best_length = float('inf')

for _ in range(iterations):
    paths = []
    lengths = []

    for _ in range(ants):
        visited = [0]  # start at city 0 for simplicity
        while len(visited) < n:
            current = visited[-1]
            probs = []
            for city in range(n):
                if city not in visited:
                    # Calculate attractiveness = pheromone^alpha * (1/distance)^beta
                    tau = pheromone[current][city] ** alpha
                    eta = (1 / distance[current][city]) ** beta
                    probs.append((city, tau * eta))
            # Normalize probabilities
            total = sum(p[1] for p in probs)
            probs = [(c, p/total) for c, p in probs]

            # Choose next city based on probabilities
            r = np.random.random()
            cumulative = 0
            for city, prob in probs:
                cumulative += prob
                if r <= cumulative:
                    visited.append(city)
                    break

        visited.append(visited[0])  # return to start

        # Calculate length of the path
        length = sum(distance[visited[i]][visited[i+1]] for i in range(n))
        paths.append(visited)
        lengths.append(length)

        if length < best_length:
            best_length = length
            best_path = visited

    # Evaporate pheromones
    for i in range(n):
        for j in range(n):
            pheromone[i][j] *= (1 - evaporation)

    # Deposit new pheromones
    for path, length in zip(paths, lengths):
        for i in range(n):
            pheromone[path[i]][path[i+1]] += Q / length

print("Best path:", best_path)
print("Best length:", best_length)

O/p:
Best path: [0, 3, 2, 1, 0]
Best length: 4.0

O/p
