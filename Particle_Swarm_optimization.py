
import random

# Objective Function (maximize f(x))
def f(x):
    return -x**2 + 5*x + 20    

# Particle class
class Particle:
    def __init__(self, x_init):
        self.position = x_init              # Current position
        self.velocity = 0                   # Current velocity
        self.best_position = x_init         # Personal best position
        self.best_value = f(x_init)         # Personal best fitness

# PSO Algorithm
def PSO(num_particles=3, w=0.5, c1=1.0, c2=1.0, iterations=10):
    # Initialize swarm with random positions 
    swarm = [Particle(random.uniform(0, 6)) for _ in range(num_particles)]

    # Find initial global best
    gbest = max(swarm, key=lambda p: p.best_value)
    gbest_position = gbest.best_position
    gbest_value = gbest.best_value

    print(f"Initial Global Best: x = {gbest_position:.4f}, f(x) = {gbest_value:.4f}")

    # Main PSO loop
    for t in range(iterations):
        for p in swarm:
            r1, r2 = random.random(), random.random()

            # Update velocity
            p.velocity = (
                w * p.velocity
                + c1 * r1 * (p.best_position - p.position)
                + c2 * r2 * (gbest_position - p.position)
            )

            # Update position
            p.position += p.velocity

            # Evaluate fitness
            fitness = f(p.position)

            # Update personal best
            if fitness > p.best_value:
                p.best_value = fitness
                p.best_position = p.position

        # Update global best
        current_best = max(swarm, key=lambda p: p.best_value)
        if current_best.best_value > gbest_value:
            gbest_value = current_best.best_value
            gbest_position = current_best.best_position

        print(f"It {t+1}: gbest = {gbest_position:.4f}, f(x) = {gbest_value:.4f}")

    print("\n Result:")
    print(f"Best position: {gbest_position:.4f}, Best value: {gbest_value:.4f}")

if __name__ == "__main__":
    PSO(num_particles=3, iterations=5)


O/p:
Initial Global Best: x = 2.2073, f(x) = 26.1643
It 1: gbest = 2.2148, f(x) = 26.1687
It 2: gbest = 2.2148, f(x) = 26.1687
It 3: gbest = 2.2148, f(x) = 26.1687
It 4: gbest = 2.3818, f(x) = 26.2360
It 5: gbest = 2.3818, f(x) = 26.2360

 Result:
Best position: 2.3818, Best value: 26.2360
