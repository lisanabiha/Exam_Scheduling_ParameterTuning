import random
import matplotlib.pyplot as plt

# Problem parameters
NUM_EXAMS = 10  # Number of exams
NUM_SLOTS = 5   # Number of available time slots
POPULATION_SIZE = 20
NUM_GENERATIONS = 50
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8

# Generate a dummy conflict matrix (example: 1 if two exams conflict, 0 otherwise)
CONFLICT_MATRIX = [[random.randint(0, 1) for _ in range(NUM_EXAMS)] for _ in range(NUM_EXAMS)]
for i in range(NUM_EXAMS):
    CONFLICT_MATRIX[i][i] = 0  # No self-conflict

# Functions remain the same
def generate_individual():
    """Generates a random individual (chromosome)."""
    return [random.randint(0, NUM_SLOTS - 1) for _ in range(NUM_EXAMS)]


def generate_population(size):
    """Generates the initial population."""
    return [generate_individual() for _ in range(size)]


def fitness(individual):
    """Calculates the fitness of an individual based on clash and fairness."""
    clashes = 0
    fairness_penalty = 0

    # Calculate clashes
    for i in range(NUM_EXAMS):
        for j in range(i + 1, NUM_EXAMS):
            if CONFLICT_MATRIX[i][j] and individual[i] == individual[j]:
                clashes += 1

    # Calculate fairness penalty (example: penalize too large time gaps)
    gaps = [abs(individual[i] - individual[j]) for i in range(NUM_EXAMS) for j in range(i + 1, NUM_EXAMS)]
    fairness_penalty = sum(gap > 2 for gap in gaps)

    return -(clashes + fairness_penalty)  # Negative because we minimize penalty


def selection(population):
    """Selects two parents using tournament selection."""
    tournament_size = 3
    selected = random.sample(population, tournament_size)
    selected.sort(key=fitness, reverse=True)
    return selected[:2]


def crossover(parent1, parent2):
    """
    Performs two-point crossover between two parents to produce two offspring.
    
    This method swaps a segment of genes (exam time slots) between two parent chromosomes.
    Two crossover points are selected randomly, and the portion of the genes 
    between these points is exchanged between the parents to create offspring.

    Args:
        parent1 (list): The first parent chromosome (exam schedule).
        parent2 (list): The second parent chromosome (exam schedule).

    Returns:
        tuple: Two offspring chromosomes (child1, child2) created after crossover.

    Process:
        1. Check if crossover happens based on CROSSOVER_RATE.
        2. Randomly select two points (point1 and point2) for crossover.
        3. Ensure point1 < point2 to define a valid segment for swapping.
        4. Exchange the segments between the two parents to produce children:
            - child1 inherits genes from parent1 outside the crossover range
              and genes from parent2 inside the crossover range.
            - child2 inherits genes from parent2 outside the crossover range
              and genes from parent1 inside the crossover range.
        5. Return the two resulting offspring.
    """
    # Perform crossover only if a random value is below the crossover rate
    if random.random() > CROSSOVER_RATE:
        return parent1, parent2  # No crossover, return parents unchanged

    # Select two random crossover points
    point1 = random.randint(0, NUM_EXAMS - 2)  # Ensure at least one gene to swap
    point2 = random.randint(point1 + 1, NUM_EXAMS - 1)  # Ensure point2 > point1

    # Create children by combining segments from both parents
    child1 = (
        parent1[:point1]        # Genes from parent1 before the first crossover point
        + parent2[point1:point2]  # Genes from parent2 within the crossover range
        + parent1[point2:]      # Genes from parent1 after the second crossover point
    )
    child2 = (
        parent2[:point1]        # Genes from parent2 before the first crossover point
        + parent1[point1:point2]  # Genes from parent1 within the crossover range
        + parent2[point2:]      # Genes from parent2 after the second crossover point
    )

    return child1, child2
    


def mutate(individual):
    """
    Mutates an individual chromosome to introduce genetic diversity.


    Args:
        individual (list): A single chromosome (exam schedule) represented as a list,
                           where each index corresponds to an exam, and the value 
                           is the assigned time slot.

    Returns:
        list: The mutated chromosome (exam schedule).

    Mutation Process:
        1. Check if mutation occurs based on MUTATION_RATE.
           - A random number is generated, and if it is below the mutation rate,
             mutation is performed.
        2. Randomly select an exam (gene) in the chromosome to mutate.
           - The index of the gene to mutate is chosen randomly using `random.randint`.
        3. Assign a new random time slot to the selected exam.
           - Replace the value at the chosen index with a new value, randomly chosen
             between 0 and NUM_SLOTS - 1, representing the available time slots.
        4. Return the modified individual. If no mutation occurs, the original
           chromosome is returned unchanged.

    Key Parameters:
        - MUTATION_RATE: The probability of mutation occurring for a given chromosome.
        - NUM_EXAMS: Total number of exams (length of the chromosome).
        - NUM_SLOTS: Total number of available time slots.

    Notes:
        - Mutation introduces randomness to avoid premature convergence.
        - Excessive mutation can disrupt good solutions, so the MUTATION_RATE
          should be tuned carefully.
        - Mutations are typically small changes (e.g., reassigning a single exam
          to a different slot) to preserve the structure of the solution.
    """
    # Perform mutation with a probability defined by MUTATION_RATE
    if random.random() < MUTATION_RATE:
        # Randomly select an exam (gene) in the chromosome to mutate
        exam = random.randint(0, NUM_EXAMS - 1)
        
        # Assign a new random time slot to the selected exam
        individual[exam] = random.randint(0, NUM_SLOTS - 1)
    
    # Return the potentially mutated individual
    return individual


def genetic_algorithm():
    """Main genetic algorithm for exam scheduling."""
    population = generate_population(POPULATION_SIZE)
    best_fitness_over_time = []  # Track best fitness per generation

    for generation in range(NUM_GENERATIONS):
        # Evaluate population fitness
        population.sort(key=fitness, reverse=True)

        # Log best fitness
        best_fitness = fitness(population[0])
        best_fitness_over_time.append(best_fitness)
        print(f"Generation {generation}, Best Fitness: {best_fitness}")

        # Create new population
        new_population = population[:2]  # Elitism: retain top 2 individuals

        while len(new_population) < POPULATION_SIZE:
            # Selection
            parent1, parent2 = selection(population)

            # Crossover
            child1, child2 = crossover(parent1, parent2)

            # Mutation
            new_population.append(mutate(child1))
            if len(new_population) < POPULATION_SIZE:
                new_population.append(mutate(child2))

        population = new_population

    # Return the best solution and the fitness history
    population.sort(key=fitness, reverse=True)
    return population[0], best_fitness_over_time


# Run the genetic algorithm
best_solution, fitness_history = genetic_algorithm()
print("Best Exam Schedule:", best_solution)
print("Fitness:", fitness(best_solution))

# Plot the fitness history
plt.plot(fitness_history, label="Best Fitness")
plt.title("Fitness Progress Over Generations")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.grid()
plt.show()