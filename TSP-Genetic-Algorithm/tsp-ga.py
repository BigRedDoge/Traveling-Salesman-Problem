import random
import math
import matplotlib.pyplot as plt
import argparse


# get cities info
# city name, x, y
def get_cities(path):
    with open(path) as f:
        cities = [[city_node[0], float(city_node[1]), float(city_node[2])] for city_node in (line.split() for line in f)]
    return cities


# calculating distance of the cities
def calculate_distance(graph, cities):
    total_sum = 0
    for index in range(len(cities)):
        try:
            city_a = cities[index][1]
            city_b = cities[index + 1][1]
            distance = graph[city_a][city_b]
        except IndexError:
            city_a = cities[-1][1]
            city_b = cities[0][1]
            distance = graph[city_a][city_b]
        total_sum += distance
    return total_sum


# selecting the population
def select_population(graph, names, size):
    population = []

    for _ in range(size):
        cities = list(names.items())
        random.shuffle(cities)
        distance = calculate_distance(graph, cities)
        population.append([distance, cities])
    fitest = sorted(population)[0]

    return population, fitest


# the genetic algorithm
def genetic_algorithm(population, graph, parent_selection_size, mutation_rate, target, max_gen, mutation_max):
    len_cities = len(graph)
    for gen_number in range(1, max_gen + 1):
        new_population = []

        # selecting two of the best options we have
        sorted_pop = sorted(population)
        new_population.append(sorted_pop[0])
        new_population.append(sorted_pop[1])

        for _ in range((len(population) - 2) // 2):
            # crossover
            # selecting parents from best of random selection of size parent_selection_size
            parent_chromosome1 = sorted(random.choices(population, k=parent_selection_size))[0]
            parent_chromosome2 = sorted(random.choices(population, k=parent_selection_size))[0]

            # select a random crossover point
            crossover_point = random.randint(0, len_cities - 1)

            # split the chromosomes at crossover point and combine them
            child_chromosome1 = parent_chromosome1[1][0:crossover_point]
            for gene in parent_chromosome2[1]:
                if gene not in child_chromosome1:
                    child_chromosome1.append(gene)

            child_chromosome2 = parent_chromosome2[1][0:crossover_point]
            for gene in parent_chromosome1[1]:
                if gene not in child_chromosome2:
                    child_chromosome2.append(gene)

            # mutation
            if random.random() < mutation_rate:
                # random amount of mutations between 1 and mutation_max
                for _ in range(random.randint(1, mutation_max)):
                    point1 = random.randint(0, len_cities - 1)
                    point2 = random.randint(0, len_cities - 1)
                    child_chromosome1[point1], child_chromosome1[point2] = (child_chromosome1[point2], child_chromosome1[point1])

                    point1 = random.randint(0, len_cities - 1)
                    point2 = random.randint(0, len_cities - 1)
                    child_chromosome2[point1], child_chromosome2[point2] = (child_chromosome2[point2], child_chromosome2[point1])

            new_population.append([calculate_distance(graph, child_chromosome1), child_chromosome1])
            new_population.append([calculate_distance(graph, child_chromosome2), child_chromosome2])

        population = new_population

        sorted_pop = sorted(population)
        if gen_number % 10 == 0 or gen_number == 1:
            print("Generation " + str(gen_number) + " - Distance: " + str(sorted_pop[0][0]))

        if sorted_pop[0][0] < target:
            break

    answer = sorted_pop[0]

    return answer, gen_number


# draw cities and answer map
def draw_map(draw, answer):
    for city, coord in draw.items():
        if city == answer[1][0][0]:
            plt.plot(coord[0], coord[1], "go")
            first = False
        else:
            plt.plot(coord[0], coord[1], "ro")
        plt.annotate(city, (coord[0], coord[1]))

    for i in range(len(answer[1])):
        try:
            first = answer[1][i][0]
            second = answer[1][i + 1][0]
            plt.plot([draw[first][0], draw[second][0]], [draw[first][1], draw[second][1]], "gray")
        except IndexError:
            first = answer[1][0][0]
            second = answer[1][-1][0]
            plt.plot([draw[first][0], draw[second][0]], [draw[first][1], draw[second][1]], "gray")

    plt.show()


def main(args, graph, names, draw):
    first_population, first_fitest = select_population(graph, names, args.population_size)
    print("Fittest chromosome before training: Distance - ", first_fitest)
    print("population: ", first_population)
    answer, gen_number = genetic_algorithm(
        first_population,
        graph,
        args.parent_selection_size,
        args.mutation_rate,
        args.target,
        args.max_gen,
        args.mutation_max
    )

    print("Fittest chromosome distance before training:", first_fitest[0])
    print("Fittest chromosome distance after training:", answer[0])
    print("Target distance: ", args.target)
    print("Answer chromosome: ", [item[0] for item in answer[1]])

    draw_map(draw, answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSP using Genetic Algorithm")
    parser.add_argument("--population_size", type=int, default=5)
    parser.add_argument("--parent_selection_size", type=int, default=2)
    parser.add_argument("--mutation_rate", type=float, default=0.15)
    parser.add_argument("--target", type=float, default=20.0)
    parser.add_argument("--max_gen", type=int, default=50)
    parser.add_argument("--mutation_max", type=int, default=3)
    args = parser.parse_args()
    for arg in  vars(args):
        if vars(args)[arg] < 0:
            raise ValueError(f"{arg} must be greater than 0")
        

    graph = [[0, 16, 11, 6],
             [8, 0, 13, 16],
             [4, 7, 0, 9],
             [5, 12, 2, 0]]
    names = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3
    }
    
    draw = {key: (random.randint(0, 100), random.randint(0, 100)) for key in names.keys()}

    main(args, graph, names, draw)

    # pip install -r requirements.txt
    # python tsp-ga.py --path "cities.txt"