#!/usr/bin/env python
from City import City
import random
import matplotlib.pyplot as plt

__author__ = "will"


def read_in_cities(file_name):
    f = open(file_name, "r")
    list_of_cities = []
    for row in f:
        comps = row.rstrip("\n")
        comps = comps.split("\t")
        list_of_cities.append(City(int(comps[0]), int(comps[1])))
    f.close()
    return list_of_cities


def tour_length(tour):
    """
    :type tour: list of City objects
    :rtype calculates the length (or cost) of the tour (not # of elements in list)
    """
    length_so_far = 0.0
    for i in range(0, len(tour) - 2):
        length_so_far += tour[i].distance_to(tour[i + 1])
    return length_so_far


def generate_random_tour(list_of_cities):
    """
    Used to generate a random tour for the initial population creation.
    :param list_of_cities: list of City objects
    :rtype a random tour of cities where tour[0] == tour[len(tour)-1].
        return type:list of City objects
    """
    tour = list_of_cities[1:]
    random.shuffle(tour)
    tour.insert(0, list_of_cities[0])
    tour.append(list_of_cities[0])
    return tour


def generate_population(list_of_cities):
    """
    Generates the initial population of the genetic algorithm
    :param list_of_cities: list of City objects
    :rtype list of a list of City objects.
        a population is a list of tours, which are lists of City objects
    """
    pop_size = 50
    pop = []
    for i in range(0, pop_size):
        random_tour = generate_random_tour(list_of_cities)
        pop.append(random_tour)
    return pop


def evolve_population(pop, mutation_rate):
    """
    :param pop: The population (list of tours, which are lists of City objects).
    :rtype a population.  Type: list of tours, which are lists of City objects.
    """
    # must create parents
    # must call and write crossover operator
    population_size = len(pop)
    survivors = []  # the generation undergoes "natural selection"
    tour_costs = []

    for i in range(0, population_size):
        tour_costs.append(1.0 / tour_length(pop[i]))

    current_value = 0
    probability_ranges = []
    for i in range(0, len(tour_costs)):
        range_tuple = (current_value, current_value + tour_costs[i])
        probability_ranges.append(range_tuple)
        current_value += tour_costs[i]

    # select individuals from population
    assert sum(tour_costs) == current_value
    for i in range(0, population_size):
        selection_number = random.random() * current_value  # number for semi-random selection
        index = 0
        for prob_range in probability_ranges:
            if prob_range[0] <= selection_number <= prob_range[1]:
                survivors.append(pop[index])
            index += 1

    assert len(survivors) == len(pop)
    survivors.sort(key=tour_length)

    next_generation = []
    for index in range(0, population_size, 2):
        # index is parent1's index, index+1 is parent2's index
        parent1 = survivors[index]
        parent2 = survivors[index+1]
        # rand = random.random()
        '''
        if rand > .5:
            child1 = cycle_crossover(parent1, parent2)
            child2 = cycle_crossover(parent2, parent1)
        else:
            child1 = crossover(tour1=parent1, tour2=parent2)
            child2 = crossover(tour1=parent1, tour2=parent2)
        '''
        #child1 = cycle_crossover(parent1, parent2)
        #child2 = cycle_crossover(parent2, parent1)

        child1 = crossover(tour1=parent1, tour2=parent2)
        child2 = crossover(tour1=parent1, tour2=parent2)
        # '''
        if random.random() < mutation_rate:
                mutate_tour(child1)

        if random.random() < mutation_rate:
                mutate_tour(child2)
        # '''
        next_generation.append(child1)
        next_generation.append(child2)

    return next_generation


def cycle_crossover(parent1, parent2):

    def index_of(the_city, tour):
        for index in range(1, len(tour) - 1):
            if tour[index] == the_city:
                return index
        return -1
    len_of_tour = len(parent1)
    child = [None]*len_of_tour
    random_index = random.randint(1, len_of_tour - 1)

    value_at_initial_index_of_parent1 = parent1[random_index]
    value_at_index_of_parent1 = parent1[random_index]
    child[random_index] = value_at_index_of_parent1
    value_at_index_of_parent2 = parent2[random_index]
    set_of_indices = {random_index}
    while value_at_index_of_parent2 != value_at_initial_index_of_parent1:
        #print("iteration")
        index_in_parent1 = index_of(value_at_index_of_parent2, parent1)
        child[index_in_parent1] = parent1[index_in_parent1]
        value_at_index_of_parent2 = parent2[index_in_parent1]
        set_of_indices.add(index_in_parent1)

    for i in range(1, len_of_tour - 1):
        if i not in set_of_indices:
            child[i] = parent2[i]

    child[0] = parent1[0]
    child[len_of_tour - 1] = parent1[len_of_tour - 1]
    return child


def test_cycle_crossover():
    tour1 = [0, 1, 2, 3, 4, 5, 6, 7, 0]
    tour2 = [0, 5, 4, 6, 7, 2, 3, 1, 0]
    child = cycle_crossover(tour1, tour2)
    # child should be [1, 2, 6, 4, 5, 3, 7]
    print(child)


def crossover(tour1, tour2):
    """
    :param tour1:
    :param tour2:
    :rtype
    """
    len_of_tours = len(tour1)
    start_pos = random.randint(1, len_of_tours - 1)
    end_pos = random.randint(1, len_of_tours - 1)

    while start_pos == end_pos:
        end_pos = random.randint(1, len_of_tours - 1)
    temp = start_pos
    if start_pos > end_pos:
        start_pos = end_pos
        end_pos = temp
    # print("start pos: ", start_pos)
    # print("end pos: ", end_pos)
    # now I have gotten the range that will be the crossover section
    # this is the section that will be taken from tour 1 and given to the child
    child_tour = [None] * len_of_tours
    child_tour[0] = tour1[0]
    child_tour[-1] = tour1[-1]
    for i in range(start_pos, end_pos + 1):
        child_tour[i] = tour1[i]

    tour2_index = 1
    # child_index = 1
    for i in range(tour2_index, len_of_tours - 1):  # loop through tour2
        child_contains_city_of_tour2_at_i = False
        for k in range(1, len_of_tours - 1):
            if child_tour[k] is None:
                continue
            if child_tour[k] is tour2[i]:# I CHANGED BECAUSE OF DUPLICATE CITY ERROR; was giving error with tour[i] == tour2[i]
                child_contains_city_of_tour2_at_i = True
                break
        if not child_contains_city_of_tour2_at_i:
            for j in range(1, len_of_tours - 1):
                if child_tour[j] is None:
                    child_tour[j] = tour2[i]
                    break
    return child_tour


def mutate_tour(tour):
    number_of_cities_in_tour = len(tour)
    i = random.randint(1, number_of_cities_in_tour - 2)
    k = random.randint(1, number_of_cities_in_tour - 2)
    while i == k:
        k = random.randint(1, number_of_cities_in_tour - 1)

    temp = tour[i]
    tour[i] = tour[k]
    tour[k] = temp


def tour_has_None(tour):
    for index in range(0, len(tour)):
        if tour[index] is None:
            return True, index
    return False, -1
#test_cycle_crossover()


cities_list = read_in_cities("cities.txt")
#cities_list = cities_list[:] # breaks

MUTATION_RATE = 0.05

cost_over_time = []

population = generate_population(cities_list)
the_best = population[0]
len_pop = len(population)
for i in range(1, len_pop - 1):
    if tour_length(population[i]) < tour_length(the_best):
        the_best = population[i]
iterations = 0
limit = 250

while iterations < limit:
    # print("iteration: ", iterations)
    population = evolve_population(pop=population, mutation_rate=MUTATION_RATE)

    if tour_length(population[0]) < tour_length(the_best):
        the_best = population[0]
    iterations += 1
    cost_over_time.append(tour_length(population[0]))
    print(iterations)

for city in the_best:
    print(city)
print("best tour cost: ", tour_length(the_best))

plt.plot(cost_over_time)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

