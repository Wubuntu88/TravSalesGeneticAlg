#!/usr/bin/env python
from City import City
import random
__author__ = "will"


def read_in_cities(file_name):
    f = open(file_name, "r")
    list_of_cities = []
    for row in f:
        comps = row.rstrip("\n")
        comps = comps.split("\t")
        list_of_cities.append(City(comps[0], comps[1]))
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


def evolve_population(pop):
    """
    :param: The population (list of tours, which are lists of City objects).
    :rtype: a population.  Type: list of tours, which are lists of City objects.
    """
    # must create parents
    # must call and write crossover operator
    return None


def crossover(tour1, tour2):
    """

    :param tour1:
    :param tour2:
    :rtype
    """
    len_of_tours = len(tour1)
    start_pos = random.randint(1, len_of_tours - 1)
    end_pos = random.randint(1, len_of_tours - 1)
    while start_pos != end_pos:
        end_pos = random.randint(1, len_of_tours - 1)
    temp = start_pos
    if start_pos > end_pos:
        start_pos = end_pos
        end_pos = temp
    # now I have gotten the range that will be the crossover section
    # this is the section that will be taken from tour 1 and given to the child
    child_tour = [None] * len_of_tours
    for i in range(start_pos, end_pos + 1):
        child_tour[i] = tour1[i]
    index = 1
    for i in range(index, len_of_tours - 1):
        if not child_tour.__contains__(tour2[i]):
            child_tour[index] = tour2[i]
            index += 1



cities_list = read_in_cities("cities.txt")
city = City(6, 21)
if city in cities_list:
    print("contains")

population = generate_population(cities_list)
print(len(population))

