# -*- coding: utf-8 -*-

import numpy as np

import array
import random

from deap import base
from deap import creator
from deap import tools

FirstCall = True


def uniform(bounds):
    return [random.uniform(b[0], b[1]) for b in bounds]


def NSGAII(NObj, objective, individual, create_random, crossover, mutation , seed=None, NGEN=30, MU=100, CXPB=0.9):
    random.seed(seed)

    global FirstCall
    if FirstCall:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,)*NObj)
        creator.create("Individual", individual,
                       fitness=creator.FitnessMin)
        FirstCall = False
    toolbox = base.Toolbox()

    #    NDIM = len(pbounds)


    toolbox.register("individual", create_random, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", objective)

    toolbox.register("mate",
                     crossover)

    toolbox.register("mutate",
                     mutation)
                     
    toolbox.register("select", tools.selNSGA2)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)

    # Begin the generational process
    for gen in range(1, NGEN):
        print("+", end="", flush=True)
        if gen == 49:
            print()
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        # print(logbook.stream)

    front = np.array([ind.fitness.values for ind in pop])

    return pop, logbook, front
