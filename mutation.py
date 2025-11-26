from random import randint as rnd
import random
from random import shuffle
import numpy as np
import pandas as pd

def mutation_syn(population_list, n, ps,nnew, ncross, nmut, bounds, f_n, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # choosen_ones = list(range(len(population_list) - nnew, len(population_list)))
    choosen_ones = list(range(len(population_list)))

    np.random.shuffle(choosen_ones)
    choosen_ones = choosen_ones[:nmut]

    for idx in choosen_ones:
        cell = np.random.randint(0, n)
        sigma = np.random.uniform(bounds[cell, 0], bounds[cell, 1])
        population_list[idx][cell] = sigma

    return population_list