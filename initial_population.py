from scipy.stats.qmc import LatinHypercube, scale
import pandas as pd
import numpy as np
import random

def init_population_syn(ps, bounds):
    seed=42
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    n= bounds.shape[0]
    # if seed is not None:
    #     np.random.seed(SEED)
    sampler = LatinHypercube(d=n, seed=seed)
    sample = sampler.random(n=ps)  
    scaled_sample = scale(sample, bounds[:, 0], bounds[:, 1])  # Scale to bounds
    population_list = [list(member) + [None] for member in scaled_sample]
    return population_list