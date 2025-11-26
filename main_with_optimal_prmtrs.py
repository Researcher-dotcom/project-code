from matplotlib import pyplot as plt
# from initial_population import init_population_syn
import initial_population
import matplotlib.pyplot as plt
import numpy as np
import cross_over
import mutation
import fitness
import pareto_fitness_diversity
from partition_samplings import partition_best_solutions
from initial_samplings import random_sampling,greedy_sampling
from optimal_samplings import optimal_sampling
import copy
from collections import defaultdict
import csv
import os
from parameters_optimal import *

def main(epoch,
         init_population,
         m,ngroups,
         sampling_function,
         hf_fun,
         lf_fun,
         cross_over,
         mutation,bounds,
         a,
         ps,f_n,n_aa,n_con,
         seed,
         T,
         DELTA
         ):
    np.random.seed(seed)
    random.seed(seed)
    # print('current_population', current_population)
    if ps is not None and bounds is not None:
        current_population = init_population(ps, bounds)
    else:
        current_population = init_population(n_aa,n_con,seed)
        ps = current_population.shape[0]
        lower_bounds = current_population.iloc[:, :-1].min().values
        upper_bounds = current_population.iloc[:, :-1].max().values
        bounds = np.column_stack((lower_bounds, upper_bounds))
        f_n = current_population.columns
        a=f_n
        
    n= bounds.shape[0]
    n_cross = 2*round(ps*CR)
    print('NCROSS',n_cross)
    n_mut = round(ps*MR)

    print('n',n)
    for i in range(epoch):
        start_len = len(current_population)
        current_population = cross_over(copy.deepcopy(current_population), ps, n_cross, ALPHA, bounds,seed)
        added = len(current_population) - start_len
        current_population = mutation(copy.deepcopy(current_population), n, ps,added, n_cross,n_mut, bounds,f_n,seed)
        current_population = lf_fun(copy.deepcopy(current_population),n,a)
        # print('current_population', current_population)
        current_population = sorted(copy.deepcopy(current_population), key=lambda x: x[-1])
        # print('current_population', current_population)
        # current_population = copy.deepcopy(current_population)[:ps]
        current_population = pareto_fitness_diversity.pareto_fitness_diversity_selection_simple(copy.deepcopy(current_population), ps, min_diff_threshold=0.001, seed=seed)

    all_groups = partition_best_solutions(copy.deepcopy(current_population),ngroups)
    # print('all_groups', all_groups)
    # print('current_population', current_population)
    # 1/0

    updated_sampling = [[] for _ in range(ngroups)] 
    n_all = []# Copy the original n_all
    n_all.append(m)
    average_last_values_t = []
    best_last_values_t = []
    std_last_values_t = []
    updated_sampling, updated_group,mu,sd = sampling_function(all_groups,m, updated_sampling,hf_fun,n,seed)
    # print('updated_sampling',updated_sampling)
    last_values_t = [group[-1] for sublist in updated_sampling for group in sublist]
    average_last_values_t.append(np.mean(last_values_t))
    best_last_values_t.append(np.min(last_values_t))
    # print('best_last_values_t',best_last_values_t)
    std_last_values_t.append(np.std(last_values_t))
    all_groups = updated_group
    p=0

    # while (T - sum(sum(n_all, []))) >= n_init:
    while sum(sum(n_all, [])) <T:
        p+=1
        print('p',p)
        remaining_budget =T - sum(sum(n_all, []))
        print('remaining_budget',remaining_budget)

        D_stage = min(DELTA, remaining_budget)

        m ,n_all= optimal_sampling(grps=all_groups, mu=mu, sd=sd, D=D_stage,n_all=n_all)
        
        updated_sampling, updated_group,mu,sd = sampling_function(all_groups,m, updated_sampling,hf_fun,n,seed)
        # print('updated_sampling',updated_sampling)
        last_values_t = [group[-1] for sublist in updated_sampling for group in sublist]
        average_last_values_t.append(np.mean(last_values_t))
        best_last_values_t.append(np.min(last_values_t))
        # print('best_last_values_t',best_last_values_t)
        std_last_values_t.append(np.std(last_values_t))
        all_groups = updated_group
    print('n_all',n_all)

    return average_last_values_t,best_last_values_t, std_last_values_t,n_all,updated_sampling