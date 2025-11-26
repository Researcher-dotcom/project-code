from random import shuffle
import numpy as np
import pandas as pd
import random

def cross_over_syn(population_list, ps, ncross, alpha, bounds, seed=None):
   if seed is not None:
      np.random.seed(seed)
      random.seed(seed)
   print('len_population_list',len(population_list))
   print('PS',ps)
   choosen_ones = list(range(ps))
   np.random.shuffle(choosen_ones)
   choosen_ones = choosen_ones[:ncross]

   if len(choosen_ones) % 2 != 0:
      choosen_ones = choosen_ones[:-1]

   for i in range(0, len(choosen_ones), 2):

      u1 = np.array(population_list[choosen_ones[i]][:-1])
      u2 = np.array(population_list[choosen_ones[i+1]][:-1])
      
      e1 = u1 - alpha * (u2 - u1)
      e2 = u2 + alpha * (u2 - u1)

      child = np.empty_like(u1)

      for j in range(len(u1)):
         while True:
               r = np.random.rand()
               val = e1[j] + r * (e2[j] - e1[j])
               if bounds[j, 0] <= val <= bounds[j, 1]:
                  child[j] = val
                  break

      population_list.append(list(child) + [None])

   return population_list