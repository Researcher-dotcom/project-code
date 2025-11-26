import random
import numpy as np
import copy

PS = 1000
CR = 0.5  # crossover rate
MR = 0.02 # mutation rate
GAMA = 0.7
EPOCH = 20 # for most of them it was # number of epochs/iterations
BOUNDS_BRAN = np.array([[-5, 10], [0, 15]]) # lower and upper bounds of the search space of synthetic functions
BOUNDS_HART3 = np.array([[0, 1], [0, 1], [0, 1]])
BOUNDS_HART6 = np.array([[0, 1], [0, 1], [0, 1],[0, 1], [0, 1], [0, 1]])
BOUNDS_TANG = np.array([[-5, 5]] * 8)
BOUNDS_MATH8 = np.array([[-1, 1]] * 8)
BOUNDS_F13_4 = np.array([[-10, 10]] * 4)
BOUNDS_F15_6 = np.array([[0, 1]] * 6)
BOUNDS_F14_5 = np.array([[-1, 1]] * 5)
BOUNDS_MATH16 = np.array([[-5, 5]] * 16)
BOUNDS_MATH10 = np.array([[-5, 5]] * 10)
BOUNDS_GRIEWANK3 = np.array([[-5, 5]] * 3)
BOUNDS_GRIEWANK5 = np.array([[-5, 5]] * 5)
BOUNDS_GRIEWANK8 = np.array([[-5, 5]] * 8)
BOUNDS_MICHA3 = np.array([[0, np.pi]] * 3)
BOUNDS_MICHA5 = np.array([[0, np.pi]] * 5)
BOUNDS_MICHA8 = np.array([[0, np.pi]] * 8)
BOUNDS_XU = np.array([[0, 100]])

ALPHA = 0.5 # for generating offsprings
A_BRAN = 0
A_HART3 = 0.5
A_HART6 = 0.5
A_GRIEWANK = 1

NGROUPS = 4
TINIT =20 # initial budget for each group
T = 400# total budget
K = [int(x * TINIT) for x in np.ones(NGROUPS)] # budget at each group 
N_INIT = sum(K) # whole budget at each iteration 
DELTA=5
