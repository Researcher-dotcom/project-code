import numpy as np
import pandas as pd


################################ Branin-Function #####################################

def lf_fun_bran(population_list, n,a):
    pi = np.pi
    A = a
    for i in range(len(population_list)):
        term1 = (population_list[i][n-1] - ((1.275 * population_list[i][n-2]**2) / pi**2) + (5 * population_list[i][n-2] / pi) - 6) ** 2
        term2 = (10 - (5 / (4 * pi))) * np.cos(population_list[i][n-2])+ 10
        term3 = (population_list[i][n-1] - ((1.275 * population_list[i][n-2]**2) / pi**2) + (5 * population_list[i][n-2] / pi) - 6) ** 2
        final_term = term1+term2-(A+0.5)*(term3)
        population_list[i][n] = final_term
    return population_list
######################
def hf_fun_bran(initial_sampling, n):
    pi = np.pi
    for i in range(len(initial_sampling)):     
        term1 = (initial_sampling[i][n-1] - ((1.275 * initial_sampling[i][n-2]**2) / pi**2) + (5 * initial_sampling[i][n-2] / pi) - 6) ** 2
        term2 = (10 - (5 / (4 * pi))) * np.cos(initial_sampling[i][n-2])+ 10
        final_term = term1+term2
        initial_sampling[i][n] = final_term
    return initial_sampling
################################ Hartmann3-Function #####################################

def lf_fun_hart3(population_list, n, a):
    alpha = np.array([0.5, 0.8, 2.0, 2.5])
    beta_1 = np.array([
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0]
    ])
    beta = 0.8 * beta_1
    p = np.array([
    [0.3689, 0.1170, 0.2673],
    [0.4699, 0.4387, 0.7470],
    [0.1091, 0.8732, 0.5547],
    [0.0381, 0.5743, 0.8828]
    ])
    A = a
    for q in range(len(population_list)):
        term1 = -sum(alpha[i]* np.exp(-sum(beta[i,j]*(population_list[q][j]-(p[i,j]))**2 for j in range(3)))for i in range(4))
        population_list[q][n] = term1
    return population_list
######################
def hf_fun_hart3(initial_sampling, n):
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    beta = np.array([
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0]
    ])
    p = np.array([
    [0.3689, 0.1170, 0.2673],
    [0.4699, 0.4387, 0.7470],
    [0.1091, 0.8732, 0.5547],
    [0.0381, 0.5743, 0.8828]
    ])
    for q in range(len(initial_sampling)):
        term1 = -sum(alpha[i]* np.exp(-sum(beta[i,j]*(initial_sampling[q][j]-p[i,j])**2 for j in range(3)))for i in range(4))
        initial_sampling[q][n] = term1
    return initial_sampling


################################ F14-Function #####################################
# Low Fidelity
def lf_fun_f14_5(population_list, n, a):
    for q in range(len(population_list)):
        term1 =  sum(0.3 +np.sin(((13 * population_list[q][j]) / 15) - 1) + 
                          ((np.sin(((13 * population_list[q][j]) / 15) - 1)) ** 2 )
                          for j in range(n))
        population_list[q][n] = term1
    return population_list
#####################
# High Fidelity
def hf_fun_f14_5(initial_sampling, n):
    for q in range(len(initial_sampling)):
        term1 =  sum(0.3 + np.sin(((16 * initial_sampling[q][j]) / 15) - 1) + 
                          ((np.sin(((16 * initial_sampling[q][j]) / 15) - 1)) ** 2 )
                          for j in range(n))
        initial_sampling[q][n] = term1
    return initial_sampling

################################ Michalewicz LF Functions #####################################
def lf_fun_griewank2_3(population_list, n,a):
    phi = a
    theta_phi = np.exp(-0.00025*phi)
    w_phi = 10 * np.pi * theta_phi
    b_phi = 0.5 * np.pi * theta_phi
    for q in range(len(population_list)):
        x = population_list[q]

        y_l =  sum(theta_phi*np.cos(w_phi*x[i] + b_phi + np.pi) for i in range(n))

        population_list[q][n] = y_l
    return population_list

def lf_fun_griewank2_5(population_list, n,a):
    phi = a
    theta_phi = np.exp(-0.00025*phi)
    w_phi = 10 * np.pi * theta_phi
    b_phi = 0.5 * np.pi * theta_phi
    for q in range(len(population_list)):
        x = population_list[q]

        y_l =  sum(theta_phi*np.cos(w_phi*x[i] + b_phi + np.pi) for i in range(n))

        population_list[q][n] = y_l
    return population_list

def lf_fun_griewank2_8(population_list, n,a):
    phi = a
    theta_phi = np.exp(-0.00025*phi)
    w_phi = 10 * np.pi * theta_phi
    b_phi = 0.5 * np.pi * theta_phi
    for q in range(len(population_list)):
        x = population_list[q]

        y_l =  sum(theta_phi*np.cos(w_phi*x[i] + b_phi + np.pi) for i in range(n))

        population_list[q][n] = y_l
    return population_list

######################  Michalewicz HF Functions #####################################
def hf_fun_Michalewicz_3(initial_sampling, n):
    for q in range(len(initial_sampling)):
        x = initial_sampling[q]
        y_h = -sum(np.sin(x[i])*np.sin((i+1) * x[i]**2 / np.pi)**20 for i in range(n))
        initial_sampling[q][n] = y_h
    return initial_sampling

def hf_fun_Michalewicz_5(initial_sampling, n):
    for q in range(len(initial_sampling)):
        x = initial_sampling[q]
        y_h = -sum(np.sin(x[i])*np.sin((i+1) * x[i]**2 / np.pi)**20 for i in range(n))
        initial_sampling[q][n] = y_h
    return initial_sampling
def hf_fun_Michalewicz_8(initial_sampling, n):
    for q in range(len(initial_sampling)):
        x = initial_sampling[q]
        y_h = -sum(np.sin(x[i])*np.sin((i+1) * x[i]**2 / np.pi)**20 for i in range(n))
        initial_sampling[q][n] = y_h
    return initial_sampling

################################ F15-6Function #####################################
def lf_fun_f15_6(population_list, n,a):
    for q in range(len(population_list)):
        x = population_list[q]
        y_l =  sum(100*(x[i]-(x[i-1]**2))**2 + 4*(x[i-1]-1)**4 for i in range(1, 6))
        population_list[q][n] = y_l
    return population_list
######################
def hf_fun_f15_6(initial_sampling, n):
    for q in range(len(initial_sampling)):
        x = initial_sampling[q]
        y_h = sum(100*(x[i]-(x[i-1]**2))**2 + (x[i-1]-1)**2 for i in range(1, 6))
        initial_sampling[q][n] = y_h
    return initial_sampling


