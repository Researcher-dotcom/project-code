import random
import numpy as np
from initial_samplings import random_sampling,greedy_sampling
import math
from parameters_optimal import *
def optimal_sampling(grps, mu, sd, D, n_all,seed=None):
    print(f"\n[DEBUG] Starting optimal_sampling | D={D} | num_groups={len(grps)} | seed={seed}")

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    # print('D',D)
    n_total = np.zeros(len(grps))
        
    if np.std(mu) == 0:  # All values in mu are equal
        print("All mu values are equal. Sampling evenly across groups.")
        print(f"[DEBUG] All mu values equal. Even distribution applied. mu={mu}")
        
        n_grps = len(grps)
        # Distribute D samples as evenly as possible across groups
        base = D // n_grps        # integer division
        remainder = D % n_grps    # remaining samples
        
        n_total = np.full(n_grps, base)
        n_total[:remainder] += 1    # distribute the leftover samples
        
        n_all.append(n_total.tolist())
        return n_total, n_all

    
    best_indx = np.argmin(mu)
    print('mu',mu) 
    print('sd',sd)
    # print('best_indx',best_indx) 
    available_indices = [i for i in range(len(grps)) if i != best_indx]
    # print('grpsnew',grps)
    # print('grps',grps)

    r = random.choice(available_indices)
    print('r',r) 
    remaining_indices = [j for j in available_indices if j != r]
    epsilon = 1e-10  # A small non-trivial value

    rb = [((sd[j] * (mu[best_indx] - mu[r])) / (((sd[r]+epsilon) * (mu[best_indx] - mu[j]))+epsilon))**2 for j in remaining_indices]
    print('rb',rb)  
    e = 1 + sum(rb)
    # print('e',e)
    f = sd[r]**(-2)+ sum((c_val/(sd[j]+epsilon))**2 for c_val, j in zip(rb, remaining_indices))
    # print('f',f)
    a = (f*sd[best_indx]**2/e**2)-1
    # print('a',a)
    b = -2*D*f*(sd[best_indx]/e)**2
    c = f*(D*sd[best_indx]/e)**2
    # --- Safety check before using a, b, c ---
    if not np.isfinite(a) or not np.isfinite(b) or not np.isfinite(c):
        print("Invalid a, b, c values detected. Using fallback allocation.")
        n_total = np.full(len(grps), D // len(grps))
        n_all.append(n_total.tolist())
        return n_total, n_all
    delta = b**2-4*a*c
    # --- Safety check for delta ---
    if not np.isfinite(delta) or delta < 0:
        print(f"Invalid delta={delta}, using fallback allocation.")
        print(f"[DEBUG] Invalid delta={delta}. Triggering fallback allocation.")

        n_total = np.full(len(grps), D // len(grps))
        n_all.append(n_total.tolist())
        return n_total, n_all
    # print('delta',delta)
    # print('a',a)
    n_b1 = (-b+np.sqrt(delta))/(2*a)
    n_b2 = (-b-np.sqrt(delta))/(2*a)
    # print('n_b1',n_b1)
    # print('n_b2',n_b2)
    n_b = n_b1 if 0 < n_b1 < D else (n_b2 if 0 < n_b2 < D else None)
    # print('n_b',n_b)
    # --- Safety check for n_b and n_f ---
    if n_b is None or not np.isfinite(n_b) or n_b <= 0 or n_b >= D:
        print("Invalid n_b, using fallback allocation")
        n_total = np.full(len(grps), D // len(grps))
        n_all.append(n_total.tolist())
        return n_total, n_all
    n_f = (D-n_b)/e
    if not np.isfinite(n_f) or n_f <= 0:
        print("Invalid n_f, using fallback allocation")
        n_total = np.full(len(grps), D // len(grps))
        n_all.append(n_total.tolist())
        return n_total, n_all
    # print('n_b',n_b)
    # print('n_f',n_f)
    n_total[best_indx] =min(round(n_b), len(grps[best_indx]))
    n_total[r] = min(round(n_f), len(grps[r]))
    for c_val,j in zip(rb,remaining_indices):
        n_total[j] = min(round(c_val * n_f), len(grps[j]))


    # print('n_total',n_total)
    n_total = np.array(n_total, dtype=int)

        # --- NEW: Redistribute leftover samples if total < D ---
    total_assigned = np.sum(n_total)
    remaining_bud = D - total_assigned
    if remaining_bud > 0:
        # Example: allocate leftovers to groups with higher sd
        for i in np.argsort(-np.array(sd)):  # descending order of sd
            available = len(grps[i]) - n_total[i]
            allocate = min(remaining_bud, available)
            n_total[i] += allocate
            remaining_bud -= allocate
            if remaining_bud == 0:
                break
    # --- END NEW ---


    n_all.append(n_total.tolist())
    print(f"[DEBUG] Final n_total={n_total}, sum={np.sum(n_total)}, expected D={D}")

    # print('n_all',n_all)
    return n_total, n_all