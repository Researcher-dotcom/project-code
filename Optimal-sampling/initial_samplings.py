import numpy as np
import random
from scipy.spatial.distance import cdist
from parameters_optimal import *

def random_sampling(groups,d, updated_sampling,hf_fun,n,seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed) 
    updated_groups = [[] for _ in range(len(groups))]  # List of empty listsn_all =[]
    avg_fitness_per_group = []  # Store average fitness values per group
    std_fitness_per_group = [] 

    for i, group in enumerate(groups):
        print('len(group_rand)',len(group))
        print('d',d)

        num_samples = d[i]
        print('num_samples',num_samples)

        if num_samples == 0:
            # print(f"Skipping group {i} as no samples are selected.")
            updated_groups[i].extend(group)  # Keep all samples in the group
            last_values = [sample[-1] for sample in updated_sampling[i]]  
            avg_fitness_per_group.append(np.mean(last_values))
            std_fitness_per_group.append(np.std(last_values))
            continue  # Skip the rest of the loop for this group

        shuffled_group = random.sample(group, len(group))
        # print('shuffled_group',shuffled_group)
        selected_samples = shuffled_group[:num_samples]  # Select `num_samples` samples
        # print('selected_samples',selected_samples)

        # remaining_samples = [sample for sample in group if sample not in selected_samples]
        remaining_samples = [sample for sample in group if tuple(sample) not in {tuple(s) for s in selected_samples}]

        selected_samples = hf_fun(selected_samples, n)
        updated_sampling[i].extend(selected_samples)
        updated_groups[i].extend(remaining_samples)  # Store remaining samples

        last_values = [sample[-1] for sample in updated_sampling[i]]  
        avg_fitness_per_group.append(np.mean(last_values))
        std_fitness_per_group.append(np.std(last_values))

    return updated_sampling, updated_groups,avg_fitness_per_group,std_fitness_per_group


# greedy_counter = 0
def greedy_sampling(groups,d,updated_sampling,hf_fun,n,seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    updated_groups = [[] for _ in range(len(groups))]  # List of empty lists
    # n_all = []
    avg_fitness_per_group = []  # Store average fitness values per group
    std_fitness_per_group = []

    for g_idx, group in enumerate(groups):
        kk = d[g_idx]
        # k=min(2,kk)
       
        group = np.array(group)

        # if kk == 0 or len(group)==0:
        if kk == 0:
            updated_groups[g_idx] = group.tolist()
            last_values = [sample[-1] for sample in updated_sampling[g_idx]]
            avg_fitness_per_group.append(np.mean(last_values))
            std_fitness_per_group.append(np.std(last_values))
            continue  # Skip the rest of the loop for this group

        if group.ndim == 1:
            group = group.reshape(1, -1)  # convert 1D to 2D
    
        centroid = np.mean(group[:, :-1], axis=0)
        distances = np.linalg.norm(group[:, :-1] - centroid, axis=1)
        first_idx = np.argmin(distances)

        selected_samples = [group[first_idx].tolist()]
        group = np.delete(group, first_idx, axis=0)  # Remove the selected sample

        # Step 2: Incrementally select the rest (k-1) samples
        for _ in range(kk - 1):
            group_X = group[:, :-1]
            group_y = group[:, -1]
            selected_X = np.array([samp[:-1] for samp in selected_samples])
            selected_y = np.array([samp[-1] for samp in selected_samples])


            # Compute pairwise distances and find the min distance to any selected sample
            dx = cdist(group_X, selected_X)
            dy = np.abs(group_y[:, np.newaxis] - selected_y[np.newaxis, :])  # shape (N-k, k)
            # Normalize dx and dy to handle scale differences
            # dx = (dx - dx.mean()) / (dx.std() + 1e-8)
            # dy = (dy - dy.mean()) / (dy.std() + 1e-8)
            # gama=0.1
            # dxy = (1-gama)*dx + gama*dy
            dxy = dx *dy
            dxy_min = np.min(dxy, axis=1)
            # min_distances = np.min(pairwise_distances, axis=1)

            # Select the sample furthest from current selected ones
            next_sample_idx = np.argmax(dxy_min)
            selected_samples.append(group[next_sample_idx].tolist())
            group = np.delete(group, next_sample_idx, axis=0)

        selected_samples = hf_fun(selected_samples, n)
        updated_sampling[g_idx].extend(selected_samples)
        # print('updated_sampling', updated_sampling)
         # Store selected samples for this group
        updated_groups[g_idx].extend(group.tolist()) # Store remaining samples

        last_values = [sample[-1] for sample in updated_sampling[g_idx]]
        # print('last_values', last_values)
        avg_fitness_per_group.append(np.mean(last_values))
        std_fitness_per_group.append(np.std(last_values))

    return updated_sampling, updated_groups, avg_fitness_per_group, std_fitness_per_group