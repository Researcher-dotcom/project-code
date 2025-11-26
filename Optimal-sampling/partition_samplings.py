def partition_best_solutions(population, ngroups):
    # Sort population by the last value (e.g., fitness)
    sorted_population = sorted(population, key=lambda x: x[-1])
    
    # Compute size of each group (some will be one item larger to handle remainder)
    k2, m2 = divmod(len(sorted_population), ngroups)
    
    groups = []
    start = 0
    for i in range(ngroups):
        end = start + k2 + (1 if i < m2 else 0)  # distribute the remainder
        groups.append(sorted_population[start:end])
        start = end
    
    return groups