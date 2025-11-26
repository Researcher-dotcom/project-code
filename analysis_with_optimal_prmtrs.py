from matplotlib import pyplot as plt
import pandas as pd
import initial_population
import matplotlib.pyplot as plt
import numpy as np
import cross_over
import mutation
import fitness
import argparse
from initial_samplings import random_sampling,greedy_sampling
import os
import main_with_optimal_prmtrs
from parameters_optimal import *

# ---------------- Function configurations ---------------- #
function_configs = {
    "Branin-2": {
        "hf_fun": fitness.hf_fun_bran,
        "lf_fun": fitness.lf_fun_bran,
        "bounds": BOUNDS_BRAN,
        "a": A_BRAN
    },

    "Hartmann-3": {
        "hf_fun": fitness.hf_fun_hart3,
        "lf_fun": fitness.lf_fun_hart3,
        "bounds": BOUNDS_HART3,
        "a": A_HART3
    },

    "F14-5": {
        "hf_fun": fitness.hf_fun_f14_5,
        "lf_fun": fitness.lf_fun_f14_5,
        "bounds": BOUNDS_F14_5,
        "a": None
    },

     "F15-6": {
        "hf_fun": fitness.hf_fun_f15_6,
        "lf_fun": fitness.lf_fun_f15_6,
        "bounds": BOUNDS_F15_6,
        "a": None
    },

        "michalewicz-3": {
        "hf_fun": fitness.hf_fun_Michalewicz_3,
        "lf_fun": fitness.lf_fun_griewank2_3,
        "bounds": BOUNDS_MICHA3,
        "a": A_GRIEWANK
    },
        "michalewicz-5": {
        "hf_fun": fitness.hf_fun_Michalewicz_5,
        "lf_fun": fitness.lf_fun_griewank2_5,
        "bounds": BOUNDS_MICHA5,
        "a": A_GRIEWANK
    },
        "michalewicz-8": {
        "hf_fun": fitness.hf_fun_Michalewicz_8,
        "lf_fun": fitness.lf_fun_griewank2_8,
        "bounds": BOUNDS_MICHA8,
        "a": A_GRIEWANK
    },
}


# ---------------- Save results ---------------- #
def save_stage_results(updated_sampling, test_function, method, seed,
                       col_names=None, avg_val=None, best_val=None, std_val=None, n_all=None):
    """Save flattened sampling data and metrics for a given test function and method."""
    os.makedirs("results", exist_ok=True)

    base_name = f"{test_function}_{method}_seed_{seed}"
    csv_path = f"results/{base_name}.csv"
    metrics_path = f"results/{base_name}_summary.txt"

    # Flatten and save main data
    flat_data = [item for sublist in updated_sampling for item in sublist]
    df_stage = pd.DataFrame(flat_data, columns=col_names if col_names else None)
    df_stage.to_csv(csv_path, index=False)

    # Save summary stats
    if avg_val is not None and best_val is not None and std_val is not None:
        with open(metrics_path, "w") as f:
            f.write(f"Test Function: {test_function}\n")
            f.write(f"Sampling Method: {method}\n")
            f.write(f"Average: {avg_val}\n")
            f.write(f"Best: {best_val}\n")
            f.write(f"Std Dev: {std_val}\n")
            f.write(f"n_all: {n_all}\n")

    print(f"Saved data → {csv_path}")
    print(f"Saved metrics → {metrics_path}")

# ---------------- Main block ---------------- #
if __name__ == "__main__":
    # Parse command-line arguments from SLURM
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_function", type=str, required=True)
    parser.add_argument("--sampling_function", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    TEST_FUNCTION = args.test_function
    SAMPLING_METHOD = args.sampling_function
    SEED = args.seed

    print(f"Running: {TEST_FUNCTION} using {SAMPLING_METHOD} sampling")

    # Map string to actual function
    sampling_function_map = {
        "random": random_sampling,
        "greedy": greedy_sampling
    }
    SAMPLING_FUNCTION = sampling_function_map[SAMPLING_METHOD]

    cfg = function_configs[TEST_FUNCTION]

    avg_last_values, best_last_values, std_last_values_t, n_all_value, updated_sampling = main_with_optimal_prmtrs.main(
        epoch=EPOCH,
        init_population=initial_population.init_population_syn,
        m=K,
        ngroups=NGROUPS,
        sampling_function=SAMPLING_FUNCTION,  # <-- pass the function here
        hf_fun=cfg["hf_fun"],
        lf_fun=cfg["lf_fun"],
        cross_over=cross_over.cross_over_syn,
        mutation=mutation.mutation_syn,
        bounds=cfg["bounds"],
        a=cfg["a"],
        ps=PS,
        f_n=None,
        n_aa=None,
        n_con=None,
        seed=SEED,
        T=T,
        DELTA=DELTA
    )

    save_stage_results(
        updated_sampling,
        test_function=TEST_FUNCTION,
        method=SAMPLING_METHOD,
        seed = SEED,
        avg_val=avg_last_values,
        best_val=best_last_values,
        std_val=std_last_values_t,
        n_all=n_all_value
    )

