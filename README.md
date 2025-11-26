# Multi-fidelity Optimisation via Hybrid Genetic-Greedy Search
This repository contains Python scripts for performing Multi-fidelity Optimisation via Hybrid Genetic-Greedy Search experiments on benchmark test functions using various sampling methods. The project supports reproducible experiments with multiple seeds and can be run on high-performance computing (HPC) clusters via SLURM.
# Repository Structure
```bash
├── analysis_with_optimal_prmtrs.py       # Main analysis script
├── analysis_with_optimal_prmtrs.slurm   # SLURM batch script
├── cross_over.py                         # Crossover operators for GA
├── fitness.py                            # Fitness evaluation functions
├── initial_population.py                 # Initial population generation
├── initial_samplings.py                  # Sampling strategies
├── main_with_optimal_prmtrs.py           # Main script for running with parameters
├── mutation.py                           # Mutation operators
├── optimal_samplings.py                   # Optimal samplings
├── parameters_optimal.py                 # Optimal parameter definitions
├── pareto_fitness_diversity.py           # Pareto-based fitness & diversity calculations
└── partition_samplings.py                # Partitioning of samples
```
# Requirements
Python 3.9+
Required Python packages:
numpy
scipy
matplotlib
HPC environment with SLURM (optional)

# Usage
## Running on HPC (SLURM)
The script analysis_with_optimal_prmtrs.slurm is set up for a job array. Example usage:
```bash
sbatch analysis_with_optimal_prmtrs.slurm
```
The script automatically selects the test function, sampling method, and seed based on the SLURM array ID:
- Test functions: Branin-2, Hartmann-3, F14-5, F15-6, michalewicz-3, michalewicz-5, michalewicz-8
- Sampling methods: random, greedy
- Seeds: 1–30
## Running Locally
You can also run the analysis directly:
```bash
python -u analysis_with_optimal_prmtrs.py \
    --test_function Branin-2 \
    --sampling_function random \
    --seed 1
```
