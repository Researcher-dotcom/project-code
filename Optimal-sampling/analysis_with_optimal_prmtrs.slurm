#!/bin/bash

#SBATCH --job-name=experiment
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=7G
#SBATCH --time=01:00:00

# Write output to local files
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

# Optional job array
#SBATCH --array=0-419

# Example lists (modify as needed)
test_functions=("Branin-2" "Hartmann-3" "F14-5" "F15-6" "michalewicz-3" "michalewicz-5" "michalewicz-8")
sampling_methods=("random" "greedy")
NUM_SEEDS=30

NUM_TEST=${#test_functions[@]}
NUM_SAMPLING=${#sampling_methods[@]}

ARRAY_ID=$SLURM_ARRAY_TASK_ID
SEED_INDEX=$(( ARRAY_ID % NUM_SEEDS ))
SAMPLING_INDEX=$(( (ARRAY_ID / NUM_SEEDS) % NUM_SAMPLING ))
TEST_INDEX=$(( ARRAY_ID / (NUM_SEEDS * NUM_SAMPLING) ))

TEST_FUNCTION=${test_functions[$TEST_INDEX]}
SAMPLING_METHOD=${sampling_methods[$SAMPLING_INDEX]}
SEED=$((SEED_INDEX + 1))

echo "Running: $TEST_FUNCTION | $SAMPLING_METHOD | Seed: $SEED"

# Load generic modules (no institution-specific ones)
module load python/3.9

# If using conda:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate myenv

python -u analysis_with_optimal_prmtrs.py \
  --test_function $TEST_FUNCTION \
  --sampling_function $SAMPLING_METHOD \
  --seed $SEED
