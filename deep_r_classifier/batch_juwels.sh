#!/bin/bash

# use project specified via jutil
#SBATCH --account=structuretofunction

# set the number of nodes
#SBATCH --nodes=1

# task per GPU
#SBATCH --ntasks-per-node=4

# big partition
#SBATCH --partition=booster

# set max wallclock time
#SBATCH --time=24:00:00

# set number of GPUs
#SBATCH --gres=gpu:4

# enable email notifications
#SBATCH --mail-user=J.C.Knight@sussex.ac.uk
#SBATCH --mail-type=END,FAIL

# run the application
srun ./train_evaluate_args_juwels.sh
