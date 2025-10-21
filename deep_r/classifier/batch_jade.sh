#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# big partition
#SBATCH --partition=small

# set max wallclock time
#SBATCH --time=24:00:00

# set number of GPUs
#SBATCH --gres=gpu:1

# enable email notifications
#SBATCH --mail-user=J.C.Knight@sussex.ac.uk
#SBATCH --mail-type=END,FAIL

# run the application
./train_evaluate_args_jade.sh
