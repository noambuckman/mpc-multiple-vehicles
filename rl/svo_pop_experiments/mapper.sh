#!/bin/bash

#SBATCH --cpus-per-task 1
#SBATCH -a 1-2
#SBATCH --partition=xeon-p8


conda activate mpc
export PYTHONPATH=$PYTHONPATH:/home/gridsan/nbuckman/mpc-multiple-vehicles
cd /home/gridsan/nbuckman/mpc-multiple-vehicles

# # Initialize Modules
# source /etc/profile

# # Load Anaconda Module
# module load anaconda/2020a

EXPERIMENT_RANDOM_SEED=$(($SLURM_ARRAY_TASK_ID + $RANDOM))

echo $1
echo $2
echo $3
echo $4

# Call your script as you would from the command line, passing in $1 and $2 as arugments
# Note that $1 and $2 are the arguments passed into this script
python sim_mapper_population.py $1 $2 --experiment-random-seed EXPERIMENT_RANDOM_SEED