#!/bin/bash

conda activate mpc
export PYTHONPATH=$PYTHONPATH:/home/gridsan/nbuckman/mpc-multiple-vehicles
cd /home/gridsan/nbuckman/mpc-multiple-vehicles

EXPERIMENT_RANDOM_SEED=$(($SLURM_ARRAY_TASK_ID + $RANDOM))

echo $LLSUB_RANK 
echo $LLSUB_SIZE
python sim_mapper_population.py $LLSUB_RANK $LLSUB_SIZE --experiment-random-seed EXPERIMENT_RANDOM_SEED 