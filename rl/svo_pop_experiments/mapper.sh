#!/bin/bash

conda activate mpc
export PYTHONPATH=$PYTHONPATH:/home/gridsan/nbuckman/mpc-multiple-vehicles
cd /home/gridsan/nbuckman/mpc-multiple-vehicles/rl/svo_pop_experiments

EXPERIMENT_RANDOM_SEED=$(($SLURM_ARRAY_TASK_ID + $RANDOM))
echo $1
echo $LLSUB_RANK 
echo $LLSUB_SIZE
python sim_mapper_population.py $LLSUB_RANK $LLSUB_SIZE --input-params $1 --experiment-random-seed $EXPERIMENT_RANDOM_SEED 