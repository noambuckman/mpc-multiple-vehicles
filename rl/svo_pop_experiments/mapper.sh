#!/bin/bash

conda activate mpc
export PYTHONPATH=$PYTHONPATH:/home/gridsan/nbuckman/mpc-multiple-vehicles
cd /home/gridsan/nbuckman/mpc-multiple-vehicles/rl/svo_pop_experiments

EXPERIMENT_RANDOM_SEED=$RANDOM
INPUT_PARAMS="svo_population_experiments.json"
echo $LLSUB_RANK 
echo $LLSUB_SIZE
python sim_mapper_population.py $LLSUB_RANK $LLSUB_SIZE --input-params $INPUT_PARAMS --experiment-random-seed $EXPERIMENT_RANDOM_SEED 