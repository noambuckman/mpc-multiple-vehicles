#!/bin/bash

source /etc/profile
module load anaconda/2021a
pip install --user casadi
pip install --user p_tqdm
pip install --user shapely
export PYTHONPATH=$PYTHONPATH:/home/gridsan/nbuckman/mpc-multiple-vehicles

EXPERIMENT_RANDOM_SEED=100400
INPUT_PARAMS="experiment.json"
RESULTS_DIR="results/"
N_PROCESSORS=16


python -u $HOME/mpc-multiple-vehicles/experiments/06_23_systematic_experiments.py $LLSUB_RANK $LLSUB_SIZE --input-params $INPUT_PARAMS --experiment-random-seed $EXPERIMENT_RANDOM_SEED  --results-dir $RESULTS_DIR --n-processors $N_PROCESSORS 

