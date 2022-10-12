#!/bin/bash

# source /etc/profile

module load anaconda3/2021.11
pip install --user casadi
pip install --user p_tqdm
pip install --user shapely
export PYTHONPATH=$PYTHONPATH:~/mpc-multiple-vehicles

EXPERIMENT_RANDOM_SEED=2100400
INPUT_PARAMS="experiment.json"
RESULTS_DIR="results/"


python -u $HOME/mpc-multiple-vehicles/experiments/06_23_systematic_experiments.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT --input-params $INPUT_PARAMS --experiment-random-seed $EXPERIMENT_RANDOM_SEED  --results-dir $RESULTS_DIR --n-processors $N_PROCESSORS --dry-run

