#!/bin/bash

#SBATCH --cpus-per-task 8
#SBATCH -a 1-4

# echo $(date +%H%m%s)
printf -v date '%(%Y-%m-%d)T\n' -1 

# put current date as yyyy-mm-dd HH:MM:SS in $date
printf -v date '%(%m-%d-%Y-%H%M%S)T\n' -1 
echo $date


conda activate mpc
export PYTHONPATH=$PYTHONPATH:/home/gridsan/nbuckman/mpc-multiple-vehicles
cd mpc-multiple-vehicles

N_CNTRLD=2



N_OTHER=20
N_LANES=2
RANDOM_SVO=1
SAVE_IBR=1
N_MPC=30

CAR_DENSITY=3000
SVO_THETA=0.0
SEED=1

LOG_SUBDIR='cntrld_test' + $SLURM_ARRAY_TASK_ID
echo $LOG_SUBDIR

# python src/iterative_best_response.py --seed $SLURM_ARRAY_TASK_ID --n-mpc $N_MPC --car-density $CAR_DENSITY --n-other $N_OTHER --n-lanes $N_LANES --n-cntrld $N_CNTRLD --random-svo $RANDOM_SVO --svo-theta $SVO_THETA --log-subdir $LOG_SUBDIR --save-ibr $SAVE_IBR