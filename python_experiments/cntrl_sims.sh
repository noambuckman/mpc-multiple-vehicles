#!/bin/bash

#SBATCH --cpus-per-task 8
#SBATCH -a 1-2

# echo $(date +%H%m%s)
printf -v date '%(%Y-%m-%d)T\n' -1 

# put current date as yyyy-mm-dd HH:MM:SS in $date
printf -v date '%(%m-%d-%Y-%H%M%S)T\n' -1 
echo $date


conda activate mpc
export PYTHONPATH=$PYTHONPATH:/home/gridsan/nbuckman/mpc-multiple-vehicles
cd mpc-multiple-vehicles

N_CNTRLD=3



N_OTHER=20
N_LANES=2
SAVE_IBR=1
N_MPC=30

CAR_DENSITY=3000

RANDOM_SVO=1
SVO_THETA=0.0

LOG_SUBDIR='cntrld/cntrld_'${SLURM_ARRAY_TASK_ID}_${date}
echo $LOG_SUBDIR

python src/iterative_best_response.py --seed $SLURM_ARRAY_TASK_ID --log-subdir $LOG_SUBDIR --n-cntrld $N_CNTRLD --n-processors $SLURM_CPUS_PER_TASK --n-mpc $N_MPC --car-density $CAR_DENSITY --n-other $N_OTHER --n-lanes $N_LANES --random-svo $RANDOM_SVO --svo-theta $SVO_THETA --save-ibr $SAVE_IBR
