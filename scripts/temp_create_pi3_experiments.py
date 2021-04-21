### 
# Purpose:  Create svo experiments which vary the svo types of the agents
# We should also save the logs for each one and save it somewhere

import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Run iterative best response with SVO')
parser.add_argument('svo_dir',type=str, default=None, help="Load log")
# parser.add_argument('seed_start', type=int, default=-1)
# parser.add_argument('seed_end', type=int, default=-1)
parser.add_argument('--dry-run', action='store_true' )
args = parser.parse_args()

## Settings we will override during each call
n_other = 30
n_lanes = 2
n_cntrld = 2
random_svo = 0
save_ibr = 0
##
batch_subdir = args.svo_dir
######## Egoistic
all_cmds = ""
results_parent_dir = os.path.expanduser("~") + "/mpc_results/"
os.makedirs(results_parent_dir + batch_subdir, exist_ok=True)

experiment_seed_strings = ['014', '007', '016', '015', '008', '009', '001', '020', '006', '000']
experiment_seeds = [int(s) for s in experiment_seed_strings]

########### Pi/6 (s prefix)
svo_theta = np.pi/6.0
for idx in experiment_seeds:
    log_subdir = batch_subdir + "s%03d"%idx
    cmd = "python iterative_best_response.py --seed %d --n-other %s --n-lanes %s --n-cntrld %s --random-svo %s --svo-theta %0.06f --log-subdir %s --save-ibr %d &"%(idx, n_other, n_lanes, n_cntrld, random_svo, 
                                                                                                                    svo_theta, log_subdir, save_ibr)
    print(cmd)
    all_cmds += cmd
    all_cmds += "\n"
    if args.dry_run:
        continue
    else:
        os.system(cmd)
