### 
# Purpose:  Create svo experiments which vary the svo types of the agents
# We should also save the logs for each one and save it somewhere

import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Run iterative best response with SVO')
parser.add_argument('svo_dir',type=str, default=None, help="Load log")
parser.add_argument('seed-start', type=int, default=-1)
parser.add_argument('seed-end', type=int, default=-1)
args = parser.parse_args()

## Settings we will override during each call
n_other = 30
n_lanes = 2
n_cntrld = 2
random_svo = 0
##
batch_subdir = args.svo_dir
######## Egoistic
svo_theta = 0.0
all_cmds = ""
results_parent_dir = os.path.expanduser("~") + "/mpc_results/"
os.makedirs(results_parent_dir + batch_subdir)

for idx in range(args.seed_start, args.seed_end + 1):
    log_subdir = batch_subdir + "e%03d"%idx
    cmd = "python iterative_best_response.py --seed %d --n-other %s --n-lanes %s --n-cntrld %s --random-svo %s --svo-theta %0.06f --log-subdir %s &"%(idx, n_other, n_lanes, n_cntrld, random_svo, 
                                                                                                                    svo_theta, log_subdir)
    all_cmds += cmd
    all_cmds += "\n"
    print(cmd)
    os.system(cmd)

########### Prosocial
svo_theta = np.pi/4.0
for idx in range(args.seed_start, args.seed_end + 1):
    log_subdir = batch_subdir + "p%03d"%idx
    cmd = "python iterative_best_response.py --seed %d --n-other %s --n-lanes %s --n-cntrld %s --random-svo %s --svo-theta %0.06f --log-subdir %s &"%(idx, n_other, n_lanes, n_cntrld, random_svo, 
                                                                                                                    svo_theta, log_subdir)
    print(cmd)
    all_cmds += cmd
    all_cmds += "\n"
    os.system(cmd)

######## Altruistic
svo_theta = np.pi/2.01
for idx in range(args.seed_start, args.seed_end + 1):
    log_subdir = batch_subdir + "a%03d"%idx
    cmd = "python iterative_best_response.py --seed %d --n-other %s --n-lanes %s --n-cntrld %s --random-svo %s --svo-theta %0.06f --log-subdir %s &"%(idx, n_other, n_lanes, n_cntrld, random_svo, 
                                                                                                                    svo_theta, log_subdir)
    print(cmd)
    all_cmds += cmd
    all_cmds += "\n"
    os.system(cmd)

os.makedirs(os.path.expanduser("~") + "/mpc_results/" + batch_subdir, exist_ok=True)
with open(os.path.expanduser("~") + "/mpc_results/" + batch_subdir + "cmds.txt",'w') as f:
    f.write(all_cmds)


