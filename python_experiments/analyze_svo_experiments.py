import time, datetime, argparse, os, sys, pickle, psutil, logging
import numpy as np
# np.set_printoptions(precision=2)
import matplotlib.pyplot as plt
import casadi as cas
import copy as cp
PROJECT_PATHS = ['/home/nbuckman/Dropbox (MIT)/DRL/2020_01_cooperative_mpc/mpc-multiple-vehicles/', '/Users/noambuckman/mpc-multiple-vehicles/']
for p in PROJECT_PATHS:
    sys.path.append(p)
# import src.traffic_world as tw
import src.multiagent_mpc as mpc
# import src.car_plotting_multiple as cmplot
# import src.solver_helper as helper

import json
# import string, random
import glob

parser = argparse.ArgumentParser(description='Run iterative best response with SVO')
parser.add_argument('svo_dir',type=str, default=None, help="Load log")
parser.add_argument('--end-mpc', type=int, default=-1)
args = parser.parse_args()
params = vars(args)

list_of_experiments = glob.glob(args.svo_dir + "*/", recursive=False)
max_end_mpc = np.infty
max_dir = ""
for log_directory in list_of_experiments:
    list_of_mpc_data = glob.glob(log_directory + 'data/all_*xamb.npy')
    n_mpc_runs = len(list_of_mpc_data)
    end_mpc = n_mpc_runs - 1
    max_end_mpc = min(max_end_mpc, end_mpc)
    if max_end_mpc == end_mpc:
        max_dir = log_directory

assert args.end_mpc <= max_end_mpc

if args.end_mpc == -1:
    end_mpc = max_end_mpc
else:
    end_mpc = args.end_mpc
    ### Find the max end_mpc

print("End MPC", end_mpc, max_dir)
prosocial_experiments = glob.glob(args.svo_dir + "p*/", recursive=False)

egoistic_experiments = glob.glob(args.svo_dir + "e*/", recursive=False)

altruistic_experiments = glob.glob(args.svo_dir + "a*/", recursive=False)

experiment_dirs = {
    "e": egoistic_experiments,
    "p": prosocial_experiments,
    "a": altruistic_experiments,
}
x_traveled_experiments = {}
for svo_type, experiments in experiment_dirs.items():
    x_traveled_experiments[svo_type] = np.zeros(shape=(1, len(experiments)))

    for idx, log_directory in enumerate(experiments):
        with open(log_directory + "params.json",'rb') as fp:
                params = json.load(fp)    
        data_filename = log_directory + 'data/all_%02d'%end_mpc
        xamb_actual, _, _, xothers_actual, _, _, = mpc.load_state(data_filename, params['n_other'], ignore_des=True)
        end_frame = xamb_actual.shape[1] #Not exactly sure why we need minus 1

        x_traveled_experiments[svo_type][0,idx] = xamb_actual[0, end_frame-1]
    print(svo_type, x_traveled_experiments[svo_type])


##### Print Table
print("SVO:  Mean   Median")
for svo_type, x_traveled in x_traveled_experiments.items():
    print("%s:  %0.03f %0.03f"%(svo_type, np.mean(x_traveled), np.median(x_traveled)))


# print(prosocial_experiments)
# print(egoistic_experiments)
# print(altruistic_experiments)
