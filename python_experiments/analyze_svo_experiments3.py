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
np.set_printoptions(precision=3)
import json
# import string, random
import glob

parser = argparse.ArgumentParser(description='Run iterative best response with SVO')
parser.add_argument('svo_dir',type=str, default=None, help="Load log")
parser.add_argument('--min-end-mpc', type=int, default=-1, help="Only include directories with a minimum rnd of iterations")
parser.add_argument('--end-mpc', type=int, default=-1)
args = parser.parse_args()
params = vars(args)

list_of_experiments = glob.glob(args.svo_dir + "*/", recursive=False)
max_end_mpc = np.infty
max_dir = ""
list_of_experiments_ = []
for log_directory in list_of_experiments:
    list_of_mpc_data = glob.glob(log_directory + 'data/all_*xamb.npy')
    n_mpc_runs = len(list_of_mpc_data)
    end_mpc = n_mpc_runs - 1

    if end_mpc < args.min_end_mpc:
        continue
    max_end_mpc = min(max_end_mpc, end_mpc)
    if max_end_mpc == end_mpc:
        max_dir = log_directory
    list_of_experiments_ += [log_directory]
list_of_experiments = list_of_experiments_

print("Experiments with min end mpc of %d"%args.min_end_mpc)
print(list_of_experiments)
assert args.end_mpc <= max_end_mpc

if args.end_mpc == -1:
    end_mpc = max_end_mpc
else:
    end_mpc = args.end_mpc
    ### Find the max end_mpc

print("End MPC:", end_mpc, " Directory that reached end_mpc: ", max_dir)
experiment_dirs = {"e": [], "p": [], "a": [], "s":[]}
experiment_seeds = {"e": [], "p": [], "a": [], "s": []}
all_seeds = set()
for log_directory in list_of_experiments:
    string_split = log_directory.split('/')
    log_name = string_split[-2]
    svo_type = log_name[0]
    seed_number = log_name[1:]
    experiment_dirs[svo_type] += [log_directory]
    experiment_seeds[svo_type] += [seed_number]
    all_seeds.add(seed_number)
### Check that all the seeds we have e, a, p for each one

set_of_experiments = set(list_of_experiments)

x_traveled_experiments = {"e": [], "p": [], "a": [], "s":[]}
final_set_of_experiments = {"e": [], "p": [], "a": [], "s":[]}
final_seeds = []
for seed in all_seeds:
    if (seed in experiment_seeds["e"]) and (seed in experiment_seeds["a"]) and (seed in experiment_seeds["p"]) and (seed in experiment_seeds["s"]):
        for svo_type in x_traveled_experiments.keys():
            log_directory = args.svo_dir + svo_type + seed + "/"
            with open(log_directory + "params.json",'rb') as fp:
                params = json.load(fp)    
        
            data_filename = log_directory + 'data/all_%03d'%end_mpc
            if os.path.isfile(data_filename + "xamb.npy"):
                pass
            else:        
                data_filename = log_directory + 'data/all_%02d'%end_mpc
            xamb_actual, _, _, xothers_actual, _, _, = mpc.load_state(data_filename, params['n_other'], ignore_des=True)
            end_frame = xamb_actual.shape[1] #Not exactly sure why we need minus 1

            x_traveled_experiments[svo_type] += [xamb_actual[0, end_frame-1]]
            final_set_of_experiments[svo_type] += [log_directory]
        final_seeds += [seed]

print(final_set_of_experiments["a"])
print(" ")
print(final_set_of_experiments["e"])
print(" ")
print(final_set_of_experiments["p"])

print(" ")
print(final_set_of_experiments["s"])

for svo_type in {"a", "e", "p", "s"}:
    x_traveled_experiments[svo_type] = np.array(x_traveled_experiments[svo_type])

# for svo_type, experiments in experiment_dirs.items():
#     x_traveled_experiments[svo_type] = np.zeros(shape=(1, len(experiments)))
#     for idx, log_directory in enumerate(experiments):
#         with open(log_directory + "params.json",'rb') as fp:
#                 params = json.load(fp)    
        
#         data_filename = log_directory + 'data/all_%03d'%end_mpc
#         if os.path.isfile(data_filename + "xamb.npy"):
#             pass
#         else:        
#             data_filename = log_directory + 'data/all_%02d'%end_mpc
#         xamb_actual, _, _, xothers_actual, _, _, = mpc.load_state(data_filename, params['n_other'], ignore_des=True)
#         end_frame = xamb_actual.shape[1] #Not exactly sure why we need minus 1

#         x_traveled_experiments[svo_type][0,idx] = xamb_actual[0, end_frame-1]
#     print(svo_type, x_traveled_experiments[svo_type])




# x_traveled_experiments = {"a": [], "e":[], "p":[]}
# for svo_type, experiments in experiment_dirs.items():
#     x_traveled_experiments[svo_type] = np.zeros(shape=(1, len(experiments)))
#     for idx, log_directory in enumerate(experiments):
#         with open(log_directory + "params.json",'rb') as fp:
#                 params = json.load(fp)    
        
#         data_filename = log_directory + 'data/all_%03d'%end_mpc
#         if os.path.isfile(data_filename + "xamb.npy"):
#             pass
#         else:        
#             data_filename = log_directory + 'data/all_%02d'%end_mpc
#         xamb_actual, _, _, xothers_actual, _, _, = mpc.load_state(data_filename, params['n_other'], ignore_des=True)
#         end_frame = xamb_actual.shape[1] #Not exactly sure why we need minus 1

#         x_traveled_experiments[svo_type][0,idx] = xamb_actual[0, end_frame-1]
#     print(svo_type, x_traveled_experiments[svo_type])
print(final_seeds)
svo_types = ["e", "s", "p", "a"]
##### Print Table
print("SVO:  Mean   Median")
for svo_type in svo_types:
    x_traveled = x_traveled_experiments[svo_type]
    print("%s:  %0.03f %0.03f"%(svo_type, np.mean(x_traveled), np.median(x_traveled)))

print("Latex")



print("& SVO: &Mean   &Median  &Std")
for svo_type in svo_types:
    x_traveled = x_traveled_experiments[svo_type]
    print("& %s:  & %0.02f & %0.02f & %0.02f"%(svo_type, np.mean(x_traveled), np.median(x_traveled), np.std(x_traveled)))

print("Number of Experiments:")
for svo_type in svo_types:
    x_traveled = x_traveled_experiments[svo_type]
    print("%s: %d"%(svo_type, x_traveled.shape[0]))


# print(prosocial_experiments)
# print(egoistic_experiments)
# print(altruistic_experiments)
