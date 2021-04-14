import time, datetime, argparse, os, sys, pickle, psutil, logging
from numpy.core.arrayprint import str_format
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
parser.add_argument('svo_dir',type=str, default=None, nargs='+', help="Load log")
parser.add_argument('--min-end-mpc', type=int, default=-1, help="Only include directories with a minimum rnd of iterations")
parser.add_argument('--end-mpc', type=int, default=-1)
args = parser.parse_args()
params = vars(args)

# list_of_experiments = glob.glob(args.svo_dir + "*/", recursive=False)
max_end_mpc = np.infty
max_dir = ""
list_of_experiments_ = []


def get_svo_type(svo):
    epsilon = np.pi/36 # +/- 5 deg
    if 0 - epsilon <= svo <= 0 + epsilon:
        return "e"
    elif np.pi/6 - epsilon <= svo <= np.pi/6 + epsilon:
        return "s"
    elif np.pi/4 - epsilon <= svo <= np.pi/4 + epsilon:
        return "p"
    elif np.pi/2 - epsilon <= svo <= np.pi/2 + epsilon:
        return "a"
    else:
        raise Exception("SVO is not withing range")


### 1) Make sure a seed is acceptable by checking that represented in svo and densities
all_svos = ["e", "p"]
all_densities = [1250, 2500, 5000]

experiment_dirs = {}
experiment_seeds = {}
for svo_type in all_svos:
    experiment_dirs[svo_type] = {}
    experiment_seeds[svo_type] = {}
    for density in all_densities:
        experiment_dirs[svo_type][density] = set()
        experiment_seeds[svo_type][density] = set()

all_seeds = set()
list_of_experiments = []
# 2) Let's fine the number of mpc rounds we will include

for log_directory in args.svo_dir:
    with open(log_directory + "params.json",'rb') as fp:
        params = json.load(fp)    
    
    seed_number = params["seed"]
    svo = params["svo_theta"]
    svo_type = get_svo_type(svo)
    density = params["car_density"]
    random_svo = params["random_svo"]
    # print(seed_number, svo_type, density)
    # Initialize the dicts
    if (svo_type not in all_svos) or (density not in all_densities) or random_svo == 1:
        continue
    experiment_dirs[svo_type][density].add(log_directory)
    experiment_seeds[svo_type][density].add(seed_number)
    
    all_seeds.add(seed_number)
    list_of_experiments += [log_directory]
### Check that all the seeds we have e, a, p for each one

set_of_experiments = set(list_of_experiments)

acceptable_logs = {}

ignored_seeds = set()
ignored_density = set()
for seed in all_seeds:
    for svo_type in all_svos:
        for density in all_densities:
            if seed not in experiment_seeds[svo_type][density]:
                ignored_seeds.add(seed)


list_of_experiments_ = []
for log_directory in list_of_experiments:
    with open(log_directory + "params.json",'rb') as fp:
            params = json.load(fp)    
    seed_number = params["seed"]
    if seed_number not in ignored_seeds:
        list_of_experiments_ += [log_directory]
        list_of_mpc_data = glob.glob(log_directory + 'data/all_*xamb.npy')
        n_mpc_runs = len(list_of_mpc_data)
        end_mpc = n_mpc_runs - 1
        if end_mpc < args.min_end_mpc:
            continue
        max_end_mpc = min(max_end_mpc, end_mpc)
        if max_end_mpc == end_mpc:
            max_dir = log_directory
list_of_experiments = list_of_experiments_


print("Experiments with min end mpc of %d"%args.min_end_mpc)

if args.end_mpc == -1:
    end_mpc = max_end_mpc
else:
    end_mpc = args.end_mpc
    ### Find the max end_mpc

print("End MPC:", end_mpc, " Directory that reached end_mpc: ", max_dir)


print("Seeds", all_seeds)
print("Ignored Seeds", ignored_seeds)
final_seeds = all_seeds - ignored_seeds
print("Final seeds", final_seeds)
### Store all the final distances here
x_traveled_experiments = {}
for svo_type in all_svos:
    x_traveled_experiments[svo_type] = {}
    for density in all_densities:
        x_traveled_experiments[svo_type][density] = []

### Get the directory data if its an admissable seed
for log_directory in list_of_experiments:
    with open(log_directory + "params.json",'rb') as fp:
        params = json.load(fp)    
    
    seed_number = params["seed"]
    if seed_number in ignored_seeds:
        continue
    
    svo = params["svo_theta"]
    svo_type = get_svo_type(svo)
    density = params["car_density"]
    
    data_filename = log_directory + 'data/all_%03d'%end_mpc
    if os.path.isfile(data_filename + "xamb.npy"):
        pass
    else:        
        data_filename = log_directory + 'data/all_%02d'%end_mpc
    xamb_actual, _, _, xothers_actual, _, _, = mpc.load_state(data_filename, params['n_other'], ignore_des=True)
    end_frame = xamb_actual.shape[1] #Not exactly sure why we need minus 1

    x_traveled_experiments[svo_type][density] += [xamb_actual[0, end_frame-1]]



for svo_type in x_traveled_experiments.keys():
    for density in x_traveled_experiments[svo_type].keys():
        x_traveled_experiments[svo_type][density] = np.array(x_traveled_experiments[svo_type][density])

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
# svo_types = ["e", "s", "p", "a"]
##### Print Table
print("SVO:  Dens.", *all_densities)

for svo_type in all_svos:
    # for density in all_densities:
    x_mean_distance = [np.mean(x_traveled_experiments[svo_type][density]) for density in all_densities]
    strFormat = len(x_mean_distance) * '& {:.2f}'
    formattedList = strFormat.format(*x_mean_distance)    
    print("%s: "%svo_type,  formattedList)

# print("Latex")



# print("& SVO: &Mean   &Median  &Std")
# for svo_type in svo_types:
#     x_traveled = x_traveled_experiments[svo_type]
#     print("& %s:  & %0.02f & %0.02f & %0.02f"%(svo_type, np.mean(x_traveled), np.median(x_traveled), np.std(x_traveled)))

# print("Number of Experiments:")
# for svo_type in svo_types:
#     x_traveled = x_traveled_experiments[svo_type]
#     print("%s: %d"%(svo_type, x_traveled.shape[0]))


# # print(prosocial_experiments)
# # print(egoistic_experiments)
# # print(altruistic_experiments)
