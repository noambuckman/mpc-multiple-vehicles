import os, pickle, json, glob, argparse
import numpy as np
import matplotlib.pyplot as plt


CONV_MS_TO_MPH = 2.23694


def find_sims_by_param(experiment_dir, param_names, param_values):
    ''' Load the directory and print the'''
    

    sim_dirs = glob.glob(os.path.join(experiment_dir,'results/*-*'))

    dirs = []
    
    for sim_dir in sim_dirs:
        all_params_match = True
        params_path = glob.glob(sim_dir + '/params.json')
        
        params = json.load(open(params_path[0],'rb'))
        for i in range(len(param_names)):
            if float(params[param_names[i]]) != param_values[i]:
                all_params_match = False
                break

        if all_params_match:                  
            dirs.append(sim_dir)

    for d in dirs:
        print(d)
    print("Total sims", len(dirs))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=str, help="experiment directory to analyze")
    parser.add_argument("--param-names", type=str, nargs="*", help="param names to find")
    parser.add_argument("--param-values", type=float, nargs="*", help="Value to find")
    args = parser.parse_args()

    find_sims_by_param(args.experiment_dir, args.param_names, args.param_values)
