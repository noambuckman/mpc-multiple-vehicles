import numpy as np
from src.ibr_nonamb import run_iterative_best_response
from scripts.post_processing import load_initial_vars

from argparse import ArgumentParser





def rerun(log_dir: str):
    params, vehicles, world, x0 = load_initial_vars(log_dir)

    xothers_actual, uothers_actual = run_iterative_best_response(vehicles, world, x0, params, log_dir)

    all_trajectories = np.array(xothers_actual)
    all_control = np.array(uothers_actual)
    np.save(open(log_dir + "/trajectories.npy", 'wb'), all_trajectories)
    np.save(open(log_dir + "/controls.npy", 'wb'), all_control)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="Add the log_dir you want to rerun")
    
    args = parser.parse_args()

    print("Rerunning %s"%args.log_dir)
    # args.log_dir = "/home/nbuckman/mpc_results/supercloud/070322_multiple_trajs/results/6fdf-g115-20220703-122228_rerun2"
    rerun(args.log_dir)


    # log_dir = "/home/nbuckman/mpc_results/supercloud/070322_multiple_trajs/results/c7ga-5ega-20220703-122232"

    # rerun(log_dir)