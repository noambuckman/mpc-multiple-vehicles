import argparse
import os, pickle, json
import numpy as np
from src.ibr_nonamb import run_iterative_best_response

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=str, help="log directory for experiment to rerun")
    parser.add_argument("--i-mpc-start", int=0, help="Which iteration of mpc to start")
    args = parser.parse_args()

    print("Preloading settings from log %s" % args.log_dir)

    with open(args.log_dir + "params.json", "rb") as fp:
        params = json.load(fp)
    params["pid"] = os.getpid()

    vehicles = pickle.load(open(args.log_dir + "other_vehicles.p", "rb"))
    world = pickle.load(open(args.log_dir + "world.p", "rb"))

    # Load trajectory or start from save x0
    if args.mpc_start_iteration > 0:
        try:
            TRAJ_PATH = os.path.join(args.log_dir, "trajectories.npy")
            traj = np.load(TRAJ_PATH)
        except:
            raise Exception("Missing saved trajectory at %s" % TRAJ_PATH)

        time_idx = params["N"] * args.mpc_start_iteration
        x0 = traj[:, :, time_idx:time_idx + 1]
    else:
        try:
            X0_PATH = os.path.join(args.log_dir, "x0.p")
            x0 = pickle.load(open(X0_PATH, "rb"))
        except:
            raise Exception("Missing save initial conditions (x0) at %d" % X0_PATH)

    # Run the simulation
    x, u = run_iterative_best_response(vehicles, world, x0, params, args.log_dir, False, 0)

    all_trajectories = np.array(x)
    all_control = np.array(u)
    np.save(open(args.log_dir + "/trajectories.npy", 'wb'), all_trajectories)
    np.save(open(args.log_dir + "/controls.npy", 'wb'), all_control)