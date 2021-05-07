import sys, random, itertools, string
import numpy as np

import os, pickle, datetime, random
import numpy as np
import json

from src.utils.ibr_argument_parser import IBRParser
from src.traffic_world import TrafficWorld
from src.iterative_best_response import run_iterative_best_response

import src.utils.solver_helper as helper


def run_simulation(log_dir, params, theta_ij):
    """ Runs a simulation with IBR using the determined svo matrix theta_ij
    Returns:  List of utilities and individual rewards
    """

    # Generate directory structure and log name
    params["start_time_string"] = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Determine number of control points in the optimization
    params["N"] = max(1, int(params["T"] / params["dt"]))
    params["number_ctrl_pts_executed"] = max(1, int(np.floor(params["N"] * params["p_exec"])))

    # Create the world and vehicle objects
    world = TrafficWorld(params["n_lanes"], 0, 999999)

    # Create the vehicle placement based on a Poisson distribution
    position_list = helper.poission_positions(cars_per_hour=params["car_density"],
                                              total_number_cars=params["n_other"],
                                              position_random_seed=params["seed"])

    (ambulance, amb_x0, all_other_vehicles,
     all_other_x0) = helper.initialize_cars_from_positions(params["N"],
                                                           params["dt"],
                                                           world,
                                                           list_of_positions=position_list,
                                                           list_of_svo=theta_ij)

    # Save the vehicles and world for this simulation
    os.makedirs(log_dir, exist_ok=True)
    pickle.dump(all_other_vehicles, open(log_dir + "/other_vehicles.p", "wb"))
    pickle.dump(ambulance, open(log_dir + "/ambulance.p", "wb"))
    pickle.dump(world, open(log_dir + "/world.p", "wb"))
    print("Results saved in log %s:" % log_dir)

    # Initialize the state and control arrays
    params["pid"] = os.getpid()
    if params["n_other"] != len(all_other_vehicles):
        raise Exception("n_other larger than  position list")
    with open(log_dir + "params.json", "w") as fp:
        json.dump(params, fp, indent=2)

    xamb_actual, xothers_actual = run_iterative_best_response(ambulance, all_other_vehicles, world, amb_x0,
                                                              all_other_x0, params, log_dir)

    all_trajectories = [xamb_actual] + xothers_actual
    all_trajectories = np.array(all_trajectories)
    return all_trajectories


if __name__ == "__main__":
    parser = IBRParser()
    parser.add_argument("my_task_id", type=int)
    parser.add_argument("num_tasks", type=int)
    parser.add_argument("--experiment_random_seed", type=int, default=0)

    args = parser.parse_args()
    default_params = vars(args)

    # Grab the arguments that are passed in
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])
    experiment_random_seed = default_params["experiment_random_seed"]

    all_params_dict = json.load('params.json')

    # Add the seeds based on number of experiments
    all_params_dict["seeds"] = [experiment_random_seed + ix for ix in range(all_params_dict["n_experiments"])]
    all_params = itertools.product(all_params_dict)

    my_params = all_params[my_task_id:len(all_params):num_tasks]

    sim_svos = []
    sim_trajs = []

    for params in my_params:

        for param in params:
            default_params[param] = params[param]

        alpha_num = string.ascii_lowercase[:8] + string.digits

        experiment_string = ("".join(random.choice(alpha_num)
                                     for j in range(4)) + "-" + "".join(random.choice(alpha_num) for j in range(4)) +
                             "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        experiment_dir = (os.path.expanduser("~") + "/mpc_results/" + experiment_string)
        log_dir = experiment_dir + "/" + experiment_string + '_%05d' % ep_ix + "/"

        # Generate the SVOs for the vehicles
        p_cooperative = params["p_cooperative"]
        n_cooperative = int(p_cooperative * params["n_other"])
        cooperative_agents = np.random.choice(range(params["n_other"]), size=n_cooperative, replace=False)
        theta_ij = [np.pi / 4.0 if i in cooperative_agents else 0.001 for i in range(params["n_other"])]

        all_trajectories = run_simulation(None, params, theta_ij)
        np.save(open(log_dir + "/trajectories.npy", 'wb'), all_trajectories)

        sim_trajs += [all_trajectories]
        sim_svos += [theta_ij]

    sim_trajs = np.array(sim_trajs)
    sim_svos = np.array(sim_svos)
    np.save(open("trajectories_%05d_%02d.npy" % (experiment_random_seed, my_task_id, 'wb')), sim_trajs)
    np.save(open("svos_%05d_%02d.npy" % (experiment_random_seed, my_task_id, 'wb')), sim_svos)
    print(" Done with experiments")