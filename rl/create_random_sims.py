import os, pickle, datetime, string, random
import numpy as np
from src.ibr_argument_parser import IBRParser
from tqdm import trange

from src.traffic_world import TrafficWorld
import src.solver_helper as helper

from src.iterative_best_response import run_iterative_best_response
import json


def run_simulation(log_dir, params, theta_ij):
    """ Runs a simulation with IBR using the determined svo matrix theta_ij
    Returns:  List of utilities and individual rewards
    """

    # Generate directory structure and log name
    params["start_time_string"] = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Determine number of control points in the optimization
    i_mpc_start = 0
    params["N"] = max(1, int(params["T"] / params["dt"]))
    params["number_ctrl_pts_executed"] = max(1, int(np.floor(params["N"] * params["p_exec"])))

    # Create the world and vehicle objects
    world = TrafficWorld(params["n_lanes"], 0, 999999)

    # Create the vehicle placement based on a Poisson distribution
    MAX_VELOCITY = 25 * 0.447  # m/s
    VEHICLE_LENGTH = 4.5  # m
    time_duration_s = (params["n_other"] * 3600.0 / params["car_density"]) * 10  # amount of time to generate traffic
    initial_vehicle_positions = helper.poission_positions(
        params["car_density"],
        int(time_duration_s),
        params["n_lanes"],
        MAX_VELOCITY,
        VEHICLE_LENGTH,
        position_random_seed=params["seed"],
    )
    position_list = initial_vehicle_positions[:params["n_other"]]

    list_of_svo = theta_ij
    (
        ambulance,
        amb_x0,
        all_other_vehicles,
        all_other_x0,
    ) = helper.initialize_cars_from_positions(params["N"], params["dt"], world, True, position_list, list_of_svo)

    # Save the vehicles and world for this simulation
    data_dir = log_dir + "data"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)  # this should move into run_ibr

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

    xamb_actual, xothers_actual = run_iterative_best_response(
        params,
        log_dir,
        False,
        i_mpc_start,
        amb_x0,
        all_other_x0,
        ambulance,
        all_other_vehicles,
        world,
    )

    all_trajectories = [xamb_actual] + xothers_actual
    all_trajectories = np.array(all_trajectories)
    np.save(open(log_dir + "/trajectories.npy", 'wb'), all_trajectories)
    return all_trajectories


if __name__ == "__main__":
    parser = IBRParser()
    parser.add_argument("--max-svo", type=float, default=np.pi / 2.0, help="Max SVO we allow for random")
    parser.add_argument("--n-sims", type=int, default=10, help="Number of simulation")
    parser.add_argument('--svo-file',
                        type=str,
                        default=None,
                        help="load a pickl file with list of svos for non ambulance")

    parser.set_defaults(
        n_other=4,
        n_mpc=5,
        T=4,
    )
    args = parser.parse_args()
    params = vars(args)

    alpha_num = string.ascii_lowercase[:8] + string.digits

    experiment_string = ("".join(random.choice(alpha_num)
                                 for j in range(4)) + "-" + "".join(random.choice(alpha_num) for j in range(4)) + "-" +
                         datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    experiment_dir = (os.path.expanduser("~") + "/mpc_results/" + experiment_string)

    history_file = experiment_dir + "/" + experiment_string + "_history.p"
    print("History Saved in %s" % history_file)
    history = []
    for ep_ix in trange(params["n_sims"]):

        log_dir = experiment_dir + "/" + experiment_string + '_%05d' % ep_ix + "/"
        theta_ij = np.random.rand(params["n_other"], 1) * params["max_svo"]

        if args.svo_file is not None:
            with open(args.svo_file, 'rb') as f:
                svo_list = pickle.load(f)
                svo_ix = svo_list[ep_ix % len(svo_list)]
                theta_ij = np.array(svo_ix)

        all_vehicle_trajectories = run_simulation(log_dir, params, theta_ij)

        V_i_list = -(all_vehicle_trajectories[:, 0, -1] - all_vehicle_trajectories[:, 0, 0])
        ego_theta = 0.0  #current default in sims
        theta_ij = np.insert(theta_ij, 0, ego_theta)
        # Train a network to learn a function V(\theta_ij)
        history.append((theta_ij, V_i_list))

        with open(history_file, "wb") as f:
            pickle.dump(history, f)
