import sys, itertools, os, pickle, datetime, json, copy
import numpy as np

from src.utils.ibr_argument_parser import IBRParser
from src.utils.log_management import random_date_string
from src.traffic_world import TrafficWorld
from src.ibr_nonamb import run_iterative_best_response
import src.utils.solver_helper as helper


def run_simulation(log_dir, params):
    """ Runs a simulation with IBR using the determined svo matrix theta_ij
    Returns:  List of utilities and individual rewards
    """

    # Generate directory structure and log name
    params["start_time_string"] = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    params["N"] = max(1, int(params["T"] / params["dt"]))
    params["number_ctrl_pts_executed"] = max(1, int(np.floor(params["N"] * params["p_exec"])))
    params["pid"] = os.getpid()

    # Create the world and vehicle objects
    world = TrafficWorld(params["n_lanes"], 0, 999999)

    # Create the vehicle placement based on a Poisson distribution
    initial_velocity = 25 * 0.447

    try:
        position_list = helper.poission_positions(cars_per_hour=params["car_density"],
                                                  total_number_cars=params["n_other"] + 1,
                                                  average_velocity=initial_velocity,
                                                  position_random_seed=params["seed"])
    except Exception:
        print("Too many vehicles?")
        with open(log_dir + "params.json", "w") as fp:
            json.dump(params, fp, indent=2)
        return None
    # Create the vehicles and initial positions
    (_, _, all_other_vehicles, all_other_x0) = helper.initialize_cars_from_positions(params["N"],
                                                                                     params["dt"],
                                                                                     world,
                                                                                     no_grass=True,
                                                                                     list_of_positions=position_list)

    # Generate the SVOs for the vehicles
    p_cooperative = current_sim_params["p_cooperative"]
    n_cooperative = int(p_cooperative * current_sim_params["n_other"])
    cooperative_agents = np.random.choice(range(current_sim_params["n_other"]), size=n_cooperative, replace=False)

    for vehicle_it in range(len(all_other_vehicles)):
        vehicle = all_other_vehicles[vehicle_it]
        if vehicle_it in cooperative_agents:
            svo = np.pi / 4.0
        else:
            svo = 0.00001

        vehicle.theta_ij[-1] = svo
        for vehicle_j in all_other_vehicles:
            vehicle.theta_ij[vehicle_j.agent_id] = svo

    # Set the max velocity and initial velocity
    for vehicle_it in range(len(all_other_vehicles)):
        all_other_vehicles[vehicle_it].max_v = np.random.uniform(25, 30) * 0.447  #20 to 25 mph
        all_other_x0[vehicle_it][-2] = initial_velocity

    # Save the vehicles and world for this simulation
    os.makedirs(log_dir, exist_ok=True)
    pickle.dump(all_other_vehicles, open(log_dir + "/other_vehicles.p", "wb"))
    pickle.dump(world, open(log_dir + "/world.p", "wb"))
    print("Results saved in log %s:" % log_dir)

    # Initialize the state and control arrays

    with open(log_dir + "params.json", "w") as fp:
        json.dump(params, fp, indent=2)

    xothers_actual = run_iterative_best_response(all_other_vehicles, world, all_other_x0, params, log_dir)

    all_trajectories = np.array(xothers_actual)
    return all_trajectories


if __name__ == "__main__":
    parser = IBRParser()
    parser.add_argument("my_task_id", type=int)
    parser.add_argument("num_tasks", type=int)
    parser.add_argument("--experiment-random-seed", type=int, default=0)
    parser.add_argument("--input-params", type=str, default=None, help="Path to jason")
    parser.add_argument("--results-dir", type=str, default=None)

    args = parser.parse_args()
    default_params = vars(args)

    # Grab the arguments that are passed in
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])
    experiment_random_seed = args.experiment_random_seed

    if args.input_params is None:
        all_params_dict = {"n_experiments": 1, "p_cooperative": [0.5]}
    else:
        all_params_dict = json.load(open(args.input_params, 'rb'))
    # Add the seeds based on number of experiments
    all_params_dict["seed"] = [experiment_random_seed + ix for ix in range(all_params_dict["n_experiments"])]

    # Generate a list of all experiment's param dicts
    all_params = []
    for param, value in all_params_dict.items():
        if type(value) is not list:
            all_params_dict[param] = [value]
    keys, values = zip(*all_params_dict.items())
    for experiment_pairings in itertools.product(*values):
        exp_params = dict(zip(keys, experiment_pairings))
        all_params += [exp_params]

    # Select a subset of paramaters and update the params file
    my_params = all_params[my_task_id:len(all_params):num_tasks]

    for params in my_params:
        current_sim_params = copy.deepcopy(default_params)
        for param in params:
            current_sim_params[param] = params[param]

        experiment_string = random_date_string()

        if args.results_dir is None:
            all_results_dir = os.path.expanduser("~") + "/mpc_results/" + experiment_string + "/"
        else:
            all_results_dir = args.results_dir
        log_dir = all_results_dir + experiment_string + "/"

        # Run the simulation with current sim params
        all_trajectories = run_simulation(log_dir, current_sim_params)
        if all_trajectories is None:  #We got an exception
            continue
        # Save the results within the log_dir
        np.save(open(log_dir + "/trajectories.npy", 'wb'), all_trajectories)

    print(" Done with experiments")