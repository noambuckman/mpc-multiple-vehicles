import itertools, os, pickle, datetime, json, copy
import numpy as np

from src.utils.ibr_argument_parser import IBRParser
from src.utils.log_management import random_date_string
from src.traffic_world import TrafficWorld
from src.ibr_nonamb import run_iterative_best_response
import src.utils.solver_helper as helper
import subprocess
from os.path import expanduser
def get_git_revision_hash() -> str:
    cwd = os.getcwd()

    home = expanduser("~")
    os.chdir(os.path.join(home, "mpc-multiple-vehicles"))
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    os.chdir(cwd)
    return commit


def run_simulation(log_dir, params):
    """ Runs a simulation with IBR using the determined svo matrix theta_ij
    Returns:  List of utilities and individual rewards
    """

    # Generate directory structure and log name
    params["start_time_string"] = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    params["N"] = max(1, int(params["T"] / params["dt"]))
    params["number_ctrl_pts_executed"] = max(1, int(np.floor(params["N"] * params["p_exec"])))
    params["pid"] = os.getpid()
    params["git_hash"] = get_git_revision_hash()
    # Create the world and vehicle objects
    world = TrafficWorld(params["n_lanes"], 0, 999999, lane_width = params["lane_width"])

    # Create the vehicle placement based on a Poisson distribution
    initial_velocity_mph = 25.0
    initial_velocity = initial_velocity_mph * 0.447
    try:
        position_list = helper.poission_positions(cars_per_hour=params["car_density"],
                                                  total_number_cars=params["n_other"] + 1,
                                                  average_velocity=initial_velocity,
                                                  position_random_seed=params["seed"])
    except Exception:
        print("Exception: Too many vehicles?")
        # with open(log_dir + "params.json", "w") as fp:
        #     json.dump(params, fp, indent=2)
        return None, None
    # Create the vehicles and initial positions

    if len(position_list) != params["n_other"] + 1:
        print("Returned not enough vehicles")
        # with open(log_dir + "params.json", "w") as fp:
        #     json.dump(params, fp, indent=2)
        return None, None

    (_, _, all_other_vehicles, all_other_x0) = helper.initialize_cars_from_positions(params["N"],
                                                                                     params["dt"],
                                                                                     world,
                                                                                     no_grass=True,
                                                                                     list_of_positions=position_list)

    for vehicle in all_other_vehicles:

        vehicle.grass_max_y = world.get_top_grass_y()[0] - vehicle.W/2.0
        vehicle.grass_min_y = world.get_bottom_grass_y()[2] + vehicle.W/2.0

        if params["strict_wall_constraint"]:
            vehicle.max_y = vehicle.grass_max_y
            vehicle.min_y = vehicle.grass_min_y
        else:
            vehicle.max_y = 99999.99
            vehicle.min_y = -99999.99            

    # Generate the SVOs for the vehicles
    setting_rng = np.random.default_rng(params["seed"])

    p_cooperative = params["p_cooperative"]
    n_cooperative = int(p_cooperative * params["n_other"])

    all_agent_ids = list(range(params["n_other"]))
    setting_rng.shuffle(all_agent_ids)
    cooperative_agents = all_agent_ids[:n_cooperative]

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
        all_other_vehicles[vehicle_it].max_v = setting_rng.uniform(initial_velocity_mph + 0.1, 30) * 0.447  #20 to 25 mph
        all_other_x0[vehicle_it][-2] = initial_velocity

        if 'k_lat' in params:
            all_other_vehicles[vehicle_it].k_lat = params['k_lat']
        if 'k_phi_dot' in params:
            all_other_vehicles[vehicle_it].k_phi_dot = params['k_phi_dot']
        if 'k_x_dot' in params:
            all_other_vehicles[vehicle_it].k_x_dot = params['k_x_dot']
        if 'k_on_grass' in params:
            all_other_vehicles[vehicle_it].k_on_grass = params['k_on_grass']


    # Save the vehicles and world for this simulation
    os.makedirs(log_dir, exist_ok=True)
    pickle.dump(all_other_vehicles, open(log_dir + "/other_vehicles.p", "wb"))
    pickle.dump(world, open(log_dir + "/world.p", "wb"))
    pickle.dump(all_other_x0, open(log_dir + "/x0.p", "wb"))
    print("Results saved in log %s:" % log_dir)
    # if args.plot_initial_positions:
    #     plot_initial_positions(log_dir, world, all_other_vehicles, all_other_x0)

    with open(log_dir + "params.json", "w") as fp:
        json.dump(params, fp, indent=2, sort_keys=True)

    if args.dry_run:
        return None, None
    xothers_actual, uothers_actual = run_iterative_best_response(all_other_vehicles, world, all_other_x0, params,
                                                                 log_dir)

    all_trajectories = np.array(xothers_actual)
    all_control = np.array(uothers_actual)

    return all_trajectories, all_control


if __name__ == "__main__":
    parser = IBRParser()
    parser.add_argument("my_task_id", default=0, type=int)
    parser.add_argument("num_tasks", default=1, type=int)
    parser.add_argument("--experiment-random-seed", type=int, default="0719")
    parser.add_argument("--input-params", type=str, default="experiments/experiment.json", help="Path to jason")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--dry-run", action='store_true')

    args = parser.parse_args()
    default_params = vars(args)

    # Grab the arguments that are passed in
    my_task_id = int(args.my_task_id)
    num_tasks = int(args.num_tasks)
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
        all_trajectories, all_control = run_simulation(log_dir, current_sim_params)
        if all_trajectories is None:  #We got an exception
            continue
        # Save the results within the log_dir
        np.save(open(log_dir + "/trajectories.npy", 'wb'), all_trajectories)
        np.save(open(log_dir + "/controls.npy", 'wb'), all_control)

    print(" Done with experiments")