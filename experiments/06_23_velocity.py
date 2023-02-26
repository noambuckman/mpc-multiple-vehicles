import datetime, os, pickle, json, string, random
import numpy as np
from src.traffic_world import TrafficWorld
from src.utils.ibr_argument_parser import IBRParser
import src.utils.solver_helper as helper
from src.ibr_nonamb import run_iterative_best_response

if __name__ == "__main__":
    parser = IBRParser()
    args = parser.parse_args()
    params = vars(args)

    # Generate directory structure and log name
    params["start_time_string"] = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    alpha_num = string.ascii_lowercase[:8] + string.digits
    if params["log_subdir"] is None:
        subdir_name = ("".join(random.choice(alpha_num) for j in range(4)) + "-" +
                       "".join(random.choice(alpha_num) for j in range(4)) + "-" + params["start_time_string"])
    else:
        subdir_name = params["log_subdir"]
    log_dir = os.path.expanduser("~") + "/mpc_results/" + subdir_name + "/"
    for f in [log_dir + "imgs/", log_dir + "data/", log_dir + "vids/", log_dir + "plots/"]:
        os.makedirs(f, exist_ok=True)

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
    initial_vehicle_positions = helper.poission_positions(params["car_density"],
                                                          params["n_other"],
                                                          params["n_lanes"],
                                                          MAX_VELOCITY,
                                                          VEHICLE_LENGTH,
                                                          position_random_seed=params["seed"])
    position_list = initial_vehicle_positions[:params["n_other"]]

    # Create the SVOs for each vehicle
    if params["random_svo"] == 1:
        list_of_svo = [np.random.choice([0, np.pi / 4.0, np.pi / 2.01]) for i in range(params["n_other"])]
    else:
        list_of_svo = [params["svo_theta"] for i in range(params["n_other"])]

    (_, _, all_other_vehicles,
     all_other_x0) = helper.initialize_cars_from_positions(params["N"], params["dt"], world, True, position_list,
                                                           list_of_svo)

    # ambulance = cp.deepcopy(all_other_vehicles[0])
    # ambulance.fd = ambulance.gen_f_desired_lane(world, 0, True)  # reset the desired lane since x0,y0 is different

    for vehicle in all_other_vehicles:
        # Set theta_ij randomly for all vehicles
        vehicle.theta_ij[-1] = np.random.uniform(0.001, np.pi / 2.01)
        for vehicle_j in all_other_vehicles:
            vehicle.theta_ij[vehicle_j.agent_id] = np.random.uniform(0.001, np.pi / 2.01)

        vehicle.max_v = np.random.uniform(25, 30) * 0.447  #20 to 25 mph

    # For fun, let's make the first vehicle much slower
    all_other_vehicles[0].max_v = 20.0 * 0.447
    all_other_vehicles[-2].max_v = 20.0 * 0.447
    all_other_vehicles[-1].max_v = 20.0 * 0.447

    for i in range(len(all_other_vehicles)):
        all_other_x0[i][-2] = 0.99 * all_other_vehicles[i].max_v  # set initial speed to the limit

    if params["k_lat"]:
        for vehicle in all_other_vehicles:
            vehicle.k_lat = params["k_lat"]
    # Save the vehicles and world for this simulation
    pickle.dump(all_other_vehicles, open(log_dir + "/other_vehicles.p", "wb"))
    pickle.dump(world, open(log_dir + "/world.p", "wb"))
    print("Results saved in log %s:" % log_dir)

    # Initialize the state and control arrays
    params["pid"] = os.getpid()
    if params["n_other"] != len(all_other_vehicles):
        raise Exception("n_other larger than  position list")
    with open(log_dir + "params.json", "w") as fp:
        json.dump(params, fp, indent=2)

    xothers_actual = run_iterative_best_response(all_other_vehicles, world, all_other_x0, params, log_dir,
                                                 args.load_log_dir, i_mpc_start)
    all_trajectories = xothers_actual
    all_trajectories = np.array(all_trajectories)
    np.save(open(log_dir + "/trajectories.npy", 'wb'), all_trajectories)
