import numpy as np
import matplotlib.pyplot as plt
import src.solver_helper as helper
from src.traffic_world import TrafficWorld
from src.car_plotting_multiple import plot_multiple_cars
from src.multiagent_mpc import MultiMPC, generate_warm_starts
from src.idm import IDM_acceleration, get_lead_vehicle, MOBIL_lanechange
from contextlib import redirect_stdout
from src.ibr_argument_parser import IBRParser
import datetime, string, random, os, pickle, json, time
import copy as cp


def run_idm_baseline(params, k_politeness, amb_x0, all_other_x0, ambulance, all_other_vehicles, world):
    # Current default params
    t_start_time = time.time()
    idm_params = {
        "desired_time_gap": 0.1,
        "jam_distance": 4,
    }

    params["wall_CA"] = 0  # wall_CA by default is zero for IDM since it's only being used for steering
    ######################################3
    i_mpc_start = 0
    params["N"] = max(1, int(params["T"] / params["dt"]))

    params["number_ctrl_pts_executed"] = max(1, int(np.floor(params["N"] * params["p_exec"])))

    n_mpc = params["n_mpc"]
    n_sim = params["n_mpc"] * params["number_ctrl_pts_executed"]  # corrected since

    X_other = [np.zeros((6, n_sim + 1)) for i in range(len(all_other_vehicles))]
    X_amb = np.zeros((6, n_sim + 1))

    X_amb[:, 0] = amb_x0
    for i in range(len(X_other)):
        X_other[i][:, 0] = all_other_x0[i]

    out_file = open(log_dir + "out.txt", "w")

    for t in range(n_sim):
        current_other_x0 = [X_other[i][:, t] for i in range(len(X_other))]
        current_amb_x0 = X_amb[:, t]
        current_other_veh = [all_other_vehicles[i] for i in range(len(X_other))]

        # Mobile lane change
        driver_x0 = current_amb_x0
        driver_veh = ambulance
        all_other_x0 = current_other_x0
        MOBIL_params = {
            "politeness_factor": 0.001,  #ambulance is never polite
        }

        # Compute whether lane change occurs
        desired_lane = True
        new_lane, accel = MOBIL_lanechange(driver_x0, driver_veh, all_other_x0, current_other_veh, world, desired_lane,
                                           MOBIL_params, idm_params)
        if new_lane is not None:
            driver_veh.update_desired_lane(world, new_lane, True)

        # Compute IDM acceleration (ambulance)
        lead_veh = get_lead_vehicle(current_amb_x0, current_other_x0, world)
        if lead_veh is None:
            x_lead = current_amb_x0 + 999999
            v_lead = 999999
        else:
            x_lead = current_other_x0[lead_veh]
            v_lead = x_lead[4] * np.cos(x_lead[2])

        v_current = current_amb_x0[4] * np.cos(current_amb_x0[2])
        v_desired = ambulance.max_v
        bumper_distance = x_lead[0] - current_amb_x0[0]

        idm_params["maximum_acceleration"] = ambulance.max_acceleration
        a_IDM = IDM_acceleration(bumper_distance, v_lead, v_current, v_desired, idm_params)

        # Use MPC to solve the ambulance steering

        # Increase k_lat and k_lan since our only goal is to execute the lane change (i.e. kinematic)
        ambulance.k_lat = 5.0
        ambulance.k_lan = 2.0
        ambulance.k_x_dot = 0.0
        ambulance.k_final = 0.0

        solver_params = {
            "k_slack": 1000,
            "k_CA": 0.05,
            "k_CA_power": 1.0,
        }

        warm_starts = generate_warm_starts(ambulance, world, current_amb_x0, [], params)
        u = None
        for k_warm in warm_starts:
            try:
                temp_veh = cp.deepcopy(ambulance)
                temp_veh.max_v = np.infty  #max the optimization a little easier
                temp_veh.min_v = -np.infty  #max the optimization a little easier
                temp_veh.min_y = -np.infty  #max the optimization a little easier
                temp_veh.max_y = np.infty  #max the optimization a little easier

                steering_mpc = MultiMPC(temp_veh, [], [], world, solver_params)
                steering_mpc.generate_optimization(params["N"],
                                                   current_amb_x0, [], [],
                                                   params=params,
                                                   ipopt_params={'print_level': 0})

                u_warm, x_warm, x_des_warm = warm_starts[k_warm]

                steering_mpc.opti.set_initial(steering_mpc.u_ego, u_warm)
                steering_mpc.opti.set_initial(steering_mpc.x_ego, x_warm)
                steering_mpc.opti.set_initial(steering_mpc.xd_ego, x_des_warm)
                # with redirect_stdout(out_file):
                with redirect_stdout(out_file):
                    steering_mpc.solve(None, None)
                _, u, _ = steering_mpc.get_bestresponse_solution()
            except RuntimeError:
                print("Solver didn't work")

            if u is not None:
                break
        if u is None:
            print("Control input was not returned")
            return X_amb, X_other

        # Update control and step the simulator
        u_ego = np.array([[u[0, 0]], [a_IDM * ambulance.dt]])

        # Update state with new control inputs
        x_ego_traj, _ = ambulance.forward_simulate_all(current_amb_x0, u_ego)
        X_amb[:, t + 1] = x_ego_traj[:, 1]

        # Other vehicles
        for ego_idx in range(len(all_other_vehicles)):

            # Mobile lane change
            driver_x0 = current_other_x0[ego_idx]
            driver_veh = all_other_vehicles[ego_idx]
            all_other_x0 = current_other_x0[:ego_idx] + current_other_x0[ego_idx + 1:] + [current_amb_x0]
            all_other_veh = all_other_vehicles[:ego_idx] + all_other_vehicles[ego_idx + 1:] + [ambulance]

            if k_politeness is None:
                k_politeness = driver_veh.theta_ij[-1] / (np.pi / 2.0)  #this rescales between 0 and 1
            MOBIL_params = {
                "politeness_factor": k_politeness,
            }
            new_lane, _ = MOBIL_lanechange(driver_x0, driver_veh, all_other_x0, all_other_veh, world, desired_lane,
                                           MOBIL_params, idm_params)
            if new_lane is not None:
                driver_veh.update_desired_lane(world, new_lane, True)

            driver_x0 = current_other_x0[ego_idx]
            dummy_x0 = current_other_x0[ego_idx] - 10000
            ado_x0s = current_other_x0[:ego_idx] + [dummy_x0] + current_other_x0[ego_idx + 1:] + [current_amb_x0]

            lead_veh = get_lead_vehicle(current_other_x0[ego_idx], ado_x0s, world)
            if lead_veh is None:
                x_lead = driver_x0 + 999999
                v_lead = 999999
            else:
                x_lead = ado_x0s[lead_veh]
                v_lead = x_lead[4] * np.cos(x_lead[2])

            v_current = driver_x0[4] * np.cos(driver_x0[2])
            v_desired = driver_veh.max_v
            #         v_desired = 0.001
            v_desired = driver_veh.max_v
            bumper_distance = x_lead[0] - driver_x0[0]

            idm_params["maximum_acceleration"] = driver_veh.max_acceleration
            a_IDM = IDM_acceleration(bumper_distance, v_lead, v_current, v_desired, idm_params)

            # Solve for steering angle
            driver_veh.k_lat = 5.0
            driver_veh.k_lan = 2.0
            driver_veh.k_x_dot = 0.0
            driver_veh.k_final = 0.0

            solver_params = {
                "k_slack": 1000,
                "k_CA": 0.05,
                "k_CA_power": 1.0,
                #         "constant_v": True,
            }

            warm_starts = generate_warm_starts(driver_veh, world, driver_x0, [], params)
            u = None
            for k_warm in warm_starts:
                try:
                    temp_veh = cp.deepcopy(driver_veh)
                    temp_veh.max_v = np.infty  #max the optimization a little easier
                    temp_veh.min_v = -np.infty  #max the optimization a little easier
                    temp_veh.min_y = -np.infty  #max the optimization a little easier
                    temp_veh.max_y = np.infty  #max the optimization a little easier

                    steering_mpc = MultiMPC(temp_veh, [], [], world, solver_params)
                    n_mpc = 5
                    steering_mpc.generate_optimization(params["N"],
                                                       driver_x0, [], [],
                                                       params=params,
                                                       ipopt_params={'print_level': 0})

                    u_warm, x_warm, x_des_warm = warm_starts[k_warm]

                    steering_mpc.opti.set_initial(steering_mpc.u_ego, u_warm)
                    steering_mpc.opti.set_initial(steering_mpc.x_ego, x_warm)
                    steering_mpc.opti.set_initial(steering_mpc.xd_ego, x_des_warm)
                    with redirect_stdout(out_file):
                        steering_mpc.solve(None, None)
                    _, u, _ = steering_mpc.get_bestresponse_solution()
                except RuntimeError:
                    print("Solver didn't work")

                if u is not None:
                    break

            if u is None:
                print("Control input was not returned")
                return X_amb, X_other

            # Update control and step the simulator
            u_ego = np.array([[u[0, 0]], [a_IDM * driver_veh.dt]])

            # Update state with new control inputs
            x_ego_traj, _ = driver_veh.forward_simulate_all(driver_x0, u_ego)
            X_other[ego_idx][:, t + 1] = x_ego_traj[:, 1]

    X_amb = X_amb[:, :t + 1]
    X_other = [x[:, :t + 1] for x in X_other]
    out_file.close()
    print("Simulation Done!  Runtime: %s" % (datetime.timedelta(seconds=(time.time() - t_start_time))))

    return X_amb, X_other


if __name__ == "__main__":
    parser = IBRParser()
    args = parser.parse_args()
    params = vars(args)

    ### Generate directory structure and log name
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

    ### Determine number of control points in the optimization
    i_mpc_start = 0
    params["N"] = max(1, int(params["T"] / params["dt"]))
    params["number_ctrl_pts_executed"] = max(1, int(np.floor(params["N"] * params["p_exec"])))

    ### Create the world and vehicle objects
    world = TrafficWorld(params["n_lanes"], 0, 999999)

    ### Create the vehicle placement based on a Poisson distribution
    MAX_VELOCITY = 25 * 0.447  # m/s
    VEHICLE_LENGTH = 4.5  # m
    time_duration_s = (params["n_other"] * 3600.0 / params["car_density"]) * 10  # amount of time to generate traffic
    initial_vehicle_positions = helper.poission_positions(params["car_density"],
                                                          int(time_duration_s),
                                                          params["n_lanes"],
                                                          MAX_VELOCITY,
                                                          VEHICLE_LENGTH,
                                                          position_random_seed=params["seed"])
    position_list = initial_vehicle_positions[:params["n_other"]]

    ### Create the SVOs for each vehicle
    if params["random_svo"] == 1:
        list_of_svo = [np.random.choice([0, np.pi / 4.0, np.pi / 2.01]) for i in range(params["n_other"])]
    else:
        list_of_svo = [params["svo_theta"] for i in range(params["n_other"])]

    (ambulance, amb_x0, all_other_vehicles,
     all_other_x0) = helper.initialize_cars_from_positions(params["N"], params["dt"], world, True, position_list,
                                                           list_of_svo)
    print("svos", [v.theta_ij[-1] for v in all_other_vehicles])
    if params["k_lat"]:
        for vehicle in all_other_vehicles:
            vehicle.k_lat = params["k_lat"]
    ### Save the vehicles and world for this simulation
    for i in range(len(all_other_vehicles)):
        pickle.dump(all_other_vehicles[i], open(log_dir + "data/mpcother%03d.p" % i, "wb"))
    pickle.dump(all_other_vehicles, open(log_dir + "/other_vehicles.p", "wb"))
    pickle.dump(ambulance, open(log_dir + "/ambulance.p", "wb"))
    pickle.dump(world, open(log_dir + "data/world.p", "wb"))
    print("Results saved in log %s:" % log_dir)

    #### Initialize the state and control arrays
    params["pid"] = os.getpid()
    if params["n_other"] != len(all_other_vehicles):
        raise Exception("n_other larger than  position list")
    with open(log_dir + "params.json", "w") as fp:
        json.dump(params, fp, indent=2)

    k_politeness = None
    xamb_actual, xothers_actual = run_idm_baseline(params, k_politeness, amb_x0, all_other_x0, ambulance,
                                                   all_other_vehicles, world)

    all_trajectories = [xamb_actual] + xothers_actual
    all_trajectories = np.array(all_trajectories)
    np.save(open(log_dir + "/trajectories.npy", 'wb'), all_trajectories)
