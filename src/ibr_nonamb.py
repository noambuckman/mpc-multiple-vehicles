import time, datetime, os, pickle, psutil, json, string, random
import numpy as np
import copy as cp
from contextlib import redirect_stdout
from typing import List

from src.traffic_world import TrafficWorld
from src.warm_starts import generate_warm_starts, warm_profiles_subset
from src.utils.ibr_argument_parser import IBRParser
from src.best_response import solve_warm_starts
from src.utils.solver_helper import poission_positions, extend_last_mpc_and_follow, initialize_cars_from_positions
from src.vehicle_mpc_information import VehicleMPCInformation
from src.utils.plotting.car_plotting import plot_initial_positions
from src.utils.sim_utils import get_best_responding_vehicles_idxs, assign_shared_control, get_within_range_other_vehicle_idxs, save_ibr, check_collisions, update_sim_states, load_log_data


def run_iterative_best_response(other_vehicles,
                                world: TrafficWorld,
                                other_x0: List[np.array],
                                params: dict,
                                log_dir: str = None,
                                load_log_dir: bool = False,
                                i_mpc_start: int = 0):
    """ 
        Runs iterative best response for a system of ambulance and other vehicles.
        TODO:  Add something that checks required params.  Or has default params somewhere.
    """
    out_file = open(log_dir + "out.txt", "w")
    if params["save_ibr"]:
        os.makedirs(log_dir + "data/", exist_ok=True)
    # Initialize all state and control arrays for entire simulation
    t_start_time = time.time()
    actual_t = 0
    n_ctrl_total = params["n_mpc"] * params["number_ctrl_pts_executed"]
    xothers_actual = [
        np.zeros((6, n_ctrl_total + 1)) for _ in range(params["n_other"])
    ]
    uothers_actual = [
        np.zeros((2, n_ctrl_total)) for _ in range(params["n_other"])
    ]

    if load_log_dir:
        all_other_x_executed, all_other_u_mpc, xothers_actual, uothers_actual, actual_t = load_log_data(
            params, log_dir, i_mpc_start, xothers_actual, uothers_actual)
    else:
        all_other_x_executed = [None for i in range(params["n_other"])]
        all_other_u_mpc = [None for i in range(params["n_other"])]

    # Run the simulation and solve mpc for all vehicles for each round of MPC
    for i_mpc in range(i_mpc_start, params["n_mpc"]):

        # 1) Update the initial conditions for all vehicles and normalize wrt ambulance position
        if i_mpc > 0:
            all_other_x0_g = [
                cp.deepcopy(all_other_x_executed[i]
                            [:, params["number_ctrl_pts_executed"]])
                for i in range(len(all_other_x_executed))
            ]
            other_x0 = cp.deepcopy(all_other_x0_g)

        # 3) Generate (if needed) the control inputs of other vehicles
        with redirect_stdout(out_file):
            try:
                if i_mpc == 0:
                    previous_all_other_u_mpc = [
                        np.zeros((2, params["N"]))
                        for i in range(len(other_vehicles))
                    ]
                    other_u_ibr_initial, other_x_ibr_initial, other_x_des_ibr_initial = extend_last_mpc_and_follow(
                        previous_all_other_u_mpc, params["N"] - 1,
                        other_vehicles, other_x0, params, world)
                else:
                    other_u_ibr_initial, other_x_ibr_initial, other_x_des_ibr_initial = extend_last_mpc_and_follow(
                        all_other_u_mpc, params["number_ctrl_pts_executed"],
                        other_vehicles, other_x0, params, world)
            except RuntimeError as e:
                print(e)
                print(
                    "Simulation Ended Early due to infeasible solution!  Runtime: %s"
                    %
                    (datetime.timedelta(seconds=(time.time() - t_start_time))))
                return xothers_actual, uothers_actual

        othervehs_ibr_info = []
        for i in range(len(other_vehicles)):
            othervehs_ibr_info.append(
                VehicleMPCInformation(other_vehicles[i], other_x0[i],
                                      other_u_ibr_initial[i],
                                      other_x_ibr_initial[i],
                                      other_x_des_ibr_initial[i]))

        vehs_ibr_info_predicted = cp.deepcopy(othervehs_ibr_info)

        for i_rounds_ibr in range(params["n_ibr"]):
            print("MPC %d, IBR %d / %d" %
                  (i_mpc, i_rounds_ibr, params["n_ibr"] - 1))

            vehicles_index_best_responders = get_best_responding_vehicles_idxs(
                other_vehicles)
            for response_i in vehicles_index_best_responders:

                # Assign the response veh, shared cntrl vehicles, and non-response vehicles
                response_veh_info = othervehs_ibr_info[response_i]
                veh_idxs_in_mpc = get_within_range_other_vehicle_idxs(
                    response_i, othervehs_ibr_info)
                cntrld_vehicle_info, nonresponse_veh_info, cntrld_i = assign_shared_control(
                    params, i_rounds_ibr, veh_idxs_in_mpc,
                    vehicles_index_best_responders, response_veh_info,
                    vehs_ibr_info_predicted)

                warm_starts = generate_warm_starts(
                    response_veh_info.vehicle, world, response_veh_info.x0,
                    vehs_ibr_info_predicted, params,
                    all_other_u_mpc[response_i],
                    othervehs_ibr_info[response_i].u)

                solve_again, solve_number, max_slack_ibr, debug_flag, = (
                    True, 0, np.infty, False)

                #TODO 11/23/21:  We should also warm start the trajectories for controlled vehicle by cp.copy cntrld_veh_info and updating trajectory

                # 6/9/21:  Wha are these ambulance parameters? Will they matter anymore?
                params["k_solve_amb_min_distance"] = 50
                solver_params = {}
                solver_params["solve_amb"] = (True if (
                    i_rounds_ibr < params["k_solve_amb_max_ibr"]) else False)
                solver_params["slack"] = (
                    True if i_rounds_ibr <= params["k_max_round_with_slack"]
                    else False)
                solver_params["n_warm_starts"] = params[
                    "default_n_warm_starts"]
                solver_params["k_CA"] = params["k_CA_d"]
                solver_params["k_CA_power"] = params["k_CA_power"]

                # TODO:  Some response vehicles are not being solved for and we can just set them to constant velocity
                # if (response_i not in veh_idxs_in_amb_mpc):
                #     solver_params["constant_v"] = True

                print("...Veh %02d Solver:" % response_i)
                while solve_number < params["k_max_solve_number"]:
                    solver_params["k_slack"] = (params["k_slack_d"] *
                                                10**solve_number)
                    solver_params["n_warm_starts"] = (
                        solver_params["n_warm_starts"] + 5 * solve_number)
                    if solve_number > 2:
                        debug_flag = True
                    warmstarts_subset = warm_profiles_subset(
                        solver_params["n_warm_starts"], warm_starts)

                    if psutil.virtual_memory().percent >= 95.0:
                        raise Exception(
                            "Virtual Memory is too high, exiting to save computer"
                        )
                    start_ipopt_time = time.time()

                    ipopt_params = {"print_level": params["print_level"]}

                    with redirect_stdout(out_file):
                        print("MPC %d, IBR %d / %d" %
                              (i_mpc, i_rounds_ibr, params["n_ibr"] - 1))
                        print("....# Cntrld Vehicles: %d # Non-Response: %d " %
                              (len(cntrld_vehicle_info),
                               len(nonresponse_veh_info)))
                        solved, min_cost_ibr, max_slack_ibr, x_ibr, x_des_ibr, u_ibr, key_ibr, debug_list, cntrld_vehicle_trajectories = solve_warm_starts(
                            warmstarts_subset, response_veh_info, world,
                            solver_params, params, ipopt_params,
                            nonresponse_veh_info, cntrld_vehicle_info, params["k_max_slack"], 
                            debug_flag)
                    end_ipopt_time = time.time()

                    if max_slack_ibr <= params[
                            "k_max_slack"] and max_slack_ibr < np.infty:
                        othervehs_ibr_info[response_i].update_state(
                            u_ibr, x_ibr, x_des_ibr)
                        vehs_ibr_info_predicted[response_i].update_state(
                            u_ibr, x_ibr, x_des_ibr)
                        # New 11/18: Also update the controlled veh info within the predicted positions
                        for cntrld_i_idx, vehicle_id in enumerate(cntrld_i):
                            x_jc, x_des_jc, u_jc = cntrld_vehicle_trajectories[
                                cntrld_i_idx]
                            vehs_ibr_info_predicted[vehicle_id].update_state(
                                u_jc, x_jc, x_des_jc)

                        print(
                            "......Solved.  veh: %02d | mpc: %d | ibr: %d | solver time: %0.1f s"
                            % (response_i, i_mpc, i_rounds_ibr,
                               end_ipopt_time - start_ipopt_time))
                        break
                    else:
                        print(
                            "......Re-solve. veh: %02d | mpc: %d | ibr: %d | Slack %.05f > thresh %.05f  | solver time: %0.1f"
                            % (response_i, i_mpc, i_rounds_ibr, max_slack_ibr,
                               params["k_max_slack"],
                               end_ipopt_time - start_ipopt_time))
                        solve_number += 1
                        if (solve_number == params["k_max_solve_number"]
                            ) and solver_params["solve_amb"]:
                            print("Re-solving without ambulance")
                            solve_number = 0
                            solver_params["solve_amb"] = False

                if solve_number == params["k_max_solve_number"]:
                    if i_rounds_ibr > 0:
                        default_traj = warm_starts['previous_ibr']
                    else:
                        default_traj = warm_starts['previous_mpc_hold']

                    othervehs_ibr_info[response_i].update_state(
                        default_traj.u, default_traj.x, default_traj.xd)
                    vehs_ibr_info_predicted[response_i].update_state(
                        default_traj.u, default_traj.x, default_traj.xd)

                    print(
                        "......Reached max # resolves. veh: %02d | mpc: %d | ibr: %d | Slack %.05f > thresh %.05f  | solver time: %0.1f"
                        % (response_i, i_mpc, i_rounds_ibr, max_slack_ibr,
                           params["k_max_slack"],
                           end_ipopt_time - start_ipopt_time))

                if params["save_ibr"]:
                    save_ibr(log_dir, i_mpc, i_rounds_ibr, response_i,
                             vehs_ibr_info_predicted)

            # TODO: Scale the units to a region of interest so that xj is always in reference to xi
            other_x_ibr_temp = np.stack([veh.x for veh in othervehs_ibr_info])
            all_other_x_ibr_g = [x for x in other_x_ibr_temp]

            if params["save_ibr"]:
                save_ibr(log_dir, i_mpc, i_rounds_ibr, None,
                         vehs_ibr_info_predicted)

        actual_t, all_other_u_mpc, all_other_x_executed, all_other_u_executed, xothers_actual, uothers_actual, = update_sim_states(
            othervehs_ibr_info, all_other_x_ibr_g, params, actual_t, log_dir,
            i_mpc, xothers_actual, uothers_actual)

        collision = check_collisions(other_vehicles, all_other_x_executed)
        if collision:
            out_file.close()
            print("Simulation Ended Early due to collision!  Runtime: %s" %
                  (datetime.timedelta(seconds=(time.time() - t_start_time))))
            return xothers_actual, uothers_actual

    out_file.close()

    print("Simulation Done!  Runtime: %s" %
          (datetime.timedelta(seconds=(time.time() - t_start_time))))

    return xothers_actual, uothers_actual


if __name__ == "__main__":
    parser = IBRParser()
    args = parser.parse_args()
    params = vars(args)
    # args.load_log_dir = "/home/nbuckman/mpc_results/investigation/h777-4664-20211031-204805/"
    if args.load_log_dir is None:
        # Generate directory structure and log name
        print("New stuff")
        params["start_time_string"] = datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S")
        alpha_num = string.ascii_lowercase[:8] + string.digits
        if params["log_subdir"] is None:
            subdir_name = ("".join(random.choice(alpha_num)
                                   for j in range(4)) + "-" +
                           "".join(random.choice(alpha_num)
                                   for j in range(4)) + "-" +
                           params["start_time_string"])
        else:
            subdir_name = params["log_subdir"]
        log_dir = os.path.expanduser("~") + "/mpc_results/" + subdir_name + "/"
        for f in [
                log_dir + "imgs/", log_dir + "data/", log_dir + "vids/",
                log_dir + "plots/"
        ]:
            os.makedirs(f, exist_ok=True)

        # Determine number of control points in the optimization
        i_mpc_start = 0
        params["N"] = max(1, int(params["T"] / params["dt"]))
        params["number_ctrl_pts_executed"] = max(
            1, int(np.floor(params["N"] * params["p_exec"])))

        # Create the world and vehicle objects
        world = TrafficWorld(params["n_lanes"], 0, 999999)

        # Create the vehicle placement based on a Poisson distribution
        MAX_VELOCITY = 25 * 0.447  # m/s
        VEHICLE_LENGTH = 4.5  # m
        time_duration_s = (params["n_other"] * 3600.0 / params["car_density"]
                           ) * 10  # amount of time to generate traffic
        initial_vehicle_positions = poission_positions(
            params["car_density"],
            params["n_other"] + 1,
            params["n_lanes"],
            MAX_VELOCITY,
            2 * VEHICLE_LENGTH,
            position_random_seed=params["seed"])
        position_list = initial_vehicle_positions[:params["n_other"] + 1]

        # Create the SVOs for each vehicle
        if params["random_svo"] == 1:
            list_of_svo = [
                np.random.choice([0, np.pi / 4.0, np.pi / 2.01])
                for i in range(params["n_other"])
            ]
        else:
            list_of_svo = [
                params["svo_theta"] for i in range(params["n_other"])
            ]

        (_, _, all_other_vehicles,
         all_other_x0) = initialize_cars_from_positions(
             params["N"], params["dt"], world, True, position_list,
             list_of_svo)

        # ambulance = cp.deepcopy(all_other_vehicles[0])
        # ambulance.fd = ambulance.gen_f_desired_lane(world, 0, True)  # reset the desired lane since x0,y0 is different

        for vehicle in all_other_vehicles:
            # Set theta_ij randomly for all vehicles
            vehicle.theta_ij[-1] = np.random.uniform(0.001, np.pi / 2.01)
            for vehicle_j in all_other_vehicles:
                vehicle.theta_ij[vehicle_j.agent_id] = np.random.uniform(
                    0.001, np.pi / 2.01)

        if params["k_lat"]:
            for vehicle in all_other_vehicles:
                vehicle.k_lat = params["k_lat"]
        # Save the vehicles and world for this simulation
        pickle.dump(all_other_vehicles,
                    open(log_dir + "/other_vehicles.p", "wb"))
        pickle.dump(world, open(log_dir + "/world.p", "wb"))
        pickle.dump(all_other_x0, open(log_dir + "/x0.p", "wb"))
        if args.plot_initial_positions:
            plot_initial_positions(log_dir, world, all_other_vehicles,
                                   all_other_x0)

        print("Results saved in log %s:" % log_dir)
    else:
        print("Preloading settings from log %s" % args.load_log_dir)
        log_dir = args.load_log_dir
        with open(args.load_log_dir + "params.json", "rb") as fp:
            params = json.load(fp)
        i_mpc_start = args.mpc_start_iteration
        all_other_vehicles = pickle.load(
            open(log_dir + "other_vehicles.p", "rb"))
        world = pickle.load(open(log_dir + "world.p", "rb"))
        params["pid"] = os.getpid()
        # all_trajectories = np.load(open(log_dir + "trajectories.npy", 'rb'), allow_pickle=True)
        all_other_x0 = pickle.load(open(log_dir + "x0.p", "rb"))
        # all_other_x0 = [all_trajectories[i, :, 0] for i in range(all_trajectories.shape[0])]
        args.load_log_dir = False
        # We need to get initial conditions for the iterative best response

    # Initialize the state and control arrays
    params["pid"] = os.getpid()
    if params["n_other"] != len(all_other_vehicles):
        raise Exception("n_other larger than  position list")
    with open(log_dir + "params.json", "w") as fp:
        json.dump(params, fp, indent=2)

    xothers_actual, uothers_actual = run_iterative_best_response(
        all_other_vehicles, world, all_other_x0, params, log_dir,
        args.load_log_dir, i_mpc_start)

    all_trajectories = np.array(xothers_actual)
    all_control = np.array(uothers_actual)

    np.save(open(log_dir + "/trajectories.npy", 'wb'), all_trajectories)
    np.save(open(log_dir + "/controls.npy", 'wb'), all_control)