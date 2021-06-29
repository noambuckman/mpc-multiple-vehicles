import time, datetime, os, pickle, psutil, json, string, random
import numpy as np
import copy as cp
from contextlib import redirect_stdout

from src.traffic_world import TrafficWorld
from src.multiagent_mpc import load_state, save_state
from src.warm_starts import generate_warm_starts

from src.utils.ibr_argument_parser import IBRParser
import src.utils.solver_helper as helper
from src.utils.solver_helper import warm_profiles_subset
from typing import List


def convert_to_global_units(ambulance_x0_global, x):
    """ During solving, the x-dimension is normalized to the ambulance initial position ambulance_x0_global """

    x_global = cp.deepcopy(x)
    x_global[0, :] += ambulance_x0_global[0]

    return x_global


class VehicleMPCInformation:
    ''' Helper class that holds the state of each vehicle and vehicle information'''
    def __init__(self, vehicle, x0, u=None, x=None, xd=None):
        self.vehicle = vehicle
        self.x0 = x0

        self.u = u
        self.x = x
        self.xd = xd

    def update_state(self, u, x, xd):
        self.u = u
        self.x = x
        self.xd = xd


def run_iterative_best_response(ambulance,
                                other_vehicles,
                                world: TrafficWorld,
                                amb_x0: np.array,
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

    # Initialize all state and control arrays for entire simulation

    t_start_time = time.time()
    actual_t = 0
    n_ctrl_total = params["n_mpc"] * params["number_ctrl_pts_executed"]
    xamb_actual = np.zeros((6, n_ctrl_total + 1))
    uamb_actual = np.zeros((2, n_ctrl_total))
    xothers_actual = [np.zeros((6, n_ctrl_total + 1)) for i in range(params["n_other"])]
    uothers_actual = [np.zeros((2, n_ctrl_total)) for i in range(params["n_other"])]

    if load_log_dir:  # TODO:  This could use some cleanup
        previous_mpc_file = log_dir + "data/mpc_%03d" % (i_mpc_start - 1)
        print("Loaded initial positions from %s" % (previous_mpc_file))
        xamb_executed, uamb_executed, _, all_other_x_executed, all_other_u_executed, _ = load_state(
            previous_mpc_file, params["n_other"])
        all_other_u_mpc = all_other_u_executed
        uamb_mpc = uamb_executed
        previous_all_file = log_dir + "data/all_%03d" % (i_mpc_start - 1)
        xamb_actual_prev, uamb_actual_prev, _, xothers_actual_prev, uothers_actual_prev, _, = load_state(
            previous_all_file, params["n_other"], ignore_des=True)
        t_end = xamb_actual_prev.shape[1]
        xamb_actual[:, :t_end] = xamb_actual_prev[:, :t_end]
        uamb_actual[:, :t_end] = uamb_actual_prev[:, :t_end]
        for i in range(len(xothers_actual_prev)):
            xothers_actual[i][:, :t_end] = xothers_actual_prev[i][:, :t_end]
            uothers_actual[i][:, :t_end] = uothers_actual_prev[i][:, :t_end]
        actual_t = i_mpc_start * params["number_ctrl_pts_executed"]
    else:
        xamb_executed = None
        uamb_mpc = None
        all_other_x_executed = [None for i in range(params["n_other"])]
        all_other_u_mpc = [None for i in range(params["n_other"])]

    # Run the simulation and solve mpc for all vehicles for each round of MPC
    for i_mpc in range(i_mpc_start, params["n_mpc"]):
        ##### len(other_x0) == len(other_vehicles)
        ##### other_x0 gets updated at each round of mpc

        # 1) Update the initial conditions for all vehicles and normalize wrt ambulance position
        if i_mpc > 0:
            amb_x0_g = cp.deepcopy(xamb_executed[:, params["number_ctrl_pts_executed"]])
            all_other_x0_g = [
                cp.deepcopy(all_other_x_executed[i][:, params["number_ctrl_pts_executed"]])
                for i in range(len(all_other_x_executed))
            ]
            # Amb_x0 is localized in renference to ambulance
            amb_x0 = cp.deepcopy(amb_x0_g)
            amb_x0[0] -= amb_x0_g[0]
            amb_x0[-1] = 0
            other_x0 = cp.deepcopy(all_other_x0_g)
            for j in range(len(all_other_x0_g)):
                other_x0[j][0] -= amb_x0_g[0]
                other_x0[j][-1] = 0
        else:
            amb_x0_g = cp.deepcopy(amb_x0)  # for use later on

        # 2)  Select which vehicles will be solved in IBR
        vehicles_index_best_responders = vehicles_within_range(other_x0, amb_x0, 10 * ambulance.L)
        print("Vehicle in IBR:", vehicles_index_best_responders)

        # 3) Generate (if needed) the control inputs of other vehicles for the ambulance
        with redirect_stdout(out_file):
            if i_mpc == 0:
                previous_all_other_u_mpc = [np.zeros((2, params["N"])) for i in range(len(other_vehicles))]
                other_u_ibr_initial, other_x_ibr_initial, other_x_des_ibr_initial = helper.extend_last_mpc_and_follow(
                    previous_all_other_u_mpc, params["N"] - 1, params["N"], other_vehicles, other_x0, params, world)
            else:
                other_u_ibr_initial, other_x_ibr_initial, other_x_des_ibr_initial = helper.extend_last_mpc_and_follow(
                    all_other_u_mpc, params["number_ctrl_pts_executed"], params["N"], other_vehicles, other_x0, params,
                    world)

        othervehs_ibr_info = []
        for i in range(len(other_vehicles)):
            othervehs_ibr_info.append(
                VehicleMPCInformation(other_vehicles[i], other_x0[i], other_u_ibr_initial[i], other_x_ibr_initial[i],
                                      other_x_des_ibr_initial[i]))

        amb_ibr_info = VehicleMPCInformation(ambulance, amb_x0)

        for i_rounds_ibr in range(params["n_ibr"]):
            print("MPC %d, IBR %d / %d" % (i_mpc, i_rounds_ibr, params["n_ibr"] - 1))

            # Define which vehicles are best response vehicles, shared control vehicles, and fixed control vehicles (for collision avoidance)
            # in the ambulance's best response

            response_veh_info = amb_ibr_info
            veh_idxs_in_amb_mpc = vehicles_within_range(other_x0, amb_x0, 20 * ambulance.L)

            # Select which cars the ambulance should imagine shared control
            fake_amb_i = -1
            cntrld_vehicle_info = []
            if params["plan_fake_ambulance"]:  #TODO:  Add comments or description
                other_x0_temp = [v.x0 for v in othervehs_ibr_info]
                fake_amb_i = helper.get_min_dist_i(amb_x0, other_x0_temp, restrict_greater=True)

                cntrld_vehicle_info = [othervehs_ibr_info[fake_amb_i]]
                veh_idxs_in_amb_mpc = [i for i in veh_idxs_in_amb_mpc if i != fake_amb_i]

            nonresponse_veh_info = [othervehs_ibr_info[i] for i in veh_idxs_in_amb_mpc]

            # Generate the warm starts
            warm_starts = generate_warm_starts(response_veh_info.vehicle, world, response_veh_info.x0,
                                               othervehs_ibr_info, params, uamb_mpc, response_veh_info.u)

            # Initialize parameters relevant to solving the mpc
            solver_params = {}
            solver_params["slack"] = (True if i_rounds_ibr <= params["k_max_round_with_slack"] else False)
            solver_params["solve_amb"] = (False if len(cntrld_vehicle_info) == 0
                                          or i_rounds_ibr >= params["k_solve_amb_max_ibr"] else True)
            solver_params["n_warm_starts"] = params["default_n_warm_starts"]
            solve_number, max_slack_ibr, debug_flag = 0, np.infty, False
            print("...Amb Solver:")
            ipopt_params = {"print_level": params["print_level"]}
            while solve_number < params["k_max_solve_number"]:
                solver_params["k_slack"] = params["k_slack_d"] * 10**solve_number
                solver_params["n_warm_starts"] = (solver_params["n_warm_starts"] + 5 * solve_number)

                solver_params["k_CA"] = params["k_CA_d"]  # TODO: Replace this as a param and not solver param
                solver_params["k_CA_power"] = params["k_CA_power"]

                debug_flag = True if solve_number > 2 else False
                warmstarts_subset = warm_profiles_subset(solver_params["n_warm_starts"], warm_starts)
                if psutil.virtual_memory().percent >= 95.0:
                    raise Exception("Virtual Memory is too high, exiting to save computer")

                start_ipopt_time = time.time()
                with redirect_stdout(out_file):
                    if params["save_solver_input"]:
                        with open(
                                log_dir + "data/inputs_amb_mpc_%d_ibr_%d_s_%d.p" % (i_mpc, i_rounds_ibr, solve_number),
                                "wb",
                        ) as fp:
                            list_of_inputs = [
                                # warmstarts_subset, response_veh_info.vehicle, cntrld_vehicles, nonresponse_vehicle_list,
                                # response_veh_info.x0, cntrld_x0, nonresponse_x0_list, world, solver_params, params,
                                # nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list, cntrld_u, cntrld_x,
                                # cntrld_xd, debug_flag
                            ]
                            pickle.dump(list_of_inputs, fp)
                    print("....# Vehicles in Amb's Best Response: %d " % (len(nonresponse_veh_info)))
                    solved, min_cost_ibr, max_slack_ibr, x_ibr, x_des_ibr, u_ibr, key_ibr, debug_list = helper.solve_warm_starts(
                        warmstarts_subset, response_veh_info, world, solver_params, params, ipopt_params,
                        nonresponse_veh_info, cntrld_vehicle_info, debug_flag)
                if max_slack_ibr <= params["k_max_slack"]:
                    print("......Solver converged.  Solver time: %0.1f s" % (time.time() - start_ipopt_time))
                    amb_ibr_info.update_state(u_ibr, x_ibr, x_des_ibr)
                    break
                else:
                    print(
                        "......Re-solve %d/%d:  Slack too large: %.05f > Max Threshold (%.05f).  Solver time: %0.1f s" %
                        (solve_number + 1, params["k_max_solve_number"], max_slack_ibr, params["k_max_slack"],
                         time.time() - start_ipopt_time))
                    solve_number += 1

            if solve_number == params["k_max_solve_number"]:
                raise Exception(
                    "Reached maximum number of re-solves @ Veh %s MPC Rd %d IBR %d.   Max Slack = %.05f > thresh %.05f"
                    % ("Amb", i_mpc, i_rounds_ibr, max_slack_ibr, params["k_max_slack"]))
            if params["save_ibr"]:
                xamb_ibr_g = convert_to_global_units(amb_x0_g, amb_ibr_info.x)

                other_u_ibr_temp = [veh.u for veh in othervehs_ibr_info]
                other_x_des_ibr_temp = [veh.xd for veh in othervehs_ibr_info]

                all_other_x_ibr_g = [convert_to_global_units(amb_x0_g, veh.x) for veh in othervehs_ibr_info]
                save_state(log_dir + "data/" + "ibr_m%03di%03damb" % (i_mpc, i_rounds_ibr), xamb_ibr_g, amb_ibr_info.u,
                           amb_ibr_info.xd, all_other_x_ibr_g, other_u_ibr_temp, other_x_des_ibr_temp)

            # Solve the Best Response for the other vehicles
            for response_i in vehicles_index_best_responders:
                response_veh_info = othervehs_ibr_info[response_i]

                # Select which vehicles should be included in the ado's MPC (3 cars ahead and 2 car behind)
                veh_idxs_in_mpc = [
                    j for j in range(len(other_x0)) if j != response_i and (-20 * response_veh_info.vehicle.L <= (
                        other_x0[j][0] - response_veh_info.x0[0]) <= 20 * response_veh_info.vehicle.L)
                ]

                cntrld_vehicle_info = []

                # Choose which vehicles should be in the shared control
                cntrld_scheduler = params["shrd_cntrl_scheduler"]
                if cntrld_scheduler == "constant":
                    if i_rounds_ibr >= params["rnds_shrd_cntrl"]:
                        n_cntrld = 0
                    else:
                        n_cntrld = params["n_cntrld"]
                elif cntrld_scheduler == "linear":
                    n_cntrld = max(0, params["n_cntrld"] - i_rounds_ibr)
                else:
                    raise Exception("Shrd Controller Not Specified")

                if n_cntrld > 0:
                    delta_x = [response_veh_info.x0[0] - x[0] for x in other_x0]
                    sorted_i = [
                        i for i in np.argsort(delta_x)
                        if (delta_x[i] > 0 and i in veh_idxs_in_mpc and i in vehicles_index_best_responders)
                    ]  # This necessary but could limit fringe best response interactions with outside best response
                    cntrld_i = sorted_i[:params["n_cntrld"] - 1]
                    if len(cntrld_i) > 0:
                        cntrld_vehicle_info = [
                            othervehs_ibr_info[i] for i in range(len(othervehs_ibr_info)) if i in cntrld_i
                        ]
                    # For now ambulance is always included
                    cntrld_vehicle_info += [amb_ibr_info]

                    veh_idxs_in_mpc = [idx for idx in veh_idxs_in_mpc if idx not in cntrld_i]

                nonresponse_veh_info = [othervehs_ibr_info[i] for i in veh_idxs_in_mpc]
                if len(cntrld_vehicle_info) == 0:
                    nonresponse_veh_info += [amb_ibr_info]

                warm_starts = generate_warm_starts(response_veh_info.vehicle, world, response_veh_info.x0,
                                                   othervehs_ibr_info, params, all_other_u_mpc[response_i],
                                                   othervehs_ibr_info[response_i].u)

                solve_again, solve_number, max_slack_ibr, debug_flag, = (True, 0, np.infty, False)
                params["k_solve_amb_min_distance"] = 50
                initial_distance_to_ambulance = np.sqrt((response_veh_info.x0[0] - amb_x0[0])**2 +
                                                        (response_veh_info.x0[1] - amb_x0[1])**2)
                solver_params = {}

                solver_params["solve_amb"] = (True if
                                              (i_rounds_ibr < params["k_solve_amb_max_ibr"]
                                               and initial_distance_to_ambulance < params["k_solve_amb_min_distance"]
                                               and response_veh_info.x0[0] > amb_x0[0] - 0) else False)
                solver_params["slack"] = (True if i_rounds_ibr <= params["k_max_round_with_slack"] else False)
                solver_params["n_warm_starts"] = params["default_n_warm_starts"]
                solver_params["k_CA"] = params["k_CA_d"]
                solver_params["k_CA_power"] = params["k_CA_power"]
                if (response_i not in veh_idxs_in_amb_mpc) and (response_i != fake_amb_i):
                    solver_params["constant_v"] = True

                print("...Veh %02d Solver:" % response_i)
                while solve_number < params["k_max_solve_number"]:
                    solver_params["k_slack"] = (params["k_slack_d"] * 10**solve_number)
                    solver_params["n_warm_starts"] = (solver_params["n_warm_starts"] + 5 * solve_number)
                    if solve_number > 2:
                        debug_flag = True
                    warmstarts_subset = warm_profiles_subset(solver_params["n_warm_starts"], warm_starts)

                    if psutil.virtual_memory().percent >= 95.0:
                        raise Exception("Virtual Memory is too high, exiting to save computer")
                    start_ipopt_time = time.time()
                    # TODO:  Also solve amb in the nonresponse MPC list
                    with redirect_stdout(out_file):
                        print("....# Cntrld Vehicles: %d # Non-Response: %d " %
                              (len(cntrld_vehicle_info), len(nonresponse_veh_info)))
                        solved, min_cost_ibr, max_slack_ibr, x_ibr, x_des_ibr, u_ibr, key_ibr, debug_list = helper.solve_warm_starts(
                            warmstarts_subset, response_veh_info, world, solver_params, params, ipopt_params,
                            nonresponse_veh_info, cntrld_vehicle_info, debug_flag)
                    end_ipopt_time = time.time()
                    if max_slack_ibr <= params["k_max_slack"]:
                        othervehs_ibr_info[response_i].update_state(u_ibr, x_ibr, x_des_ibr)
                        print("......Solved.  veh: %02d | mpc: %d | ibr: %d | solver time: %0.1f s" %
                              (response_i, i_mpc, i_rounds_ibr, end_ipopt_time - start_ipopt_time))
                        break
                    else:
                        print(
                            "......Re-solve. veh: %02d | mpc: %d | ibr: %d | Slack %.05f > thresh %.05f  | solver time: %0.1f"
                            % (response_i, i_mpc, i_rounds_ibr, max_slack_ibr, params["k_max_slack"],
                               end_ipopt_time - start_ipopt_time))
                        solve_number += 1
                        if (solve_number == params["k_max_solve_number"]) and solver_params["solve_amb"]:
                            print("Re-solving without ambulance")
                            solve_number = 0
                            solver_params["solve_amb"] = False
                if solve_number == params["k_max_solve_number"]:
                    raise Exception(
                        "Reached maximum number of re-solves @ Veh %02d MPC Rd %d IBR %d.   Max Slack = %.05f > thresh %.05f"
                        % (response_i, i_mpc, i_rounds_ibr, max_slack_ibr, params["k_max_slack"]))
                if params["save_ibr"]:
                    other_x_ibr_temp = [veh.x for veh in othervehs_ibr_info]
                    other_u_ibr_temp = [veh.u for veh in othervehs_ibr_info]
                    other_xd_ibr_temp = [veh.xd for veh in othervehs_ibr_info]

                    save_state(log_dir + "data/" + "ibr_m%03di%03da%03d" % (i_mpc, i_rounds_ibr, response_i),
                               amb_ibr_info.x, amb_ibr_info.u, amb_ibr_info.xd, other_x_ibr_temp, other_u_ibr_temp,
                               other_xd_ibr_temp)

            xamb_ibr_g = convert_to_global_units(amb_x0_g, amb_ibr_info.x)
            other_x_ibr_temp = [veh.x for veh in othervehs_ibr_info]
            other_u_ibr_temp = [veh.u for veh in othervehs_ibr_info]
            other_xd_ibr_temp = [veh.xd for veh in othervehs_ibr_info]
            all_other_x_ibr_g = [convert_to_global_units(amb_x0_g, x) for x in other_x_ibr_temp]
            if params["save_ibr"]:
                save_state(log_dir + "data/" + "ibr_m%03di%03d" % (i_mpc, i_rounds_ibr), xamb_ibr_g, amb_ibr_info.u,
                           amb_ibr_info.xd, all_other_x_ibr_g, other_u_ibr_temp, other_xd_ibr_temp)

        # SAVE EXECUTED MPC SOLUTION TO HISTORY

        all_other_u_mpc = [cp.deepcopy(veh.u) for veh in othervehs_ibr_info]
        uamb_mpc = cp.deepcopy(amb_ibr_info.u)

        xamb_executed = xamb_ibr_g[:, :params["number_ctrl_pts_executed"] + 1]
        uamb_executed = amb_ibr_info.u[:, :params["number_ctrl_pts_executed"]]
        all_other_x_executed = [
            all_other_x_ibr_g[i][:, :params["number_ctrl_pts_executed"] + 1] for i in range(params["n_other"])
        ]
        all_other_u_executed = [
            othervehs_ibr_info[i].u[:, :params["number_ctrl_pts_executed"]] for i in range(params["n_other"])
        ]

        # Append to the executed trajectories history of x
        xamb_actual[:, actual_t:actual_t + params["number_ctrl_pts_executed"] + 1] = xamb_executed
        uamb_actual[:, actual_t:actual_t + params["number_ctrl_pts_executed"]] = uamb_executed
        for i in range(params["n_other"]):
            xothers_actual[i][:, actual_t:actual_t + params["number_ctrl_pts_executed"] + 1] = all_other_x_executed[i]
            uothers_actual[i][:, actual_t:actual_t + params["number_ctrl_pts_executed"]] = all_other_u_executed[i]

        # SAVE STATES AND PLOT
        file_name = log_dir + "data/" + "mpc_%03d" % (i_mpc)
        print("Saving MPC Rd %03d / %03d to ... %s" % (i_mpc, params["n_mpc"] - 1, file_name))
        if params["save_state"]:
            other_u_ibr_temp = [veh.u for veh in othervehs_ibr_info]
            other_xd_ibr_temp = [veh.xd for veh in othervehs_ibr_info]

            save_state(file_name, xamb_ibr_g, amb_ibr_info.u, amb_ibr_info.xd, all_other_x_ibr_g, other_u_ibr_temp,
                       other_xd_ibr_temp)
            save_state(log_dir + "data/" + "all_%03d" % (i_mpc),
                       xamb_actual,
                       uamb_actual,
                       None,
                       xothers_actual,
                       uothers_actual,
                       None,
                       end_t=actual_t + params["number_ctrl_pts_executed"] + 1)

        actual_t += params["number_ctrl_pts_executed"]

    out_file.close()

    print("Simulation Done!  Runtime: %s" % (datetime.timedelta(seconds=(time.time() - t_start_time))))

    return xamb_actual, xothers_actual


def vehicles_within_range(other_x0, amb_x0, distance_from_ambulance):

    veh_idxs = [i for i in range(len(other_x0)) if abs(other_x0[i][0] - amb_x0[0]) <= distance_from_ambulance]

    return veh_idxs


if __name__ == "__main__":
    parser = IBRParser()
    args = parser.parse_args()
    params = vars(args)

    if args.load_log_dir is None:
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
        time_duration_s = (params["n_other"] * 3600.0 /
                           params["car_density"]) * 10  # amount of time to generate traffic
        initial_vehicle_positions = helper.poission_positions(params["car_density"],
                                                              params["n_other"] + 1,
                                                              params["n_lanes"],
                                                              MAX_VELOCITY,
                                                              VEHICLE_LENGTH,
                                                              position_random_seed=params["seed"])
        position_list = initial_vehicle_positions[:params["n_other"] + 1]

        # Create the SVOs for each vehicle
        if params["random_svo"] == 1:
            list_of_svo = [np.random.choice([0, np.pi / 4.0, np.pi / 2.01]) for i in range(params["n_other"])]
        else:
            list_of_svo = [params["svo_theta"] for i in range(params["n_other"])]

        (ambulance, amb_x0, all_other_vehicles,
         all_other_x0) = helper.initialize_cars_from_positions(params["N"], params["dt"], world, True, position_list,
                                                               list_of_svo)

        if params["k_lat"]:
            for vehicle in all_other_vehicles:
                vehicle.k_lat = params["k_lat"]
        # Save the vehicles and world for this simulation
        for i in range(len(all_other_vehicles)):
            pickle.dump(all_other_vehicles[i], open(log_dir + "data/mpcother%03d.p" % i, "wb"))
        pickle.dump(all_other_vehicles, open(log_dir + "/other_vehicles.p", "wb"))
        pickle.dump(ambulance, open(log_dir + "/ambulance.p", "wb"))
        pickle.dump(world, open(log_dir + "data/world.p", "wb"))
        print("Results saved in log %s:" % log_dir)
    else:
        print("Preloading settings from log %s" % args.load_log_dir)
        log_dir = args.load_log_dir
        with open(args.load_log_dir + "params.json", "rb") as fp:
            params = json.load(fp)
        i_mpc_start = args.mpc_start_iteration
        ambulance = pickle.load(open(log_dir + "data/mpcamb.p", "rb"))
        all_other_vehicles = [
            pickle.load(open(log_dir + "data/mpcother%03d.p" % i, "rb")) for i in range(params["n_other"])
        ]
        world = pickle.load(open(log_dir + "data/world.p", "rb"))
        params["pid"] = os.getpid()

        # We need to get initial conditions for the iterative best response

    # Initialize the state and control arrays
    params["pid"] = os.getpid()
    if params["n_other"] != len(all_other_vehicles):
        raise Exception("n_other larger than  position list")
    with open(log_dir + "params.json", "w") as fp:
        json.dump(params, fp, indent=2)

    xamb_actual, xothers_actual = run_iterative_best_response(ambulance, all_other_vehicles, world, amb_x0,
                                                              all_other_x0, params, log_dir, args.load_log_dir,
                                                              i_mpc_start)
    all_trajectories = [xamb_actual] + xothers_actual
    all_trajectories = np.array(all_trajectories)
    np.save(open(log_dir + "/trajectories.npy", 'wb'), all_trajectories)
