import time, datetime, os, pickle, json, string, random
import numpy as np
import copy as cp
from contextlib import redirect_stdout
from typing import List



from src.traffic_world import TrafficWorld
from src.warm_starts import generate_warmstarts, get_subset_warmstarts
from src.best_response import generate_solver_params, MPCSolverLogger, MPCSolverLog, parallel_mpc_solve_w_trajs, MPCProblemCall
from src.vehicle_mpc_information import Trajectory, VehicleMPCInformation

from src.utils.ibr_argument_parser import IBRParser
from src.utils.solver_helper import poission_positions, extend_last_mpc_and_follow, initialize_cars_from_positions
from src.utils.plotting.car_plotting import plot_initial_positions
from src.utils.sim_utils import ExperimentHelper, get_closest_n_obstacle_vehs, get_obstacle_vehs_closeby, get_max_dist_traveled, get_ibr_vehs_idxs, assign_shared_control, get_within_range_other_vehicle_idxs
from src.desired_trajectories import generate_lane_changing_desired_trajectories

def run_iterative_best_response(vehicles,
                                world: TrafficWorld,
                                x0: List[np.array],
                                params: dict,
                                log_dir: str = None,
                                load_log_dir: bool = False,
                                i_mpc_start: int = 0):
    """ 
        Runs iterative best response for a system of ambulance and other vehicles.
    """
    assert params["n_other"] == len(vehicles)  
    ipopt_out_file = open(log_dir + "ipopt_out.txt", "w")

    experiment = ExperimentHelper(log_dir, params)
    solution_logger = MPCSolverLogger(log_dir)
    ipopt_params = {"print_level": params["print_level"], "max_cpu_time": params["max_cpu_time"]}

    if load_log_dir:
        x_executed, u_mpc, x_actual, u_actual, t_actual = experiment.load_log_data(i_mpc_start)
    else:
        x_executed, u_mpc, x_actual, u_actual, t_actual = experiment.initialize_states()

    # Run the simulation and solve mpc for all vehicles for each round of MPC
    for i_mpc in range(i_mpc_start, params["n_mpc"]):

        # 1) Update the initial conditions for all
        #    vehicles and normalize wrt ambulance position
        if i_mpc > 0:
            x0 = [cp.deepcopy(x_executed[i][:, params["number_ctrl_pts_executed"]]) for i in range(len(x_executed))]
            for i in range(len(x_executed)): # set the s variable to zero each round of mpc
                x0[i][-1] = 0
        
        # 3) Generate (if needed) the control inputs of other vehicles
        experiment.print_initializing_trajectories(i_mpc)
        for i in range(len(vehicles)):
            vehicles[i].update_default_desired_lane_traj(world, x0[i])     

        try:
            with redirect_stdout(ipopt_out_file):
                if i_mpc == 0:
                    u_mpc_prev = [np.zeros((2, params["N"])) for _ in range(len(vehicles))]
                    u_ibr, x_ibr, xd_ibr = extend_last_mpc_and_follow(u_mpc_prev, params["N"] - 1, vehicles, x0, params,
                                                                        world)
                else:
                    u_ibr, x_ibr, xd_ibr = extend_last_mpc_and_follow(u_mpc, params["number_ctrl_pts_executed"],
                                                                    vehicles, x0, params, world)
        except RuntimeError:
            experiment.print_sim_exited_early("infeasible solution")
            ipopt_out_file.close()
            return x_actual, u_actual
        

        for i in range(len(vehicles)):
            # vehicles[i].update_desired_lane_from_x0(world, x0[i])
            vehicles[i].update_default_desired_lane_traj(world, x0[i])

        # Convert all states to VehicleMPCInformation for ease
        vehsinfo_ibr = []
        for i in range(len(vehicles)):
            vehsinfo_ibr.append(VehicleMPCInformation(vehicles[i], x0[i], u_ibr[i], x_ibr[i], xd_ibr[i]))
        vehsinfo_ibr_pred = cp.deepcopy(vehsinfo_ibr)

        # Run Iterative Best Response
        for i_ibr in range(params["n_ibr"]):
            experiment.print_mpc_ibr_round(i_mpc, i_ibr, params)

            vehicles_idx_best_responders = get_ibr_vehs_idxs(vehicles)
            for ag_idx in vehicles_idx_best_responders:

                ego_x0 = vehsinfo_ibr[ag_idx].x0

                # delta_y_following = vehsinfo_ibr[ag_idx].vehicle.desired_lane - ego_x0[1]
                # desired_trajectories = [LaneChangeManueverPiecewise(5, 5, 999999, delta_y_following)]

                desired_trajectories = generate_lane_changing_desired_trajectories(world, ego_x0, params["T"], n_speed_increments=3)

                # Convert all positions into local frame of best responding (ego) vehicle
                fake_ego_x0_transforming = np.zeros(shape=(6,))
                vehsinfo_ibr_egoframe = [vi.transform_to_local(fake_ego_x0_transforming) for vi in vehsinfo_ibr]
                vehsinfo_ibr_pred_egoframe = [vi.transform_to_local(fake_ego_x0_transforming) for vi in vehsinfo_ibr_pred]

                response_vehinfo_egoframe = vehsinfo_ibr_egoframe[ag_idx]

                max_dist_traveled = get_max_dist_traveled(response_vehinfo_egoframe, params)
                veh_idxs_in_mpc = get_within_range_other_vehicle_idxs(ag_idx, vehsinfo_ibr_egoframe, max_dist_traveled)

                ctrld_vehsinfo_egoframe, obstacle_vehsinfo_egoframe, cntrld_i = assign_shared_control(params, i_ibr, veh_idxs_in_mpc,
                                                                                    vehicles_idx_best_responders,
                                                                                    response_vehinfo_egoframe, vehsinfo_ibr_pred_egoframe)



                obstacle_vehsinfo_egoframe = get_obstacle_vehs_closeby(response_vehinfo_egoframe, ctrld_vehsinfo_egoframe, obstacle_vehsinfo_egoframe, distance_threshold=40.0)
                obstacle_vehsinfo_egoframe = get_closest_n_obstacle_vehs(response_vehinfo_egoframe, ctrld_vehsinfo_egoframe, obstacle_vehsinfo_egoframe, max_num_obstacles=params["max_num_obstacles"],
                                                                min_num_obstacles_ego=3)




                warmstarts_dict = generate_warmstarts(response_vehinfo_egoframe, world, vehsinfo_ibr_pred_egoframe, params, u_mpc[ag_idx],
                                                      vehsinfo_ibr_egoframe[ag_idx].u)

                experiment.print_vehicle_id(ag_idx)
                # Try to solve MPC multiple times
                # each time increasing # warmstarts and slack cost
                solve_number = 0
                while solve_number < params["k_max_solve_number"]:
                    s_params = generate_solver_params(params, i_ibr, solve_number)
                    warmstarts_subset = get_subset_warmstarts(s_params["n_warm_starts"], warmstarts_dict)

                    experiment.check_machine_memory()

                    t_start_ipopt = time.time()
                    experiment.print_mpc_ibr_round(i_mpc, i_ibr, params)
                    experiment.print_nc_nnc(ctrld_vehsinfo_egoframe, obstacle_vehsinfo_egoframe, verbose=True)
                    # print("Trying to solve...")
                    # print("X_ego", response_vehinfo_egoframe.x[0:,:], "...")
                    # print("Xd_ego", response_vehinfo_egoframe.xd[0:,:], "...")

                    # print("X_ctrl_egoframe", [vi.x[:,:] for vi in ctrld_vehsinfo_egoframe], "...")
                    # print("Xd_ctrl_egoframe", [vi.xd[:,:] for vi in ctrld_vehsinfo_egoframe], "...")

                    # print("X_Obst_egoframe", [vi.x[:,:] for vi in obstacle_vehsinfo_egoframe], "...")
                    # print("Xd_Obst_egoframe", [vi.xd[:,:] for vi in obstacle_vehsinfo_egoframe], "...")

                    warmstart_w_traj_dict = {}
                    for warm_key, warm_traj in warmstarts_subset.items():
                        for traj_idx, desired_trajectory in enumerate(desired_trajectories):
                            warm_traj_key = warm_key + "%d"%traj_idx
                            warmstart_w_traj_dict[warm_traj_key] = (warm_traj, desired_trajectory)

                    problem_call = MPCProblemCall(warmstart_w_traj_dict, 
                                                response_vehinfo_egoframe, 
                                                world, 
                                                s_params, 
                                                params, 
                                                ipopt_params,
                                                obstacle_vehsinfo_egoframe, 
                                                ctrld_vehsinfo_egoframe)
                    # problem_call = None

                    with redirect_stdout(ipopt_out_file):
                        sol = parallel_mpc_solve_w_trajs(warmstart_w_traj_dict, response_vehinfo_egoframe, world, s_params, params, ipopt_params,
                            obstacle_vehsinfo_egoframe, ctrld_vehsinfo_egoframe)
                        solution_logger.add_log(MPCSolverLog(sol, i_mpc, i_ibr, response_vehinfo_egoframe.vehicle.agent_id, solve_number, problem_call))

                    if sol.max_slack < min(params["k_max_slack"], np.infty):
                        # print("Local Ego Solution")
                        # print("X", sol.trajectory.x[0:2,:5], "...")
                        # print("Xd", sol.trajectory.xd[0:2,:5], "...")

                        # Convert the units back to the world frame
                        u_solved = sol.trajectory.u
                        x_sim, xd_sim = response_vehinfo_egoframe.vehicle.forward_simulate_all(response_vehinfo_egoframe.x0, u_solved)
                        solution_traj = Trajectory(u=u_solved, x=x_sim, xd=xd_sim)
                        solution_traj_world = solution_traj.transform_to_global(fake_ego_x0_transforming)
                        
                        # print("Saving Ego Veh Traj to Pred", solution_traj_world.x[0:2,:5], "...")
                        vehsinfo_ibr[ag_idx].update_state_from_traj(solution_traj_world)
                        vehsinfo_ibr_pred[ag_idx].update_state_from_traj(solution_traj_world)

                        for cntrld_i_idx, veh_id in enumerate(cntrld_i):
                            u_ctrl = sol.cntrld_vehicle_trajectories[cntrld_i_idx].u
                            ctrld_veh_info = ctrld_vehsinfo_egoframe[cntrld_i_idx]

                            x_ctrl, xd_ctrl = ctrld_veh_info.vehicle.forward_simulate_all(ctrld_veh_info.x0, u_ctrl)
                            c_veh_traj = Trajectory(u=u_ctrl, x=x_ctrl, xd=xd_ctrl)

                            c_veh_traj_world = c_veh_traj.transform_to_global(fake_ego_x0_transforming)
                            vehsinfo_ibr_pred[veh_id].update_state_from_traj(c_veh_traj_world)
                        experiment.print_solved_status(ag_idx, i_mpc, i_ibr, t_start_ipopt)
                        # experiment.save_plot(sol)
                        break
                    else:
                        experiment.print_not_solved_status(ag_idx, i_mpc, i_ibr, sol.max_slack, t_start_ipopt)
                        solve_number += 1

                if solve_number == params["k_max_solve_number"]:
                    if i_ibr > 0:
                        default_traj = warmstarts_dict['previous_ibr']
                    else:
                        default_traj = warmstarts_dict['previous_mpc_hold']

                    default_traj_world = default_traj.transform_to_global(fake_ego_x0_transforming)

                    vehsinfo_ibr[ag_idx].update_state_from_traj(default_traj_world)
                    vehsinfo_ibr_pred[ag_idx].update_state_from_traj(default_traj_world)
                    experiment.print_max_solved_status(ag_idx, i_mpc, i_ibr, sol.max_slack, t_start_ipopt)

                experiment.save_ibr(i_mpc, i_ibr, ag_idx, vehsinfo_ibr_pred)

            other_x_ibr_temp = np.stack([veh.x for veh in vehsinfo_ibr])
            all_other_x_ibr_g = [x for x in other_x_ibr_temp]

            experiment.save_ibr(i_mpc, i_ibr, None, vehsinfo_ibr)

        t_actual, u_mpc, x_executed, _, x_actual, u_actual, = experiment.update_sim_states(
            vehsinfo_ibr, all_other_x_ibr_g, t_actual, i_mpc, x_actual, u_actual)

        collision = experiment.check_collisions(vehicles, x_executed)
        experiment.save_trajectory(np.array(x_actual), np.array(u_actual))
        solution_logger.write()
        if collision:
            experiment.print_sim_exited_early("collision!")
            ipopt_out_file.close()
            return x_actual, u_actual

    experiment.print_sim_finished()
    ipopt_out_file.close()
    return x_actual, u_actual


if __name__ == "__main__":
    parser = IBRParser()
    args = parser.parse_args()
    params = vars(args)
    # args.load_log_dir = "/home/nbuckman/mpc_results/investigation/h777-4664-20211031-204805/"
    if args.load_log_dir is None:
        # Generate directory structure and log name
        print("New stuff")
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
        initial_vehicle_positions = poission_positions(params["car_density"],
                                                       params["n_other"] + 1,
                                                       params["n_lanes"],
                                                       MAX_VELOCITY,
                                                       2 * VEHICLE_LENGTH,
                                                       position_random_seed=params["seed"])
        position_list = initial_vehicle_positions[:params["n_other"] + 1]

        # Create the SVOs for each vehicle
        if params["random_svo"] == 1:
            list_of_svo = [np.random.choice([0, np.pi / 4.0, np.pi / 2.01]) for i in range(params["n_other"])]
        else:
            list_of_svo = [params["svo_theta"] for i in range(params["n_other"])]

        (_, _, all_other_vehicles, all_other_x0) = initialize_cars_from_positions(params["N"], params["dt"], world,
                                                                                  True, position_list, list_of_svo)

        # ambulance = cp.deepcopy(all_other_vehicles[0])
        # ambulance.fd = ambulance.gen_f_desired_lane(world, 0, True)  # reset the desired lane since x0,y0 is different

        for vehicle in all_other_vehicles:
            # Set theta_ij randomly for all vehicles
            vehicle.theta_ij[-1] = np.random.uniform(0.001, np.pi / 2.01)
            for vehicle_j in all_other_vehicles:
                vehicle.theta_ij[vehicle_j.agent_id] = np.random.uniform(0.001, np.pi / 2.01)

        if params["k_lat"]:
            for vehicle in all_other_vehicles:
                vehicle.k_lat = params["k_lat"]
        # Save the vehicles and world for this simulation
        pickle.dump(all_other_vehicles, open(log_dir + "/other_vehicles.p", "wb"))
        pickle.dump(world, open(log_dir + "/world.p", "wb"))
        pickle.dump(all_other_x0, open(log_dir + "/x0.p", "wb"))
        if args.plot_initial_positions:
            plot_initial_positions(log_dir, world, all_other_vehicles, all_other_x0)

        print("Results saved in log %s:" % log_dir)
    else:
        print("Preloading settings from log %s" % args.load_log_dir)
        log_dir = args.load_log_dir
        with open(args.load_log_dir + "params.json", "rb") as fp:
            params = json.load(fp)
        i_mpc_start = args.mpc_start_iteration
        all_other_vehicles = pickle.load(open(log_dir + "other_vehicles.p", "rb"))
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

    xothers_actual, uothers_actual = run_iterative_best_response(all_other_vehicles, world, all_other_x0, params,
                                                                 log_dir, args.load_log_dir, i_mpc_start)

    all_trajectories = np.array(xothers_actual)
    all_control = np.array(uothers_actual)

    np.save(open(log_dir + "/trajectories.npy", 'wb'), all_trajectories)
    np.save(open(log_dir + "/controls.npy", 'wb'), all_control)