import numpy as np
import random
import copy as cp
import multiprocessing, functools
from collections import deque

import src.multiagent_mpc as mpc
import src.vehicle as vehicle

# def warm_solve_subroutine(k_warm, ux_warm_profiles, response_MPC, amb_MPC, nonresponse_MPC_list, k_slack, k_CA, k_CA_power, world, wall_CA, N, T, response_x0, amb_x0, nonresponse_x0_list, slack, solve_amb, nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list, uamb=None, xamb=None, xamb_des=None):
#     u_warm, x_warm, x_des_warm = ux_warm_profiles[k_warm]
#     temp, current_cost, max_slack, bri, x, x_des, u = solve_best_response(response_MPC, amb_MPC, nonresponse_MPC_list, k_slack, k_CA, k_CA_power, world, wall_CA, N, T, response_x0, amb_x0, nonresponse_x0_list, slack, solve_amb, k_warm, u_warm, x_warm, x_des_warm, nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list, uamb, xamb, xamb_des)
#     return (temp, current_cost, max_slack, x, x_des, u)


def nonresponse_subset(list_of_veh_idxs, all_other_x0, all_other_MPC, all_other_u, all_other_x, all_other_x_des):
    ''' Returns subsets of previous lists '''

    nonresponse_x0_list = [all_other_x0[i] for i in list_of_veh_idxs]
    nonresponse_MPC_list = [all_other_MPC[i] for i in list_of_veh_idxs]
    nonresponse_u_list = [all_other_u[i] for i in list_of_veh_idxs]
    nonresponse_x_list = [all_other_x[i] for i in list_of_veh_idxs]
    nonresponse_xd_list = [all_other_x_des[i] for i in list_of_veh_idxs]

    return nonresponse_x0_list, nonresponse_MPC_list, nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list


def nonresponse_slice(i, all_other_x0, all_other_MPC, all_other_u, all_other_x, all_other_x_des):
    ''' Returns all vehicles except i'''

    nonresponse_x0_list = all_other_x0[:i] + all_other_x0[i + 1:]
    nonresponse_MPC_list = all_other_MPC[:i] + all_other_MPC[i + 1:]
    nonresponse_u_list = all_other_u[:i] + all_other_u[i + 1:]
    nonresponse_x_list = all_other_x[:i] + all_other_x[i + 1:]
    nonresponse_xd_list = all_other_x_des[:i] + all_other_x_des[i + 1:]

    return nonresponse_x0_list, nonresponse_MPC_list, nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list


def warm_profiles_subset(n_warm_keys, ux_warm_profiles):
    '''choose randomly n_warm_keys keys from ux_warm_profiles and return the subset'''

    warm_keys = list(ux_warm_profiles.keys())
    random.shuffle(warm_keys)
    warm_subset_keys = warm_keys[:n_warm_keys]
    ux_warm_profiles_subset = dict((k, ux_warm_profiles[k]) for k in warm_subset_keys)

    return ux_warm_profiles_subset


def get_min_dist_i(ambulance_x0, all_other_x0, restrict_greater=False):
    all_dist_sqrd = [(ambulance_x0[0] - x0[0])**2 + (ambulance_x0[1] - x0[1])**2 for x0 in all_other_x0]

    if restrict_greater:
        for i in range(len(all_other_x0)):
            if all_other_x0[i][0] < ambulance_x0[0]:
                all_dist_sqrd[i] = 999999999

    if len(all_dist_sqrd) == 0:
        return []
    else:
        return np.argmin(all_dist_sqrd)


def extend_last_mpc_ctrl(all_other_u_mpc, number_ctrl_pts_executed, N, all_other_MPC, all_other_x0):
    '''Copy the previous mpc and extend the last values with all zeros'''
    all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr = [
        np.zeros(shape=(2, N)) for i in range(len(all_other_MPC))
    ], [np.zeros(shape=(6, N + 1))
        for i in range(len(all_other_MPC))], [np.zeros(shape=(3, N + 1)) for i in range(len(all_other_MPC))]
    for i in range(len(all_other_MPC)):
        all_other_u_ibr[i] = np.concatenate((all_other_u_mpc[i][:, number_ctrl_pts_executed:],
                                             np.tile(np.zeros(shape=(2, 1)), (1, number_ctrl_pts_executed))),
                                            axis=1)  ##
        all_other_x_ibr[i], all_other_x_des_ibr[i] = all_other_MPC[i].forward_simulate_all(
            all_other_x0[i].reshape(6, 1), all_other_u_ibr[i])
    return all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr


def extend_last_mpc_and_follow(all_other_u_mpc, number_ctrl_pts_executed, N, all_other_MPC, all_other_x0, params,
                               world):
    '''Copy the previous mpc and extend the last values with lane following control'''

    all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr = [
        np.zeros(shape=(2, N)) for i in range(len(all_other_MPC))
    ], [np.zeros(shape=(6, N + 1))
        for i in range(len(all_other_MPC))], [np.zeros(shape=(3, N + 1)) for i in range(len(all_other_MPC))]
    print("Ambulance guess of ado...")
    for i in range(len(all_other_MPC)):
        print("...veh %03d" % i)

        # Ctrl and traj from previous MPC
        prev_ctrl = all_other_u_mpc[i][:, number_ctrl_pts_executed:-1]  #take off the last position
        prev_traj, prev_traj_des = all_other_MPC[i].forward_simulate_all(all_other_x0[i].reshape(6, 1), prev_ctrl)
        # Predicted portion of just lane following.  This is an estimated ctrl of ado vehicles.
        initial_pt = prev_traj[:, -1]

        lane_following_ctrl, lane_following_traj, lane_following_traj_des = lane_following_optimizations(
            number_ctrl_pts_executed + 1, all_other_MPC[i], initial_pt, params, world)

        # Lane following traj's initial pt is redundant (since it is also in prev traj)
        lane_following_traj = lane_following_traj[:, 1:]
        lane_following_traj_des = lane_following_traj_des[:, 1:]
        all_other_u_ibr[i] = np.concatenate((prev_ctrl, lane_following_ctrl), axis=1)
        all_other_x_ibr[i] = np.concatenate((prev_traj, lane_following_traj), axis=1)
        all_other_x_des_ibr[i] = np.concatenate((prev_traj_des, lane_following_traj_des), axis=1)
    return all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr


def pullover_guess(N, all_other_MPC, all_other_x0):
    ''' Provide a 2deg turn for all the other vehicles on the road'''

    all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr = [
        np.zeros(shape=(2, N)) for i in range(len(all_other_MPC))
    ], [np.zeros(shape=(6, N + 1))
        for i in range(len(all_other_MPC))], [np.zeros(shape=(3, N + 1)) for i in range(len(all_other_MPC))]
    for i in range(len(all_other_x0)):
        if all_other_x0[i][1] <= 0.5:
            all_other_u_ibr[i][0, 0] = -2 * np.pi / 180  # This is a hack and should be explicit that it's lane change
        else:
            all_other_u_ibr[i][0, 0] = 2 * np.pi / 180  # This is a hack and should be explicit that it's lane change
        all_other_x_ibr[i], all_other_x_des_ibr[i] = all_other_MPC[i].forward_simulate_all(
            all_other_x0[i].reshape(6, 1), all_other_u_ibr[i])
    return all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr


def lane_following_optimizations(N, response_MPC, response_x0, params, world):
    cp_MPC = cp.deepcopy(response_MPC)
    bri = mpc.MultiMPC(cp_MPC, [], [], world)
    bri.generate_optimization(N, response_x0, [], [], params=params, ipopt_params={'print_level': 0})

    bri.opti.subject_to(bri.u_ego[1, :] == 0)
    bri.solution = bri.opti.solve()
    x, u, x_des, _, _, _, _, _, _ = bri.get_solution()
    del bri, cp_MPC
    return u[:, :N], x[:, :N + 1], x_des[:, :N + 1]


def solve_best_response(warm_key,
                        warm_trajectory,
                        response_MPC,
                        cntrld_vehicles,
                        nonresponse_MPC_list,
                        response_x0,
                        cntrld_x0,
                        nonresponse_x0_list,
                        world,
                        solver_params,
                        params,
                        ipopt_params,
                        nonresponse_u_list,
                        nonresponse_x_list,
                        nonresponse_xd_list,
                        cntrld_u=[],
                        cntrld_x=[],
                        cntrld_xd=[],
                        return_bri=False):
    '''Create the iterative best response object and solve.  Assumes that it receives warm start profiles.
    This really should only require a u_warm, x_warm, x_des_warm and then one level above we generate those values'''

    u_warm, x_warm, x_des_warm = warm_trajectory
    bri = mpc.MultiMPC(response_MPC, cntrld_vehicles, nonresponse_MPC_list, world, solver_params)
    params["collision_avoidance_checking_distance"] = 100
    bri.generate_optimization(params["N"],
                              response_x0,
                              cntrld_x0,
                              nonresponse_x0_list,
                              params=params,
                              ipopt_params=ipopt_params)
    # print("Succesffully generated optimzation %d %d %d"%(len(nonresponse_MPC_list), len(nonresponse_x_list), len(nonresponse_u_list)))
    # u_warm, x_warm, x_des_warm = ux_warm_profiles[k_warm]
    bri.opti.set_initial(bri.u_ego, u_warm)
    bri.opti.set_initial(bri.x_ego, x_warm)
    bri.opti.set_initial(bri.xd_ego, x_des_warm)

    # Set initial trajectories of cntrld vehicles
    for ic in range(len(bri.vehs_ctrld)):
        bri.opti.set_initial(bri.x_ctrld[ic], cntrld_x[ic])
        bri.opti.set_initial(bri.xd_ctrld[ic], cntrld_xd[ic])

    # Set trajectories of non-cntrld vehicles
    for j in range(len(nonresponse_x_list)):
        bri.opti.set_value(bri.x_other[j], nonresponse_x_list[j])
        bri.opti.set_value(bri.xd_other[j], nonresponse_xd_list[j])

    try:
        if "constant_v" in solver_params and solver_params["constant_v"]:
            bri.opti.subject_to(bri.u_ego[1, :] == 0)

        bri.solve(cntrld_u, nonresponse_u_list, solver_params['solve_amb'])
        x_ibr, u_ibr, x_des_ibr, _, _, _, _, _, _ = bri.get_solution()
        current_cost = bri.solution.value(bri.total_svo_cost)

        # all_slack_vars = [bri.slack_i_jnc] + [bri.slack_i_jc] + bri.slack_ic_jnc + bri.slack_ic_jc
        # all_slack_vars = [bri.solution.value(s) for s in all_slack_vars]
        all_slack_vars = [bri.solution.value(bri.slack_i_jnc),
                          bri.solution.value(bri.slack_i_jc)] + [bri.solution.value(s) for s in bri.slack_ic_jc
                                                                 ] + [bri.solution.value(s) for s in bri.slack_ic_jnc]
        # print(len(all_slack_vars))
        max_slack = np.max([np.max(s) for s in all_slack_vars] + [0.000000000000])
        # print("Max slack", max_slack)
        # max_slack = np.max([np.max(bri.solution.value(s)) for s in bri.slack_vars_list])
        min_response_warm_ibr = None  #<This used to return k_warm

        debug_list = [
            current_cost,
            bri.solution.value(bri.response_svo_cost),
            bri.solution.value(bri.k_CA * bri.collision_cost),
            bri.solution.value(bri.k_slack * bri.slack_cost), all_slack_vars
        ]
        solved = True
        if return_bri:
            debug_list += [bri]
        return solved, current_cost, max_slack, x_ibr, x_des_ibr, u_ibr, warm_key, debug_list
    except RuntimeError:
        # print("Infeasibility: k_warm %s"%k_warm)
        if return_bri:
            return bri
        return False, np.infty, np.infty, None, None, None, None, []
        # ibr_sub_it +=1


def solve_warm_starts(ux_warm_profiles,
                      response_veh_info,
                      world,
                      solver_params,
                      params,
                      ipopt_params,
                      nonresponse_veh_info,
                      cntrl_veh_info,
                      debug_flag=False):

    response_MPC = response_veh_info.vehicle
    response_x0 = response_veh_info.x0

    nonresponse_MPC_list = [veh_info.vehicle for veh_info in nonresponse_veh_info]
    nonresponse_x0_list = [veh_info.x0 for veh_info in nonresponse_veh_info]
    nonresponse_u_list = [veh_info.u for veh_info in nonresponse_veh_info]
    nonresponse_x_list = [veh_info.x for veh_info in nonresponse_veh_info]
    nonresponse_xd_list = [veh_info.xd for veh_info in nonresponse_veh_info]

    cntrld_vehicles = [veh_info.vehicle for veh_info in cntrl_veh_info]
    cntrld_x0 = [veh_info.x0 for veh_info in cntrl_veh_info]
    cntrld_u = [veh_info.u for veh_info in cntrl_veh_info]
    cntrld_x = [veh_info.x for veh_info in cntrl_veh_info]
    cntrld_xd = [veh_info.xd for veh_info in cntrl_veh_info]

    warm_solve_partial = functools.partial(solve_best_response,
                                           response_MPC=response_MPC,
                                           cntrld_vehicles=cntrld_vehicles,
                                           nonresponse_MPC_list=nonresponse_MPC_list,
                                           response_x0=response_x0,
                                           cntrld_x0=cntrld_x0,
                                           nonresponse_x0_list=nonresponse_x0_list,
                                           world=world,
                                           solver_params=solver_params,
                                           params=params,
                                           ipopt_params=ipopt_params,
                                           nonresponse_u_list=nonresponse_u_list,
                                           nonresponse_x_list=nonresponse_x_list,
                                           nonresponse_xd_list=nonresponse_xd_list,
                                           cntrld_u=cntrld_u,
                                           cntrld_x=cntrld_x,
                                           cntrld_xd=cntrld_xd,
                                           return_bri=False)

    if params['n_processors'] > 1:
        pool = multiprocessing.Pool(processes=params['n_processors'])
        solve_costs_solutions = pool.starmap(warm_solve_partial,
                                             ux_warm_profiles.items())  #will apply k=1...N to plot_partial
        pool.terminate()
    else:
        solve_costs_solutions = []
        for k_warm in ux_warm_profiles:
            solve_costs_solutions += [warm_solve_partial(k_warm, ux_warm_profiles[k_warm])]

    if debug_flag:
        for ki in range(len(solve_costs_solutions)):
            debug_list = solve_costs_solutions[ki][7]
            print(debug_list)
            if len(debug_list) == 0:
                print("Infeasible")
            else:
                # print("Feasible")
                print("Costs: Total Cost %.04f Vehicle-Only Cost:  %.04f Collision Cost %0.04f  Slack Cost %0.04f" %
                      (tuple(debug_list[0:4])))

    min_cost_solution = min(solve_costs_solutions, key=lambda r: r[1])

    return min_cost_solution


def initialize_cars(n_other,
                    N,
                    dt,
                    world,
                    svo_theta,
                    no_grass=False,
                    random_lane=False,
                    x_variance=1.0,
                    list_of_positions=None):
    '''x_variance is in terms of number of min_dist'''
    ## Create the Cars in this Problem
    all_other_x0 = []
    all_other_u = []
    all_other_MPC = []
    next_x0_0 = 0
    next_x0_1 = 0
    for i in range(n_other):
        x1_MPC = vehicle.Vehicle(dt)
        x1_MPC.n_circles = 3
        x1_MPC.theta_i = svo_theta
        x1_MPC.N = N

        x1_MPC.k_change_u_v = 0.001
        x1_MPC.max_delta_u = 50 * np.pi / 180 * x1_MPC.dt
        x1_MPC.k_u_v = 0.01
        x1_MPC.k_u_delta = .00001
        x1_MPC.k_change_u_v = 0.01
        x1_MPC.k_change_u_delta = 0.001
        x1_MPC.k_s = 0
        x1_MPC.k_x = 0
        x1_MPC.k_x_dot = -1.0 / 100.0
        x1_MPC.k_lat = 0.001
        x1_MPC.k_lon = 0.0
        x1_MPC.k_phi_error = 0.001
        x1_MPC.k_phi_dot = 0.01

        x1_MPC.min_y = world.y_min
        x1_MPC.max_y = world.y_max
        if no_grass:
            x1_MPC.min_y += world.grass_width
            x1_MPC.max_y -= world.grass_width
        x1_MPC.strict_wall_constraint = True

        # Vehicle Initial Conditions
        lane_offset = np.random.uniform(0, 1) * x_variance * 2 * x1_MPC.min_dist
        if random_lane:
            lane_number = np.random.randint(2)
        else:
            lane_number = 0 if i % 2 == 0 else 1
        if lane_number == 0:
            next_x0_0 += x1_MPC.L + 2 * x1_MPC.min_dist + lane_offset
            next_x0 = next_x0_0
        else:
            next_x0_1 += x1_MPC.L + 2 * x1_MPC.min_dist + lane_offset
            next_x0 = next_x0_1

        if list_of_positions:  #list of positions overrides everything
            lane_number, next_x0 = list_of_positions[i]

        initial_speed = 0.99 * x1_MPC.max_v
        x1_MPC.fd = x1_MPC.gen_f_desired_lane(world, lane_number, True)
        x0 = np.array([next_x0, world.get_lane_centerline_y(lane_number), 0, 0, initial_speed, 0]).T
        all_other_MPC += [x1_MPC]
        all_other_x0 += [x0]

    # Settings for Ambulance
    amb_MPC = cp.deepcopy(x1_MPC)
    amb_MPC.theta_i = 0.0

    amb_MPC.k_u_v = 0.0000
    amb_MPC.k_u_delta = .01
    amb_MPC.k_change_u_v = 0.0000
    amb_MPC.k_change_u_delta = 0

    amb_MPC.k_s = 0
    amb_MPC.k_x = 0
    amb_MPC.k_x_dot = -1.0 / 100.0
    # amb_MPC.k_x = -1.0/100
    # amb_MPC.k_x_dot = 0
    amb_MPC.k_lat = 0.00001
    amb_MPC.k_lon = 0.0
    # amb_MPC.min_v = 0.8*initial_speed
    amb_MPC.max_v = 30 * 0.447  # m/s
    amb_MPC.k_phi_error = 0.1
    amb_MPC.k_phi_dot = 0.01
    amb_MPC.min_y = world.y_min
    amb_MPC.max_y = world.y_max
    if no_grass:
        amb_MPC.min_y += world.grass_width
        amb_MPC.max_y -= world.grass_width
    amb_MPC.fd = amb_MPC.gen_f_desired_lane(world, 0, True)
    x0_amb = np.array([0, 0, 0, 0, initial_speed, 0]).T

    return amb_MPC, x0_amb, all_other_MPC, all_other_x0


def initialize_cars_from_positions(N, dt, world, no_grass=False, list_of_positions=None, list_of_svo=None):
    '''x_variance is in terms of number of min_dist'''
    ## Create the Cars in this Problem
    if list_of_svo is None:
        list_of_svo = [0 for _ in range(len(list_of_positions) - 1)]
    # assert len(list_of_positions) == len(list_of_svo)

    all_other_x0 = []
    all_other_MPC = []
    for i in range(len(list_of_positions) - 1):
        x1_MPC = vehicle.Vehicle(dt)
        x1_MPC.agent_id = i
        x1_MPC.n_circles = 3
        x1_MPC.theta_i = 0
        # We begin by assuming theta_ij is only towards the ambulance
        x1_MPC.theta_ij[-1] = list_of_svo[i]
        x1_MPC.N = N

        x1_MPC.k_change_u_v = 0.001
        x1_MPC.max_delta_u = 50 * np.pi / 180 * x1_MPC.dt
        x1_MPC.k_u_v = 0.01
        x1_MPC.k_u_delta = .00001
        x1_MPC.k_change_u_v = 0.01
        x1_MPC.k_change_u_delta = 0.001
        x1_MPC.k_s = 0
        x1_MPC.k_x = 0
        x1_MPC.k_x_dot = -1.0 / 100.0
        x1_MPC.k_lat = 0.001
        x1_MPC.k_lon = 0.0
        x1_MPC.k_phi_error = 0.001
        x1_MPC.k_phi_dot = 0.01

        x1_MPC.min_y = world.y_min
        x1_MPC.max_y = world.y_max
        if no_grass:
            x1_MPC.min_y += world.grass_width
            x1_MPC.max_y -= world.grass_width
        x1_MPC.strict_wall_constraint = False

        lane_number, next_x0 = list_of_positions[i + 1]  #index off by one since ambulance is index 0

        initial_speed = 0.99 * x1_MPC.max_v
        x1_MPC.fd = x1_MPC.gen_f_desired_lane(world, lane_number, True)
        x0 = np.array([next_x0, world.get_lane_centerline_y(lane_number), 0, 0, initial_speed, 0]).T
        all_other_MPC += [x1_MPC]
        all_other_x0 += [x0]

    # Settings for Ambulance
    amb_MPC = cp.deepcopy(x1_MPC)
    amb_MPC.agent_id = -1
    amb_MPC.theta_i = 0.0

    amb_MPC.k_u_v = 0.0000
    amb_MPC.k_u_delta = .01
    amb_MPC.k_change_u_v = 0.0000
    amb_MPC.k_change_u_delta = 0

    amb_MPC.k_s = 0
    amb_MPC.k_x = 0
    amb_MPC.k_x_dot = -1.0 / 100.0
    # amb_MPC.k_x = -1.0/100
    # amb_MPC.k_x_dot = 0
    amb_MPC.k_lat = 0.00001
    amb_MPC.k_lon = 0.0
    # amb_MPC.min_v = 0.8*initial_speed
    amb_MPC.max_v = 30 * 0.447  # m/s
    amb_MPC.k_phi_error = 0.1
    amb_MPC.k_phi_dot = 0.01
    amb_MPC.min_y = world.y_min
    amb_MPC.max_y = world.y_max
    amb_MPC.strict_wall_constraint = True
    if no_grass:
        amb_MPC.min_y += world.grass_width
        amb_MPC.max_y -= world.grass_width

    lane_number, next_x0 = list_of_positions[0]
    amb_MPC.fd = amb_MPC.gen_f_desired_lane(world, lane_number, True)
    x0_amb = np.array([next_x0, world.get_lane_centerline_y(lane_number), 0, 0, initial_speed, 0]).T

    return amb_MPC, x0_amb, all_other_MPC, all_other_x0


def poission_positions(cars_per_hour: float,
                       total_number_cars: int,
                       n_lanes: int = 2,
                       average_velocity: float = 25 * 0.447,
                       intervehicle_spacing: float = 9.0,
                       position_random_seed: int = None):
    '''Description:  Generate the initial spacing of vehicles based on a desired vehicle density
    Simulate a poisson queuing process where vehicles arrive according to a poisson distribution
    and are assigned lanes.  Single arrival queue, n-queue departures.
    Users specifiy a density in terms of cars per hour, number of lanes, and total number of desired cars    
    Args:
        cars_per_hour:  Desired car density, specified by # cars per hour
        total_number_cars: Total desired number of cars in entire experiments
        n_lanes (int):  Number of lanes used to return lane assignment and x position
        average_velocity: int
        intervehicle_spacing: float default = 9.0 (i.e. 1 car space between)
        position_random_seed:  int

    '''
    cars_per_second = cars_per_hour / 3600.0
    #     if cars_per_second > (average_velocity / intervehicle_spacing * n_lanes):
    #         raise Exception("Car density is too large for given car velocity, number lanes, and inter-vehicle spacing")

    # Spatial resolution should be around 0.5 meters
    spatial_resolution = 0.50  #m
    dt = spatial_resolution / average_velocity
    cars_per_dt = cars_per_second * dt
    total_seconds = total_number_cars * 3600.0 / cars_per_hour * 10
    total_dt = int(total_seconds / dt)

    rng = np.random.default_rng(position_random_seed)
    n_cars_arrival_per_dt = rng.poisson(cars_per_dt, int(total_dt))  # number cars arriving at each dt
    # Exclude arrivals after we have achieved our desired number of vehicles
    n_cars_arrival_per_dt = n_cars_arrival_per_dt[:np.argmax(np.cumsum(n_cars_arrival_per_dt) > total_number_cars)]

    lane_ids = list(range(n_lanes))
    lane_car_positions = {}
    for lane in lane_ids:
        lane_car_positions[lane] = []

    road_queue = deque()
    agent_id = 0

    si = 0
    while np.sum([len(lane_car_positions[lane]) for lane in lane_ids]) < total_number_cars:
        if si >= 10 * len(n_cars_arrival_per_dt):
            raise Exception("Too many vehicles in the system ??")

        # Add incoming cars to the system
        if si < len(n_cars_arrival_per_dt) and n_cars_arrival_per_dt[si] > 0:
            # Cars are arriving to the system
            number_cars_arriving = n_cars_arrival_per_dt[si]
            for idx in range(number_cars_arriving):
                agent_id += 1
                road_queue.append(agent_id)

        # Attempt to assign arriving cars into lane
        if len(road_queue) > 0:
            rng.shuffle(lane_ids)
            for lane in lane_ids:
                if len(lane_car_positions[lane]) > 0:
                    closest_car = lane_car_positions[lane][0]
                    if closest_car > intervehicle_spacing:
                        road_queue.popleft()
                        lane_car_positions[lane] = [0.0] + lane_car_positions[lane]
                else:
                    road_queue.popleft()
                    lane_car_positions[lane] = [0.0] + lane_car_positions[lane]

                if len(road_queue) == 0:
                    break

        # All cars move forward assuming constant velocity and update their position
        for lane in lane_car_positions:
            lane_car_positions[lane] = [pos + dt * average_velocity for pos in lane_car_positions[lane]]
        si += 1

    initial_vehicle_positions = []
    for lane in lane_car_positions:
        initial_vehicle_positions += [(lane, float(x)) for x in lane_car_positions[lane]]

    initial_vehicle_positions = sorted(initial_vehicle_positions, key=lambda l: l[1])

    return initial_vehicle_positions


def poission_positions_multiple(cars_per_hour_list,
                                n_other,
                                total_seconds,
                                n_lanes,
                                average_velocity,
                                car_length,
                                position_random_seed=None):
    # n_segments = len(cars_per_hour_list)
    # cars_per_segment = np.ceil(float(n_other)/n_segments)

    initial_vehicle_positions = []
    prev_car_lane = -9999 * np.ones((n_lanes, 1))
    prev_car_lane[0] = 0.0
    segment_start_x = 0
    for lane_segment_idx in range(len(cars_per_hour_list)):
        cars_per_hour, p_cars_per_segment = cars_per_hour_list[lane_segment_idx]
        print(cars_per_hour, p_cars_per_segment)
        cars_per_segment = np.ceil(n_other * p_cars_per_segment)
        cars_per_second = cars_per_hour / 3600.0
        dt = 0.20
        cars_per_dt = cars_per_second * dt
        rng = np.random.default_rng(position_random_seed)
        n_cars_per_second = rng.poisson(cars_per_second, int(total_seconds))
        total_dt = int(total_seconds / dt)
        n_cars_per_dt = rng.poisson(cars_per_dt, int(total_dt))

        all_vehicle_positions = []

        ### Random place vehicles in the lanes
        for s in range(len(n_cars_per_dt)):
            if n_cars_per_dt[s] == 0:
                continue
            else:
                for j in range(n_cars_per_dt[s]):
                    lane_number = rng.integers(0, n_lanes)
                    x_position = segment_start_x + (average_velocity * rng.uniform(0.95, 1.05)) * (s * dt)
                    all_vehicle_positions += [(lane_number, x_position)]

        car_counter = 0
        # Remove cars that would have collided
        for (lane, x) in all_vehicle_positions:
            if x > prev_car_lane[lane] + 1.4 * car_length:  # Collision free, add vehicle
                initial_vehicle_positions += [(lane, float(x))]
                prev_car_lane[lane] = x
                car_counter += 1
            if car_counter == cars_per_segment:
                break
        segment_start_x = x

    return initial_vehicle_positions
