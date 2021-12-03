import numpy as np
import multiprocessing, functools
from typing import List, Tuple
from src.multiagent_mpc import MultiMPC
from casadi import nlpsol
from src.multiagent_mpc import MultiMPC, mpcp_to_nlpp, nlpx_to_mpcx
from vehicle_parameters import VehicleParameters


def solve_best_response(
        warm_key,
        warm_trajectory,
        response_vehicle,
        cntrld_vehicles,
        nonresponse_vehicle_list,
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
        return_bri=False) -> Tuple[bool, float, float, np.array, np.array, np.array, str, List, List[Tuple[np.array]]]:
    '''Create the iterative best response object and solve.  Assumes that it receives warm start profiles.
    This really should only require a u_warm, x_warm, x_des_warm and then one level above we generate those values'''

    u_warm, x_warm, x_des_warm = warm_trajectory
    bri = MultiMPC(response_vehicle, cntrld_vehicles, nonresponse_vehicle_list, world, solver_params)
    params["collision_avoidance_checking_distance"] = 100
    bri.generate_optimization(params["N"],
                              response_x0,
                              cntrld_x0,
                              nonresponse_x0_list,
                              params=params,
                              ipopt_params=ipopt_params)
    # print("Succesffully generated optimzation %d %d %d"%(len(nonresponse_vehicle_list), len(nonresponse_x_list), len(nonresponse_u_list)))
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

        bri.solve(cntrld_u, nonresponse_u_list)
        x_ibr, u_ibr, x_des_ibr, x_cntrld, u_cntrld, x_des_cntrld, _, _, _ = bri.get_solution()
        current_cost = bri.solution.value(bri.total_svo_cost)

        all_slack_vars = [bri.solution.value(bri.slack_i_jnc),
                          bri.solution.value(bri.slack_i_jc)] + [bri.solution.value(s) for s in bri.slack_ic_jc
                                                                 ] + [bri.solution.value(s) for s in bri.slack_ic_jnc]
        max_slack = np.max([np.max(s) for s in all_slack_vars] + [0.000000000000])

        debug_list = [
            current_cost,
            bri.solution.value(bri.response_svo_cost),
            bri.solution.value(bri.k_CA * bri.collision_cost),
            bri.solution.value(bri.k_slack * bri.slack_cost), all_slack_vars
        ]
        if return_bri:
            debug_list += [bri]
        cntrld_vehicle_trajectories = [(x_cntrld[j], x_des_cntrld[j], u_cntrld[j]) for j in range(len(x_cntrld))]
        return True, current_cost, max_slack, x_ibr, x_des_ibr, u_ibr, warm_key, debug_list, cntrld_vehicle_trajectories
    except RuntimeError:
        if return_bri:
            return bri
        return False, np.infty, np.infty, None, None, None, None, [], None


def solve_warm_starts(
        ux_warm_profiles,
        response_veh_info,
        world,
        solver_params,
        params,
        ipopt_params,
        nonresponse_veh_info,
        cntrl_veh_info,
        debug_flag=False) -> Tuple[bool, float, float, np.array, np.array, np.array, str, List, List[Tuple[np.array]]]:

    response_vehicle = response_veh_info.vehicle
    response_x0 = response_veh_info.x0

    nonresponse_vehicle_list = [veh_info.vehicle for veh_info in nonresponse_veh_info]
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
                                           response_vehicle=response_vehicle,
                                           cntrld_vehicles=cntrld_vehicles,
                                           nonresponse_vehicle_list=nonresponse_vehicle_list,
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
                print("Costs: Total Cost %.04f Vehicle-Only Cost:  %.04f Collision Cost %0.04f  Slack Cost %0.04f" %
                      (tuple(debug_list[0:4])))

    min_cost_solution = min(solve_costs_solutions, key=lambda r: r[1])

    return min_cost_solution


def solve_best_response_c(
        warm_key,
        warm_trajectory,
        response_vehicle,
        cntrld_vehicles,
        nonresponse_vehicle_list,
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
        return_bri=False) -> Tuple[bool, float, float, np.array, np.array, np.array, str, List, List[Tuple[np.array]]]:
    '''Create the iterative best response object and solve.  Assumes that it receives warm start profiles.
    This really should only require a u_warm, x_warm, x_des_warm and then one level above we generate those values'''

    u_warm, x_warm, x_des_warm = warm_trajectory

    nc = len(cntrld_vehicles)
    nnc = len(nonresponse_vehicle_list)
    params["collision_avoidance_checking_distance"] = 100

    x0warm = get_warm_start_x0(x_warm, u_warm, x_des_warm, cntrld_x, cntrld_u, cntrld_xd, nonresponse_x_list,
                               nonresponse_u_list, nonresponse_xd_list)

    # This doesn't change and could be moved outside this method
    mpc = MultiMPC(params["N"], world, nc, nnc, solver_params, params, ipopt_params)
    lbg_mpc, ubg_mpc = get_bounds_from_mpc(mpc)
    solver = load_solver_from_file("./gen.so")
    # Get the constants for the MPC solver

    p_ego, p_cntrld, p_nc = convert_vehicles_to_parameters(nc, nnc, response_vehicle, cntrld_vehicles,
                                                           nonresponse_vehicle_list)
    theta_ego_i, theta_ic, theta_i_nc = get_svo_values()
    p0 = mpcp_to_nlpp(response_x0, p_ego, theta_ego_i, theta_ic, theta_i_nc, cntrld_x0, p_cntrld, nonresponse_x0_list,
                      p_nc)
    set_constant_v_constraint(params)  #this is carry over and should be removed

    # Call the solver
    try:
        solution = solver(x0=x0warm, p=p0, lbg=lbg_mpc, ubg=ubg_mpc)
        x_ego, u_ego, xd_ego, cntrld_vehicle_trajectories, max_slack, current_cost = get_trajectories_from_solution(
            solution, params["N"], nc, nnc)

        debug_list = []
        return True, current_cost, max_slack, x_ego, xd_ego, u_ego, warm_key, debug_list, cntrld_vehicle_trajectories
    except RuntimeError:
        return False, np.infty, np.infty, None, None, None, None, [], None


def get_trajectories_from_solution(nlp_solution, N, nc, nnc):
    current_cost = nlp_solution['f']

    traj = nlp_solution['x']

    x_ego, u_ego, xd_ego, x_ctrl, u_ctrl, xd_ctrl, _, _, _, s_i_jnc, s_ic_jnc, s_i_jc, s_ic_jc, _, _, _, _ = nlpx_to_mpcx(
        traj, N, nc, nnc)

    cntrld_vehicle_trajectories = [(x_ctrl[j], xd_ctrl[j], u_ctrl[j]) for j in range(len(x_ctrl))]

    # Compute the maximum slack
    all_slack_vars = [s_i_jnc, s_i_jc] + [s for s in s_ic_jc] + [s for s in s_ic_jnc]
    max_slack = np.max([np.max(s) for s in all_slack_vars] + [0.000000000000])

    return x_ego, u_ego, xd_ego, cntrld_vehicle_trajectories, max_slack, current_cost


def get_svo_values():
    return NotImplementedError


def load_solver_from_file(filename):
    ''' Load the .so file'''
    nlpsolver = nlpsol('solver', 'ipopt', filename)

    return nlpsolver


def get_bounds_from_mpc(mpc):
    lbg = mpc._lbg_list
    ubg = mpc._ubg_list

    return lbg, ubg


def get_warm_start_x0(nx):
    ''' Convert warm starts for the ego vehicle and control vehicle into a nx X 1 array'''
    raise NotImplementedError
    x0warm = np.zeros((nx))

    return None


def get_non_response_p0():
    ''' Convert non response trajectories to np x 1 parameter array that will be loaded'''
    raise NotImplementedError
    return None


def set_constant_v_constraint(solver_params):
    raise NotImplementedError
    if "constant_v" in solver_params and solver_params["constant_v"]:
        bri.opti.subject_to(bri.u_ego[1, :] == 0)


def convert_vehicles_to_parameters(nc, nnc, ego_vehicle, cntrld_vehicles, nonresponse_vehicles):
    ''' Converts each vehicle into a list of parameters
    
        nc:  # of controlled vehicles
        nnc: # of non response vehicles 
    '''

    p_ego = convert_vehicle_to_parameter(nc, nnc, ego_vehicle)
    p_cntrld = [convert_vehicle_to_parameter(nc, nnc, v) for v in cntrld_vehicles]
    p_nc = [convert_vehicle_to_parameter(nc, nnc, v) for v in nonresponse_vehicles]

    return p_ego, p_cntrld, p_nc


def convert_vehicle_to_parameter(nc, nnc, vehicle):

    vp = VehicleParameters(nc, nnc)
    vp.set_param_values(vehicle)

    return vp