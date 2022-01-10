import numpy as np
import multiprocessing, functools
from typing import List, Tuple, Dict
from src.multiagent_mpc import MultiMPC, mpcx_to_nlpx, nlpp_to_mpcp
from casadi import nlpsol
from src.multiagent_mpc import MultiMPC, mpcp_to_nlpp, nlpx_to_mpcx
from src.vehicle_parameters import VehicleParameters
from src.vehicle_mpc_information import Trajectory
import time, os, pickle
from casadi import nlpsol


def _solve_best_response(
    warm_key: str,
    warm_trajectory: Trajectory,
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
    cntrld_xd=[]
) -> Tuple[bool, float, float, np.array, np.array, np.array, str, List,
           List[Tuple[np.array]]]:
    '''Create the iterative best response object and solve.  Assumes that it receives warm start profiles.
    This really should only require a u_warm, x_warm, x_des_warm and then one level above we generate those values'''

    u_warm, x_warm, x_des_warm = warm_trajectory
    bri = MultiMPC(response_vehicle, cntrld_vehicles, nonresponse_vehicle_list,
                   world, solver_params)
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
        x_ibr, u_ibr, x_des_ibr, x_cntrld, u_cntrld, x_des_cntrld, _, _, _ = bri.get_solution(
        )
        current_cost = bri.solution.value(bri.total_svo_cost)

        all_slack_vars = [
            bri.solution.value(bri.slack_i_jnc),
            bri.solution.value(bri.slack_i_jc)
        ] + [bri.solution.value(s) for s in bri.slack_ic_jc
             ] + [bri.solution.value(s) for s in bri.slack_ic_jnc]
        max_slack = np.max([np.max(s)
                            for s in all_slack_vars] + [0.000000000000])

        debug_list = [
            current_cost,
            bri.solution.value(bri.response_svo_cost),
            bri.solution.value(bri.k_CA * bri.collision_cost),
            bri.solution.value(bri.k_slack * bri.slack_cost), all_slack_vars
        ]
        # if return_bri:
        #     debug_list += [bri]
        cntrld_vehicle_trajectories = [(x_cntrld[j], x_des_cntrld[j],
                                        u_cntrld[j])
                                       for j in range(len(x_cntrld))]
        return True, current_cost, max_slack, x_ibr, x_des_ibr, u_ibr, warm_key, debug_list, cntrld_vehicle_trajectories
    except RuntimeError:
        # if return_bri:
        #     return bri
        return False, np.infty, np.infty, None, None, None, None, [], None


def solve_warm_starts(
    warm_start_trajectories: Dict[str, Trajectory],
    response_veh_info,
    world,
    solver_params,
    params,
    ipopt_params,
    nonresponse_veh_info,
    cntrl_veh_info,
    max_slack=np.infty,
    debug_flag=False
) -> Tuple[bool, float, float, np.array, np.array, np.array, str, List,
           List[Tuple[np.array]]]:

    response_vehicle = response_veh_info.vehicle
    response_x0 = response_veh_info.x0

    nonresponse_vehicle_list = [
        veh_info.vehicle for veh_info in nonresponse_veh_info
    ]
    nonresponse_x0_list = [veh_info.x0 for veh_info in nonresponse_veh_info]
    nonresponse_x_list = [veh_info.x for veh_info in nonresponse_veh_info]

    cntrld_vehicles = [veh_info.vehicle for veh_info in cntrl_veh_info]
    cntrld_x0 = [veh_info.x0 for veh_info in cntrl_veh_info]
    cntrld_u_warm = [veh_info.u for veh_info in cntrl_veh_info]
    cntrld_x_warm = [veh_info.x for veh_info in cntrl_veh_info]
    cntrld_xd_warm = [veh_info.xd for veh_info in cntrl_veh_info]

    warm_solve_partial = functools.partial(
        solve_best_response_c,
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
        nonresponse_x_list=nonresponse_x_list,
        cntrld_u_warm=cntrld_u_warm,
        cntrld_x_warm=cntrld_x_warm,
        cntrld_xd_warm=cntrld_xd_warm)

    if params['n_processors'] > 1:
        pool = multiprocessing.Pool(processes=params['n_processors'])
        solve_costs_solutions = pool.starmap(
            warm_solve_partial, warm_start_trajectories.items(
            ))  #will apply k=1...N to plot_partial
        pool.terminate()
    else:
        solve_costs_solutions = []
        for k_warm in warm_start_trajectories:
            solve_costs_solutions += [
                warm_solve_partial(k_warm, warm_start_trajectories[k_warm])
            ]

    if debug_flag:
        for ki in range(len(solve_costs_solutions)):
            debug_list = solve_costs_solutions[ki][7]
            print(debug_list)
            if len(debug_list) == 0:
                print("Infeasible")
            else:
                print(
                    "Costs: Total Cost %.04f Vehicle-Only Cost:  %.04f Collision Cost %0.04f  Slack Cost %0.04f"
                    % (tuple(debug_list[0:4])))

    below_max_slack_sols = [
        s for s in solve_costs_solutions if s[2] <= max_slack
    ]
    if len(below_max_slack_sols) > 0:
        print("# Feasible Solutions Below Max Slack: %d"%len(below_max_slack_sols))
        return min(below_max_slack_sols, key=lambda r: r[1])
    else:
        min_cost_solution = min(solve_costs_solutions, key=lambda r: r[1])
        print("# No Feasible Solutions Below Max Slack. Returning with Slack = %.02f"%min_cost_solution[2])

    return min_cost_solution


def pre_caller(warmstart_dict: Dict[str, Trajectory], response_veh_info, world,
               solver_params, params, ipopt_params, nonresponse_veh_info,
               cntrl_veh_info, pickled:bool=True):
    # Grab components needed for generating MPC
    response_vehicle = response_veh_info.vehicle
    response_x0 = response_veh_info.x0

    nonresponse_vehicles = [vi.vehicle for vi in nonresponse_veh_info]
    nonresponse_x0_list = [vi.x0 for vi in nonresponse_veh_info]
    nonresponse_x_list = [vi.x for vi in nonresponse_veh_info]

    cntrld_vehicles = [veh_info.vehicle for veh_info in cntrl_veh_info]
    cntrld_x0 = [veh_info.x0 for veh_info in cntrl_veh_info]
    cntrld_u_warm = [veh_info.u for veh_info in cntrl_veh_info]
    cntrld_x_warm = [veh_info.x for veh_info in cntrl_veh_info]
    cntrld_xd_warm = [veh_info.xd for veh_info in cntrl_veh_info]

    nc = len(cntrld_vehicles)
    nnc = len(nonresponse_vehicles)
    # Generate an MPC

    # Convert vehicle paramaters into an array
    p_ego, p_cntrld, p_nc = convert_vehicles_to_parameters(
        nc, nnc, response_vehicle, cntrld_vehicles, nonresponse_vehicles)
    theta_ego_i, theta_ic, theta_i_nc = get_fake_svo_values(nc, nnc)

    # nlp_solver = load_solver_from_mpc(mpc, precompiled=False)
    precompiled_code_dir = params["precompiled_solver_dir"]

    call_mpc_solver_on_warm = functools.partial(
        call_mpc_solver,
        pickled=pickled,
        precompiled_code_dir=precompiled_code_dir,
        cntrld_u_warm=cntrld_u_warm,
        cntrld_x_warm=cntrld_x_warm,
        cntrld_xd_warm=cntrld_xd_warm,
        world=world,
        nc=nc,
        nnc=nnc,
        p_ego=p_ego,
        p_cntrld=p_cntrld,
        p_nc=p_nc,
        theta_ego_i=theta_ego_i,
        theta_ic=theta_ic,
        theta_i_nc=theta_i_nc,
        response_x0=response_x0,
        cntrld_x0=cntrld_x0,
        nonresponse_x0_list=nonresponse_x0_list,
        nonresponse_x_list=nonresponse_x_list,
        params=params,
        solver_params=solver_params,
        ipopt_params=ipopt_params)

    if params["n_processors"] > 1:
        pool = multiprocessing.Pool(processes=params['n_processors'])
        solve_costs_solutions = pool.starmap(call_mpc_solver_on_warm,
                                             warmstart_dict.items())
        pool.terminate()
    else:
        solve_costs_solutions = []
        for warm_key, warm_trajectory in warmstart_dict.items():
            solve_costs_solutions += [
                call_mpc_solver_on_warm(warm_key, warm_trajectory)
            ]

    below_max_slack_sols = [
        s for s in solve_costs_solutions if s[2] <= params["k_max_slack"]
    ]


    if len(below_max_slack_sols) > 0:
        print("# Feasible Solutions Below Max Slack: %d"%len(below_max_slack_sols))
        min_cost_solution = min(below_max_slack_sols, key=lambda r: r[1])
    else:
        min_cost_solution = min(solve_costs_solutions, key=lambda r: r[1])
        print("# No Feasible Solutions Below Max Slack. Returning with Slack = %.02f"%min_cost_solution[2])
    
    return min_cost_solution


def call_mpc_solver(
    warm_key: str,
    warm_trajectory: Trajectory,
    precompiled_code_dir: str,
    pickled: bool,
    cntrld_u_warm,
    cntrld_x_warm,
    cntrld_xd_warm,
    world,
    nc,
    nnc,
    p_ego,
    p_cntrld,
    p_nc,
    theta_ego_i,
    theta_ic,
    theta_i_nc,
    response_x0,
    cntrld_x0,
    nonresponse_x0_list,
    nonresponse_x_list,
    params,
    solver_params,
    ipopt_params,
) -> Tuple[bool, float, float, np.array, np.array, np.array, str, List,
           List[Tuple[np.array]]]:
    '''Create the iterative best response object and solve.  Assumes that it receives warm start profiles.
    This really should only require a u_warm, x_warm, x_des_warm and then one level above we generate those values'''
    nlp_p = mpcp_to_nlpp(response_x0, p_ego, theta_ego_i, theta_ic, theta_i_nc,
                         cntrld_x0, p_cntrld, nonresponse_x0_list, p_nc,
                         nonresponse_x_list)
    nlp_x0 = get_warm_start_x0(nc, nnc, warm_trajectory, cntrld_x_warm,
                               cntrld_u_warm, cntrld_xd_warm)



    # Call the solver
    # Load pre-compiled solver and bounds
    if pickled:
        nlp_solver_name = "mpc_%02d_%02d.p" % (nc, nnc)
        nlp_solver_path = os.path.join(precompiled_code_dir, nlp_solver_name)      
        nlp_solver = pickle.load(open(nlp_solver_path, "rb"))  
    else:
        nlp_solver_name = "mpc_%02d_%02d.so" % (nc, nnc)
        nlp_solver_path = os.path.join(precompiled_code_dir, nlp_solver_name)
        nlp_solver = nlpsol('solver', 'ipopt', nlp_solver_path)
    nlp_lbg, nlp_ubg = get_bounds_from_compiled_mpc(nc, nnc, precompiled_code_dir)


    try:
        solution = nlp_solver(x0=nlp_x0, p=nlp_p, lbg=nlp_lbg, ubg=nlp_ubg)
        x_ego, u_ego, xd_ego, cntrld_vehicle_trajectories, max_slack, current_cost = get_trajectories_from_solution(
            solution, params["N"], nc, nnc)
        debug_list = []
        return True, current_cost, max_slack, x_ego, xd_ego, u_ego, warm_key, debug_list, cntrld_vehicle_trajectories
    except Exception as e:
        print(e)
        return False, np.infty, np.infty, None, None, None, None, [], None


def solve_best_response_c(
    warm_key: str,
    warm_trajectory: Trajectory,
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
    nonresponse_x_list,
    cntrld_u_warm=None,
    cntrld_x_warm=None,
    cntrld_xd_warm=None
) -> Tuple[bool, float, float, np.array, np.array, np.array, str, List,
           List[Tuple[np.array]]]:
    '''Create the iterative best response object and solve.  Assumes that it receives warm start profiles.
    This really should only require a u_warm, x_warm, x_des_warm and then one level above we generate those values'''
    nc = len(cntrld_vehicles)
    nnc = len(nonresponse_vehicle_list)
    params["collision_avoidance_checking_distance"] = 100

    nlp_x0 = get_warm_start_x0(nc, nnc, warm_trajectory, cntrld_x_warm,
                               cntrld_u_warm, cntrld_xd_warm)
    mpc = MultiMPC(params["N"], world, nc, nnc, solver_params, params,
                   ipopt_params)

    nlp_lbg, nlp_ubg = get_bounds_from_mpc(mpc)
    nlp_solver = load_solver_from_mpc(mpc, precompiled=False)
    p_ego, p_cntrld, p_nc = convert_vehicles_to_parameters(
        nc, nnc, response_vehicle, cntrld_vehicles, nonresponse_vehicle_list)
    theta_ego_i, theta_ic, theta_i_nc = get_fake_svo_values(nc, nnc)
    nlp_p = mpcp_to_nlpp(response_x0, p_ego, theta_ego_i, theta_ic, theta_i_nc,
                         cntrld_x0, p_cntrld, nonresponse_x0_list, p_nc,
                         nonresponse_x_list)

    set_constant_v_constraint(
        params)  #this is carry over and should be removed
    # response_x0t, p_egot, theta_ego_it, theta_ict, theta_i_nct, cntrld_xt, p_cntrldt, nonresponse_x0_listt, p_nct, nonresponse_x_listt = nlpp_to_mpcp(
    #     nlp_p, params["N"], nc, nnc, 6, mpc.p_size)
    # Call the solver
    try:
        solution = nlp_solver(x0=nlp_x0, p=nlp_p, lbg=nlp_lbg, ubg=nlp_ubg)
        x_ego, u_ego, xd_ego, cntrld_vehicle_trajectories, max_slack, current_cost = get_trajectories_from_solution(
            solution, params["N"], nc, nnc)
        debug_list = []
        return True, current_cost, max_slack, x_ego, xd_ego, u_ego, warm_key, debug_list, cntrld_vehicle_trajectories
    except Exception as e:
        print(e)
        return False, np.infty, np.infty, None, None, None, None, [], None


def get_trajectories_from_solution(nlp_solution, N, nc, nnc):
    current_cost = nlp_solution['f']

    traj = nlp_solution['x']

    x_ego, u_ego, xd_ego, x_ctrl, u_ctrl, xd_ctrl, s_i_jnc, s_ic_jnc, s_i_jc, s_ic_jc, _, _, _, _ = nlpx_to_mpcx(
        traj, N, nc, nnc)

    x_ego = np.array(x_ego)
    u_ego = np.array(u_ego)
    xd_ego = np.array(xd_ego)
    cntrld_vehicle_trajectories = [
        Trajectory(x=np.array(x_ctrl[j]),
                   xd=np.array(xd_ctrl[j]),
                   u=np.array(u_ctrl[j])) for j in range(len(x_ctrl))
    ]
    # cntrld_vehicle_trajectories = [(x_ctrl[j], xd_ctrl[j], u_ctrl[j]) for j in range(len(x_ctrl))]

    # Compute the maximum slack

    all_slack_vars = [s_i_jnc, s_i_jc] + [s for s in s_ic_jc
                                          ] + [s for s in s_ic_jnc]
    max_slack = np.max([np.max(s) for s in all_slack_vars] + [0.000000000000])

    return x_ego, u_ego, xd_ego, cntrld_vehicle_trajectories, max_slack, current_cost


def get_fake_svo_values(n_ctrl, n_other):
    ''' Generate fake svo parameters with correct output for testing'''

    theta_i_ego = np.pi / 4
    theta_ic = [np.pi / 4 for _ in range(n_ctrl)]
    theta_inc = [np.pi / 4 for _ in range(n_other)]
    return theta_i_ego, theta_ic, theta_inc


def load_solver_from_file(filename):
    ''' Load the .so file'''
    nlpsolver = nlpsol('solver', 'ipopt', filename)

    return nlpsolver


def load_solver_from_mpc(mpc, precompiled: bool = True):
    ''' Get name of sovler from mpc'''
    if precompiled:
        solver_name_prefix = mpc.solver_prefix
        solver = load_solver_from_file("./%s.so" % solver_name_prefix)
    else:
        solver = mpc.solver

    return solver

def get_bounds_from_compiled_mpc(nc, nnc, precompiled_code_dir):
    bounds_path_name = "bounds_%02d%02d.p"%(nc, nnc)
    bounds_full_path = os.path.join(precompiled_code_dir, bounds_path_name)
    lbg, ubg = pickle.load(open(bounds_full_path, 'rb'))

    return lbg, ubg

def get_bounds_from_mpc(mpc):

    lbg = mpc._lbg_list
    ubg = mpc._ubg_list

    return lbg, ubg


def init_slack_vars_zero(N, n_vehs_cntrld, n_other_vehicle):
    ''' Return zeros for slack vars'''
    s_i_jnc = np.zeros(shape=(n_other_vehicle, N + 1))
    s_ic_jnc = [
        np.zeros(shape=(n_other_vehicle, N + 1)) for i in range(n_vehs_cntrld)
    ]
    s_i_jc = np.zeros(shape=(n_vehs_cntrld, N + 1))
    s_ic_jc = [
        np.zeros(shape=(n_vehs_cntrld, N + 1)) for ic in range(n_vehs_cntrld)
    ]

    s_top = np.zeros(shape=(1, N + 1))
    s_bottom = np.zeros(shape=(1, N + 1))
    s_c_top = [np.zeros(shape=(1, N + 1)) for i in range(n_vehs_cntrld)]
    s_c_bottom = [np.zeros(shape=(1, N + 1)) for i in range(n_vehs_cntrld)]

    return s_i_jnc, s_ic_jnc, s_i_jc, s_ic_jc, s_top, s_bottom, s_c_top, s_c_bottom


def init_random_warm_starts(n_vehs_cntrld, x_warm, u_warm, x_des_warm):
    # initialize with random warm starts
    x_ctrld_warm = [
        np.random.uniform(size=x_warm.shape) for _ in range(n_vehs_cntrld)
    ]
    u_ctrld_warm = [
        np.random.uniform(size=u_warm.shape) for _ in range(n_vehs_cntrld)
    ]
    xd_ctrld_warm = [
        np.random.uniform(size=x_des_warm.shape) for _ in range(n_vehs_cntrld)
    ]

    return x_ctrld_warm, u_ctrld_warm, xd_ctrld_warm


def get_warm_start_x0(n_vehs_cntrld,
                      n_other,
                      ego_warm_trajectory,
                      x_ctrld_warm=None,
                      u_ctrld_warm=None,
                      xd_ctrld_warm=None):
    ''' Convert warm starts for the ego vehicle and control vehicle into a nx X 1 array'''
    x_warm = ego_warm_trajectory.x
    u_warm = ego_warm_trajectory.u
    x_des_warm = ego_warm_trajectory.xd

    N = u_warm.shape[1]

    s_i_jnc, s_ic_jnc, s_i_jc, s_ic_jc, s_top, s_bottom, s_c_top, s_c_bottom = init_slack_vars_zero(
        N, n_vehs_cntrld, n_other)

    if x_ctrld_warm is None:
        x_ctrld_warm, u_ctrld_warm, xd_ctrld_warm = init_random_warm_starts(
            n_vehs_cntrld, x_warm, u_warm, x_des_warm)

    nlp_x = mpcx_to_nlpx(n_other, x_warm, u_warm, x_des_warm, x_ctrld_warm,
                         u_ctrld_warm, xd_ctrld_warm, s_i_jnc, s_ic_jnc,
                         s_i_jc, s_ic_jc, s_top, s_bottom, s_c_top, s_c_bottom)

    return nlp_x


def get_non_response_p0():
    ''' Convert non response trajectories to np x 1 parameter array that will be loaded'''
    raise NotImplementedError
    return None


def set_constant_v_constraint(solver_params):
    return None
    raise NotImplementedError
    if "constant_v" in solver_params and solver_params["constant_v"]:
        bri.opti.subject_to(bri.u_ego[1, :] == 0)


def convert_vehicles_to_parameters(nc, nnc, ego_vehicle, cntrld_vehicles,
                                   nonresponse_vehicles):
    ''' Converts each vehicle into a list of parameters
    
        nc:  # of controlled vehicles
        nnc: # of non response vehicles 
    '''

    p_ego = convert_vehicle_to_parameter(nc, nnc, ego_vehicle)
    p_cntrld = [
        convert_vehicle_to_parameter(nc, nnc, v) for v in cntrld_vehicles
    ]
    p_nc = [
        convert_vehicle_to_parameter(nc, nnc, v) for v in nonresponse_vehicles
    ]

    return p_ego, p_cntrld, p_nc


def convert_vehicle_to_parameter(nc, nnc, vehicle):

    vp = VehicleParameters(nc, nnc)
    vp.set_param_values(vehicle)

    return vp