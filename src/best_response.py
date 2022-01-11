import numpy as np
import multiprocessing, functools, time, os, pickle
from typing import List, Tuple, Dict
from casadi import nlpsol

from src.multiagent_mpc import MultiMPC, mpcp_to_nlpp, nlpx_to_mpcx, mpcx_to_nlpx
from src.vehicle_parameters import VehicleParameters
from src.traffic_world import TrafficWorld
from src.vehicle import Vehicle
from src.vehicle_mpc_information import VehicleMPCInformation, Trajectory




def parallel_mpc_solve(warmstart_dict: Dict[str, Trajectory],
                       response_veh_info: VehicleMPCInformation,
                       world: TrafficWorld, solver_params: dict, params: dict,
                       ipopt_params: dict,
                       nonresponse_veh_info: List[VehicleMPCInformation],
                       cntrl_veh_info: List[VehicleMPCInformation]):
    ''' 1. Generate and reformat some variables to be fed into the call_mpc_solver method
        2. Parallelize and call the MPC solver for the dictionary of warm starts for the ego vehicle
    '''

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

    # Convert vehicle paramaters into an array
    p_ego, p_cntrld, p_nc = convert_vehicles_to_parameters(
        nc, nnc, response_vehicle, cntrld_vehicles, nonresponse_vehicles)
    theta_ego_i, theta_ic, theta_i_nc = get_svo_values(response_vehicle, cntrld_vehicles, nonresponse_vehicles)

    precompiled_code_dir = params["precompiled_solver_dir"]
    solver_mode = params["solver_mode"]

    call_mpc_solver_on_warm = functools.partial(
        call_mpc_solver,
        solver_mode=solver_mode,
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
        print("# Feasible Solutions Below Max Slack: %d" %
              len(below_max_slack_sols))
        min_cost_solution = min(below_max_slack_sols, key=lambda r: r[1])
    else:
        min_cost_solution = min(solve_costs_solutions, key=lambda r: r[1])
        print(
            "# No Feasible Solutions Below Max Slack. Returning with Slack = %.02f"
            % min_cost_solution[2])

    return min_cost_solution


def call_mpc_solver(
    warm_key: str,
    warm_trajectory: Trajectory,
    precompiled_code_dir: str,
    solver_mode: str,
    cntrld_u_warm: np.array,
    cntrld_x_warm: np.array,
    cntrld_xd_warm: np.array,
    world: TrafficWorld,
    nc: int,
    nnc: int,
    p_ego: VehicleParameters,
    p_cntrld: List[VehicleParameters],
    p_nc: List[VehicleParameters],
    theta_ego_i,
    theta_ic,
    theta_i_nc,
    response_x0: np.array,
    cntrld_x0: List[np.array],
    nonresponse_x0_list: List[np.array],
    nonresponse_x_list: List[np.array],
    params: dict,
    solver_params: dict,
    ipopt_params: dict,
) -> Tuple[bool, float, float, np.array, np.array, np.array, str, List,
           List[Tuple[np.array]]]:
    '''Create the iterative best response object and solve.  Assumes that it receives warm start profiles.
    This really should only require a u_warm, x_warm, x_des_warm and then one level above we generate those values'''
    nlp_p = mpcp_to_nlpp(response_x0, p_ego, theta_ego_i, theta_ic, theta_i_nc,
                         cntrld_x0, p_cntrld, nonresponse_x0_list, p_nc,
                         nonresponse_x_list)
    nlp_x0 = get_warm_start_x0(nc, nnc, warm_trajectory, cntrld_x_warm,
                               cntrld_u_warm, cntrld_xd_warm)

    if solver_mode.lower() == "pickled":
        nlp_solver, nlp_lbg, nlp_ubg = get_pickled_solver(
            precompiled_code_dir, nc, nnc)
    elif solver_mode.lower() == "compiled":
        nlp_solver, nlp_lbg, nlp_ubg = get_compiled_solver(
            precompiled_code_dir, nc, nnc)
    else:
        # Generate the solver from scratch. This can take ~8s for large
        mpc = MultiMPC(params["N"], world, nc, nnc, solver_params, params,
                       ipopt_params)
        nlp_solver = load_solver_from_mpc(mpc, precompiled=False)
        nlp_lbg, nlp_ubg = get_bounds_from_mpc(mpc)

    # Try to solve the mpc
    try:
        solution = nlp_solver(x0=nlp_x0, p=nlp_p, lbg=nlp_lbg, ubg=nlp_ubg)
        x_ego, u_ego, xd_ego, cntrld_vehicle_trajectories, max_slack, current_cost = get_trajectories_from_solution(
            solution, params["N"], nc, nnc)
        debug_list = []
        return True, current_cost, max_slack, x_ego, xd_ego, u_ego, warm_key, debug_list, cntrld_vehicle_trajectories
    except Exception as e:
        print(e)
        return False, np.infty, np.infty, None, None, None, None, [], None


def get_compiled_solver(precompiled_code_dir, nc, nnc):
    ''' Use a compiled .so version of the solver '''

    nlp_solver_name = "mpc_%02d_%02d.so" % (nc, nnc)
    nlp_solver_path = os.path.join(precompiled_code_dir, nlp_solver_name)

    nlp_solver = nlpsol('solver', 'ipopt', nlp_solver_path)

    nlp_lbg, nlp_ubg = get_bounds_from_compiled_mpc(nc, nnc,
                                                    precompiled_code_dir)

    return nlp_solver, nlp_lbg, nlp_ubg


def get_pickled_solver(precompiled_code_dir, nc, nnc):
    ''' Load the NLP Solver from a pickled version of it'''

    nlp_solver_name = "mpc_%02d_%02d.p" % (nc, nnc)
    nlp_solver_path = os.path.join(precompiled_code_dir, nlp_solver_name)

    nlp_solver = pickle.load(open(nlp_solver_path, "rb"))

    nlp_lbg, nlp_ubg = get_bounds_from_compiled_mpc(nc, nnc,
                                                    precompiled_code_dir)

    return nlp_solver, nlp_lbg, nlp_ubg


def get_trajectories_from_solution(nlp_solution, N, nc, nnc):
    ''' Converts the returned solution from Casadi.NLPSolver object to trajectories and costs'''

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

def get_svo_values(response_vehicle: Vehicle, cntrld_vehicles: List[Vehicle], noncontrolled_vehicles: List[Vehicle]):
    ''' Return the theta_ij w.r.t the control vehicles'''
    theta_ic = []
    for cntrld_vehicle in cntrld_vehicles:
        theta_ic += [response_vehicle.get_theta_ij(cntrld_vehicle.agent_id)]

    theta_inc = []
    for vehicle in noncontrolled_vehicles:
        theta_inc += [response_vehicle.get_theta_ij(vehicle)]
    
    theta_i_ego = response_vehicle.theta_ij[-1]

    return theta_i_ego, theta_ic, theta_inc

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
    bounds_path_name = "bounds_%02d%02d.p" % (nc, nnc)
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



def generate_solver_params(params, i_ibr, solve_number):
    solver_params = {}
    solver_params["slack"] = (True if i_ibr <= params["k_max_round_with_slack"]
                              else False)
    solver_params[
        "n_warm_starts"] = params["n_processors"]  + 5 * solve_number
    solver_params["k_CA"] = params["k_CA_d"]
    solver_params["k_CA_power"] = params["k_CA_power"]

    solver_params["k_slack"] = (params["k_slack_d"] * 10**solve_number)

    return solver_params