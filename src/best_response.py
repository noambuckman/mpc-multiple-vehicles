import numpy as np
import multiprocessing, functools, time, os, pickle
from typing import List, Tuple, Dict

from src.multiagent_mpc import MultiMPC, mpcp_to_nlpp, nlpx_to_mpcx, mpcx_to_nlpx, get_pickled_solver, get_compiled_solver, load_solver_from_mpc, get_bounds_from_mpc
from src.vehicle_parameters import VehicleParameters
from src.traffic_world import TrafficWorld
from src.vehicle import Vehicle
from src.vehicle_mpc_information import VehicleMPCInformation, Trajectory
from src.desired_trajectories import PiecewiseDesiredTrajectory
import tqdm
class MPCSolverReturn(object):
    def __init__(self, solved_status:bool, 
                        current_cost:float, 
                        max_slack:float, trajectory: Trajectory, warm_key: str, desired_traj: PiecewiseDesiredTrajectory, debug_list: List, cntrld_vehicle_trajectories: List[np.array], 
                        max_cpu_limit: bool = False, g_violation: float = 0.0):
        self.solved_status = solved_status
        self.current_cost = current_cost 
        self.max_slack = max_slack
        self.trajectory = trajectory
        self.warm_key = warm_key
        self.desired_traj = desired_traj
        self.debug_list = debug_list
        self.cntrld_vehicle_trajectories = cntrld_vehicle_trajectories
        self.max_cpu_limit = max_cpu_limit
        self.g_violation = g_violation

class MPCSolverReturnException(MPCSolverReturn):
    def __init__(self):
        MPCSolverReturn.__init__(self, False, np.infty, np.infty, None, None, None, [], None)

class MPCSolverLog(object):
    def __init__(self, solver_return:MPCSolverReturn, i_mpc:int, i_ibr:int, agent_id:int, solve_i:int):
        self.solver_return = solver_return
        self.i_mpc = i_mpc
        self.i_ibr = i_ibr
        self.agent_id = agent_id
        self.solve_i = solve_i

class MPCSolverLogger(object):
    def __init__(self, log_dir):
        self.logs = []
        self.filename = os.path.join(log_dir, "mpc_solver_logger.p")

        self.logs_indexed = {}

    def add_log(self, log : MPCSolverLog):
        self.logs.append(log)

    def write(self):
        with open(self.filename, 'wb') as fp:
            pickle.dump(self.logs, fp)

    def read(self):
        with open(self.filename, 'rb') as fp:
            self.logs = pickle.load(fp)
    
    def index_logs(self):
        for log in self.logs:
            self.logs_indexed[(log.i_mpc, log.i_ibr, log.agent_id, log.solve_i)] = log.solver_return
    
    def get_log(self, i_mpc, i_ibr, agent_id, solve_i):
        return self.logs_indexed[(i_mpc, i_ibr, agent_id, solve_i)]

    
# def parallel_mpc_solve(warmstart_dict: Dict[str, Trajectory],
#                        response_veh_info: VehicleMPCInformation,
#                        world: TrafficWorld, solver_params: dict, params: dict,
#                        ipopt_params: dict,
#                        nonresponse_veh_info: List[VehicleMPCInformation],
#                        cntrl_veh_info: List[VehicleMPCInformation]) -> MPCSolverReturn:

#     ''' Only allow a single desired trajectory. This is the same before as before'''
#     response_vehicle = response_veh_info.vehicle

#     # desired_trajectories = LaneFollowingPiecewiseTrajectory(response_vehicle, world)

#     # Generate dictionary with the single x_coeff_d
#     warmstart_w_traj_dict = {}
#     for warm_key, warm_traj in warmstart_dict.items():
#         warmstart_w_traj_dict[warm_key] = (warm_traj, desired_trajectories)

#     return parallel_mpc_solve_w_trajs(warmstart_w_traj_dict, response_veh_info, world, solver_params, params, ipopt_params, nonresponse_veh_info, cntrl_veh_info)


def parallel_mpc_solve_w_trajs(warmstart_traj_dict: Dict[str, Tuple[Trajectory, PiecewiseDesiredTrajectory]],
                                response_veh_info: VehicleMPCInformation,
                                world: TrafficWorld, solver_params: dict, params: dict,
                                ipopt_params: dict,
                                nonresponse_veh_info: List[VehicleMPCInformation],
                                cntrl_veh_info: List[VehicleMPCInformation]) -> MPCSolverReturn:
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


    # ego_desired_traj = generate_desired_polynomial_coeff(response_vehicle, world)
    # x_coeff_d, y_coeff_d, phi_coeff_d = ego_desired_traj.x_coeff, ego_desired_traj.y_coeff, ego_desired_traj.phi_coeff
    # print("ego coeff x, y, phi", x_coeff_d, y_coeff_d, phi_coeff_d)


    # Generate lane following trajectories for ctrld vehicles 
    x_coeff_d_ctrld = [None for i in range(len(cntrld_vehicles))]
    y_coeff_d_ctrld = [None for i in range(len(cntrld_vehicles))]
    phi_coeff_d_ctrld = [None for i in range(len(cntrld_vehicles))]
    spline_lengths_ctrld = [None for i in range(len(cntrld_vehicles))]
    for i, ctrld_vehicle in enumerate(cntrld_vehicles):
        # cntrld_veh_desired_traj = LaneFollowingPiecewiseTrajectory(ctrld_vehicle, world)
        cntrld_veh_desired_traj = ctrld_vehicle.desired_traj
        x_coeff_d_ctrld[i], y_coeff_d_ctrld[i], phi_coeff_d_ctrld[i], spline_lengths_ctrld[i] = cntrld_veh_desired_traj.to_array()


 
    precompiled_code_dir = params["precompiled_solver_dir"]
    solver_mode = params["solver_mode"]

    list_of_args = []
    for warm_key, (warm_traj, desired_traj) in warmstart_traj_dict.items():
        list_of_args += [(warm_key, warm_traj, desired_traj)]     

    call_mpc_solver_on_warm_and_desired = functools.partial(
        call_mpc_solver,
        list_of_warm_trajs = list_of_args,   
        solver_mode=solver_mode,
        precompiled_code_dir=precompiled_code_dir,
        cntrld_u_warm=cntrld_u_warm,
        cntrld_x_warm=cntrld_x_warm,
        cntrld_xd_warm=cntrld_xd_warm,
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
        x_coeff_d_ctrld = x_coeff_d_ctrld,
        y_coeff_d_ctrld = y_coeff_d_ctrld,
        phi_coeff_d_ctrld = phi_coeff_d_ctrld,
        spline_lengths_ctrld = spline_lengths_ctrld,       
        params=params,
        solver_params=solver_params,
        ipopt_params=ipopt_params)
  
    if params["n_processors"] > 1:        
        with multiprocessing.Pool(processes=params['n_processors']) as pool:
            
            # solve_costs_solutions = pool.imap_unordered(call_mpc_solver_on_warm_and_desired, [i for i in range(len(list_of_args))])

            solve_costs_solutions = pool.map(call_mpc_solver_on_warm_and_desired,
                                                [i for i in range(len(list_of_args))])
            pool.close()

            # below_max_slack_sols = []
            
            
            
            # below_max_slack_counter = 0
            # for sol in tqdm.tqdm(solve_costs_solutions, total=len(list_of_args)):
            #     if sol.max_slack <= params["k_max_slack"]:
            #         below_max_slack_sols.append(sol)
            #     #     if not sol.max_cpu_limit:
            #     #         below_max_slack_counter += 1
            #     #     elif sol.g_violation <= params["k_max_slack"]:
            #     #         below_max_slack_counter += 1
                
            #     # params["number_solutions_for_early_exit"] = 10
            #     # if below_max_slack_counter > params["number_solutions_for_early_exit"]:
            #     #     pool.terminate()
            #     #     print("Received 10 feasible solutions, exiting early")
            #     #     break
                
            pool.terminate()
            # pool.join()
    else:
        solve_costs_solutions = []
        for idx in range(len(warmstart_traj_dict)):
            solve_costs_solutions += [
                call_mpc_solver_on_warm_and_desired(idx)
            ]

    below_max_slack_sols = [
        s for s in solve_costs_solutions if s.max_slack <= params["k_max_slack"]
    ]



    if len(below_max_slack_sols) > 0:
        print("# Feasible Solutions Below Max Slack: %d" %
              len(below_max_slack_sols))

        returned_before_max_cpu_sols = [s for s in below_max_slack_sols if not s.max_cpu_limit]
        n_not_max_cpu = len(returned_before_max_cpu_sols)
        
        reached_max_cpu_sols = [s for s in below_max_slack_sols if s.max_cpu_limit]
        n_max_cpu = len(reached_max_cpu_sols)

        if n_not_max_cpu > 0:
            min_cost_solution = min(returned_before_max_cpu_sols, key=lambda r: r.current_cost)
        else:       
            min_cost_solution = min(reached_max_cpu_sols, key=lambda r: r.current_cost)
        
        below_g_violation = len([s for s in reached_max_cpu_sols if s.g_violation <= 0.0001])
        print("# Max CPU Reached (All) %d   # Max CPU Reached (Feasible) %d   # Fully Solved %d"%(n_max_cpu, below_g_violation, n_not_max_cpu))
                    
    else:
        min_cost_solution = min(solve_costs_solutions, key=lambda r: r.current_cost)
        print(
            "# No Feasible Solutions Below Max Slack. Returning with Slack = %.02f"
            % min_cost_solution.max_slack)

    return min_cost_solution


def call_mpc_solver(
    traj_idx:int,
    list_of_warm_trajs: List[Tuple[str, Trajectory, PiecewiseDesiredTrajectory]],
    precompiled_code_dir: str,
    solver_mode: str,
    cntrld_u_warm: np.array,
    cntrld_x_warm: np.array,
    cntrld_xd_warm: np.array,
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
    x_coeff_d_ctrld: List[List[float]],
    y_coeff_d_ctrld: List[List[float]],
    phi_coeff_d_ctrld: List[List[float]],
    spline_lengths_ctrld, 
    params: dict,
    solver_params: dict,
    ipopt_params: dict,
) -> MPCSolverReturn:
    '''Create the iterative best response object and solve.  Assumes that it receives warm start profiles.
    This really should only require a u_warm, x_warm, x_des_warm and then one level above we generate those values'''
    start_time = time.time()
    
    warm_key, warm_trajectory, desired_traj = list_of_warm_trajs[traj_idx]
    
    k_slack = solver_params["k_slack"]
    k_CA = solver_params["k_CA"]
    k_CA_power = solver_params["k_CA_power"]
    k_ttc = params['k_ttc']
    ttc_threshold = params['ttc_threshold']
    dt = params["dt"]


    x_coeff_d, y_coeff_d, phi_coeff_d, spline_lengths_d = desired_traj.to_array()
    n_coeff_d = desired_traj.polynomial1.n_coeff



    nlp_p = mpcp_to_nlpp(dt, response_x0, p_ego, theta_ego_i, theta_ic, theta_i_nc,
                         cntrld_x0, p_cntrld, nonresponse_x0_list, p_nc,
                         nonresponse_x_list, k_slack, k_CA, k_CA_power, k_ttc, ttc_threshold,
                         x_coeff_d, y_coeff_d, phi_coeff_d, spline_lengths_d, 
                         x_coeff_d_ctrld, y_coeff_d_ctrld, phi_coeff_d_ctrld, spline_lengths_ctrld)
    nlp_x0 = get_warm_start_x0(nc, nnc, warm_trajectory, cntrld_x_warm,
                               cntrld_u_warm, cntrld_xd_warm)
    
    if solver_mode.lower() == "pickled":
        nlp_solver, nlp_lbg, nlp_ubg = get_pickled_solver(
            precompiled_code_dir, params["N"], nc, nnc, params["safety_constraint"])
    elif solver_mode.lower() == "compiled":
        nlp_solver, nlp_lbg, nlp_ubg = get_compiled_solver(
            precompiled_code_dir, params["N"], nc, nnc, params["safety_constraint"])
    else:
        mpc_call_start = time.time()
        # Generate the solver from scratch. This can take ~8s for large
        mpc = MultiMPC(params["N"], nc, nnc, n_coeff_d, params,
                       ipopt_params, safety_constraint=params["safety_constraint"])
        mpc_duration = time.time() - mpc_call_start
        nlp_solver = load_solver_from_mpc(mpc, precompiled=False)
        nlp_lbg, nlp_ubg = get_bounds_from_mpc(mpc)

    # Try to solve the mpc
    try:
        pre_call_tic = time.time()
        solution = nlp_solver(x0=nlp_x0, p=nlp_p, lbg=nlp_lbg, ubg=nlp_ubg)
        solver_call_duration = time.time() - pre_call_tic
        x_ego, u_ego, xd_ego, cntrld_vehicle_trajectories, max_slack, current_cost = get_trajectories_from_solution(
            solution, params["N"], nc, nnc)
        max_g = 0.0
        max_cpu_limit = False
        trajectory = Trajectory(u=u_ego, x=x_ego, xd=xd_ego)

        debug_list = []
        if -0.00001 < current_cost < 0.00001:
            # For now, do not use any solutions that reached max cpu
            print("Cost = 0.000: Max CPU suspected")
            last_solution = mpc.callback.last_solution

            g = np.array(last_solution['g'])
            ubg, lbg = np.array(nlp_ubg), np.array(nlp_lbg)
            ub_gap = np.clip(g - ubg, 0, np.infty)
            lb_gap = np.clip(lbg - g, 0, np.infty)
            gap_magnitude =  (ub_gap.T @ ub_gap + lb_gap.T @ lb_gap)
            slack_cost = 100 * gap_magnitude
            print("Constraint Gap %.05f"%gap_magnitude)
            x_ego, u_ego, xd_ego, cntrld_vehicle_trajectories, max_slack, current_cost = get_trajectories_from_solution(last_solution, params["N"], nc, nnc)
            current_cost += slack_cost #we don't know the cost yet, so we'll make it pretty high
            max_cpu_limit = True
            max_g = gap_magnitude
        print("MPC Duration %.03f Solver Call %.03f   Total Call %.03f"%(mpc_duration, solver_call_duration, time.time()-start_time))
        return MPCSolverReturn(True, current_cost, max_slack, trajectory, warm_key, desired_traj, debug_list, cntrld_vehicle_trajectories, max_cpu_limit, max_g)
    except Exception as e:
        print(e)
        return MPCSolverReturnException()

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


