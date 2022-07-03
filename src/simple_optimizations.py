import numpy as np
import casadi as cas
import copy as cp
from src.multiagent_mpc import NonconvexOptimization
from src.best_response import parallel_mpc_solve_w_trajs, generate_solver_params
from src.vehicle_mpc_information import Trajectory, VehicleMPCInformation
from src.traffic_world import TrafficWorld
from typing import List
from src.geometry_helper import minkowski_ellipse_collision_distance


def feasible_guess(N, vehicle, x0, params, world,
                   other_vehicle_info: List[VehicleMPCInformation]):
    ''' Try to get a feasible solution for a single vehicle without 
        considering the actual cost.

        We warm-start the ego vehicle with a solution for a state-only
        optimization without dynamics.  
    '''

    # We need to deepcopy and del vehicles to allow multiprocesssing (see Note elsewhere)
    cp_vehicle = cp.deepcopy(vehicle)
    cp_other_vehicle_info = cp.deepcopy(other_vehicle_info)
    
    for j, vinfo in enumerate(cp_other_vehicle_info):
        if vinfo.u is None:
            u_j = np.zeros((2, N))
            x0_j = vinfo.x0.reshape(6, 1)
            x_j, xd_j = vinfo.vehicle.forward_simulate_all(x0_j, u_j)

            vinfo.update_state(u_j, x_j, xd_j)
            cp_other_vehicle_info[j] = vinfo


    cp_params = cp.deepcopy(params)
    cp_params["N"] = N
    # turnoff any pre-compiled since we're changing N
    cp_params["solver_mode"] = "regenerate"
    solver_params = generate_solver_params(params, 0, 0)

    ipopt_params = {'print_level': 5, 'max_cpu_time': 20}


    # warm start with no control inputs
    u_warm_intial = np.zeros((2, N))
    x_warm_initial, x_des_warm_initial = cp_vehicle.forward_simulate_all(
        x0.reshape(6, 1), u_warm_intial)

    # solve for a spatially feasible x that we will use as a warm start
    x_other = [v.x for v in cp_other_vehicle_info]
    cp_other_vehicles = [cp.deepcopy(vi.vehicle) for vi in cp_other_vehicle_info]
    x_warm, _ = spatial_only_optimization(x_warm_initial, [], x_other,
                                          cp_vehicle, None, cp_other_vehicles,
                                          3 * cp_vehicle.L)
    print("Spatial Only Optimization Solution", x_warm)
    # warm start with feasible x (not dynamically feasible)
    cp_vehicle.update_default_desired_lane_traj(world, x0)
    warm_traj_dict = {
        'mix spatial none':
        (Trajectory(u=u_warm_intial, x=x_warm, xd=x_des_warm_initial), cp_vehicle.desired_traj)
    }

    response_veh_info = VehicleMPCInformation(cp_vehicle, x0)
    print("Solving Vehicle %d"% response_veh_info.vehicle.agent_id)

    

    sol = parallel_mpc_solve_w_trajs(
        warm_traj_dict, response_veh_info, world, solver_params, cp_params,
        ipopt_params, cp_other_vehicle_info, [])

    del cp_vehicle, response_veh_info  # needed to fix: TypeError: cannot pickle 'SwigPyObject' object
    del cp_other_vehicle_info, cp_other_vehicles

    if sol.max_slack < np.infty:
        return sol.trajectory.u, sol.trajectory.x, sol.trajectory.xd
    else:
        print(
            "Warning...Bad prediction of remaining MPC trajectory before IBR")
        return u_warm_intial, x_warm_initial, x_des_warm_initial


# spatial free space
def spatial_only_optimization(x_initial: np.array,
                              cntrld_x_initial: List[np.array],
                              non_response_x: List[np.array], response_veh,
                              cntrld_veh, non_response_veh,
                              distance_threshold):
    ''' Solve a geometry only optimization to warm start where feasible X are required. 
        x_initial: 6xN trajectory
        cntrld_x_initial: List[6xN] trajectory
    
        We ingore vehicle dynamics
    '''

    opt = NonconvexOptimization()
    # Decision Variables
    x = cas.MX.sym('x', x_initial.shape[0], x_initial.shape[1])
    cntrld_x = [
        cas.MX.sym('x_%02d' % j, cntrld_x_initial[j].shape[0],
                   cntrld_x_initial[j].shape[1])
        for j in range(len(cntrld_x_initial))
    ]
    x_list = [cas.reshape(x, x.shape[0] * x.shape[1], 1)] + [
        cas.reshape(cntrld_x[j], cntrld_x[j].shape[0] * cntrld_x[j].shape[1],
                    1) for j in range(len(cntrld_x_initial))
    ]
    opt._x_list = cas.vcat(x_list)

    # Collision Avoidance
    if len(non_response_x)>0:
        initial_close_vehs = vehicles_close([x_initial] + cntrld_x_initial,
                                            non_response_x, distance_threshold)
    else:
        initial_close_vehs = []

    # Optimize for minimum deviations from initial trajectories
    cost = cas.sumsqr(x - x_initial)
    for j in range(len(cntrld_x_initial)):
        cost += cas.sumsqr(cntrld_x_initial[j] - cntrld_x[j])
    opt._f = cost

    epsilon = 0.001
    for k in range(x_initial.shape[1]):
        for j in range(len(cntrld_x_initial)):

            dist = minkowski_ellipse_collision_distance(
                response_veh, cntrld_veh[j], x[0, k], x[1, k], x[2, k],
                cntrld_x[j][0, k], cntrld_x[j][1, k], cntrld_x[j][2, k])
            opt.add_bounded_constraint(1 + epsilon, dist, None)

        for j in range(len(non_response_x)):
            if initial_close_vehs[j]:
                dist = minkowski_ellipse_collision_distance(
                    response_veh, non_response_veh[j], x[0, k], x[1, k],
                    x[2, k], non_response_x[j][0, k], non_response_x[j][1, k],
                    non_response_x[j][2, k])

                opt.add_bounded_constraint(1 + epsilon, dist, None)

                for jc in range(len(cntrld_x_initial)):
                    dist = minkowski_ellipse_collision_distance(
                        cntrld_veh[jc], response_veh[j],
                        non_response_x[j][0, k], non_response_x[j][1, k],
                        non_response_x[j][2, k], cntrld_x[jc][0, k],
                        cntrld_x[jc][1, k], cntrld_x[jc][2, k])
                    opt.add_bounded_constraint(1 + epsilon, dist, None)
    opt._p_list = None
    solver, _ = opt.get_nlpsol()

    x_guess_list = [
        cas.reshape(x_initial, x_initial.shape[0] * x_initial.shape[1], 1)
    ] + [
        cas.reshape(
            cntrld_x_initial[j],
            cntrld_x_initial[j].shape[0] * cntrld_x_initial[j].shape[1], 1)
        for j in range(len(cntrld_x_initial))
    ]
    x_guess = cas.vcat(x_guess_list)

    solution = solver(x0=x_guess, lbg=opt._lbg_list, ubg=opt._ubg_list)

    # Convert the nlp_x to individual x_new and xc_new
    idx = 0
    x_new = cas.reshape(
        solution['x'][idx:x_initial.shape[0] * x_initial.shape[1]],
        x_initial.shape[0], x_initial.shape[1])
    idx += x_initial.shape[0] * x_initial.shape[1]
    xc_new = []
    for j in range(len(cntrld_x_initial)):
        x_temp = solution['x'][idx:cntrld_x_initial[j].shape[0] *
                               cntrld_x_initial[j].shape[1]]
        xc_new += [
            cas.reshape(x_temp, cntrld_x_initial[j].shape[0],
                        cntrld_x_initial[j].shape[1])
        ]
        idx += cntrld_x_initial[j].shape[0] * cntrld_x_initial[j].shape[1]

    del opt
    return x_new, xc_new


def lane_following_optimizations(N: int, vehicle, x0: np.array, params: dict,
                                 world: TrafficWorld):
    cp_vehicle = cp.deepcopy(vehicle)
    cp_params = cp.deepcopy(params)
    cp_params["N"] = N

    # Try not to change the velocity of the car (i.e. assume velocity zero)
    cp_vehicle.k_u_v = 1000
    cp_vehicle.strict_wall_constraint = False

    solver_params = generate_solver_params(params, 0, 0)
    ipopt_params = {'print_level': 5, 'max_cpu_time': 20}

    # warm start with no control inputs
    u_warm = np.zeros((2, N))
    x_warm, x_des_warm = cp_vehicle.forward_simulate_all(
        x0.reshape(6, 1), u_warm)

    response_veh_info = VehicleMPCInformation(cp_vehicle, x0)
    warm_traj = Trajectory(u=u_warm, x=x_warm, xd=x_des_warm)

    desired_traj = vehicle.piecewise_desired
    warm_start_dict = {"test": (warm_traj, desired_traj)}
    sol = parallel_mpc_solve_w_trajs(
        warm_start_dict, response_veh_info, world, solver_params, params,
        ipopt_params, [], [])

    del response_veh_info, cp_vehicle

    if sol.max_slack < np.infty:
        return sol.trajectory.u, sol.trajectory.x, sol.trajectory.xd
    else:
        return u_warm, x_warm, x_des_warm


def vehicles_close(planning_vehicles_x: List[np.array],
                   other_vehicles_x: List[np.array], dist_threshold):
    ''' returns array [nvehicles x 1] which is true if the other vehicle is every close to one of the planning vehicles'''
    planning_vehicles_x = np.stack(planning_vehicles_x,
                                   axis=0)  #np.array everything
    other_vehicles_x = np.stack(other_vehicles_x, axis=0)  #np.array everything
    vehs_close = np.array([False for j in range(other_vehicles_x.shape[0])])

    for x_initial in planning_vehicles_x:
        delta_xy = (x_initial - other_vehicles_x)[:, :2, :]
        dist_sqrd = np.sqrt(np.sum(delta_xy * delta_xy, axis=1))
        close = (dist_sqrd <= dist_threshold)
        vehs_close_i = np.any(close, axis=1)  #update
        vehs_close = np.logical_or(vehs_close, vehs_close_i)

    return vehs_close
