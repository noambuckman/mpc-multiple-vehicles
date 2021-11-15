import numpy as np
import casadi as cas
import copy as cp
from src.best_response import solve_best_response

from typing import List
from src.geometry_helper import minkowski_ellipse_collision_distance


def feasible_guess(N, vehicle, x0, params, world, other_vehicle_info):
    ''' Solve for the ego vehicle. '''

    # We need to deepcopy and del vehicles to allow multiprocesssing (see Note elsewhere)
    cp_vehicle = cp.deepcopy(vehicle)
    other_vehicles = [cp.deepcopy(vi.vehicle) for vi in other_vehicle_info]

    cp_params = cp.deepcopy(params)
    cp_params["N"] = N

    # Try not to change the velocity of the car (i.e. assume velocity zero)
    # cp_vehicle.k_u_v = 1000
    # cp_vehicle.strict_wall_constraint = True

    # TODO:  Allow for default values for this. Distinguish between solver, params, and ipopt params
    solver_params = {}
    solver_params["slack"] = True
    solver_params["k_CA"] = params["k_CA_d"]
    solver_params["k_CA_power"] = params["k_CA_power"]
    solver_params["k_slack"] = params["k_slack_d"]

    ipopt_params = {'print_level': 0}
    x_other = [v.x for v in other_vehicle_info]
    xd_other = [v.xd for v in other_vehicle_info]
    u_other = [v.u for v in other_vehicle_info]
    x0_other_vehicles = [v.x0 for v in other_vehicle_info]

    for j in range(len(other_vehicle_info)):

        if other_vehicle_info[j].u is None:
            u_other[j] = np.zeros((2, N))
            x_other[j], xd_other[j] = other_vehicles[j].forward_simulate_all(other_vehicle_info[j].x0.reshape(6, 1),
                                                                             u_other[j])

    # warm start with no control inputs
    u_warm_intial = np.zeros((2, N))
    x_warm_initial, x_des_warm_initial = cp_vehicle.forward_simulate_all(x0.reshape(6, 1), u_warm_intial)

    # solve for a spatially feasible x that we will use as a warm start
    x_warm, _ = spatial_only_optimization(x_warm_initial, [], x_other, cp_vehicle, None, other_vehicles,
                                          3 * cp_vehicle.L)

    # warm start with feasible x (not dynamically feasible)
    warm_traj = u_warm_intial, x_warm, x_des_warm_initial
    # max_slack = np.infty
    _, _, max_slack, x, x_des, u, _, _, _ = solve_best_response("mix spatial none", warm_traj, cp_vehicle, [],
                                                                other_vehicles, x0, [], x0_other_vehicles, world,
                                                                solver_params, cp_params, ipopt_params, u_other,
                                                                x_other, xd_other)

    del cp_vehicle  # needed to fix: TypeError: cannot pickle 'SwigPyObject' object
    del other_vehicles

    if max_slack < np.infty:
        return u, x, x_des
    else:
        print("Warning...Bad prediction of remaining MPC trajectory before IBR")
        return u_warm_intial, x_warm_initial, x_des_warm_initial


# spatial free space
def spatial_only_optimization(x_initial: np.array, cntrld_x_initial: List[np.array], non_response_x: List[np.array],
                              response_veh, cntrld_veh, non_response_veh, distance_threshold):
    ''' Solve a geometry only optimization to warm start where feasible X are required. We ingore vehicle dynamics'''

    opt = cas.Opti()
    # Decision Variables
    x = opt.variable(x_initial.shape[0], x_initial.shape[1])
    cntrld_x = [
        opt.variable(cntrld_x_initial[j].shape[0], cntrld_x_initial[j].shape[1]) for j in range(len(cntrld_x_initial))
    ]

    # Collision Avoidance
    initial_close_vehs = vehicles_close([x_initial] + cntrld_x_initial, non_response_x, distance_threshold)

    # Optimize for minimum deviations from initial trajectories
    cost = cas.sumsqr(x - x_initial)
    for j in range(len(cntrld_x_initial)):
        cost += cas.sumsqr(cntrld_x_initial[j] - cntrld_x[j])
    opt.minimize(cost)

    epsilon = 0.001
    for k in range(x_initial.shape[1]):
        for j in range(len(cntrld_x_initial)):

            dist = minkowski_ellipse_collision_distance(response_veh, cntrld_veh[j], x[0, k], x[1, k], x[2, k],
                                                        cntrld_x[j][0, k], cntrld_x[j][1, k], cntrld_x[j][2, k])
            opt.subject_to(dist >= 1 + epsilon)

        for j in range(len(non_response_x)):
            if initial_close_vehs[j]:
                dist = minkowski_ellipse_collision_distance(response_veh, non_response_veh[j], x[0, k], x[1, k],
                                                            x[2, k], non_response_x[j][0, k], non_response_x[j][1, k],
                                                            non_response_x[j][2, k])

                opt.subject_to(dist >= 1 + epsilon)

                for jc in range(len(cntrld_x_initial)):
                    dist = minkowski_ellipse_collision_distance(cntrld_veh[jc], response_veh[j], non_response_x[j][0,
                                                                                                                   k],
                                                                non_response_x[j][1, k], non_response_x[j][2, k],
                                                                cntrld_x[jc][0, k], cntrld_x[jc][1, k], cntrld_x[jc][2,
                                                                                                                     k])
                    opt.subject_to(dist >= 1 + epsilon)

    opt.solver("ipopt", {}, {'print_level': 0})
    opt.solve()

    x_new = opt.value(x)
    xc_new = [opt.value(x) for x in cntrld_x]

    del opt
    return x_new, xc_new


def vehicles_close(planning_vehicles_x: List[np.array], other_vehicles_x: List[np.array], dist_threshold):
    ''' returns array [nvehicles x 1] which is true if the other vehicle is every close to one of the planning vehicles'''
    planning_vehicles_x = np.array(planning_vehicles_x)  #np.array everything
    other_vehicles_x = np.array(other_vehicles_x)  #np.array everything
    vehs_close = np.array([False for j in range(other_vehicles_x.shape[0])])

    for x_initial in planning_vehicles_x:
        delta_xy = (x_initial - other_vehicles_x)[:, :2, :]
        dist_sqrd = np.sqrt(np.sum(delta_xy * delta_xy, axis=1))
        close = (dist_sqrd <= dist_threshold)
        vehs_close_i = np.any(close, axis=1)  #update
        vehs_close = np.logical_or(vehs_close, vehs_close_i)

    return vehs_close


# if __name__ == "__main__":
#     n_vehs = 5
#     N = 25
#     other_x0 = [np.random.uniform(size=(6, 1)) for _ in range(n_vehs)]
#     other_vehicles = [vehicle.Vehicle(0.2) for _ in range(n_vehs)]
#     world = TrafficWorld(2, 0)

#     for i in range(len(other_vehicles)):
#         other_vehicles[i].update_desired_lane(world, 0, right_direction=True)

#     parser = IBRParser()
#     args = parser.parse_args()
#     params = vars(args)

#     prev_u_mpc = [np.random.uniform(size=(2, N)) for _ in range(n_vehs)]
#     new_u_mpc = extend_last_mpc_and_follow(prev_u_mpc, 12, other_vehicles, other_x0, params, world)