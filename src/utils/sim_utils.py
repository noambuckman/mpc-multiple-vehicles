import numpy as np
from src.vehicle import Vehicle
from typing import List
import copy as cp
from src.multiagent_mpc import load_state, save_state
from shapely.geometry import box
from shapely.affinity import rotate


def generate_test_scenario(n_other, N, dt, n_lanes=2, car_density=5000):
    from src.traffic_world import TrafficWorld
    from src.utils.solver_helper import poission_positions, initialize_cars_from_positions
    import numpy as np

    # Create the world and vehicle objects
    world = TrafficWorld(n_lanes, 0, 999999)

    # Create the vehicle placement based on a Poisson distribution
    MAX_VELOCITY = 25 * 0.447  # m/s
    VEHICLE_LENGTH = 4.5  # m
    time_duration_s = (n_other * 3600.0 / car_density) * \
        10  # amount of time to generate traffic
    initial_vehicle_positions = poission_positions(car_density, n_other + 1,
                                                   n_lanes, MAX_VELOCITY,
                                                   2 * VEHICLE_LENGTH)
    position_list = initial_vehicle_positions[:n_other + 1]

    # Create the SVOs for each vehicle
    list_of_svo = [
        np.random.choice([0, np.pi / 4.0, np.pi / 2.01])
        for i in range(n_other)
    ]

    (_, _, all_vehicles,
     all_x0) = initialize_cars_from_positions(N, dt, world, True,
                                              position_list, list_of_svo)

    for vehicle in all_vehicles:
        # Set theta_ij randomly for all vehicles
        vehicle.theta_ij[-1] = np.random.uniform(0.001, np.pi / 2.01)
        for vehicle_j in all_vehicles:
            vehicle.theta_ij[vehicle_j.agent_id] = np.random.uniform(
                0.001, np.pi / 2.01)

    return all_vehicles, all_x0


def generate_test_mpc_scenario(n_cntrld,
                               n_non_response,
                               N,
                               dt,
                               n_lanes=2,
                               car_density=5000):
    ''' Generate the vehicle data and initial conditions for split of cntrld and non controlled vehicles'''

    n_other = n_cntrld + n_non_response + 1
    all_vehicles, all_x0 = generate_test_scenario(n_other, N, dt, n_lanes,
                                                  car_density)

    response_idx = 0
    cntrld_idx = [0 + j + 1 for j in range(n_cntrld)]
    nonresponse_idx = [n_cntrld + 1 + j for j in range(n_non_response)]
    n_cntrld = len(cntrld_idx)
    n_non_response = len(nonresponse_idx)

    x_initial = all_x0[response_idx]
    cntrld_x_initial = [all_x0[idx] for idx in cntrld_idx]
    non_response_x = [all_x0[idx] for idx in nonresponse_idx]

    response_veh = all_vehicles[response_idx]
    cntrld_veh = [all_vehicles[idx] for idx in cntrld_idx]
    non_response_veh = [all_vehicles[idx] for idx in nonresponse_idx]

    return x_initial, cntrld_x_initial, non_response_x, response_veh, cntrld_veh, non_response_veh


def convert_to_global_units(x_reference_global: np.array, x: np.array):
    """ During solving, the x-dimension is normalized to the ambulance initial position x_reference_global """
    if x_reference_global is None:
        return x

    x_global = cp.deepcopy(x)
    x_global[0, :] += x_reference_global[0]

    return x_global


def load_log_data(params, log_dir, i_mpc_start, xothers_actual,
                  uothers_actual):
    previous_mpc_file = log_dir + "data/mpc_%03d" % (i_mpc_start - 1)
    print("Loaded initial positions from %s" % (previous_mpc_file))
    _, _, _, all_other_x_executed, all_other_u_executed, _ = load_state(
        previous_mpc_file, params["n_other"])
    all_other_u_mpc = all_other_u_executed

    previous_all_file = log_dir + "data/all_%03d" % (i_mpc_start - 1)
    _, _, _, xothers_actual_prev, uothers_actual_prev, _, = load_state(
        previous_all_file, params["n_other"], ignore_des=True)
    t_end = xothers_actual_prev[0].shape[1]

    for i in range(len(xothers_actual_prev)):
        xothers_actual[i][:, :t_end] = xothers_actual_prev[i][:, :t_end]
        uothers_actual[i][:, :t_end] = uothers_actual_prev[i][:, :t_end]
    actual_t = i_mpc_start * params["number_ctrl_pts_executed"]

    return all_other_x_executed, all_other_u_mpc, xothers_actual, uothers_actual, actual_t


def check_collisions(all_other_vehicles: List[Vehicle],
                     all_other_x_executed: List[np.array]) -> bool:
    car_collisions = get_collision_pairs(all_other_vehicles,
                                         all_other_x_executed)
    if len(car_collisions) > 0:
        return True
    else:
        return False


def check_collision(vehicle_a: Vehicle, vehicle_b: Vehicle, X_a: np.array,
                    X_b: np.array) -> bool:
    for t in range(X_a.shape[1]):
        x_a, y_a, theta_a = X_a[0, t], X_a[1, t], X_a[2, t]
        x_b, y_b, theta_b = X_b[0, t], X_b[1, t], X_b[2, t]
        box_a = box(x_a - vehicle_a.L / 2.0, y_a - vehicle_a.W / 2.0,
                    x_a + vehicle_a.L / 2.0, y_a + vehicle_a.W / 2.0)
        rotated_box_a = rotate(box_a,
                               theta_a,
                               origin='center',
                               use_radians=True)

        box_b = box(x_b - vehicle_b.L / 2.0, y_b - vehicle_b.W / 2.0,
                    x_b + vehicle_b.L / 2.0, y_b + vehicle_b.W / 2.0)
        rotated_box_b = rotate(box_b,
                               theta_b,
                               origin='center',
                               use_radians=True)

        if rotated_box_a.intersects(rotated_box_b):
            return True

    return False


def vehicles_within_range(other_x0, amb_x0, distance_from_ambulance):

    veh_idxs = [
        i for i in range(len(other_x0))
        if abs(other_x0[i][0] - amb_x0[0]) <= distance_from_ambulance
    ]

    return veh_idxs


def update_sim_states(othervehs_ibr_info, all_other_x_ibr_g, params, actual_t,
                      log_dir, i_mpc, xothers_actual, uothers_actual):
    ''' Update the simulation states with the solutions from Iterative Best Response / MPC round 
        We update multiple sim steps at a time (params["number_cntrl_pts_executed"])    
    '''

    all_other_u_mpc = [cp.deepcopy(veh.u) for veh in othervehs_ibr_info]
    all_other_x_executed = [
        all_other_x_ibr_g[i][:, :params["number_ctrl_pts_executed"] + 1]
        for i in range(params["n_other"])
    ]
    all_other_u_executed = [
        othervehs_ibr_info[i].u[:, :params["number_ctrl_pts_executed"]]
        for i in range(params["n_other"])
    ]

    # Append to the executed trajectories history of x
    for i in range(params["n_other"]):
        xothers_actual[i][:, actual_t:actual_t +
                          params["number_ctrl_pts_executed"] +
                          1] = all_other_x_executed[i]
        uothers_actual[
            i][:, actual_t:actual_t +
               params["number_ctrl_pts_executed"]] = all_other_u_executed[i]

    # SAVE STATES AND PLOT
    file_name = log_dir + "data/" + "mpc_%03d" % (i_mpc)
    print("Saving MPC Rd %03d / %03d to ... %s" %
          (i_mpc, params["n_mpc"] - 1, file_name))

    if params["save_state"]:
        other_u_ibr_temp = [veh.u for veh in othervehs_ibr_info]
        other_xd_ibr_temp = [veh.xd for veh in othervehs_ibr_info]

        save_state(file_name, None, None, None, all_other_x_ibr_g,
                   other_u_ibr_temp, other_xd_ibr_temp)
        save_state(log_dir + "data/" + "all_%03d" % (i_mpc),
                   None,
                   None,
                   None,
                   xothers_actual,
                   uothers_actual,
                   None,
                   end_t=actual_t + params["number_ctrl_pts_executed"] + 1)

    actual_t += params["number_ctrl_pts_executed"]

    return actual_t, all_other_u_mpc, all_other_x_executed, all_other_u_executed, xothers_actual, uothers_actual


def save_ibr(log_dir, i_mpc, i_rounds_ibr, response_i,
             vehs_ibr_info_predicted):
    ''' Save the trajectories from each round of IBR'''
    x_ibr = np.stack([veh.x for veh in vehs_ibr_info_predicted], axis=0)
    u_ibr = np.stack([veh.u for veh in vehs_ibr_info_predicted], axis=0)
    x_d = np.stack([veh.xd for veh in vehs_ibr_info_predicted], axis=0)
    if response_i is None:
        file_prefix = log_dir + "data/" + \
            "ibr_m%03di%03d" % (i_mpc, i_rounds_ibr)
    else:
        file_prefix = log_dir + "data/" + \
            "ibr_m%03di%03da%03d" % (i_mpc, i_rounds_ibr, response_i)
    np.save(open(file_prefix + "x.npy", 'wb'), x_ibr)
    np.save(open(file_prefix + "u.npy", 'wb'), u_ibr)
    np.save(open(file_prefix + "xd.npy", 'wb'), x_d)


def get_best_responding_vehicles_idxs(vehicle_list):
    ''' For now just return all the vehicles'''
    return list(range(len(vehicle_list)))


def get_within_range_other_vehicle_idxs(response_i, allvehs_ibr_info):
    ''' Only grab vehicles that are within 20 vehicle lengths in the X direction'''
    response_veh_info = allvehs_ibr_info[response_i]

    within_range_idxs = [
        j for j in range(len(allvehs_ibr_info))
        if j != response_i and (-20 * response_veh_info.vehicle.L <= (
            allvehs_ibr_info[j].x0[0] - response_veh_info.x0[0]) <= 20 *
                                response_veh_info.vehicle.L)
    ]

    return within_range_idxs


def assign_shared_control(params, i_rounds_ibr, veh_idxs_in_mpc,
                          vehicles_index_best_responders, response_veh_info,
                          vehs_ibr_info_predicted):
    ''' Divide up vehicles in MPC between shared control and not-shared control / non response'''

    # Determine the number of vehicles in shared control
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
        delta_x = [
            response_veh_info.x0[0] - otherveh_info.x0[0]
            for otherveh_info in vehs_ibr_info_predicted
        ]
        # This necessary but could limit fringe best response interactions with outside best response
        sorted_i = [
            i for i in np.argsort(delta_x)
            if (delta_x[i] > 0 and i in veh_idxs_in_mpc
                and i in vehicles_index_best_responders)
        ]
        cntrld_i = sorted_i[:n_cntrld]

        if len(cntrld_i) > 0:
            cntrld_vehicle_info = [
                vehs_ibr_info_predicted[i]
                for i in range(len(vehs_ibr_info_predicted)) if i in cntrld_i
            ]
        else:
            cntrld_i = []
            cntrld_vehicle_info = []

        veh_idxs_in_mpc = [
            idx for idx in veh_idxs_in_mpc if idx not in cntrld_i
        ]
        nonresponse_veh_info = [
            vehs_ibr_info_predicted[i] for i in veh_idxs_in_mpc
        ]
    else:
        cntrld_i = []
        cntrld_vehicle_info = []
        nonresponse_veh_info = [
            vehs_ibr_info_predicted[i] for i in veh_idxs_in_mpc
        ]

    return cntrld_vehicle_info, nonresponse_veh_info, cntrld_i


def get_collision_pairs(all_other_vehicles: List[Vehicle],
                        all_other_x_executed: List[np.array]) -> List[tuple]:
    car_collisions = []
    for i in range(len(all_other_vehicles)):
        for j in range(len(all_other_vehicles)):
            if i >= j:
                continue
            else:
                collision = check_collision(all_other_vehicles[i],
                                            all_other_vehicles[j],
                                            all_other_x_executed[i],
                                            all_other_x_executed[j])
                if collision:
                    car_collisions += [(i, j)]
    return car_collisions
