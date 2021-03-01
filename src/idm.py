import numpy as np
from src.vehicle import Vehicle
from src.traffic_world import TrafficWorld
from typing import List


def IDM_acceleration(bumper_distance, lead_vehicle_velocity, current_speed, desired_speed, idm_params=None):
    ''' predict the trajectory of a vehicle based on Intelligent Driver Model 
    based on: Kesting, A., Treiber, M., & Helbing, D. (2010). Enhanced intelligent driver model to access the impact of driving strategies 
    on traffic capacity. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences

    v_0:  desired speed
    delta:  free acceleration exponent
    T: desired time gap
    S_0:  jam distance
    '''
    default_idm_params = {
        "free_acceleration_exponent": 4,
        "desired_time_gap": 2.0,
        "jam_distance": 2.0,
        "maximum_acceleration": 1.4,
        "desired_deceleration": 2.0,
        "coolness_factor": 0.99,
    }
    if idm_params is not None:
        for param in idm_params:
            try:
                default_idm_params[param] = idm_params[param]
            except KeyError:
                raise Exception("Invalid IDM Param: check if param key correct")

    # set variable to match the paper for ease of reading
    v_0 = desired_speed
    v = current_speed
    s = bumper_distance
    delta = default_idm_params["free_acceleration_exponent"]
    T = default_idm_params["desired_time_gap"]
    s_0 = default_idm_params["jam_distance"]
    a = default_idm_params["maximum_acceleration"]
    b = default_idm_params["desired_deceleration"]
    c = default_idm_params["coolness_factor"]
    delta_v = v - lead_vehicle_velocity

    def s_star(v, delta_v, s_0=s_0, a=a, b=b, T=T):
        '''  Deceleration strategy [eq 2.2] 
        '''
        deceleration = s_0 + v * T + v * delta_v / (s * np.sqrt(a * b))

        return deceleration

    # print("decel", -(a * s_star(v, delta_v) / s)**2)
    # print("free accel", -a * (v / v_0)**delta)
    # print("accel", a)
    a_IDM = a * (1 - (v / v_0)**delta - (s_star(v, delta_v) / s)**2)  #[eq 2.1]
    # print("IDM accel", a_IDM)
    return a_IDM


def IDM_trajectory_prediction(veh, N, X_0, X_lead=None, desired_speed=None, idm_params=None):
    ''' Compute an IDM trajectory for a vehicle based on a speed following vehicle
    '''
    if X_lead is None:
        X_lead = X_0 + 99999

    if desired_speed is None:
        desired_speed = veh.max_v

    # idm_params = {}
    idm_params["maximum_acceleration"] = veh.max_acceleration / veh.dt  # correct for previous multiple in dt

    # for t in range(N):
    bumper_distance = X_lead[0] - X_0[0] - veh.L
    current_speed = X_0[4] * np.cos(X_0[2])
    lead_vehicle_velocity = X_lead[4] * np.cos(X_lead[2])

    # print("Dist", bumper_distance, "Lead V", lead_vehicle_velocity)
    # print("Current Speed", current_speed, "Desired Speed", desired_speed, idm_params)
    a_IDM = IDM_acceleration(bumper_distance, lead_vehicle_velocity, current_speed, desired_speed, idm_params)

    U_ego = np.zeros((2, 1))
    U_ego[0, 0] = 0  # assume no steering
    U_ego[1, 0] = a_IDM
    # print("a_idm", a_IDM)
    x, x_des = veh.forward_simulate_all(X_0, U_ego)
    # print("x", x)
    # print("x_des", x_des)
    return U_ego, x, x_des


def MOBIL_lanechange(driver_x0: np.array,
                     driver_veh: Vehicle,
                     all_other_x0: List[np.array],
                     all_other_veh: List[Vehicle],
                     world: TrafficWorld,
                     MOBIL_params: dict = None,
                     IDM_params: dict = None):
    ''' MOBIL lane changing rules '''

    default_MOBIL_params = {
        "politeness_factor": 0.5,
        "changing_threshold": 0.1,
        "maximum_safe_deceleration": 4,
        "bias_for_right_lane": 0.3
    }
    if MOBIL_params:
        for param in MOBIL_params:
            try:
                default_MOBIL_params[param] = MOBIL_params[param]
            except KeyError:
                raise Exception("Key Error:  Check if MOBIL Param is correct")

    p = default_MOBIL_params["politeness_factor"]
    a_thr = default_MOBIL_params["changing_threshold"]
    b_safe = default_MOBIL_params["maximum_safe_deceleration"]
    a_bias = default_MOBIL_params["bias_for_right_lane"]

    driver_old_lane = world.get_lane_from_x0(driver_x0)
    driver_new_lanes = [li for li in range(world.n_lanes) if li != driver_old_lane]

    best_new_lane = None
    for new_lane in driver_new_lanes:

        new_follower_idx = get_prev_vehicle_lane(driver_x0, all_other_x0, new_lane, world)
        new_leader_idx = get_next_vehicle_lane(driver_x0, all_other_x0, new_lane, world)

        old_follower_idx = get_prev_vehicle_lane(driver_x0, all_other_x0, driver_old_lane, world)
        old_leader_idx = get_next_vehicle_lane(driver_x0, all_other_x0, driver_old_lane, world)

        # Calculate the new acceleration of the new follower

        follower_lead_pairs = {
            "newfollower_after": (new_follower_idx, -1),
            "newfollower_before": (new_follower_idx, new_leader_idx),
            "oldfollower_before": (old_follower_idx, -1),
            "oldfollower_after": (old_follower_idx, old_leader_idx),
            "driver_after": (-1, new_leader_idx),
            "driver_before": (-1, old_leader_idx)
        }

        accel = {}
        lane_gap = True
        for key, (follower_idx, lead_idx) in follower_lead_pairs.items():
            if follower_idx is None:
                # no follower, just set acceleration = 0 for before & after
                accel[key] = 0
                continue
            elif follower_idx == -1:
                follower_x0 = driver_x0
                follower_veh = driver_veh
            else:
                follower_x0 = all_other_x0[follower_idx]
                follower_veh = all_other_veh[follower_idx]

            current_speed = follower_x0[4] * np.cos(follower_x0[2])
            desired_speed = follower_veh.max_v

            if lead_idx is None:  # no cars ahead
                bumper_distance = 999999 - follower_x0[0] - follower_veh.L
                lead_velocity = 999999
            elif lead_idx == -1:  # lead vehicle is the driver vehicle
                bumper_distance = driver_x0[0] - follower_x0[0] - follower_veh.L
                lead_velocity = driver_x0[4] * np.cos(driver_x0[2])
            else:
                bumper_distance = all_other_x0[lead_idx][0] - follower_x0[0] - follower_veh.L
                lead_velocity = all_other_x0[lead_idx][4] * np.cos(all_other_x0[lead_idx][2])
            if lead_idx == -1:
                print(key, follower_idx, lead_idx)
                print(bumper_distance)
            if bumper_distance < 0:
                lane_gap = False  #checks if there is even a gap between lead vehicle in next lane
            a_follower = IDM_acceleration(bumper_distance, lead_velocity, current_speed, desired_speed, IDM_params)
            accel[key] = a_follower

        safety_criteria = accel["newfollower_after"] >= -b_safe

        driver_incentive = accel["driver_after"] - accel["driver_before"]
        new_follower_incentive = accel["newfollower_after"] - accel["newfollower_before"]
        old_follower_incentive = accel["oldfollower_after"] - accel["oldfollower_before"]

        incentive_criteria = (driver_incentive + p * (new_follower_incentive + old_follower_incentive)) >= (a_thr)

        if incentive_criteria and safety_criteria and lane_gap:
            best_new_lane = new_lane

    return best_new_lane, accel


def get_lead_vehicle(x0, other_x0, world):
    n_other = len(other_x0)
    ego_lane = world.get_lane_from_x0(x0)

    veh_same_lane = [world.get_lane_from_x0(other_x0[i]) == ego_lane for i in range(n_other)]
    veh_in_front = [(other_x0[i][0] - x0[0]) > 0 for i in range(n_other)]

    vehicles_sorted = np.argsort([(other_x0[i][0] - x0[0])**2 for i in range(n_other)])
    vehicles_sorted_valid = [idx for idx in vehicles_sorted if veh_same_lane[idx] and veh_in_front[idx]]

    if len(vehicles_sorted_valid) > 0:
        return vehicles_sorted_valid[0]
    else:
        return None


def get_next_vehicle_lane(x0, all_other_x0, lane, world):
    ''' get next vehicle in a lane '''

    idx_in_lane = [world.get_lane_from_x0(all_other_x0[idx]) == lane for idx in range(len(all_other_x0))]
    idx_forward = [all_other_x0[idx][0] - x0[0] > 0 for idx in range(len(all_other_x0))]
    idx_dist_sorted = np.argsort([np.abs(all_other_x0[idx][0] - x0[0]) for idx in range(len(all_other_x0))])

    idx_dist_sorted_valid = [idx for idx in idx_dist_sorted if idx_in_lane[idx] and idx_forward[idx]]
    if len(idx_dist_sorted_valid) > 0:
        return idx_dist_sorted_valid[0]
    else:
        return None


def get_prev_vehicle_lane(x0, all_other_x0, lane, world):
    ''' get previous vehicle in a lane '''

    idx_in_lane = [world.get_lane_from_x0(all_other_x0[idx]) == lane for idx in range(len(all_other_x0))]
    idx_forward = [all_other_x0[idx][0] - x0[0] < 0 for idx in range(len(all_other_x0))]
    idx_dist_sorted = np.argsort([np.abs(all_other_x0[idx][0] - x0[0]) for idx in range(len(all_other_x0))])

    idx_dist_sorted_valid = [idx for idx in idx_dist_sorted if idx_in_lane[idx] and idx_forward[idx]]
    if len(idx_dist_sorted_valid) > 0:
        return idx_dist_sorted_valid[0]
    else:
        return None