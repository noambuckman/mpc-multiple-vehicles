import numpy as np
from src.traffic_world import TrafficWorld
from src.vehicle_mpc_information import Trajectory
import random
from typing import Dict


def generate_warm_x(car_mpc, world: TrafficWorld, x0: np.array, average_v=None):
    """ Warm starts that return a trajectory in x (control) -space
    N:  Number of control points
    car_mpc:  Vehicle instance
    car_x0:  Initial position
    
    Return:  x_warm_profiles [dict]
                keys: label of warm start [str]
                values: 6xn state vector

            ux_warm_profiles [dict]
                keys: label of warm start [str]
                values:  control vector (initialized as zero), state vector, x desired vector
    """

    x_warm_profiles = {}
    N = car_mpc.N
    lane_width = world.lane_width
    if average_v is None:
        constant_v = car_mpc.max_v
    else:
        constant_v = average_v
    t_array = np.arange(0, car_mpc.dt * (N + 1) - 0.000001, car_mpc.dt)
    x = x0[0] + t_array * constant_v
    y0 = x0[1]
    x_warm_default = np.repeat(x0.reshape(6, 1), N + 1, 1)
    x_warm_default[0, :] = x
    x_warm_default[1, :] = y0
    x_warm_default[2, :] = np.zeros((1, N + 1))
    x_warm_default[3, :] = np.zeros((1, N + 1))
    x_warm_default[4, :] = constant_v
    x_warm_default[5, :] = t_array * constant_v
    x_warm_profiles["0constant v"] = x_warm_default
    # lane change up
    y_up = y0 + lane_width
    for percent_change in [0.00, 0.5, 0.75]:
        key = "0up %d" % (int(100 * percent_change))
        x_up = np.copy(x_warm_default)
        ti_lane_change = int(percent_change * (N + 1))
        y = y_up * np.ones((1, N + 1))
        y[:, :ti_lane_change] = x0[1] * np.ones((1, ti_lane_change))
        x_up[1, :] = y
        x_warm_profiles[key] = x_up

    y_down = y0 - lane_width
    for percent_change in [0.00, 0.5, 0.75]:
        key = "0down %d" % (int(100 * percent_change))
        x_up = np.copy(x_warm_default)
        ti_lane_change = int(percent_change * (N + 1))
        y = y_down * np.ones((1, N + 1))
        y[:, :ti_lane_change] = x0[1] * np.ones((1, ti_lane_change))
        x_up[1, :] = y
        x_warm_profiles[key] = x_up

    ux_warm_profiles = {}
    for k_warm in x_warm_profiles.keys():
        u_warm = np.zeros((2, N))
        x_warm = x_warm_profiles[k_warm]
        x_des_warm = np.zeros(shape=(3, N + 1))
        for k in range(N + 1):
            x_des_warm[:, k:k + 1] = car_mpc.fd(x_warm[-1, k])
        ux_warm_profiles[k_warm] = [u_warm, x_warm, x_des_warm]

    return x_warm_profiles, ux_warm_profiles


def centerline_following(N: int, car_mpc, car_x0):
    y_follow = car_x0[1]

    u_warm = np.zeros((2, N))
    u_warm[1, :] = np.zeros(shape=(1, N))  ### No acceleration

    x = np.zeros(shape=(6, N + 1))
    x[:, 0:1] = car_x0.reshape(6, 1)
    for k in range(N):
        k_u = 0.1
        u_turn = -k_u * (x[1, k] - y_follow)
        u_turn = np.clip(u_turn, -car_mpc.max_delta_u, car_mpc.max_delta_u)
        x_k = x[:, k]
        u_warm[0, k:k + 1] = u_turn
        x_knext = car_mpc.F_kutta(car_mpc.f, x_k, u_warm[:, k])
        x[:, k + 1:k + 2] = x_knext

    x_des = np.zeros(shape=(3, N + 1))
    for k in range(N + 1):
        x_des[:, k:k + 1] = car_mpc.fd(x[-1, k])

    return [u_warm, x, x_des]


def generate_warm_u(N: int, car_mpc, car_x0):
    """ Warm starts that return a trajectory in u (control) -space
    N:  Number of control points
    car_mpc:  Vehicle instance
    car_x0:  Initial position
    
    Return:  u_warm_profiles [dict]
                keys: label of warm start [str]
                values: 2xn control vector

            ux_warm_profiles [dict]
                keys: label of warm start [str]
                values:  control vector, state vector, x desired vector
    """

    u0_warm_profiles = {}
    u1_warm_profiles = {}
    # braking
    u_warm = np.zeros((2, N))
    u_warm[0, :] = np.zeros(shape=(1, N))
    u_warm[1, :] = np.ones(shape=(1, N)) * car_mpc.min_v_u
    u1_warm_profiles["braking"] = u_warm

    # accelerate
    u_warm = np.zeros((2, N))
    u_warm[0, :] = np.zeros(shape=(1, N))
    u_warm[1, :] = np.ones(shape=(1, N)) * car_mpc.max_v_u
    u1_warm_profiles["accelerating"] = u_warm

    u_warm = np.zeros((2, N))
    u_warm[0, :] = np.zeros(shape=(1, N))
    t_half = int(N)
    u_warm[1, :t_half] = np.ones(shape=(1, t_half)) * car_mpc.max_v_u / 3.0

    # no accelerate
    u_warm = np.zeros((2, N))
    u1_warm_profiles["none"] = u_warm

    # lane change left
    u_warm = np.zeros((2, N))
    u_l1 = 0
    u_r1 = int(N / 3)
    u_l2 = int(2 * N / 3)

    # u_r2 = int(3*N/4)
    u_warm[0, u_l1] = -0.5 * car_mpc.max_delta_u
    u_warm[0, u_r1] = car_mpc.max_delta_u
    u_warm[0, u_l2] = -0.5 * car_mpc.max_delta_u

    u_warm[1, :] = np.zeros(shape=(1, N))
    u0_warm_profiles["lane_change_right"] = u_warm

    u0_warm_profiles["lane_change_left"] = -u0_warm_profiles["lane_change_right"]
    u0_warm_profiles["none"] = np.zeros(shape=(2, N))

    u_warm_profiles = {}
    for u0_k in u0_warm_profiles.keys():
        for u1_k in u1_warm_profiles.keys():
            u_k = u0_k + " " + u1_k
            u_warm_profiles[u_k] = u0_warm_profiles[u0_k] + u1_warm_profiles[u1_k]

    # Generate x, x_des from the u_warm profiles
    ux_warm_profiles = {}
    for k_warm in u_warm_profiles.keys():
        u_warm = u_warm_profiles[k_warm]
        x_warm, x_des_warm = car_mpc.forward_simulate_all(car_x0.reshape(6, 1), u_warm)
        ux_warm_profiles[k_warm] = [u_warm, x_warm, x_des_warm]

    # Generate some line following examples
    u_warm, x_warm, x_des_warm = centerline_following(N, car_mpc, car_x0)
    k_warm = "line_following"
    u_warm_profiles[k_warm] = u_warm
    ux_warm_profiles[k_warm] = [u_warm, x_warm, x_des_warm]

    return u_warm_profiles, ux_warm_profiles


def generate_warm_starts(vehicle,
                         world: TrafficWorld,
                         x0: np.array,
                         other_veh_info,
                         params: dict,
                         u_mpc_previous=None,
                         u_ibr_previous=None):
    """ Generate a dictionary of warm starts for the solver.  
    
        Returns:  Dictionary with warm_start_name: (state, control, desired_state)
    """
    other_x0 = [veh_info.x0 for veh_info in other_veh_info]

    u_warm_profiles, ux_warm_profiles = generate_warm_u(params["N"], vehicle, x0)

    if len(other_x0) > 0:
        warm_velocity = np.median([x[4] for x in other_x0])
    else:
        warm_velocity = x0[4]
    _, x_ux_warm_profiles = generate_warm_x(vehicle, world, x0, warm_velocity)
    ux_warm_profiles.update(x_ux_warm_profiles)

    if u_mpc_previous is not None:  # Try out the controls that were previous executed
        u_warm_profiles["previous_mpc_hold"] = np.concatenate(
            (
                u_mpc_previous[:, params["number_ctrl_pts_executed"]:],
                np.tile(u_mpc_previous[:, -1:], (1, params["number_ctrl_pts_executed"])),
            ),
            axis=1,
        )
        x_warm, x_des_warm = vehicle.forward_simulate_all(x0.reshape(6, 1), u_warm_profiles["previous_mpc_hold"])
        ux_warm_profiles["previous_mpc_hold"] = [
            u_warm_profiles["previous_mpc_hold"],
            x_warm,
            x_des_warm,
        ]

        u_warm_profiles["previous_mpc_none"] = np.concatenate(
            (
                u_mpc_previous[:, params["number_ctrl_pts_executed"]:],
                np.tile(np.zeros((2, 1)), (1, params["number_ctrl_pts_executed"])),
            ),
            axis=1,
        )
        x_warm, x_des_warm = vehicle.forward_simulate_all(x0.reshape(6, 1), u_warm_profiles["previous_mpc_none"])
        ux_warm_profiles["previous_mpc_none"] = [
            u_warm_profiles["previous_mpc_none"],
            x_warm,
            x_des_warm,
        ]

    if u_ibr_previous is not None:  # Try out the controller from the previous round of IBR
        u_warm_profiles["previous_ibr"] = u_ibr_previous
        x_warm, x_des_warm = vehicle.forward_simulate_all(x0.reshape(6, 1), u_warm_profiles["previous_ibr"])
        ux_warm_profiles["previous_ibr"] = [
            u_warm_profiles["previous_ibr"],
            x_warm,
            x_des_warm,
        ]

    warm_start_trajectories = {}
    for key, (u, x, xd) in ux_warm_profiles.items():

        warm_start_trajectories[key] = Trajectory(u=u, x=x, xd=xd)

    return warm_start_trajectories


def warm_profiles_subset(n_warm_keys: int, ux_warm_profiles: Dict[str, Trajectory]):
    '''choose randomly n_warm_keys keys from ux_warm_profiles and return the subset'''

    priority_warm_keys = []
    if n_warm_keys >= 1 and "previous_mpc_hold" in ux_warm_profiles:
        priority_warm_keys += ["previous_mpc_hold"]
    if n_warm_keys >= 2 and "previous_ibr" in ux_warm_profiles:
        priority_warm_keys += ["previous_ibr"]
    remaining_n_keys = n_warm_keys - len(priority_warm_keys)
    remaining_keys = [k for k in ux_warm_profiles.keys() if k not in priority_warm_keys]
    random.shuffle(remaining_keys)

    warm_subset_keys = priority_warm_keys + remaining_keys[:remaining_n_keys]

    ux_warm_profiles_subset = dict((k, ux_warm_profiles[k]) for k in warm_subset_keys)

    return ux_warm_profiles_subset