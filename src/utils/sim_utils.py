import numpy as np
from src.vehicle import Vehicle
from typing import List
import copy as cp
from src.multiagent_mpc import load_state, save_state
from shapely.geometry import box
from shapely.affinity import rotate
import psutil, time, datetime
import os, pickle


class ExperimentHelper(object):
    ''' Helper for logging various things and keeping track of experiment start time'''
    def __init__(self, log_dir, params):
        self.start_time = time.time()
        assert "n_ibr" in params
        assert "k_max_slack" in params
        self.log_dir = log_dir
        self.params = params

        if params["save_ibr"]:
            os.makedirs(log_dir + "data/", exist_ok=True)

    def save_ibr(self, i_mpc, i_rounds_ibr, response_i,
                 vehs_ibr_info_predicted):

        if self.params["save_ibr"]:
            save_ibr(self.log_dir, i_mpc, i_rounds_ibr, response_i,
                     vehs_ibr_info_predicted)

    def check_collisions(self, all_other_vehicles: List[Vehicle],
                         all_other_x_executed: List[np.array]) -> bool:
        car_collisions = self.get_collision_pairs(all_other_vehicles,
                                                  all_other_x_executed)
        if len(car_collisions) > 0:
            return True
        else:
            return False

    def get_collision_pairs(
            self, all_other_vehicles: List[Vehicle],
            all_other_x_executed: List[np.array]) -> List[tuple]:
        car_collisions = []
        for i in range(len(all_other_vehicles)):
            for j in range(len(all_other_vehicles)):
                if i >= j:
                    continue
                else:
                    collision = self.check_collision(all_other_vehicles[i],
                                                     all_other_vehicles[j],
                                                     all_other_x_executed[i],
                                                     all_other_x_executed[j])
                    if collision:
                        car_collisions += [(i, j)]
        return car_collisions

    def update_sim_states(self, othervehs_ibr_info, all_other_x_ibr_g,
                          actual_t, i_mpc, xothers_actual, uothers_actual):
        ''' Update the simulation states with the solutions from Iterative Best Response / MPC round 
            We update multiple sim steps at a time (params["number_cntrl_pts_executed"])    
        '''

        all_other_u_mpc = [cp.deepcopy(veh.u) for veh in othervehs_ibr_info]
        all_other_x_executed = [
            all_other_x_ibr_g[i][:, :self.params["number_ctrl_pts_executed"] +
                                 1] for i in range(self.params["n_other"])
        ]
        all_other_u_executed = [
            othervehs_ibr_info[i].u[:, :self.
                                    params["number_ctrl_pts_executed"]]
            for i in range(self.params["n_other"])
        ]

        # Append to the executed trajectories history of x
        for i in range(self.params["n_other"]):
            xothers_actual[i][:, actual_t:actual_t +
                              self.params["number_ctrl_pts_executed"] +
                              1] = all_other_x_executed[i]
            uothers_actual[i][:, actual_t:actual_t + self.params[
                "number_ctrl_pts_executed"]] = all_other_u_executed[i]

        # SAVE STATES AND PLOT
        file_name = self.log_dir + "data/" + "mpc_%03d" % (i_mpc)
        print("Saving MPC Rd %03d / %03d to ... %s" %
              (i_mpc, self.params["n_mpc"] - 1, file_name))

        if self.params["save_state"]:
            other_u_ibr_temp = [veh.u for veh in othervehs_ibr_info]
            other_xd_ibr_temp = [veh.xd for veh in othervehs_ibr_info]

            save_state(file_name, None, None, None, all_other_x_ibr_g,
                       other_u_ibr_temp, other_xd_ibr_temp)
            save_state(self.log_dir + "data/" + "all_%03d" % (i_mpc),
                       None,
                       None,
                       None,
                       xothers_actual,
                       uothers_actual,
                       None,
                       end_t=actual_t +
                       self.params["number_ctrl_pts_executed"] + 1)

        actual_t += self.params["number_ctrl_pts_executed"]

        return actual_t, all_other_u_mpc, all_other_x_executed, all_other_u_executed, xothers_actual, uothers_actual

    def initialize_states(self):

        x_executed = [None for i in range(self.params["n_other"])]
        u_mpc = [None for i in range(self.params["n_other"])]
        N_total = self.params["n_mpc"] * self.params["number_ctrl_pts_executed"]

        u_actual = [
            np.zeros((2, N_total)) for _ in range(self.params["n_other"])
        ]
        x_actual = [
            np.zeros((6, N_total + 1)) for _ in range(self.params["n_other"])
        ]
        t_actual = 0
        return x_executed, u_mpc, x_actual, u_actual, t_actual

    def load_log_data(self, i_mpc_start):
        N_total = self.params["n_mpc"] * self.params["number_ctrl_pts_executed"]
        uothers_actual = [
            np.zeros((2, N_total)) for _ in range(self.params["n_other"])
        ]
        xothers_actual = [
            np.zeros((6, N_total + 1)) for _ in range(self.params["n_other"])
        ]

        previous_mpc_file = self.log_dir + "data/mpc_%03d" % (i_mpc_start - 1)
        print("Loaded initial positions from %s" % (previous_mpc_file))
        _, _, _, all_other_x_executed, all_other_u_executed, _ = load_state(
            previous_mpc_file, self.params["n_other"])
        all_other_u_mpc = all_other_u_executed

        previous_all_file = self.log_dir + "data/all_%03d" % (i_mpc_start - 1)
        _, _, _, xothers_actual_prev, uothers_actual_prev, _, = load_state(
            previous_all_file, self.params["n_other"], ignore_des=True)
        t_end = xothers_actual_prev[0].shape[1]

        for i in range(len(xothers_actual_prev)):
            xothers_actual[i][:, :t_end] = xothers_actual_prev[i][:, :t_end]
            uothers_actual[i][:, :t_end] = uothers_actual_prev[i][:, :t_end]
        actual_t = i_mpc_start * self.params["number_ctrl_pts_executed"]

        return all_other_x_executed, all_other_u_mpc, xothers_actual, uothers_actual, actual_t

    def check_collision(self, vehicle_a: Vehicle, vehicle_b: Vehicle,
                        X_a: np.array, X_b: np.array) -> bool:
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

    def print_vehicle_id(self, response_i):
        print("...Veh %02d Solver:" % response_i)

    def print_mpc_ibr_round(self, i_mpc, i_ibr, params):
        print("MPC %d, IBR %d / %d" % (i_mpc, i_ibr, params["n_ibr"] - 1))

    def print_nc_nnc(self, cntrld_vehicle_info, nonresponse_veh_info):
        print("....# Cntrld Vehicles: %d # Non-Response: %d " %
              (len(cntrld_vehicle_info), len(nonresponse_veh_info)))

    def print_solved_status(self, response_i, i_mpc, i_ibr, start_ipopt_time):
        print(
            "......Solved.  veh: %02d | mpc: %d | ibr: %d | solver time: %0.1f s"
            % (response_i, i_mpc, i_ibr, time.time() - start_ipopt_time))

    def print_max_solved_status(self, response_i, i_mpc, i_ibr, max_slack,
                                start_ipopt_time):
        print(
            "......Reached max # resolves. veh: %02d | mpc: %d | ibr: %d | Slack %.05f > thresh %.05f  | solver time: %0.1f"
            % (response_i, i_mpc, i_ibr, max_slack, self.params["k_max_slack"],
               time.time() - start_ipopt_time))

    def print_not_solved_status(self, response_i, i_mpc, i_ibr, max_slack,
                                start_ipopt_time):
        print(
            "......Re-solve. veh: %02d | mpc: %d | ibr: %d | Slack %.05f > thresh %.05f  | solver time: %0.1f"
            % (response_i, i_mpc, i_ibr, max_slack, self.params["k_max_slack"],
               time.time() - start_ipopt_time))

    def print_sim_exited_early(self, reason_msg):
        print("Sim Ended Early due to %s  Runtime: %s" %
              (reason_msg,
               datetime.timedelta(seconds=(time.time() - self.start_time))))

    def print_sim_finished(self):
        print("Simulation Done!  Runtime: %s" %
              (datetime.timedelta(seconds=(time.time() - self.start_time))))

    def print_initializing_trajectories(self, i_mpc):
        print("Extending Trajectories...Solving MPC for each vehicle:  Rd %d"%i_mpc)

    def check_machine_memory(self):
        if psutil.virtual_memory().percent >= 95.0:
            raise Exception(
                "Virtual Memory is too high, exiting to save computer")
    
    def save_trajectory(self, trajectory_array, controls_array):
        ''' Save the current trajectory, controls'''

        trajectory_dir_path = os.path.join(self.log_dir, "trajectories")
        os.makedirs(trajectory_dir_path, exist_ok=True)

        trajectory_path = os.path.join(trajectory_dir_path, "trajectory_t.npy")
        controls_path = os.path.join(trajectory_dir_path, "controls_t.npy")

        with open(trajectory_path, "wb") as fp:
            np.save(fp, trajectory_array)

        with open(controls_path, 'wb') as fp:
            np.save(fp, controls_array)

        
    

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

    return all_vehicles, all_x0, world


def generate_test_mpc_scenario(n_cntrld,
                               n_non_response,
                               N,
                               dt,
                               n_lanes=2,
                               car_density=5000):
    ''' Generate the vehicle data and initial conditions for split of cntrld and non controlled vehicles'''

    n_other = n_cntrld + n_non_response + 1
    all_vehicles, all_x0, world = generate_test_scenario(
        n_other, N, dt, n_lanes, car_density)

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

    return x_initial, cntrld_x_initial, non_response_x, response_veh, cntrld_veh, non_response_veh, world


def convert_to_global_units(x_reference_global: np.array, x: np.array):
    """ During solving, the x-dimension is normalized to the ambulance initial position x_reference_global """
    if x_reference_global is None:
        return x

    x_global = cp.deepcopy(x)
    x_global[0, :] += x_reference_global[0]

    return x_global


def vehicles_within_range(other_x0, amb_x0, distance_from_ambulance):

    veh_idxs = [
        i for i in range(len(other_x0))
        if abs(other_x0[i][0] - amb_x0[0]) <= distance_from_ambulance
    ]

    return veh_idxs


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


def get_ibr_vehs_idxs(vehicle_list):
    ''' For now just return all the vehicles'''
    return list(range(len(vehicle_list)))


def get_max_dist_traveled(response_vehinfo, params):
    ''' '''
    T = params["T"]  #planning horizon
    max_speed = response_vehinfo.vehicle.max_v

    max_distance_traveled = max_speed * T

    return max_distance_traveled


def get_within_range_other_vehicle_idxs(response_i,
                                        vehsinfo,
                                        max_distance: float = 100
                                        ) -> List[int]:
    ''' Only grab vehicles that are within 20 vehicle lengths in the X direction'''
    response_veh_info = vehsinfo[response_i]

    within_range_idxs = []
    for j in range(len(vehsinfo)):
        initial_distance = vehsinfo[j].x0[0] - response_veh_info.x0[0]
        interspace = abs(initial_distance) - response_veh_info.vehicle.L
        if j != response_i and interspace <= max_distance:
            within_range_idxs += [j]

    return within_range_idxs


def get_closest_n_obstacle_vehs(
    response_vehinfo,
    cntrld_vehicle_info,
    obstacle_vehs_info,
    max_num_obstacles: int = None,
    min_num_obstacles_ego: int = None,
):
    ''' Return the closest vehicles by distance to the response and cntrld_vehicle info'''

    if max_num_obstacles is None or len(obstacle_vehs_info)==0:
        return obstacle_vehs_info
        
    ego_obstacle_info = []
    if min_num_obstacles_ego is None:
        remaining_obstacle_info = obstacle_vehs_info
    else:
        # First get the closest vehicles to to the ego
        # Get distances
        remaining_obstacle_info = []    
        x_other = np.stack([vi.x for vi in obstacle_vehs_info], axis=0)
        x_other = np.expand_dims(x_other, axis=1)  #[nother x 1 x 6 x N]
        x_planning = np.stack([response_vehinfo.x] + [], axis=0)
        # [1 x nplanning x 6 x N]
        x_planning = np.expand_dims(x_planning, axis=0)

        # [nother x nplanning x  N]
        dist = np.sqrt(
            np.sum(((x_planning - x_other)[:, :, 0:2, :]**2),
                   axis=2)) - response_vehinfo.vehicle.L

        distance_cost = np.sum(dist**2, axis=(1, 2))
        sorted_idx = np.argsort(distance_cost)
        sorted_idx = sorted_idx[:min_num_obstacles_ego]

        for idx in range(len(obstacle_vehs_info)):
            if idx in sorted_idx:
                ego_obstacle_info += [obstacle_vehs_info[idx]]
            else:
                remaining_obstacle_info += [obstacle_vehs_info[idx]]
    
    
    max_num_obstacles = max_num_obstacles - len(ego_obstacle_info)
    sorted_remaining_obstacle_info = []
    if max_num_obstacles > 0 and len(remaining_obstacle_info) > 0:
        x_other = np.stack([vi.x for vi in remaining_obstacle_info], axis=0)
        x_other = np.expand_dims(x_other, axis=1)  #[nother x 1 x 6 x N]
        x_planning = np.stack([response_vehinfo.x] +
                            [vi.x for vi in cntrld_vehicle_info],
                            axis=0)
        x_planning = np.expand_dims(x_planning, axis=0)  # [1 x nplanning x 6 x N]

        # [nother x nplanning x  N]
        dist = np.sqrt(np.sum(((x_planning - x_other)[:, :, 0:2, :]**2),
                            axis=2)) - response_vehinfo.vehicle.L

        distance_cost = np.sum(dist**2, axis=(1, 2))
        sorted_idx = np.argsort(distance_cost)
        sorted_idx = sorted_idx[:max_num_obstacles]

        for idx in sorted_idx:
            sorted_remaining_obstacle_info += [remaining_obstacle_info[idx]]
        
    return ego_obstacle_info + sorted_remaining_obstacle_info



def get_obstacle_vehs_closeby(response_vehinfo,
                              cntrld_vehicle_info,
                              obstacle_vehs_info,
                              distance_threshold=20.0):
    ''' Check whether response or cntrld vehinfo are close by
        to the obstacle cars at any point during trajectory.
        Assuming the current/previous warm start of the vehicles

        We don't do explicit geometry checking but use a distance threshold and length of car.
    '''
    if len(obstacle_vehs_info) == 0 or distance_threshold == np.infty:
        return obstacle_vehs_info

    x_other = np.stack([vi.x for vi in obstacle_vehs_info], axis=0)
    x_other = np.expand_dims(x_other, axis=1)  #[nother x 1 x 6 x N]

    x_planning = np.stack([response_vehinfo.x] +
                          [vi.x for vi in cntrld_vehicle_info],
                          axis=0)
    x_planning = np.expand_dims(x_planning, axis=0)  # [1 x nplanning x 6 x N]

    # [nother x nplanning x  N]
    dist = np.sqrt(np.sum(((x_planning - x_other)[:, :, 0:2, :]**2),
                          axis=2)) - response_vehinfo.vehicle.L
    within_dist = np.any(dist <= distance_threshold,
                         axis=(1, 2))  # [nother x 1]

    obstacles_within_dist = []
    for idx in range(len(obstacle_vehs_info)):
        if within_dist[idx]:
            obstacles_within_dist += [obstacle_vehs_info[idx]]

    return obstacles_within_dist


def assign_shared_control(params, i_rounds_ibr, idxs_in_mpc,
                          vehicles_index_best_responders, response_veh_info,
                          vehs_ibrinfo_pred):
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

    cntrld_idx = []
    cntrld_vehicle_info = []

    if n_cntrld > 0:
        delta_x = []
        for _, otherveh_info in enumerate(vehs_ibrinfo_pred):
            delta_x += [response_veh_info.x0[0] - otherveh_info.x0[0]]

        # This necessary but could limit fringe
        # best response interactions with outside best response
        sorted_all_idx = np.argsort(delta_x)
        sorted_candidate_idx = []
        for idx in sorted_all_idx:
            if delta_x[idx] < 0:
                # dont include vehicles ahead
                continue
            if idx not in idxs_in_mpc:
                # dont include vehicles not in mpc
                continue
            if idx not in vehicles_index_best_responders:
                # dont include vehs not in best responderes list
                continue

            sorted_candidate_idx += [idx]

        cntrld_idx = sorted_candidate_idx[:n_cntrld]
        cntrld_vehicle_info = [vehs_ibrinfo_pred[idx] for idx in cntrld_idx]

    nonresponse_veh_info = [
        vehs_ibrinfo_pred[idx] for idx in idxs_in_mpc if idx not in cntrld_idx
    ]

    return cntrld_vehicle_info, nonresponse_veh_info, cntrld_idx


