import numpy as np
import casadi as cas


class VehicleParameters(object):
    def __init__(self, n_cntrld: int = 0, n_other_vehicles: int = 0, agent_str: str = ""):
        self.param_counter = -1
        self.agent_str = agent_str

        self.agent_id = cas.MX.sym('p_agid' + self.get_param_name(), 1, 1)
        self.dt = cas.MX.sym('p_dt' + self.get_param_name(), 1, 1)
        self.k_total = cas.MX.sym('p_k' + self.get_param_name(), 1, 1)  # the overall weighting of the total costs
        self.theta_i = cas.MX.sym('p_thetai' + self.get_param_name(), 1, 1)  # my theta wrt to ambulance
        self.L = cas.MX.sym('p_L' + self.get_param_name(), 1, 1)
        self.W = cas.MX.sym('p_W' + self.get_param_name(), 1, 1)
        self.n_circles = cas.MX.sym('p_nc' + self.get_param_name(), 1, 1)

        self.desired_lane = cas.MX.sym('p_d_lane' + self.get_param_name(), 1, 1)
        ## State Costs Constants

        self.k_x = cas.MX.sym('p_kx' + self.get_param_name(), 1, 1)
        self.k_y = cas.MX.sym('p_ky' + self.get_param_name(), 1, 1)
        self.k_phi = cas.MX.sym('p_kphi' + self.get_param_name(), 1, 1)
        self.k_delta = cas.MX.sym('p_kd' + self.get_param_name(), 1, 1)

        self.k_v = cas.MX.sym('p_kv' + self.get_param_name(), 1, 1)
        self.k_s = cas.MX.sym('p_ks' + self.get_param_name(), 1, 1)

        ## Control Costs Constants
        self.k_u_delta = cas.MX.sym('p_kudelta' + self.get_param_name(), 1, 1)
        self.k_u_v = cas.MX.sym('p_kuv' + self.get_param_name(), 1, 1)

        ## Derived State Costs Constants
        self.k_lat = cas.MX.sym('p_klat' + self.get_param_name(), 1, 1)
        self.k_lon = cas.MX.sym('p_klon' + self.get_param_name(), 1, 1)
        self.k_phi_error = cas.MX.sym('p_phierror' + self.get_param_name(), 1, 1)
        self.k_phi_dot = cas.MX.sym('p_phidot' + self.get_param_name(), 1, 1)

        self.k_x_dot = cas.MX.sym('p_kxdot' + self.get_param_name(), 1, 1)

        self.k_change_u_v = cas.MX.sym('p_kchangeuv' + self.get_param_name(), 1, 1)
        self.k_change_u_delta = cas.MX.sym('p_changeudelta' + self.get_param_name(), 1, 1)

        self.k_final = cas.MX.sym('p_kfinal' + self.get_param_name(), 1, 1)

        self.k_on_grass = cas.MX.sym('p_kongrass' + self.get_param_name(), 1, 1)
        self.k_limit_costs = cas.MX.sym('k_klimitcosts' + self.get_param_name(), 1, 1)
        
        # Constraints
        self.max_steering_rate = cas.MX.sym('p_max_steering_rate' + self.get_param_name(), 1, 1)  # deg/sec
        self.max_delta_u = cas.MX.sym('p_max_delta_u' + self.get_param_name(), 1, 1)  # rad (change in steering angle)

        self.max_acceleration = cas.MX.sym('p_max_acceleration' + self.get_param_name(), 1, 1)  #m/s^2
        self.max_v_u = cas.MX.sym('p_max_v_u' + self.get_param_name(), 1, 1)  # m/s (change in velocity)

        self.max_deceleration = cas.MX.sym('p_max_deceleration' + self.get_param_name(), 1, 1)  #m/s^2
        self.min_v_u = cas.MX.sym('p_min_v_u' + self.get_param_name(), 1, 1)  # m/s  (change in velocity)

        # Speed limit
        self.max_v = cas.MX.sym('p_max_v' + self.get_param_name(), 1, 1)  # m/s
        self.min_v = cas.MX.sym('p_min_v' + self.get_param_name(), 1, 1)

        # Spatial constraints
        self.max_y = cas.MX.sym('p_max_y' + self.get_param_name(), 1, 1)
        self.min_y = cas.MX.sym('p_min_y' + self.get_param_name(), 1, 1)
        self.grass_max_y = cas.MX.sym('p_grass_max_y' + self.get_param_name(), 1, 1) # for adding grass cost
        self.grass_min_y = cas.MX.sym('p_grass_min_y' + self.get_param_name(), 1, 1) # for adding grass cost       
        
        # self.strict_wall_constraint = cas.MX.sym('p_' + self.get_param_name(), 1, 1)

        self.max_X_dev = cas.MX.sym('p_max_X_dev' + self.get_param_name(), 1, 1)
        self.max_Y_dev = cas.MX.sym('p_max_Y_dev' + self.get_param_name(), 1, 1)

        # Initialize vehicle dynamics
        # self.f = self.gen_f_vehicle_dynamics()
        # self.fd = None
        #
        # Distance used for collision avoidance
        self.circle_radius = cas.MX.sym('p_circle_radius' + self.get_param_name(), 1, 1)
        self.min_dist = cas.MX.sym('p_min_dist' + self.get_param_name(), 1, 1)  # 2 times the radius of 1.5
        # self.radius = cas.MX.sym('p_' + self.get_param_name(), 1, 1)

        self.ax = cas.MX.sym('p_ax' + self.get_param_name(), 1, 1)
        self.by = cas.MX.sym('p_by' + self.get_param_name(), 1, 1)

        # self.theta_i_ego = cas.MX.sym('p_' + self.get_param_name(), 1)
        # self.theta_i_jc = cas.MX.sym('p_' + self.get_param_name(), n_cntrld, 1)
        # self.theta_i_jnc = cas.MX.sym('p_' + self.get_param_name(), n_other_vehicles,
        #                               1)  # These need to be set dependent on which IDs are in jc and nc

    def get_param_name(self):
        self.param_counter += 1
        return self.agent_str + '%03d' % self.param_counter

    def get_opti_params(self):
        all_params = cas.vertcat(
            self.agent_id,
            self.dt,
            self.k_total,  # the overall weighting of the total costs
            self.theta_i,  # my theta wrt to ambulance
            self.L,
            self.W,
            self.n_circles,
            self.desired_lane,
            self.k_x,
            self.k_y,
            self.k_phi,
            self.k_delta,
            self.k_v,
            self.k_s,
            self.k_u_delta,
            self.k_u_v,
            self.k_lat,
            self.k_lon,
            self.k_phi_error,
            self.k_phi_dot,
            self.k_x_dot,
            self.k_change_u_v,
            self.k_change_u_delta,
            self.k_final,
            self.max_steering_rate,  # deg/sec
            self.max_delta_u,  # rad (change in steering angle)
            self.max_acceleration,  #m/s^2
            self.max_v_u,  # m/s (change in velocity)
            self.max_deceleration,  #m/s^2
            self.min_v_u,  # m/s  (change in velocity)
            self.max_v,  # m/s
            self.min_v,
            self.max_y,
            self.min_y,
            # self.strict_wall_constraint ,
            self.max_X_dev,
            self.max_Y_dev,
            self.circle_radius,
            self.min_dist,  # 2 times the radius of 1.5
            # self.radius ,
            self.ax,
            self.by,
            self.grass_max_y,
            self.grass_min_y,            
            self.k_on_grass,
            self.k_limit_costs,
            # self.theta_i_ego[:],
            # self.theta_i_jc[:],
            # self.theta_i_jnc[:],
        )

        return all_params

    def get_param_values(self, vehicle):

        param_vals = [
            vehicle.agent_id,
            vehicle.dt,
            vehicle.k_total,  # the overall weighting of the total costs
            vehicle.theta_i,  # my theta wrt to ambulance
            vehicle.L,
            vehicle.W,
            vehicle.n_circles,
            vehicle.desired_lane,
            vehicle.k_x,
            vehicle.k_y,
            vehicle.k_phi,
            vehicle.k_delta,
            vehicle.k_v,
            vehicle.k_s,
            vehicle.k_u_delta,
            vehicle.k_u_v,
            vehicle.k_lat,
            vehicle.k_lon,
            vehicle.k_phi_error,
            vehicle.k_phi_dot,
            vehicle.k_x_dot,
            vehicle.k_change_u_v,
            vehicle.k_change_u_delta,
            vehicle.k_final,
            vehicle.max_steering_rate,  # deg/sec
            vehicle.max_delta_u,  # rad (change in steering angle)
            vehicle.max_acceleration,  #m/s^2
            vehicle.max_v_u,  # m/s (change in velocity)
            vehicle.max_deceleration,  #m/s^2
            vehicle.min_v_u,  # m/s  (change in velocity)
            vehicle.max_v,  # m/s
            vehicle.min_v,
            vehicle.max_y,
            vehicle.min_y,
            # vehicle.strict_wall_constraint,
            vehicle.max_X_dev,
            vehicle.max_Y_dev,
            vehicle.circle_radius,
            vehicle.min_dist,  # 2 times the radius of 1.5
            # vehicle.radius,
            vehicle.ax,
            vehicle.by,
            vehicle.grass_max_y,
            vehicle.grass_min_y,
            vehicle.k_on_grass,
            vehicle.k_limit_costs,
            # vehicle.theta_i_ego,
            # vehicle.theta_i_jc,
            # vehicle.theta_i_jnc
        ]

        tall_params = []
        for p in param_vals:
            if type(p) == np.float:
                if p == cas.inf:
                    tall_params += [999999999999]
                elif p == -cas.inf:
                    tall_params += [-999999999999999]
                else:
                    tall_params += [p]
            elif type(p) == int:
                tall_params += [p]
            else:
                tall_params += [p.flatten()]

        return tall_params

    def set_param_values(self, vehicle):
        ''' Copy all values from vehicle'''

        self.__dict__.update(vehicle.__dict__)
