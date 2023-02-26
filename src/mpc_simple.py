import numpy as np
import casadi as cas
from typing import List

from src.traffic_world import TrafficWorld
from src.vehicle_parameters import VehicleParameters


class MultiMPC(object):
    ''' Optimization class contains all the object related to a best response optimization.
        We assume that each vehicle has a dynamics model (f), costs function, and constraints.

       The optimization currently solves assuming a Kinmeatic Bicycle Dynamics Model, assuming
       control inputs steering + acceleration.  
    '''
    def __init__(self,
                 N: int,
                 world: TrafficWorld,
                 n_vehs_cntrld: int = 0,
                 n_other_vehicles: int = 0,
                 solver_params: dict = None,
                 params: dict = None,
                 ipopt_params=None):

        if solver_params is None:
            solver_params = {}
            self.k_slack = 99999
            self.k_CA = 0
            self.k_CA_power = 1
            self.collision_cost = 0
        else:
            self.k_slack = solver_params["k_slack"]
            self.k_CA = solver_params["k_CA"]
            self.k_CA_power = solver_params["k_CA_power"]

        self.world = world
        self.dt = params["dt"]
        self.opti = cas.Opti()

        n_state, n_ctrl, n_desired = 6, 2, 3

        self.n_vehs_cntrld = n_vehs_cntrld
        self.n_other_vehicle = n_other_vehicles

        self.g_list = []
        self.lbg_list = []
        self.ubg_list = []

        self.x_list = []
        self.p_list = []

        # Response (planning) Vehicle Variables
        self.x_ego = cas.MX.sym('x_ego', n_state, N + 1)
        self.u_ego = cas.MX.sym('u_ego', n_ctrl, N)
        self.xd_ego = cas.MX.sym('xd_ego', n_desired, N + 1)
        self.x_list += [self.x_ego[:], self.u_ego[:], self.xd_ego[:]]

        # Parameters
        self.x0_ego = cas.MX.sym('x0_ego', n_state, 1)
        #TODO: deal with parameters
        self.p_ego = VehicleParameters(self.n_vehs_cntrld, self.n_other_vehicle, "ego")
        self.p_list += [self.x0_ego[:], self.p_ego.get_opti_params()]

        # Cntrld Vehicles become variables in the optimization
        self.x_ctrld = [cas.MX.sym('x_ctrl%02d' % i, n_state, N + 1) for i in range(self.n_vehs_cntrld)]
        self.u_ctrld = [cas.MX.sym('u_ctrl%02d' % i, n_ctrl, N) for i in range(self.n_vehs_cntrld)]
        self.xd_ctrld = [cas.MX.sym('xd_ctrl%02d' % i, n_desired, N + 1) for i in range(self.n_vehs_cntrld)]

        for i in range(self.n_vehs_cntrld):
            self.x_list += [(self.x_ctrld[i][:])]
            self.x_list += [(self.u_ctrld[i][:])]
            self.x_list += [(self.xd_ctrld[i][:])]

        # Cntrld Vehicle Parameters
        self.x0_cntrld = [cas.MX.sym('x0_ctrl%02d' % i, n_state, 1) for i in range(self.n_vehs_cntrld)]
        self.p_cntrld_list = [
            VehicleParameters(self.n_vehs_cntrld, self.n_other_vehicle, "ctrl%02d" % i)
            for i in range(self.n_vehs_cntrld)
        ]
        for i in range(len(self.x0_cntrld)):
            self.p_list += [self.x0_cntrld[i][:], self.p_cntrld_list[i].get_opti_params()]

        # Variables of surrounding vehicles assumed fixed as parameters for computation
        self.x_other = [cas.MX.sym('x_nc%02d' % i, n_state, N + 1) for i in range(self.n_other_vehicle)]
        self.u_other = [cas.MX.sym('u_nc%02d' % i, n_ctrl, N) for i in range(self.n_other_vehicle)]
        self.xd_other = [cas.MX.sym('xd_nc%02d' % i, 3, N + 1) for i in range(self.n_other_vehicle)]
        for i in range(self.n_other_vehicle):
            self.x_list.append(self.x_other[i][:])
            self.x_list.append(self.u_other[i][:])
            self.x_list.append(self.xd_other[i][:])

        # Non-Response Vehicle Parameters
        self.x0_allother = [cas.MX.sym('x0_nc%02d' % i, n_state, 1) for i in range(self.n_other_vehicle)]
        self.p_other_vehicle_list = [
            VehicleParameters(self.n_vehs_cntrld, self.n_other_vehicle, "nc%02d" % i)
            for i in range(self.n_other_vehicle)
        ]

        for i in range(len(self.x0_allother)):
            self.p_list += [self.x0_allother[i][:], self.p_other_vehicle_list[i].get_opti_params()]

        # TODO:  Add future parametrs
        # - svo_ij for cars

        # Generate costs for each vehicle
        self.response_costs, _ = self.generate_veh_costs(self.x_ego, self.u_ego, self.xd_ego, self.p_ego)

        self.all_other_costs = []
        # Generate SVO cost for other (non-cntrld) vehicles
        for idx in range(self.n_other_vehicle):
            # svo_ij = self.ego_veh.get_theta_ij(self.p_other_vehicle_list[idx].agent_id)
            svo_ij = self.p_ego.theta_i_jnc[idx]
            #TODO: Convert svo_ij into paramters that is set
            #TODO: Get rid of the if statement
            nonresponse_cost, _ = self.generate_veh_costs(self.x_other[idx], self.u_other[idx], self.xd_other[idx],
                                                          self.p_other_vehicle_list[idx])
            nonresponse_cost = nonresponse_cost * (svo_ij > 0)
            self.all_other_costs += [cas.sin(svo_ij) * nonresponse_cost]

        # SVO cost for Other (ctrld) Vehicles
        for idx in range(self.n_vehs_cntrld):
            # svo_ij = self.ego_veh.get_theta_ij(self.p_cntrld_list[idx].agent_id)
            svo_ij = self.p_ego.theta_i_jc[idx]
            nonresponse_cost, _ = self.generate_veh_costs(self.x_ctrld[idx], self.u_ctrld[idx], self.xd_ctrld[idx],
                                                          self.p_cntrld_list[idx])
            nonresponse_cost = nonresponse_cost * (svo_ij) > 0
            self.all_other_costs += [cas.sin(svo_ij) * nonresponse_cost]

        # Generate Slack Variables used as part of collision avoidance
        self.slack_cost = 0
        if self.n_other_vehicle > 0:
            self.slack_i_jnc = cas.MX.sym('s_i_jnc', self.n_other_vehicle, N + 1)
            self.x_list.append(self.slack_i_jnc[:])
            self.slack_ic_jnc = [
                cas.MX.sym('s_i%02d_jnc' % i, self.n_other_vehicle, N + 1) for i in range(self.n_vehs_cntrld)
            ]
            for i in range(len(self.slack_ic_jnc)):
                self.x_list.append(self.slack_ic_jnc[i][:])

            self.add_bounded_constraint(np.zeros(shape=self.slack_i_jnc.shape), self.slack_i_jnc, None)
            for slack_var in self.slack_ic_jnc:
                self.add_bounded_constraint(np.zeros(shape=slack_var.shape), slack_var, None)

            for j in range(self.slack_i_jnc.shape[0]):
                for t in range(self.slack_i_jnc.shape[1]):
                    self.slack_cost += self.slack_i_jnc[j, t]**2
            for ic in range(self.n_vehs_cntrld):
                for jnc in range(self.slack_ic_jnc[ic].shape[0]):
                    for t in range(self.slack_ic_jnc[ic].shape[1]):
                        self.slack_cost += self.slack_ic_jnc[ic][jnc, t]**2
        else:
            self.slack_i_jnc = 0
            self.slack_ic_jnc = []

        # Slack variables related to cntrld vehicles
        if self.n_vehs_cntrld > 0:
            self.slack_i_jc = cas.MX.sym("s_i_jc", self.n_vehs_cntrld, N + 1)
            self.x_list += [self.slack_i_jc[:]]
            self.add_bounded_constraint(np.zeros(shape=self.slack_i_jc.shape), self.slack_i_jc, None)
            for jc in range(self.slack_i_jc.shape[0]):
                for t in range(self.slack_i_jc.shape[1]):
                    self.slack_cost += self.slack_i_jc[jc, t]**2

            self.slack_ic_jc = [
                cas.MX.sym("s_ic%02d_jc" % ic, self.n_vehs_cntrld, N + 1) for ic in range(self.n_vehs_cntrld)
            ]
            for i in range(len(self.slack_ic_jc)):
                self.x_list.append(self.slack_ic_jc[i][:])
            for ic in range(self.n_vehs_cntrld):
                self.add_bounded_constraint(np.zeros(shape=self.slack_ic_jc[ic].shape), self.slack_ic_jc[ic], None)
                for jc in range(self.slack_ic_jc[ic].shape[0]):
                    for t in range(self.slack_ic_jc[ic].shape[1]):
                        self.slack_cost += self.slack_ic_jc[ic][jc, t]**2
        else:
            self.slack_i_jc = 0
            self.slack_ic_jc = []

        self.response_svo_cost = np.cos(self.p_ego.theta_i_ego) * self.response_costs

        if self.n_other_vehicle + self.n_vehs_cntrld > 0:
            self.other_svo_cost = np.sum(self.all_other_costs) / len(self.all_other_costs)
        else:
            self.other_svo_cost = 0

        ###################### Constraints
        ego_lane_number = self.p_ego.desired_lane
        fd = self.gen_f_desired_lane(world, ego_lane_number, right_direction=True)  # TODO:  This could mess things up
        f = self.gen_f_vehicle_dynamics(self.p_ego, model="kinematic_bicycle")
        self.add_dynamics_constraints_g(self.x_ego, self.u_ego, self.xd_ego, self.x0_ego, f, fd, self.dt)
        self.add_state_constraints_g(self.x_ego, self.u_ego, self.p_ego)

        # Add Constraints to cntrld Vehicles
        for j in range(self.n_vehs_cntrld):
            ado_lane_number = self.p_cntrld_list[j].desired_lane
            fd = self.gen_f_desired_lane(world, ado_lane_number, right_direction=True)  # This part could be changed
            f = self.gen_f_vehicle_dynamics(self.p_cntrld_list[j], model="kinematic_bicycle")
            self.add_dynamics_constraints_g(self.x_ctrld[j], self.u_ctrld[j], self.xd_ctrld[j], self.x0_cntrld[j], f,
                                            fd, self.dt)
            self.add_state_constraints_g(self.x_ctrld[j], self.u_ctrld[j], self.p_cntrld_list[j])

        # Compute Collision Avoidance ellipses using Minkowski sum
        self.pairwise_distances = []  # keep track of all the distances between ego and ado vehicles
        self.collision_cost = 0
        self.k_ca2 = 0.77  # TODO: HARDCODED:  when should this number change?

        self.top_wall_slack = cas.MX.sym('s_top', 1, N + 1)
        self.bottom_wall_slack = cas.MX.sym('s_bot', 1, N + 1)
        self.x_list += [self.top_wall_slack[:], self.bottom_wall_slack[:]]
        for k in range(self.top_wall_slack.shape[1]):
            self.add_bounded_constraint(0, self.top_wall_slack[0, k], None)
            self.add_bounded_constraint(0, self.bottom_wall_slack[0, k], None)

        self.top_wall_slack_c = [cas.MX.sym('sc_top_%02d' % i, 1, N + 1) for i in range(self.n_vehs_cntrld)]
        self.bottom_wall_slack_c = [cas.MX.sym('sc_bot_%02d' % i, 1, N + 1) for i in range(self.n_vehs_cntrld)]
        for i in range(len(self.top_wall_slack_c)):
            self.x_list += [self.top_wall_slack_c[i][:], self.bottom_wall_slack_c[i][:]]

        if "collision_avoidance_checking_distance" not in params:
            print("No collision avoidance checking distance")
            params["collision_avoidance_checking_distance"] = 400
        if "wall_CA" not in params:
            print("No default wall CA")
            params["wall_CA"] = True

        for ic in range(self.n_vehs_cntrld):
            for k in range(self.top_wall_slack_c[ic].shape[1]):
                self.add_bounded_constraint(0, self.top_wall_slack_c[ic][0, k], None)
                self.add_bounded_constraint(0, self.bottom_wall_slack_c[ic][0, k], None)

        for k in range(N + 1):
            # Compute response vehicles collision center points
            for i in range(self.n_other_vehicle):
                initial_displacement = self.x0_allother[i] - self.x0_ego
                initial_xy_distance = cas.sqrt(initial_displacement[0]**2 + initial_displacement[1]**2)
                within_collision_distance = (initial_xy_distance <= params["collision_avoidance_checking_distance"])

                dist = self.minkowski_ellipse_collision_distance(self.p_ego, self.p_other_vehicle_list[i],
                                                                 self.x_ego[0, k], self.x_ego[1, k], self.x_ego[2, k],
                                                                 self.x_other[i][0, k], self.x_other[i][1, k],
                                                                 self.x_other[i][2, k])

                self.pairwise_distances += [dist]
                # Only keep track if within_collision_distance == 1  (i.e. if outside, don't add cost or constraint)
                self.add_bounded_constraint(((1 - self.slack_i_jnc[i, k]) * within_collision_distance), dist, None)
                distance_clipped = cas.fmax(dist, 0.00001)  # This can be a smaller distance if we'd like
                self.collision_cost += (within_collision_distance * 1 /
                                        (distance_clipped - self.k_ca2)**self.k_CA_power)
            for j in range(self.n_vehs_cntrld):
                dist = self.minkowski_ellipse_collision_distance(self.p_ego, self.p_cntrld_list[j], self.x_ego[0, k],
                                                                 self.x_ego[1, k], self.x_ego[2, k], self.x_ctrld[j][0,
                                                                                                                     k],
                                                                 self.x_ctrld[j][1, k], self.x_ctrld[j][2, k])

                self.add_bounded_constraint(1 - self.slack_i_jc[j, k], dist, None)
                self.pairwise_distances += [dist]
                distance_clipped = cas.fmax(dist, 0.00001)
                self.collision_cost += (1 / (distance_clipped - self.k_ca2)**self.k_CA_power)

            if (params["wall_CA"] == 1):  # Add a collision cost related to distance from wall
                dist_btw_wall_bottom = self.x_ego[1, k] - (self.p_ego.min_y + self.p_ego.W / 2.0)
                dist_btw_wall_top = (self.p_ego.max_y - self.p_ego.W / 2.0) - self.x_ego[1, k]
                self.add_bounded_constraint(0 - self.bottom_wall_slack[0, k], dist_btw_wall_bottom, None)
                self.add_bounded_constraint(0 - self.top_wall_slack[0, k], dist_btw_wall_top, None)
                self.slack_cost += (self.top_wall_slack[0, k]**2 + self.bottom_wall_slack[0, k]**2)

        for ic in range(self.n_vehs_cntrld):
            for k in range(N + 1):
                # Genereate collision circles for cntrld vehicles and other car
                for j in range(self.n_other_vehicle):
                    initial_displacement = self.x0_allother[j] - self.x0_cntrld[ic]
                    initial_xy_distance = cas.sqrt(initial_displacement[0]**2 + initial_displacement[1]**2)
                    within_collision_distance = (initial_xy_distance <= params["collision_avoidance_checking_distance"])

                    dist = self.minkowski_ellipse_collision_distance(self.p_cntrld_list[ic],
                                                                     self.p_other_vehicle_list[j],
                                                                     self.x_ctrld[ic][0, k], self.x_ctrld[ic][1, k],
                                                                     self.x_ctrld[ic][2, k], self.x_other[j][0, k],
                                                                     self.x_other[j][1, k], self.x_other[j][2, k])

                    self.add_bounded_constraint(((1 - self.slack_ic_jnc[ic][j, k]) * within_collision_distance), dist,
                                                None)

                    distance_clipped = cas.fmax(dist, 0.0001)  # could be buffered if we'd like
                    self.collision_cost += (within_collision_distance * 1 /
                                            (distance_clipped - self.k_ca2)**self.k_CA_power)
                for j in range(self.n_vehs_cntrld):
                    if j <= ic:
                        self.add_equal_constraint(self.slack_ic_jc[ic][j, k], 0)
                    else:
                        initial_displacement = self.x0_cntrld[j] - self.x0_cntrld[ic]
                        initial_xy_distance = cas.sqrt(initial_displacement[0]**2 + initial_displacement[1]**2)
                        within_collision_distance = (initial_xy_distance <=
                                                     params["collision_avoidance_checking_distance"])

                        dist = self.minkowski_ellipse_collision_distance(self.p_cntrld_list[ic], self.p_cntrld_list[j],
                                                                         self.x_ctrld[ic][0, k], self.x_ctrld[ic][1, k],
                                                                         self.x_ctrld[ic][2, k], self.x_ctrld[j][0, k],
                                                                         self.x_ctrld[j][1, k], self.x_ctrld[j][2, k])

                        self.add_bounded_constraint(((1 - self.slack_ic_jc[ic][j, k]) * within_collision_distance),
                                                    dist, None)

                        distance_clipped = cas.fmax(dist, 0.0001)  # could be buffered if we'd like
                        self.collision_cost += (within_collision_distance * 1 /
                                                (distance_clipped - self.k_ca2)**self.k_CA_power)

                if params["wall_CA"] == 1:  # Compute CA cost of ambulance and wall
                    dist_btw_wall_bottom = self.x_ctrld[ic][1, k] - (self.p_cntrld_list[ic].min_y +
                                                                     self.p_cntrld_list[ic].W / 2.0)
                    dist_btw_wall_top = (self.p_cntrld_list[ic].max_y -
                                         self.p_cntrld_list[ic].W / 2.0) - self.x_ctrld[ic][1, k]

                    self.add_bounded_constraint((0 - self.bottom_wall_slack_c[ic][0, k]), dist_btw_wall_bottom, None)
                    self.add_bounded_constraint((0 - self.top_wall_slack_c[ic][0, k]), dist_btw_wall_top, None)

                    self.slack_cost += (self.top_wall_slack_c[ic][0, k]**2 + self.bottom_wall_slack_c[ic][0, k]**2)

        # Total optimization costs
        self.total_svo_cost = (self.response_svo_cost + self.other_svo_cost + self.k_slack * self.slack_cost +
                               self.k_CA * self.collision_cost)
        # self.opti.minimize(self.total_svo_cost)
        self.f = self.total_svo_cost
        # print(self.lbg_list)
        print(len(self.lbg_list))
        print(len(self.ubg_list))
        print(len(self.g_list))
        tall_g_list = []
        tall_ubg_list = []
        tall_lbg_list = []
        for ix in range(len(self.g_list)):
            if self.g_list[ix].shape[1] != 1:
                tall_g_list += [self.g_list[ix][:]]
                tall_lbg_list += [self.lbg_list[ix][:]]
                tall_ubg_list += [self.ubg_list[ix][:]]
            else:
                tall_g_list += [self.g_list[ix]]
                tall_lbg_list += [self.lbg_list[ix]]
                tall_ubg_list += [self.ubg_list[ix]]

        self.g_list = cas.vertcat(*tall_g_list)
        self.lbg_list = cas.vertcat(*tall_lbg_list)
        self.ubg_list = cas.vertcat(*tall_ubg_list)

        #### Collect all variables into one large vector

        ### Ego: Slack variables
        tall_x_list = []
        for ix in range(len(self.x_list)):
            if self.x_list[ix].shape[1] != 1:
                tall_x_list += [cas.reshape(self.x_list[ix], self.x_list[ix].shape[0] * self.x_list[ix].shape[1], 1)]
            else:
                tall_x_list += [self.x_list[ix]]
            print(tall_x_list[ix])
        self.x_list = cas.vertcat(*tall_x_list)

        tall_p_list = []
        for ix in range(len(self.p_list)):
            if self.p_list[ix].shape[1] != 1:
                tall_p_list += [cas.reshape(self.p_list[ix], self.p_list[ix].shape[0] * self.p_list[ix].shape[1], 1)]
            else:
                tall_p_list += [self.p_list[ix]]
        self.p_list = cas.vertcat(*tall_p_list)
        # Set the solver conditions
        if ipopt_params is None:
            ipopt_params = {}

        prob = {'f': self.f, 'g': self.g_list, 'x': self.x_list, 'p': self.p_list}
        solver = cas.nlpsol('solver', 'ipopt', prob)

    # def generate_optimization(self,
    #                           N: int,
    #                           x0: np.array,
    #                           x0_other_ctrl: List[np.array],
    #                           x0_other: List[np.array],
    #                           params: dict = None,
    #                           ipopt_params: dict = None):
    #     ''' Setup an optimization for the response vehicle, shared control vehicles, and surrounding vehicles
    #         Input:
    #             N:  number of control points in the optimization (N+1 state points)
    #             x0: response vehicle's initial position [np.array(n_state)]
    #             x0_other_ctrl:  list of shared control vehicle initial positions List[np.array(n_state)]
    #             x0_other:  list of non-responding vehicle initial positions List[np.array(n_state)]
    #             params:  simulation parameters
    #             ipopt_params:  paramaters for the optimizatio solver
    #     '''

    def solve(self, uctrld_warm, uother, solve_amb=False):
        for ic in range(self.n_vehs_cntrld):
            self.opti.set_initial(self.u_ctrld[ic], uctrld_warm[ic])
        for i in range(self.n_other_vehicle):
            self.opti.set_value(self.u_other[i], uother[i])
        self.solution = self.opti.solve()

    def add_equal_constraint(self, lhs, rhs):
        self.g_list += [lhs - rhs]

        self.lbg_list += [np.zeros(shape=lhs.shape)]
        self.ubg_list += [np.zeros(shape=lhs.shape)]

    def add_bounded_constraint(self, lbg, g, ubg):
        if lbg is not None:
            self.g_list += [g - lbg]
            self.lbg_list += [cas.MX.zeros(g.shape[0], g.shape[1])]
            self.ubg_list += [cas.MX.inf(g.shape[0], g.shape[1])]
            cas.MX.inf()

        if ubg is not None:
            self.g_list += [g - ubg]
            self.lbg_list += [-cas.MX.inf(g.shape[0], g.shape[1])]
            self.ubg_list += [cas.MX.zeros(g.shape[0], g.shape[1])]

    def add_state_constraints_g(self, X, U, ego_veh):
        ''' Construct vehicle specific constraints that only rely on
        the ego vehicle's own state '''
        for k in range(X.shape[1]):
            self.add_bounded_constraint(ego_veh.min_y + ego_veh.W / 2.0, X[1, k], ego_veh.max_y - ego_veh.W / 2.0)
            self.add_bounded_constraint(-np.pi / 2, X[2, k], np.pi / 2)  #no crazy angle
            self.add_bounded_constraint(ego_veh.min_v, X[4, k], ego_veh.max_v)

        # constraints on the control inputs
        for k in range(U.shape[1]):
            self.add_bounded_constraint(-ego_veh.max_delta_u, U[0, k], ego_veh.max_delta_u)
            self.add_bounded_constraint(ego_veh.min_v_u, U[1, k], ego_veh.max_v_u)  # 0-60 around 4 m/s^2

    def add_dynamics_constraints_g(self, X, U, X_desired, x0, f, fd, dt: float):
        N = U.shape[1]
        for k in range(N):
            self.add_equal_constraint(X[:, k + 1], self.F_kutta(f, X[:, k], U[:, k], dt))

        for k in range(N + 1):
            self.add_equal_constraint(X_desired[:, k], fd(X[-1, k]))

        self.add_equal_constraint(X[:, 0], x0)

    def generate_veh_costs(self, X, U, X_desired, p_car):
        ''' Compute the all the vehicle specific costs corresponding to performance
            of the vehicle as it traverse a desired  trajectory
        '''
        u_delta_cost = cas.sumsqr(U[0, :])
        u_v_cost = cas.sumsqr(U[1, :])

        lon_cost = self.generate_longitudinal_cost(X, X_desired)

        phi_error_cost = cas.sumsqr(X_desired[2, :] - X[2, :])
        X_ONLY = False
        if X_ONLY:
            s_cost = cas.sumsqr(X[0, -1])
            lat_cost = cas.sumsqr(X[1, :])
        else:
            lat_cost = self.generate_lateral_cost(X, X_desired)
            s_cost = cas.sumsqr(X[5, -1])
        final_costs = 0  # for now I've diactivated this
        v_cost = cas.sumsqr(X[4, :])
        phidot_cost = self.generate_phidot_cost(X, p_car)  #this function assumes a kinematic bicycle model
        N = U.shape[1]
        change_u_delta = cas.sumsqr(U[0, 1:N - 1] - U[0, 0:N - 2])
        change_u_v = cas.sumsqr(U[1, 1:N - 1] - U[1, 0:N - 2])
        x_cost = cas.sumsqr(X[0, :])
        x_dot_cost = cas.sumsqr(X[4, :] * cas.cos(X[2, :]))

        all_costs = [
            p_car.k_u_delta * u_delta_cost, p_car.k_u_v * u_v_cost, p_car.k_lat * lat_cost, p_car.k_lon * lon_cost,
            p_car.k_phi_error * phi_error_cost, p_car.k_phi_dot * phidot_cost, p_car.k_s * s_cost, p_car.k_v * v_cost,
            p_car.k_change_u_v * change_u_v, p_car.k_change_u_delta * change_u_delta, p_car.k_final * final_costs,
            p_car.k_x * x_cost, p_car.k_x_dot * x_dot_cost
        ]

        all_costs = np.array(all_costs)
        total_cost = np.sum(all_costs)
        return total_cost, all_costs

    def add_dynamics_constraints(self, opti, X, U, X_desired, x0, f, fd, dt: float):
        ''' Construct any dynamic constraints based on the kinematic bicycle model
            A Run-Kutta approximation is used to discretize the vehicle dynamics.
        '''
        # State Dynamics
        N = U.shape[1]
        for k in range(N):
            opti.subject_to(X[:, k + 1] == self.F_kutta(f, X[:, k], U[:, k], dt))

        for k in range(N + 1):
            opti.subject_to(X_desired[:, k] == fd(X[-1, k]))  #This should be the trajectory dynamic constraint
        opti.subject_to(X[:, 0] == x0)

    def gen_f_desired_lane(self, world, lane_number, right_direction=True):
        ''' Generates a function the vehicle progression along a desired trajectory '''
        if right_direction == False:
            raise Exception("Haven't implemented left lanes")
        self.desired_lane = lane_number
        s = cas.MX.sym('s')
        xd = s
        yd = world.get_lane_centerline_y(lane_number, right_direction)
        phid = 0
        des_traj = cas.vertcat(xd, yd, phid)
        fd = cas.Function('fd', [s], [des_traj], ['s'], ['des_traj'])

        return fd

    def gen_f_vehicle_dynamics(self, p_veh, model: str = "kinematic_bicycle"):
        ''' Vehicle dynamics using a kinematic bike model.
        Returns a CASADI function that outputs xdot.
        '''
        if model == "kinematic_bicycle":
            X = cas.MX.sym('X')  # horizontal distance
            Y = cas.MX.sym('Y')  # vertical distance
            Phi = cas.MX.sym('Phi')  # orientation (angle from x-axis)
            Delta = cas.MX.sym('Delta')  # steering angle
            V = cas.MX.sym('V')  # speed
            s = cas.MX.sym('s')  # progression along contour

            delta_u = cas.MX.sym('delta_u')  # change in steering angle
            v_u = cas.MX.sym('v_u')  # change in velocity
            x = cas.vertcat(X, Y, Phi, Delta, V, s)
            u = cas.vertcat(delta_u, v_u)

            ode = cas.vertcat(V * cas.cos(Phi), V * cas.sin(Phi), V * cas.tan(Delta) / p_veh.L, delta_u, v_u, V)

            f = cas.Function('f', [x, u], [ode], ['x', 'u'], ['ode'])
        else:
            raise Exception("Have not implemented non-kinematic bicycle: %s" % model)
        return f

    def F_kutta(self, f, x_k, u_k, dt: float):
        ''' Run-Kutta Approximation of a continuous dynamics f.
        Returns x_{k+1}= f(x_k, u_k) with timestep dt
        '''

        k1 = f(x_k, u_k)
        k2 = f(x_k + dt / 2 * k1, u_k)
        k3 = f(x_k + dt / 2 * k2, u_k)
        k4 = f(x_k + dt * k3, u_k)
        x_next = x_k + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return x_next

    def generate_lateral_cost(self, X, X_desired):
        ''' Lateral costs based on distance traversed along desired trajectory'''

        lateral_cost = np.sum([(-cas.sin(X_desired[2, k]) * (X[0, k] - X_desired[0, k]) + cas.cos(X_desired[2, k]) *
                                (X[1, k] - X_desired[1, k]))**2 for k in range(X.shape[1])])

        return lateral_cost

    def generate_longitudinal_cost(self, X, X_desired):
        ''' Longitudinal costs based on distance traversed along desired trajectory'''

        longitudinal_cost = np.sum([(cas.cos(X_desired[2, k]) * (X[0, k] - X_desired[0, k]) + cas.sin(X_desired[2, k]) *
                                     (X[1, k] - X_desired[1, k]))**2 for k in range(X.shape[1])])

        return longitudinal_cost

    def generate_phidot_cost(self, X, p_veh):
        ''' Yaw rate cost computed by the dynamics of the vehicle '''
        phid = X[4, :] * cas.tan(X[3, :]) / p_veh.L
        phid_cost = cas.sumsqr(phid)

        return phid_cost

    def get_opti_parameters(self):
        ''' Return the parameters of our MPC'''
        ego_veh_params = self.p_ego.get_opti_params()
        all_cntrld_params = [param for p_veh in self.p_cntrld_list for param in p_veh.get_opti_params()]
        all_other_params = [param for p_veh in self.p_other_vehicle_list for param in p_veh.get_opti_params()]

        print(*ego_veh_params)
        print(*all_cntrld_params)
        print(*all_cntrld_params)
        all_params = cas.vertcat(
            *ego_veh_params,
            *all_cntrld_params,
            *all_other_params,
            # self.x0_ego,
            # self.x0_cntrld,
            # self.x0_allother,
        )
        return all_params

    def get_bestresponse_solution(self):
        x1, u1, x1_des, = (self.solution.value(self.x_ego), self.solution.value(self.u_ego),
                           self.solution.value(self.xd_ego))

        return x1, u1, x1_des

    def get_solution(self):
        x1, u1, x1_des, = (self.solution.value(self.x_ego), self.solution.value(self.u_ego),
                           self.solution.value(self.xd_ego))

        cntrld_x = [self.solution.value(self.x_ctrld[i]) for i in range(self.n_vehs_cntrld)]
        cntrld_u = [self.solution.value(self.u_ctrld[i]) for i in range(self.n_vehs_cntrld)]
        cntrld_des = [self.solution.value(self.xd_ctrld[i]) for i in range(self.n_vehs_cntrld)]

        other_x = [self.solution.value(self.x_other[i]) for i in range(self.n_other_vehicle)]
        other_u = [self.solution.value(self.u_other[i]) for i in range(self.n_other_vehicle)]
        other_des = [self.solution.value(self.xd_other[i]) for i in range(self.n_other_vehicle)]

        return x1, u1, x1_des, cntrld_x, cntrld_u, cntrld_des, other_x, other_u, other_des

    def generate_collision_ellipse(self, x_e, y_e, x_o, y_o, phi_o, alpha_o, beta_o):
        """ alpha_o:  major axis of length"""
        dx = x_e - x_o
        dy = y_e - y_o

        R_o = cas.vertcat(
            cas.horzcat(cas.cos(phi_o), cas.sin(phi_o)),
            cas.horzcat(-cas.sin(phi_o), cas.cos(phi_o)),
        )
        M = cas.vertcat(cas.horzcat(1 / alpha_o**2, 0), cas.horzcat(0, 1 / beta_o**2))
        dX = cas.vertcat(dx, dy)
        prod = cas.mtimes([dX.T, R_o.T, M, R_o, dX])

        M_smaller = cas.vertcat(cas.horzcat(1 / (0.5 * alpha_o)**2, 0), cas.horzcat(0, 1 / (0.5 * beta_o)**2))
        dist_prod = cas.mtimes([dX.T, R_o.T, M_smaller, R_o, dX])

        return dist_prod, prod

    # def debug_callback(self, i, plot_range=[], file_name=False):
    #     xothers_plot = [self.opti.debug.value(xo) for xo in self.x_other]
    #     xamb_plot = self.opti.debug.value(self.x_ego)
    #     if self.ambMPC:
    #         xothers_plot += [self.opti.debug.value(self.xamb_opt)]

    #     if file_name:
    #         uamb_plot = self.opti.debug.value(self.u_ego)
    #         uothers_plot = [self.opti.debug.value(xo) for xo in self.u_other]
    #         save_state(file_name, xamb_plot, uamb_plot, None, xothers_plot, uothers_plot, None)

    #     if len(plot_range) > 0:

    #         plot_cars(self.world, self.ego_veh, xamb_plot, xothers_plot, None, "ellipse", False, 0)
    #         plt.show()

    #         plt.plot(xamb_plot[4, :], "--")
    #         plt.plot(xamb_plot[4, :] * np.cos(xamb_plot[2, :]))
    #         plt.ylabel("Velocity / Vx")
    #         plt.hlines(35 * 0.447, 0, xamb_plot.shape[1])
    #         plt.show()
    #     print("%d Total Cost %.03f J_i %.03f,  J_j %.03f, Slack %.03f, CA  %.03f" % (
    #         i,
    #         self.opti.debug.value(self.total_svo_cost),
    #         self.opti.debug.value(self.response_svo_cost),
    #         self.opti.debug.value(self.other_svo_cost),
    #         self.opti.debug.value(self.k_slack * self.slack_cost),
    #         self.opti.debug.value(self.k_CA * self.collision_cost),
    #     ))
    #     for i in range(len(self.response_costs_list)):
    #         print(" %.04f : %s" % (
    #             self.opti.debug.value(self.response_costs_list[i]),
    #             self.response_cost_titles[i],
    #         ))

    def plot_collision_slack_cost(self):
        """Make a function that plots the contours for collision and slack
        """
        x_e = np.array([0, 100])  # these need to be corrected from world
        y_e = np.array([-5, 5])  # these need to be corrected from world
        X, Y = np.meshgrid(x_e, y_e)
        for i in range(len(self.x_other)):
            # This is a test pose, we need to get it from our optimization
            x_o, y_o, phi_o, alpha_o, beta_o = (2, 1, 0, 1, 1)
            R_o = np.array([[np.cos(phi_o), np.sin(phi_o)], [-np.sin(phi_o), np.cos(phi_o)]])
            M = np.array([[1 / alpha_o**2, 0], [0, 1 / beta_o**2]])
            dx = X - x_o
            dy = Y - y_o
            dX = np.stack((dx, dy), axis=2)
            prod = cas.mtimes([dX.T, R_o.T, M, R_o, dX])

    def generate_stopping_constraint_ttc(self, x_opt, xcntrld_opt, xothers_opt, min_time_to_collision=3.0, k_ttc=0.0):
        """ Add a velocity constraint on the ego vehicle so it doesn't go too fast behind a lead vehicle.
            Constrains the vehicle so that the time to collision is always less than time_to_collision seconds
            assuming that both ego and ado vehicle mantain constant velocity
        """

        N = x_opt.shape[1]
        car_length = self.ego_veh.L
        car_width = self.ego_veh.W
        time_to_collision_cost = 0.0
        for k in range(N):
            x_ego = x_opt[0, k]
            y_ego = x_opt[1, k]
            phi_ego = x_opt[2, k]
            v_ego = x_opt[4, k]
            v_ego_components = (v_ego * cas.cos(phi_ego), v_ego * cas.sin(phi_ego))

            for j in range(len(xcntrld_opt)):
                ## Safety constraint between ego + cntrld vehicles
                x_amb = xcntrld_opt[j][0, k]
                y_amb = xcntrld_opt[j][1, k]
                phi_amb = xcntrld_opt[j][2, k]
                v_amb = xcntrld_opt[j][4, k] - 2 * self.ego_veh.max_v_u

                v_amb_components = (v_amb * cas.cos(phi_amb), v_amb * cas.sin(phi_amb))

                dxegoamb = (x_amb - x_ego) - car_length
                dyegoamb = (y_amb - y_ego) - car_width
                dot_product = (v_ego_components[0] - v_amb_components[0]) * dxegoamb + (v_ego_components[1] -
                                                                                        v_amb_components[1]) * dyegoamb
                positive_dot_product = cas.max(
                    dot_product,
                    0)  # if negative, negative_dot_product will be 0 so time_to_collision is very very large

                time_to_collision = (dxegoamb**2 + dyegoamb**2) / (dot_product + 0.000001)
                time_to_collision_cost += k_ttc * 1 / time_to_collision**2

                self.opti.subject_to(dot_product <= (dxegoamb**2 + dyegoamb**2) / (0.000001 + min_time_to_collision))

            for j in range(len(xothers_opt)):
                x_j = xothers_opt[j][0, k]
                y_j = xothers_opt[j][1, k]
                phi_j = xothers_opt[j][2, k]
                v_j = xothers_opt[j][4, k] - 2 * self.ego_veh.max_v_u

                #### Add constraint between ego and j
                dxego = (x_j - x_ego) - car_length
                dyego = (y_j - y_ego) - car_width

                v_j_components = (v_j * cas.cos(phi_j), v_j * cas.sin(phi_j))
                dot_product = (v_ego_components[0] - v_j_components[0]) * dxego + (v_ego_components[1] -
                                                                                   v_j_components[1]) * dyego
                positive_dot_product = cas.max(
                    dot_product,
                    0)  # if negative, negative_dot_product will be 0 so time_to_collision is very very large
                time_to_collision = (dxego**2 + dyego**2) / (positive_dot_product + 0.000001)
                time_to_collision_cost += k_ttc * 1 / time_to_collision**2
                self.opti.subject_to(dot_product <= (dxego**2 + dyego**2) / (0.000001 + min_time_to_collision))

                #### Add constraint betweem cntrld vehicles and j
                add_constraint_for_cntrld = False
                if add_constraint_for_cntrld:
                    for jc in range(len(xcntrld_opt)):
                        x_ctrl = xcntrld_opt[jc][0, k]
                        y_ctrl = xcntrld_opt[jc][1, k]
                        phi_ctrl = xcntrld_opt[jc][2, k]
                        v_ctrl = xcntrld_opt[jc][4, k]
                        v_ctrl_components = (
                            v_ctrl * cas.cos(phi_ctrl),
                            v_ctrl * cas.sin(phi_ctrl),
                        )

                        dxctrl = (x_j - x_ctrl) - car_length
                        dyctrl = (y_j - y_ctrl) - car_width

                        dot_product = (v_ctrl_components[0] - v_j_components[0]) * dxctrl + (v_ctrl_components[1] -
                                                                                             v_j_components[1]) * dyctrl
                        positive_dot_product = cas.max(
                            dot_product,
                            0)  # if negative, negative_dot_product will be 0 so time_to_collision is very very large
                        time_to_collision = (dxctrl**2 + dyctrl**2) / (positive_dot_product + 0.000001)
                        time_to_collision_cost += k_ttc * 1 / time_to_collision**2

                        self.opti.subject_to(dot_product <= (dxctrl**2 + dyctrl**2) /
                                             (0.00000001 + min_time_to_collision))

    def generate_stopping_constraint(self, x_opt, xamb_opt, xothers_opt, solve_amb, safety_buffer=0.50):
        """ Add a velocity constraint on the ego vehicle so it doesn't go too fast behind a lead vehicle.
        Constrains ego velocity so that ego vehicle can brake (at u_v_max) to the same velocity of the lead vehicle
        within the distance to the lead vehicle.  We add a buffer distance to ensure doesn't collide.

        safety_buffer:  Shortened distance for the braking
        """
        u_v_maxbraking = (self.ego_veh.min_v_u)  # this is max change in V, in discrete steps
        a_maxbraking = u_v_maxbraking / self.ego_veh.dt  ### find max acceleration
        N = x_opt.shape[1]
        car_length = self.ego_veh.L
        car_width = self.ego_veh.W
        x_amb, y_amb, v_amb = None, None, None
        for k in range(N):
            x_ego = x_opt[0, k]
            y_ego = x_opt[1, k]
            v_ego = x_opt[4, k]

            ### Add constraint between ego vehicle and ambulance
            if xamb_opt is not None:
                x_amb = xamb_opt[0, k]
                y_amb = xamb_opt[1, k]
                v_amb = xamb_opt[4, k]
                dxegoamb = (x_amb - x_ego) - car_length
                dyegoamb = (y_amb - y_ego) - car_width
                dist_egoamb = cas.sqrt(dxegoamb**2 + dyegoamb**2)
                egobehind_amb = cas.fmax(-dxegoamb, 0)  ## 0 if ego is beind, else should be >0

                v_max_constraint = cas.fmax(
                    (v_amb**2 - 2 * a_maxbraking * (dist_egoamb - safety_buffer)),
                    999 * egobehind_amb,
                )
                self.opti.subject_to(v_ego**2 <= v_max_constraint)

            ### Add constraints between ego vehicle and other vehicles
            for j in range(len(xothers_opt)):
                x_j = xothers_opt[j][0, k]
                y_j = xothers_opt[j][1, k]
                v_j = xothers_opt[j][4, k]

                #### Add constraint between ego and j
                dxego = (x_j - x_ego) - car_length
                dyego = (y_j - y_ego) - car_width

                dist_ego = cas.sqrt(dxego**2 + dyego**2)
                ego_behind = cas.fmax(-dxego, 0)  ###how many meters behind or 0 if ahead/same
                v_max_constraint = cas.fmax(
                    v_j**2 - 2 * a_maxbraking * (dist_ego - safety_buffer),
                    999 * ego_behind,
                )
                self.opti.subject_to(v_ego**2 <= v_max_constraint)

                #### Add constraint betweem amb and j
                if xamb_opt is not None and solve_amb:
                    dxamb = (x_j - x_amb) - car_length
                    dyamb = (y_j - y_amb) - car_width
                    dist_amb = cas.sqrt(dxamb**2 + dyamb**2)

                    amb_behind = cas.fmax(-dxamb, 0)  ## 0 if ambulance is beind, else should be >0
                    v_max_constraint = cas.fmax(
                        (v_j**2 - 2 * a_maxbraking * (dist_amb - safety_buffer)),
                        999 * amb_behind,
                    )
                    self.opti.subject_to(v_amb**2 <= v_max_constraint)

    def minkowski_ellipse_collision_distance(self, ego_veh, ado_veh, x_ego, y_ego, phi_ego, x_ado, y_ado, phi_ado):
        """ Return the squared distance between the ego vehicle and ado vehicle
        for collision avoidance 
        Halder, A. (2019). On the Parameterized Computation of Minimum Volume Outer Ellipsoid of Minkowski Sum of Ellipsoids."
        """
        # if not numpy:
        shape_matrix_ego = cas.vertcat(cas.horzcat(ego_veh.ax, 0.0), cas.horzcat(0.0, ego_veh.by))
        shape_matrix_ado = cas.vertcat(cas.horzcat(ado_veh.ax, 0.0), cas.horzcat(0.0, ado_veh.by))

        rotation_matrix_ego = cas.vertcat(
            cas.horzcat(cas.cos(phi_ego), -cas.sin(phi_ego)),
            cas.horzcat(cas.sin(phi_ego), cas.cos(phi_ego)),
        )
        rotation_matrix_ado = cas.vertcat(
            cas.horzcat(cas.cos(phi_ado), -cas.sin(phi_ado)),
            cas.horzcat(cas.sin(phi_ado), cas.cos(phi_ado)),
        )

        # Compute the Minkowski Sum
        M_e_curr = cas.mtimes([rotation_matrix_ego, shape_matrix_ego])
        Q1 = cas.mtimes([M_e_curr, cas.transpose(M_e_curr)])

        M_a_curr = cas.mtimes([rotation_matrix_ado, shape_matrix_ado])
        Q2 = cas.mtimes([M_a_curr, cas.transpose(M_a_curr)])

        beta = cas.sqrt(cas.trace(Q1) / cas.trace(Q2))
        Q_minkowski = (1 + 1.0 / beta) * Q1 + (1.0 + beta) * Q2

        X_ego = cas.vertcat(x_ego, y_ego)
        X_ado = cas.vertcat(x_ado, y_ado)
        dist_squared = cas.mtimes([cas.transpose(X_ado - X_ego), cas.inv(Q_minkowski), (X_ado - X_ego)])

        return dist_squared


def load_state(file_name, n_others, ignore_des=False):
    xamb = np.load(file_name + "xamb.npy", allow_pickle=False)
    uamb = np.load(file_name + "uamb.npy", allow_pickle=False)
    if not ignore_des:
        xamb_des = np.load(file_name + "xamb_des.npy", allow_pickle=False)
    else:
        xamb_des = None

    xothers, uothers, xothers_des = [], [], []
    for i in range(n_others):
        x = np.load(file_name + "x%0d.npy" % i, allow_pickle=False)
        u = np.load(file_name + "u%0d.npy" % i, allow_pickle=False)
        xothers += [x]
        uothers += [u]
        if not ignore_des:
            x_des = np.load(file_name + "x_des%0d.npy" % i, allow_pickle=False)
            xothers_des += [x_des]

    return xamb, uamb, xamb_des, xothers, uothers, xothers_des


def save_state(file_name, xamb, uamb, xamb_des, xothers, uothers, xothers_des, end_t=None):
    if end_t is None:
        end_t = xamb.shape[1]

    np.save(file_name + "xamb", xamb[:, :end_t], allow_pickle=False)
    np.save(file_name + "uamb", uamb[:, :end_t], allow_pickle=False)
    if xamb_des is not None:
        np.save(file_name + "xamb_des", xamb_des[:, :end_t], allow_pickle=False)

    for i in range(len(xothers)):
        x, u = xothers[i], uothers[i]
        np.save(file_name + "x%0d" % i, x[:, :end_t], allow_pickle=False)
        np.save(file_name + "u%0d" % i, u[:, :end_t], allow_pickle=False)
        if xothers_des is not None:
            x_des = xothers_des[i]
            np.save(file_name + "x_des%0d" % i, x_des[:, :end_t], allow_pickle=False)

    return file_name


def save_costs(file_name, ibr):
    ''' Get the value for each cost variable '''
    car1_costs_list = np.array([ibr.opti.debug.value(cost) for cost in ibr.car1_costs_list])
    amb_costs_list = np.array([ibr.opti.debug.value(cost) for cost in ibr.amb_costs_list])
    svo_cost = ibr.opti.debug.value(ibr.response_svo_cost)
    other_svo_cost = ibr.opti.debug.value(ibr.other_svo_cost)
    total_svo_cost = ibr.opti.debug.value(ibr.total_svo_cost)

    np.save(file_name + "car1_costs_list", car1_costs_list, allow_pickle=False)
    np.save(file_name + "amb_costs_list", amb_costs_list, allow_pickle=False)
    np.save(file_name + "svo_cost", svo_cost, allow_pickle=False)
    np.save(file_name + "other_svo_cost", other_svo_cost, allow_pickle=False)
    np.save(file_name + "total_svo_cost", total_svo_cost, allow_pickle=False)
    return file_name


def load_costs(file_name):
    ''' Load all the cost values '''
    ''' This is rarely used, maybe we should remove it '''
    car1_costs_list = np.load(file_name + "car1_costs_list.npy", allow_pickle=False)
    amb_costs_list = np.load(file_name + "amb_costs_list.npy", allow_pickle=False)
    svo_cost = np.load(file_name + "svo_cost.npy", allow_pickle=False)
    other_svo_cost = np.load(file_name + "other_svo_cost.npy", allow_pickle=False)
    total_svo_cost = np.load(file_name + "total_svo_cost.npy", allow_pickle=False)

    return car1_costs_list, amb_costs_list, svo_cost, other_svo_cost, total_svo_cost


def load_costs_int(i):
    '''TODO: This code is rarely/not used, perhaps it should be removed'''

    car1_costs_list = np.load("%03dcar1_costs_list.npy" % i, allow_pickle=False)
    amb_costs_list = np.load("%03damb_costs_list.npy" % i, allow_pickle=False)
    svo_cost = np.load("%03dsvo_cost.npy" % i, allow_pickle=False)
    other_svo_cost = np.load("%03dother_svo_cost.npy" % i, allow_pickle=False)
    total_svo_cost = np.load("%03dtotal_svo_cost.npy" % i, allow_pickle=False)

    return car1_costs_list, amb_costs_list, svo_cost, other_svo_cost, total_svo_cost
