import numpy as np
import casadi as cas
import matplotlib.pyplot as plt
from typing import List

from src.traffic_world import TrafficWorld
from src.utils.plotting.car_plotting import plot_cars
# from src.warm_starts import generate_warm_x, centerline_following, generate_warm_u, generate_warm_starts


class MultiMPC(object):
    ''' Optimization class contains all the object related to a best response optimization.
        We assume that each vehicle has a dynamics model (f), costs function, and constraints.

       The optimization currently solves assuming a Kinmeatic Bicycle Dynamics Model, assuming
       control inputs steering + acceleration.  
    '''
    def __init__(self,
                 response_vehicle,
                 cntrld_vehicles,
                 other_vehicle_list,
                 world: TrafficWorld,
                 solver_params: dict = None):

        self.ego_veh = response_vehicle
        self.vehs_ctrld = cntrld_vehicles
        self.other_vehicle_list = other_vehicle_list

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

        self.opti = cas.Opti()

    def generate_optimization(self,
                              N: int,
                              x0: np.array,
                              x0_other_ctrl: List[np.array],
                              x0_other: List[np.array],
                              params: dict = None,
                              ipopt_params: dict = None):
        ''' Setup an optimization for the response vehicle, shared control vehicles, and surrounding vehicles
            Input: 
                N:  number of control points in the optimization (N+1 state points)
                x0: response vehicle's initial position [np.array(n_state)]
                x0_other_ctrl:  list of shared control vehicle initial positions List[np.array(n_state)]
                x0_other:  list of non-responding vehicle initial positions List[np.array(n_state)]
                params:  simulation parameters
                ipopt_params:  paramaters for the optimizatio solver
        '''

        n_state, n_ctrl, n_desired = 6, 2, 3

        # Response (planning) Vehicle Variables
        self.x_ego = self.opti.variable(n_state, N + 1)
        self.u_ego = self.opti.variable(n_ctrl, N)
        self.xd_ego = self.opti.variable(n_desired, N + 1)
        p_ego = self.opti.parameter(n_state, 1)

        # Cntrld Vehicles become variables in the optimization
        self.x_ctrld = [self.opti.variable(n_state, N + 1) for i in self.vehs_ctrld]
        self.u_ctrld = [self.opti.variable(n_ctrl, N) for i in self.vehs_ctrld]
        self.xd_ctrld = [self.opti.variable(n_desired, N + 1) for i in self.vehs_ctrld]
        self.p_cntrld = [self.opti.parameter(n_state, 1) for i in self.vehs_ctrld]

        # Variables of surrounding vehicles assumed fixed as parameters for computation
        self.x_other = [self.opti.parameter(n_state, N + 1) for i in self.other_vehicle_list]
        self.u_other = [self.opti.parameter(n_ctrl, N) for i in self.other_vehicle_list]
        self.xd_other = [self.opti.parameter(3, N + 1) for i in self.other_vehicle_list]
        self.allother_p = [self.opti.parameter(n_state, 1) for i in self.other_vehicle_list]

        # Generate costs for each vehicle
        self.ego_veh.generate_costs(self.x_ego, self.u_ego, self.xd_ego)
        self.response_costs, self.response_costs_list, self.response_cost_titles = self.ego_veh.total_cost()

        if params is None:
            params = {}

        if "collision_avoidance_checking_distance" not in params:
            print("No collision avoidance checking distance")
            params["collision_avoidance_checking_distance"] = 400
        if "wall_CA" not in params:
            print("No default wall CA")
            params["wall_CA"] = True

        self.all_other_costs = []
        # Generate SVO cost for other (non-cntrld) vehicles
        for idx in range(len(self.other_vehicle_list)):
            svo_ij = self.ego_veh.get_theta_ij(self.other_vehicle_list[idx].agent_id)
            if svo_ij > 0:
                self.other_vehicle_list[idx].generate_costs(self.x_other[idx], self.u_other[idx], self.xd_other[idx])
                nonresponse_cost, _, _ = self.other_vehicle_list[idx].total_cost()
                self.all_other_costs += [np.sin(svo_ij) * nonresponse_cost]

        # SVO cost for Other (ctrld) Vehicles
        for idx in range(len(self.vehs_ctrld)):
            svo_ij = self.ego_veh.get_theta_ij(self.vehs_ctrld[idx].agent_id)
            if svo_ij > 0:
                self.vehs_ctrld[idx].generate_costs(
                    self.x_ctrld[idx],
                    self.u_ctrld[idx],
                    self.xd_ctrld[idx],
                )
                nonresponse_cost, _, _ = self.vehs_ctrld[idx].total_cost()
                self.all_other_costs += [np.sin(svo_ij) * nonresponse_cost]

        # Generate Slack Variables used as part of collision avoidance
        self.slack_cost = 0
        if len(self.other_vehicle_list) > 0:
            self.slack_i_jnc = self.opti.variable(len(self.other_vehicle_list), N + 1)
            self.slack_ic_jnc = [
                self.opti.variable(len(self.other_vehicle_list), N + 1) for ic in range(len(self.vehs_ctrld))
            ]

            self.opti.subject_to(cas.vec(self.slack_i_jnc) >= 0)
            for slack_var in self.slack_ic_jnc:
                self.opti.subject_to(cas.vec(slack_var) >= 0)

            for j in range(self.slack_i_jnc.shape[0]):
                for t in range(self.slack_i_jnc.shape[1]):
                    self.slack_cost += self.slack_i_jnc[j, t]**2
            for ic in range(len(self.vehs_ctrld)):
                for jnc in range(self.slack_ic_jnc[ic].shape[0]):
                    for t in range(self.slack_ic_jnc[ic].shape[1]):
                        self.slack_cost += self.slack_ic_jnc[ic][jnc, t]**2
        else:
            self.slack_i_jnc = 0
            self.slack_ic_jnc = []

        # Slack variables related to cntrld vehicles
        if len(self.vehs_ctrld) > 0:
            self.slack_i_jc = self.opti.variable(len(self.vehs_ctrld), N + 1)
            self.opti.subject_to(cas.vec(self.slack_i_jc) >= 0)
            for jc in range(self.slack_i_jc.shape[0]):
                for t in range(self.slack_i_jc.shape[1]):
                    self.slack_cost += self.slack_i_jc[jc, t]**2

            self.slack_ic_jc = [self.opti.variable(len(self.vehs_ctrld), N + 1) for ic in range(len(self.vehs_ctrld))]
            for ic in range(len(self.vehs_ctrld)):
                self.opti.subject_to(cas.vec(self.slack_ic_jc[ic]) >= 0)
                for jc in range(self.slack_ic_jc[ic].shape[0]):
                    for t in range(self.slack_ic_jc[ic].shape[1]):
                        self.slack_cost += self.slack_ic_jc[ic][jc, t]**2
        else:
            self.slack_i_jc = 0
            self.slack_ic_jc = []

        self.response_svo_cost = np.cos(self.ego_veh.theta_i) * self.response_costs
        if len(self.all_other_costs) > 0:
            self.other_svo_cost = np.sum(self.all_other_costs) / len(self.all_other_costs)
        else:
            self.other_svo_cost = 0.0

        # Add Constraints to cntrld Vehicles
        self.ego_veh.add_dynamics_constraints(self.opti, self.x_ego, self.u_ego, self.xd_ego, p_ego)
        self.ego_veh.add_state_constraints(self.opti, self.x_ego, self.u_ego)
        for j in range(len(self.vehs_ctrld)):
            self.vehs_ctrld[j].add_dynamics_constraints(self.opti, self.x_ctrld[j], self.u_ctrld[j], self.xd_ctrld[j],
                                                        self.p_cntrld[j])
            self.vehs_ctrld[j].add_state_constraints(self.opti, self.x_ctrld[j], self.u_ctrld[j])

        # Compute Collision Avoidance ellipses using Minkowski sum
        self.pairwise_distances = []  # keep track of all the distances between ego and ado vehicles
        self.collision_cost = 0
        self.k_ca2 = 0.77  # TODO: when should this number change?

        self.top_wall_slack = self.opti.variable(1, N + 1)
        self.bottom_wall_slack = self.opti.variable(1, N + 1)
        self.opti.subject_to(cas.vec(self.top_wall_slack) >= 0)
        self.opti.subject_to(cas.vec(self.bottom_wall_slack) >= 0)

        self.top_wall_slack_c = [self.opti.variable(1, N + 1) for i in range(len(self.vehs_ctrld))]
        self.bottom_wall_slack_c = [self.opti.variable(1, N + 1) for i in range(len(self.vehs_ctrld))]
        for ic in range(len(self.vehs_ctrld)):
            self.opti.subject_to(cas.vec(self.top_wall_slack_c[ic]) >= 0)
            self.opti.subject_to(cas.vec(self.bottom_wall_slack_c[ic]) >= 0)
        for k in range(N + 1):
            # Compute response vehicles collision center points
            for i in range(len(self.other_vehicle_list)):
                initial_displacement = x0_other[i] - x0
                initial_xy_distance = cas.sqrt(initial_displacement[0]**2 + initial_displacement[1]**2)
                if (initial_xy_distance <=
                        params["collision_avoidance_checking_distance"]):  # collision avoidance distance for other cars

                    dist = self.minkowski_ellipse_collision_distance(self.ego_veh, self.other_vehicle_list[i],
                                                                     self.x_ego[0, k], self.x_ego[1, k],
                                                                     self.x_ego[2, k], self.x_other[i][0, k],
                                                                     self.x_other[i][1, k], self.x_other[i][2, k])

                    self.pairwise_distances += [dist]
                    self.opti.subject_to(dist >= (1 - self.slack_i_jnc[i, k]))
                    distance_clipped = cas.fmax(dist, 0.00001)  # This can be a smaller distance if we'd like
                    self.collision_cost += (1 / (distance_clipped - self.k_ca2)**self.k_CA_power)
            for j in range(len(self.vehs_ctrld)):
                dist = self.minkowski_ellipse_collision_distance(self.ego_veh, self.vehs_ctrld[j], self.x_ego[0, k],
                                                                 self.x_ego[1, k], self.x_ego[2, k], self.x_ctrld[j][0,
                                                                                                                     k],
                                                                 self.x_ctrld[j][1, k], self.x_ctrld[j][2, k])

                self.opti.subject_to(dist >= 1 - self.slack_i_jc[j, k])
                self.pairwise_distances += [dist]
                distance_clipped = cas.fmax(dist, 0.00001)
                self.collision_cost += (1 / (distance_clipped - self.k_ca2)**self.k_CA_power)

            if (params["wall_CA"] == 1):  # Add a collision cost related to distance from wall
                dist_btw_wall_bottom = self.x_ego[1, k] - (self.ego_veh.min_y + self.ego_veh.W / 2.0)
                dist_btw_wall_top = (self.ego_veh.max_y - self.ego_veh.W / 2.0) - self.x_ego[1, k]

                self.opti.subject_to(dist_btw_wall_bottom >= 0 - self.bottom_wall_slack[0, k])
                self.opti.subject_to(dist_btw_wall_top >= 0 - self.top_wall_slack[0, k])
                self.slack_cost += (self.top_wall_slack[0, k]**2 + self.bottom_wall_slack[0, k]**2)

        for ic in range(len(self.vehs_ctrld)):
            for k in range(N + 1):
                # Genereate collision circles for cntrld vehicles and other car
                for j in range(len(self.other_vehicle_list)):
                    initial_displacement = x0_other[j] - x0_other_ctrl[ic]
                    initial_xy_distance = cas.sqrt(initial_displacement[0]**2 + initial_displacement[1]**2)
                    if (initial_xy_distance <= params["collision_avoidance_checking_distance"]
                        ):  # collision avoidance distance for other cars

                        dist = self.minkowski_ellipse_collision_distance(self.vehs_ctrld[ic],
                                                                         self.other_vehicle_list[j],
                                                                         self.x_ctrld[ic][0, k], self.x_ctrld[ic][1, k],
                                                                         self.x_ctrld[ic][2, k], self.x_other[j][0, k],
                                                                         self.x_other[j][1, k], self.x_other[j][2, k])

                        self.opti.subject_to(dist >= (1 - self.slack_ic_jnc[ic][j, k]))
                        distance_clipped = cas.fmax(dist, 0.0001)  # could be buffered if we'd like
                        self.collision_cost += (1 / (distance_clipped - self.k_ca2)**self.k_CA_power)
                for j in range(len(self.vehs_ctrld)):
                    if j <= ic:
                        self.opti.subject_to(self.slack_ic_jc[ic][j, k] == 0)
                    else:
                        initial_displacement = x0_other_ctrl[j] - x0_other_ctrl[ic]
                        initial_xy_distance = cas.sqrt(initial_displacement[0]**2 + initial_displacement[1]**2)
                        # collision avoidance distance for other cars
                        if (initial_xy_distance <= params["collision_avoidance_checking_distance"]):

                            dist = self.minkowski_ellipse_collision_distance(
                                self.vehs_ctrld[ic],
                                self.vehs_ctrld[j],
                                self.x_ctrld[ic][0, k],
                                self.x_ctrld[ic][1, k],
                                self.x_ctrld[ic][2, k],
                                self.x_ctrld[j][0, k],
                                self.x_ctrld[j][1, k],
                                self.x_ctrld[j][2, k],
                            )

                            self.opti.subject_to(dist >= (1 - self.slack_ic_jc[ic][j, k]))
                            distance_clipped = cas.fmax(dist, 0.0001)  # could be buffered if we'd like
                            self.collision_cost += (1 / (distance_clipped - self.k_ca2)**self.k_CA_power)

                if params["wall_CA"] == 1:  # Compute CA cost of ambulance and wall
                    dist_btw_wall_bottom = self.x_ctrld[ic][1, k] - (self.vehs_ctrld[ic].min_y +
                                                                     self.vehs_ctrld[ic].W / 2.0)
                    dist_btw_wall_top = (self.vehs_ctrld[ic].max_y - self.vehs_ctrld[ic].W / 2.0) - self.x_ctrld[ic][1,
                                                                                                                     k]

                    self.opti.subject_to(dist_btw_wall_bottom >= (0 - self.bottom_wall_slack_c[ic][0, k]))
                    self.opti.subject_to(dist_btw_wall_top >= (0 - self.top_wall_slack_c[ic][0, k]))

                    self.slack_cost += (self.top_wall_slack_c[ic][0, k]**2 + self.bottom_wall_slack_c[ic][0, k]**2)

        # Add velocity based constraints
        if "safety_constraint" in params and params["safety_constraint"] == True:
            max_deceleration = abs(self.ego_veh.max_deceleration)
            if len(self.x_ctrld) > 0:
                self.generate_circles_stopping_constraint(self.x_ego, self.x_ctrld, self.ego_veh.L, self.ego_veh.W,
                                                          max_deceleration)

                for jc in range(len(self.x_ctrld)):
                    self.generate_circles_stopping_constraint(self.ctrld[jc],
                                                              self.x_ctrld[:jc] + self.x_ctrld[jc + 1:] + self.x_other,
                                                              self.ego_veh.L, self.ego_veh.W, max_deceleration)

            if len(self.x_other) > 0:
                self.generate_circles_stopping_constraint(self.x_ego, self.x_other, self.ego_veh.L, self.ego_veh.W,
                                                          max_deceleration)

        # Total optimization costs
        self.total_svo_cost = (self.response_svo_cost + self.other_svo_cost + self.k_slack * self.slack_cost +
                               self.k_CA * self.collision_cost)
        self.opti.minimize(self.total_svo_cost)

        # Set the initial conditions
        self.opti.set_value(p_ego, x0)
        for i in range(len(self.allother_p)):
            self.opti.set_value(self.allother_p[i], x0_other[i])
        for j in range(len(x0_other_ctrl)):
            self.opti.set_value(self.p_cntrld[j], x0_other_ctrl[j])

        # Set the solver conditions
        if ipopt_params is None:
            ipopt_params = {}
        self.opti.solver("ipopt", {}, ipopt_params)

    def solve(self, uctrld_warm, uother, solve_amb=False):
        for ic in range(len(self.vehs_ctrld)):
            self.opti.set_initial(self.u_ctrld[ic], uctrld_warm[ic])
        for i in range(len(self.other_vehicle_list)):
            self.opti.set_value(self.u_other[i], uother[i])
        self.solution = self.opti.solve()

    def get_bestresponse_solution(self):
        x1, u1, x1_des, = (self.solution.value(self.x_ego), self.solution.value(self.u_ego),
                           self.solution.value(self.xd_ego))

        return x1, u1, x1_des

    def get_solution(self):
        x1, u1, x1_des, = (self.solution.value(self.x_ego), self.solution.value(self.u_ego),
                           self.solution.value(self.xd_ego))

        cntrld_x = [self.solution.value(self.x_ctrld[i]) for i in range(len(self.vehs_ctrld))]
        cntrld_u = [self.solution.value(self.u_ctrld[i]) for i in range(len(self.vehs_ctrld))]
        cntrld_des = [self.solution.value(self.xd_ctrld[i]) for i in range(len(self.vehs_ctrld))]

        other_x = [self.solution.value(self.x_other[i]) for i in range(len(self.other_vehicle_list))]
        other_u = [self.solution.value(self.u_other[i]) for i in range(len(self.other_vehicle_list))]
        other_des = [self.solution.value(self.xd_other[i]) for i in range(len(self.other_vehicle_list))]

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

    def debug_callback(self, i, plot_range=[], file_name=False):
        xothers_plot = [self.opti.debug.value(xo) for xo in self.x_other]
        xamb_plot = self.opti.debug.value(self.x_ego)
        if self.ambMPC:
            xothers_plot += [self.opti.debug.value(self.xamb_opt)]

        if file_name:
            uamb_plot = self.opti.debug.value(self.u_ego)
            uothers_plot = [self.opti.debug.value(xo) for xo in self.u_other]
            save_state(file_name, xamb_plot, uamb_plot, None, xothers_plot, uothers_plot, None)

        if len(plot_range) > 0:

            plot_cars(self.world, self.ego_veh, xamb_plot, xothers_plot, None, "ellipse", False, 0)
            plt.show()

            plt.plot(xamb_plot[4, :], "--")
            plt.plot(xamb_plot[4, :] * np.cos(xamb_plot[2, :]))
            plt.ylabel("Velocity / Vx")
            plt.hlines(35 * 0.447, 0, xamb_plot.shape[1])
            plt.show()
        print("%d Total Cost %.03f J_i %.03f,  J_j %.03f, Slack %.03f, CA  %.03f" % (
            i,
            self.opti.debug.value(self.total_svo_cost),
            self.opti.debug.value(self.response_svo_cost),
            self.opti.debug.value(self.other_svo_cost),
            self.opti.debug.value(self.k_slack * self.slack_cost),
            self.opti.debug.value(self.k_CA * self.collision_cost),
        ))
        for i in range(len(self.response_costs_list)):
            print(" %.04f : %s" % (
                self.opti.debug.value(self.response_costs_list[i]),
                self.response_cost_titles[i],
            ))

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

    def generate_corners_stopping_constraint(self,
                                             x_ego,
                                             x_others,
                                             L,
                                             W,
                                             max_deceleration=0.01,
                                             min_buffer_distance=0.001):
        ''' We assume all vehicles have same length and width.'''
        N = x_ego.shape[1]

        alpha_i = max_deceleration  #we may need this to be > 0
        alpha_j = max_deceleration

        for k in range(N):
            xi = x_ego[:, k]
            ego_corners = get_vehicle_corners(xi, L, W)

            v_i = cas.vertcat(xi[4] * cas.cos(xi[2]), xi[4] * cas.sin(xi[2]))

            for j in range(len(x_others)):
                xj = x_others[j][:, k]
                ado_corners = get_vehicle_corners(xj, L, W)
                v_j = cas.vertcat(xj[4] * cas.cos(xj[2]), xj[4] * cas.sin(xj[2]))

                for xy_i in ego_corners:
                    for xy_j in ado_corners:
                        delta_p_ij = xy_i - xy_j
                        delta_v_ij = v_i - v_j

                        rel_dist_mag = cas.sqrt(delta_p_ij.T @ delta_p_ij)

                        self.opti.subject_to(delta_p_ij.T @ delta_v_ij / rel_dist_mag +
                                             cas.sqrt(2 * (alpha_i + alpha_j) *
                                                      (rel_dist_mag - min_buffer_distance)) >= 0)

    def generate_circles_stopping_constraint(self,
                                             x_ego,
                                             x_others,
                                             L,
                                             W,
                                             max_deceleration=0.01,
                                             min_buffer_distance=0.001):
        ''' We assume all vehicles have same length and width.'''
        N = x_ego.shape[1]

        alpha_i = max_deceleration  #we may need this to be > 0
        alpha_j = max_deceleration

        for k in range(N):
            xi = x_ego[:, k]
            ego_centers, ego_radius = get_vehicle_circles(xi)

            v_i = cas.vertcat(xi[4] * cas.cos(xi[2]), xi[4] * cas.sin(xi[2]))

            for j in range(len(x_others)):
                xj = x_others[j][:, k]
                #TODO Make function to generate inscribing circles
                ado_centers, ado_radius = get_vehicle_circles(xj)
                v_j = cas.vertcat(xj[4] * cas.cos(xj[2]), xj[4] * cas.sin(xj[2]))

                min_distance = min_buffer_distance + ego_radius + ado_radius
                for xy_i in ego_centers:
                    for xy_j in ado_centers:
                        delta_p_ij = xy_i - xy_j
                        delta_v_ij = v_i - v_j

                        rel_dist_mag = cas.sqrt(delta_p_ij.T @ delta_p_ij)
                        self.opti.subject_to(delta_p_ij.T @ delta_p_ij >= min_distance**
                                             2)  # added to help prevent large gradients in next constraint
                        self.opti.subject_to(
                            delta_p_ij.T @ delta_v_ij / rel_dist_mag + cas.sqrt(2 * (alpha_i + alpha_j) *
                                                                                (rel_dist_mag - min_distance)) >= 0)

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

        # def generate_stopping_constraint_distance(self, x_opt, xcntrld_opt, xothers_opt, min_buffer_distance=0.01, k_ttc=0.0):
        #         """ Add a velocity constraint on the ego vehicle so it doesn't go too fast behind a lead vehicle.
        #             Constrains the vehicle so that the time to collision is always less than time_to_collision seconds
        #             assuming that both ego and ado vehicle mantain constant velocity
        #         """

        # N = x_opt.shape[1]
        # car_length = self.ego_veh.L
        # car_width = self.ego_veh.W
        # for k in range(N):
        #     x_ego = x_opt[0, k]
        #     y_ego = x_opt[1, k]
        #     phi_ego = x_opt[2, k]
        #     v_ego = x_opt[4, k]

        #     for j in range(len(xcntrld_opt)):
        #         ## Safety constraint between ego + cntrld vehicles
        #         x_amb = xcntrld_opt[j][0, k]
        #         y_amb = xcntrld_opt[j][1, k]
        #         phi_amb = xcntrld_opt[j][2, k]
        #         v_amb = xcntrld_opt[j][4, k] - 2 * self.ego_veh.max_v_u

        #         delta_p_ij = cas.vertcat(x_amb, y_amb) - cas.vertcat(x_ego, y_ego)
        #         delta_v_ij = cas.vertcat(v_amb * cas.cos(phi_amb), v_amb * cas.sin(phi_amb)) - cas.vertcat(v_ego * cas.cos(phi_ego), v_ego * cas.sin(phi_ego))
        #         alpha_i = self.ego_veh.max_v_u
        #         alpha_j = self.ego_veh.max_v_u # assume everyone has the same acceleration/deceleration

        #         self.opti.subject_to(- delta_p_ij.T @ delta_v_ij / (delta_p_ij.T @ delta_p_ij) <= cas.sqrt(2 * (alpha_i + alpha_j) * ((delta_p_ij.T @ delta_p_ij) - min_buffer_distance))

        #         dxegoamb = (x_amb - x_ego) - car_length
        #         dyegoamb = (y_amb - y_ego) - car_width
        #         dot_product = (v_ego_components[0] - v_amb_components[0]) * dxegoamb + (v_ego_components[1] -
        #                                                                                 v_amb_components[1]) * dyegoamb
        #         positive_dot_product = cas.max(
        #             dot_product,
        #             0)  # if negative, negative_dot_product will be 0 so time_to_collision is very very large

        #         time_to_collision = (dxegoamb**2 + dyegoamb**2) / (dot_product + 0.000001)
        #         time_to_collision_cost += k_ttc * 1 / time_to_collision**2
        #         self.opti.subject_to(dot_product <= (dxegoamb**2 + dyegoamb**2) / (0.000001 + min_time_to_collision))

        #     for j in range(len(xothers_opt)):
        #         x_j = xothers_opt[j][0, k]
        #         y_j = xothers_opt[j][1, k]
        #         phi_j = xothers_opt[j][2, k]
        #         v_j = xothers_opt[j][4, k] - 2 * self.ego_veh.max_v_u

        #         #### Add constraint between ego and j
        #         dxego = (x_j - x_ego) - car_length
        #         dyego = (y_j - y_ego) - car_width

        #         v_j_components = (v_j * cas.cos(phi_j), v_j * cas.sin(phi_j))
        #         dot_product = (v_ego_components[0] - v_j_components[0]) * dxego + (v_ego_components[1] -
        #                                                                            v_j_components[1]) * dyego
        #         positive_dot_product = cas.max(
        #             dot_product,
        #             0)  # if negative, negative_dot_product will be 0 so time_to_collision is very very large
        #         time_to_collision = (dxego**2 + dyego**2) / (positive_dot_product + 0.000001)
        #         time_to_collision_cost += k_ttc * 1 / time_to_collision**2
        #         self.opti.subject_to(dot_product <= (dxego**2 + dyego**2) / (0.000001 + min_time_to_collision))

        #         #### Add constraint betweem cntrld vehicles and j
        #         add_constraint_for_cntrld = False
        #         if add_constraint_for_cntrld:
        #             for jc in range(len(xcntrld_opt)):
        #                 x_ctrl = xcntrld_opt[jc][0, k]
        #                 y_ctrl = xcntrld_opt[jc][1, k]
        #                 phi_ctrl = xcntrld_opt[jc][2, k]
        #                 v_ctrl = xcntrld_opt[jc][4, k]
        #                 v_ctrl_components = (
        #                     v_ctrl * cas.cos(phi_ctrl),
        #                     v_ctrl * cas.sin(phi_ctrl),
        #                 )

        #                 dxctrl = (x_j - x_ctrl) - car_length
        #                 dyctrl = (y_j - y_ctrl) - car_width

        #                 dot_product = (v_ctrl_components[0] - v_j_components[0]) * dxctrl + (v_ctrl_components[1] -
        #                                                                                      v_j_components[1]) * dyctrl
        #                 positive_dot_product = cas.max(
        #                     dot_product,
        #                     0)  # if negative, negative_dot_product will be 0 so time_to_collision is very very large
        #                 time_to_collision = (dxctrl**2 + dyctrl**2) / (positive_dot_product + 0.000001)
        #                 time_to_collision_cost += k_ttc * 1 / time_to_collision**2

        #                 self.opti.subject_to(dot_product <= (dxctrl**2 + dyctrl**2) /
        #                                      (0.00000001 + min_time_to_collision))

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
        shape_matrix_ego = np.array([[float(ego_veh.ax), 0.0], [0.0, float(ego_veh.by)]])
        shape_matrix_ado = np.array([[float(ado_veh.ax), 0.0], [0.0, float(ado_veh.by)]])

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


def get_vehicle_corners(x_state, length, width):
    ''' List of x,y points corresponding to corner pts of car'''

    translation = x_state[0:2]

    phi_c = x_state[2]
    rotation_matrix_ego = cas.vertcat(
        cas.horzcat(cas.cos(phi_c), -cas.sin(phi_c)),
        cas.horzcat(cas.sin(phi_c), cas.cos(phi_c)),
    )

    corners = [(length / 2.0, width / 2.0), (-length / 2.0, width / 2.0), (-length / 2.0, -width / 2.0),
               (length / 2.0, -width / 2.0)]
    corners_array = [cas.vertcat(t[0], t[1]) for t in corners]
    list_of_xy_corners = []
    for xy in corners_array:
        xy_rotated = cas.mtimes(rotation_matrix_ego, xy)
        xy_translated = xy_rotated + translation
        list_of_xy_corners += [xy_translated]

    return list_of_xy_corners


# def get_vehicle_circles(x_state, length, width):
#     ''' List of x,y points corresponding to corner pts of car'''

#     translation = x_state[0:2]

#     phi_c = x_state[2]
#     rotation_matrix_ego = cas.vertcat(
#         cas.horzcat(cas.cos(phi_c), -cas.sin(phi_c)),
#         cas.horzcat(cas.sin(phi_c), cas.cos(phi_c)),
#     )

#     circle_centers = [(length / 4.0, 0), (-length / 4.0, 0)]
#     centers_array = [cas.vertcat(t[0], t[1]) for t in circle_centers]
#     list_of_xy_centers = []
#     for xy in centers_array:
#         xy_rotated = cas.mtimes(rotation_matrix_ego, xy)
#         xy_translated = xy_rotated + translation
#         list_of_xy_centers += [xy_translated]

#     radius = np.sqrt((length / 4)**2 + (width / 2)**2)  # analytic solution to minim. circumscribed circle

#     return list_of_xy_centers, radius


def get_vehicle_circles(x_state):
    ''' Circles that circumscribe the collision ellipse generated earlier'''
    ## Hardcoded for length L, W, ideally should be solved for at run time
    dx = 1.028
    r = 1.878

    translation = x_state[0:2]

    phi_c = x_state[2]
    rotation_matrix_ego = cas.vertcat(
        cas.horzcat(cas.cos(phi_c), -cas.sin(phi_c)),
        cas.horzcat(cas.sin(phi_c), cas.cos(phi_c)),
    )

    circle_centers = [(dx, 0), (-dx, 0)]
    centers_array = [cas.vertcat(t[0], t[1]) for t in circle_centers]
    list_of_xy_centers = []
    for xy in centers_array:
        xy_rotated = cas.mtimes(rotation_matrix_ego, xy)
        xy_translated = xy_rotated + translation
        list_of_xy_centers += [xy_translated]

    radius = r  # analytic solution to minim. circumscribed circle

    return list_of_xy_centers, radius
