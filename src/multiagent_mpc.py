import numpy as np
import casadi as cas
from typing import List

from src.traffic_world import TrafficWorld
from src.geometry_helper import minkowski_ellipse_collision_distance
from src.vehicle_parameters import VehicleParameters


class NonconvexOptimization(object):
    def __init__(self):
        # Inequality Constraints
        self._f = 0
        self._g_list = []
        self._lbg_list = []
        self._ubg_list = []
        self._x_list = []  # Decision Variables
        self._p_list = []  # Parameters

    def add_X(self, *x):
        self._x_list.extend(x)

    def add_P(self, *p):
        self._p_list.extend(p)

    def set_f(self, f):
        self._f = f

    def add_equal_constraint(self, lhs, rhs):
        self._g_list += [lhs - rhs]

        self._lbg_list += [np.zeros(shape=lhs.shape)]
        self._ubg_list += [np.zeros(shape=lhs.shape)]

    def add_bounded_constraint(self, lbg, g, ubg):
        ''' Add  lbg <= g <= ubg'''
        if lbg is not None:
            self._g_list += [g - lbg]
            self._lbg_list += [np.zeros(shape=(g.shape[0], g.shape[1]))]
            self._ubg_list += [np.infty * np.ones(shape=(g.shape[0], g.shape[1]))]

        if ubg is not None:
            self._g_list += [g - ubg]
            self._lbg_list += [-np.infty * np.ones(shape=(g.shape[0], g.shape[1]))]
            self._ubg_list += [np.zeros(shape=(g.shape[0], g.shape[1]))]

    def reshape_lists(self):

        tall_g_list = []
        tall_ubg_list = []
        tall_lbg_list = []
        for ix in range(len(self._g_list)):
            if self._g_list[ix].shape[1] != 1:
                tall_dim = self._g_list[ix].shape[0] * self._g_list[ix].shape[1]

                tall_g_list += [cas.reshape(self._g_list[ix], (tall_dim, 1))]
                tall_lbg_list += [cas.reshape(self._lbg_list[ix], (tall_dim, 1))]
                tall_ubg_list += [cas.reshape(self._ubg_list[ix], (tall_dim, 1))]
            else:
                tall_g_list += [self._g_list[ix]]
                tall_lbg_list += [self._lbg_list[ix]]
                tall_ubg_list += [self._ubg_list[ix]]

        self._g_list = cas.vertcat(*tall_g_list)
        self._lbg_list = cas.vertcat(*tall_lbg_list)
        self._ubg_list = cas.vertcat(*tall_ubg_list)

        self.nx = self._x_list.shape[0]
        self.ng = self._g_list.shape[0]
        if self._p_list is not None:
            self.np = self._p_list.shape[0]        

    def get_nlpsol(self, ipopt_params=None):
            if ipopt_params is None:
                ipopt_params = {}
            self.reshape_lists()
            if self._p_list is None:
                prob = {'f': self._f, 'g': self._g_list, 'x': self._x_list}
            else:
                prob = {'f': self._f, 'g': self._g_list, 'x': self._x_list, 'p': self._p_list}

            solver = cas.nlpsol('solver', 'ipopt', prob, {'ipopt': ipopt_params})

            solver_name_prefix = self.get_solver_name()
            return solver, solver_name_prefix      

    def get_solver_name(self):
        return "opt"


class MultiMPC(NonconvexOptimization):
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

        super(MultiMPC, self).__init__()
        ''' Setup an optimization for the response vehicle, shared control vehicles, and surrounding vehicles
            Input: 
                N:  number of control points in the optimization (N+1 state points)
                x0: response vehicle's initial position [np.array(n_state)]
                x0_other_ctrl:  list of shared control vehicle initial positions List[np.array(n_state)]
                x0_other:  list of non-responding vehicle initial positions List[np.array(n_state)]
                params:  simulation parameters
                ipopt_params:  paramaters for the optimizatio solver
        '''

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

        self.N = N
        self.n_vehs_cntrld = n_vehs_cntrld
        self.n_other_vehicle = n_other_vehicles

        n_state, n_ctrl, n_desired = 6, 2, 3

        # Response (planning) Vehicle Variables
        self.x_ego = cas.MX.sym('x_ego', n_state, N + 1)
        self.u_ego = cas.MX.sym('u_ego', n_ctrl, N)
        self.xd_ego = cas.MX.sym('xd_ego', n_desired, N + 1)

        # Cntrld Vehicles become variables in the optimization
        self.x_ctrld = [cas.MX.sym('x_ctrl%02d' % i, n_state, N + 1) for i in range(self.n_vehs_cntrld)]
        self.u_ctrld = [cas.MX.sym('u_ctrl%02d' % i, n_ctrl, N) for i in range(self.n_vehs_cntrld)]
        self.xd_ctrld = [cas.MX.sym('xd_ctrl%02d' % i, n_desired, N + 1) for i in range(self.n_vehs_cntrld)]

        # Declare the slack variables
        if self.n_other_vehicle > 0:
            self.slack_i_jnc = cas.MX.sym('s_i_jnc', self.n_other_vehicle, N + 1)
            self.slack_ic_jnc = [
                cas.MX.sym('s_i%02d_jnc' % i, self.n_other_vehicle, N + 1) for i in range(self.n_vehs_cntrld)
            ]
        else:
            self.slack_i_jnc = 0
            self.slack_ic_jnc = []

        # Slack variables related to cntrld vehicles
        if self.n_vehs_cntrld > 0:
            self.slack_i_jc = cas.MX.sym("s_i_jc", self.n_vehs_cntrld, N + 1)
            self.slack_ic_jc = [
                cas.MX.sym("s_ic%02d_jc" % ic, self.n_vehs_cntrld, N + 1) for ic in range(self.n_vehs_cntrld)
            ]
        else:
            self.slack_i_jc = 0
            self.slack_ic_jc = []

        self.top_wall_slack = cas.MX.sym('s_top', 1, N + 1)
        self.bottom_wall_slack = cas.MX.sym('s_bot', 1, N + 1)

        self.top_wall_slack_c = [cas.MX.sym('sc_top_%02d' % i, 1, N + 1) for i in range(self.n_vehs_cntrld)]
        self.bottom_wall_slack_c = [cas.MX.sym('sc_bot_%02d' % i, 1, N + 1) for i in range(self.n_vehs_cntrld)]

        self._x_list = mpcx_to_nlpx(self.n_other_vehicle, self.x_ego, self.u_ego, self.xd_ego, self.x_ctrld,
                                    self.u_ctrld, self.xd_ctrld, self.slack_i_jnc, self.slack_ic_jnc, self.slack_i_jc,
                                    self.slack_ic_jc, self.top_wall_slack, self.bottom_wall_slack,
                                    self.top_wall_slack_c, self.bottom_wall_slack_c)

        # Parameters
        self.x0_ego = cas.MX.sym('x0_ego', n_state, 1)
        self.p_ego = VehicleParameters(self.n_vehs_cntrld, self.n_other_vehicle, "ego")
        self.p_size = self.p_ego.get_opti_params().shape[0]
        self.p_theta_ic = cas.MX.sym('svo_ego', self.n_vehs_cntrld, 1)
        self.p_theta_inc = cas.MX.sym('svo_ego_nc', self.n_other_vehicle, 1)
        self.p_theta_i_ego = cas.MX.sym('svo_ego_self', 1, 1)

        # Cntrld Vehicle Parameters
        self.x0_cntrld = [cas.MX.sym('x0_ctrl%02d' % i, n_state, 1) for i in range(self.n_vehs_cntrld)]
        self.p_cntrld_list = [
            VehicleParameters(self.n_vehs_cntrld, self.n_other_vehicle, "ctrl%02d" % i)
            for i in range(self.n_vehs_cntrld)
        ]

        # Variables of surrounding vehicles assumed fixed as parameters for computation
        self.x_other = [cas.MX.sym('x_nc%02d' % i, n_state, N + 1) for i in range(self.n_other_vehicle)]

        # Non-Response Vehicle Parameters
        self.x0_allother = [cas.MX.sym('x0_nc%02d' % i, n_state, 1) for i in range(self.n_other_vehicle)]
        self.p_other_vehicle_list = [
            VehicleParameters(self.n_vehs_cntrld, self.n_other_vehicle, "nc%02d" % i)
            for i in range(self.n_other_vehicle)
        ]
        self._p_list = mpcp_to_nlpp(self.x0_ego, self.p_ego, self.p_theta_i_ego, self.p_theta_ic, self.p_theta_inc,
                                    self.x0_cntrld, self.p_cntrld_list, self.x0_allother, self.p_other_vehicle_list,
                                    self.x_other)

        if params is None:
            params = {}

        if "collision_avoidance_checking_distance" not in params:
            print("No collision avoidance checking distance")
            params["collision_avoidance_checking_distance"] = 400

        # Generate costs for each vehicle
        self.k_ca2 = 0.77  #This number must be less than 1

        self.total_svo_cost = self.compute_mpc_costs(
            self.n_vehs_cntrld, self.n_other_vehicle, self.x_ego, self.u_ego, self.xd_ego, self.p_ego, self.p_theta_ic,
            self.p_theta_i_ego, self.x_ctrld, self.u_ctrld, self.xd_ctrld, self.p_cntrld_list, self.x_other,
            self.p_other_vehicle_list, self.slack_i_jnc, self.slack_ic_jnc, self.slack_i_jc, self.slack_ic_jc,
            self.top_wall_slack, self.bottom_wall_slack, self.top_wall_slack_c, self.bottom_wall_slack_c, self.x0_ego,
            self.x0_cntrld, self.x0_allother, params, self.k_ca2, self.k_CA_power, self.k_slack, self.k_CA)
        self.set_f(self.total_svo_cost)

        ###########################3 CONSTRAINTS #########################################
        self.add_mpc_constraints(params["N"], n_vehs_cntrld, self.n_other_vehicle, world, self.x_ego, self.u_ego,
                                 self.xd_ego, self.x0_ego, self.p_ego, self.x_ctrld, self.u_ctrld, self.xd_ctrld,
                                 self.x0_cntrld, self.p_cntrld_list, self.x_other, self.x0_allother,
                                 self.p_other_vehicle_list, self.dt, self.slack_i_jnc, self.slack_ic_jnc,
                                 self.slack_i_jc, self.slack_ic_jc, self.top_wall_slack, self.bottom_wall_slack,
                                 self.top_wall_slack_c, self.bottom_wall_slack_c, params)
        # Total optimization costs
        self.solver, self.solver_prefix = self.get_nlpsol(ipopt_params)

    def add_vehicle_constraints(self, world, p_ego, x_ego, u_ego, xd_ego, x0_ego, dt):
        ego_lane_number = p_ego.desired_lane
        fd = self.gen_f_desired_lane(world, ego_lane_number, right_direction=True)  # TODO:  This could mess things up
        f = self.gen_f_vehicle_dynamics(p_ego, model="kinematic_bicycle")
        self.add_dynamics_constraints_g(x_ego, u_ego, xd_ego, x0_ego, f, fd, dt)
        self.add_state_constraints_g(x_ego, u_ego, p_ego)

    def add_mpc_constraints(self, N, n_vehs_cntrld, n_other_vehicle, world, x_ego, u_ego, xd_ego, x0_ego, p_ego,
                            x_ctrld, u_ctrld, xd_ctrld, x0_ctrld, p_ctrld_list, x_other, x0_allother,
                            p_other_vehicle_list, dt, slack_i_jnc, slack_ic_jnc, slack_i_jc, slack_ic_jc,
                            top_wall_slack, bottom_wall_slack, top_wall_slack_c, bottom_wall_slack_c, params):
        ''' Add all the constraints the MPC
            1) vehicle constraints related to their state and dynamics
            2) positive constraints on the slack variables
            3) collision avoidance and velocity safety constraints 
        
        '''
        self.add_vehicle_constraints(world, p_ego, x_ego, u_ego, xd_ego, x0_ego, dt)

        for j in range(n_vehs_cntrld):
            self.add_vehicle_constraints(world, p_ctrld_list[j], x_ctrld[j], u_ctrld[j], xd_ctrld[j], x0_ctrld[j], dt)

        self.constrain_slack_positive(n_vehs_cntrld, n_other_vehicle, slack_i_jc, slack_i_jnc, slack_ic_jc,
                                      slack_ic_jnc, top_wall_slack, bottom_wall_slack, top_wall_slack_c,
                                      bottom_wall_slack_c)

        for k in range(N + 1):
            # Compute response vehicles collision center points
            for i in range(n_other_vehicle):
                initial_displacement = x0_allother[i] - x0_ego
                initial_xy_distance = cas.sqrt(initial_displacement[0]**2 + initial_displacement[1]**2)
                within_collision_distance = (initial_xy_distance <= params["collision_avoidance_checking_distance"])

                dist = minkowski_ellipse_collision_distance(p_ego, p_other_vehicle_list[i], x_ego[0, k], x_ego[1, k],
                                                            x_ego[2, k], x_other[i][0, k], x_other[i][1, k],
                                                            x_other[i][2, k])

                # Only keep track if within_collision_distance == 1  (i.e. if outside, don't add cost or constraint)
                self.add_bounded_constraint(((1 - slack_i_jnc[i, k]) * within_collision_distance), dist, None)

            for j in range(n_vehs_cntrld):
                dist = minkowski_ellipse_collision_distance(p_ego, p_ctrld_list[j], x_ego[0, k], x_ego[1, k], x_ego[2,
                                                                                                                    k],
                                                            x_ctrld[j][0, k], x_ctrld[j][1, k], x_ctrld[j][2, k])

                self.add_bounded_constraint(1 - slack_i_jc[j, k], dist, None)

            if (params["wall_CA"] == 1):  # Add a collision cost related to distance from wall
                dist_btw_wall_bottom = x_ego[1, k] - p_ego.min_y
                dist_btw_wall_top = p_ego.max_y - x_ego[1, k]
                self.add_bounded_constraint(0 - bottom_wall_slack[0, k], dist_btw_wall_bottom, None)
                self.add_bounded_constraint(0 - top_wall_slack[0, k], dist_btw_wall_top, None)

        for ic in range(self.n_vehs_cntrld):
            for k in range(N + 1):
                # Genereate collision circles for cntrld vehicles and other car
                for j in range(self.n_other_vehicle):
                    initial_displacement = x0_allother[j] - x0_ctrld[ic]
                    initial_xy_distance = cas.sqrt(initial_displacement[0]**2 + initial_displacement[1]**2)
                    within_collision_distance = (initial_xy_distance <= params["collision_avoidance_checking_distance"])
                    dist = minkowski_ellipse_collision_distance(p_ctrld_list[ic], p_other_vehicle_list[j],
                                                                x_ctrld[ic][0, k], x_ctrld[ic][1, k], x_ctrld[ic][2, k],
                                                                x_other[j][0, k], x_other[j][1, k], x_other[j][2, k])

                    self.add_bounded_constraint(((1 - slack_ic_jnc[ic][j, k]) * within_collision_distance), dist, None)

                for j in range(n_vehs_cntrld):
                    if j <= ic:
                        self.add_equal_constraint(slack_ic_jc[ic][j, k], np.zeros(
                            (1, 1)))  #TODO: get rid of this slack variable
                    else:
                        initial_displacement = x0_ctrld[j] - x0_ctrld[ic]
                        initial_xy_distance = cas.sqrt(initial_displacement[0]**2 + initial_displacement[1]**2)
                        within_collision_distance = (initial_xy_distance <=
                                                     params["collision_avoidance_checking_distance"])

                        dist = minkowski_ellipse_collision_distance(p_ctrld_list[ic], p_ctrld_list[j], x_ctrld[ic][0,
                                                                                                                   k],
                                                                    x_ctrld[ic][1, k], x_ctrld[ic][2, k],
                                                                    x_ctrld[j][0, k], x_ctrld[j][1, k], x_ctrld[j][2,
                                                                                                                   k])

                        self.add_bounded_constraint(((1 - slack_ic_jc[ic][j, k]) * within_collision_distance), dist,
                                                    None)

                dist_btw_wall_bottom = x_ctrld[ic][1, k] - p_ctrld_list[ic].min_y
                dist_btw_wall_top = p_ctrld_list[ic].max_y - x_ctrld[ic][1, k]
                self.add_bounded_constraint((0 - bottom_wall_slack_c[ic][0, k]), dist_btw_wall_bottom, None)
                self.add_bounded_constraint((0 - top_wall_slack_c[ic][0, k]), dist_btw_wall_top, None)

        # TODO: Conver the Add velocity based constraints
        if "safety_constraint" in params and params["safety_constraint"] == True:
            max_deceleration = cas.fabs(self.p_ego.max_deceleration)
            if len(x_ctrld) > 0:
                self.generate_circles_stopping_constraint(x_ego, x_ctrld, p_ego.L, p_ego.W, max_deceleration)

                for jc in range(len(x_ctrld)):
                    self.generate_circles_stopping_constraint(x_ctrld[jc], x_ctrld[:jc] + x_ctrld[jc + 1:] + x_other,
                                                              p_ego.L, self.p_ego.W, max_deceleration)

            if len(self.x_other) > 0:
                self.generate_circles_stopping_constraint(x_ego, x_other, p_ego.L, p_ego.W, max_deceleration)

    def constrain_slack_positive(self, n_vehs_cntrld, n_other_vehicle, slack_i_jc, slack_i_jnc, slack_ic_jc,
                                 slack_ic_jnc, top_wall_slack, bottom_wall_slack, top_wall_slack_c,
                                 bottom_wall_slack_c):
        ''' Constrain the following slack to being positive'''

        if n_other_vehicle > 0:
            self.add_bounded_constraint(np.zeros(shape=slack_i_jnc.shape), slack_i_jnc, None)
            for slack_var in slack_ic_jnc:
                self.add_bounded_constraint(np.zeros(shape=slack_var.shape), slack_var, None)

        # Slack variables related to cntrld vehicles
        if n_vehs_cntrld > 0:
            self.add_bounded_constraint(np.zeros(shape=slack_i_jc.shape), slack_i_jc, None)
            for ic in range(len(slack_ic_jc)):
                self.add_bounded_constraint(np.zeros(shape=slack_ic_jc[ic].shape), slack_ic_jc[ic], None)

        for ic in range(len(top_wall_slack_c)):
            for k in range(top_wall_slack_c[ic].shape[1]):
                self.add_bounded_constraint(0, top_wall_slack_c[ic][0, k], None)
                self.add_bounded_constraint(0, bottom_wall_slack_c[ic][0, k], None)

        for k in range(top_wall_slack.shape[1]):
            self.add_bounded_constraint(0, top_wall_slack[0, k], None)
            self.add_bounded_constraint(0, bottom_wall_slack[0, k], None)

    def compute_mpc_costs(self, n_vehs_cntrld, n_other_vehicle, x_ego, u_ego, xd_ego, p_ego, p_theta_ic, p_theta_i_ego,
                          x_ctrld, u_ctrld, xd_ctrld, p_cntrld_list, x_other, p_other_vehicle_list, slack_i_jnc,
                          slack_ic_jnc, slack_i_jc, slack_ic_jc, top_wall_slack, bottom_wall_slack, top_wall_slack_c,
                          bottom_wall_slack_c, x0_ego, x0_cntrld, x0_allother, params, k_ca2, k_CA_power, k_slack,
                          k_CA):
        ''' Compute the total cost for an SVO based MPC
            For each vehicle with decision variables, compute vehicle-specific costs related to speed and control
            Compute slack costs for the slack variables
            Compute collision-based costs to penalize vehicles close to each other
        '''

        response_costs, _ = self.generate_veh_costs(x_ego, u_ego, xd_ego, p_ego)

        all_other_costs = []
        for idx in range(n_vehs_cntrld):
            # svo_ij = self.ego_veh.get_theta_ij(self.p_cntrld_list[idx].agent_id)
            svo_ij = p_theta_ic[idx]
            nonresponse_cost, _ = self.generate_veh_costs(x_ctrld[idx], u_ctrld[idx], xd_ctrld[idx], p_cntrld_list[idx])
            nonresponse_cost = nonresponse_cost * (svo_ij > 0)
            all_other_costs += [cas.sin(svo_ij) * nonresponse_cost]

        response_svo_cost = np.cos(p_theta_i_ego) * response_costs
        if len(all_other_costs) > 0:
            other_svo_cost = np.sum(all_other_costs) / len(all_other_costs)
        else:
            other_svo_cost = 0

        # Generate Slack Variables used as part of collision avoidance
        slack_cost = self.compute_quadratic_slack_cost(n_other_vehicle, n_vehs_cntrld, slack_i_jnc, slack_ic_jnc,
                                                       slack_i_jc, slack_ic_jc)
        slack_cost += self.compute_wall_slack_costs(params["N"], n_vehs_cntrld, top_wall_slack, bottom_wall_slack,
                                                    top_wall_slack_c, bottom_wall_slack_c, params)

        # Compute Collision Avoidance ellipses using Minkowski sum
        collision_cost = self.compute_collision_avoidance_costs(params["N"], n_other_vehicle, n_vehs_cntrld, x0_ego,
                                                                x0_allother, x0_cntrld, p_ego, p_other_vehicle_list,
                                                                p_cntrld_list, x_ego, x_other, x_ctrld, params, k_ca2,
                                                                k_CA_power)

        total_svo_cost = (response_svo_cost + other_svo_cost + k_slack * slack_cost + k_CA * collision_cost)
        return total_svo_cost

    def compute_quadratic_slack_cost(self, n_other_vehicle, n_vehs_cntrld, slack_i_jnc, slack_ic_jnc, slack_i_jc,
                                     slack_ic_jc):
        slack_cost = 0

        if n_other_vehicle > 0:
            for j in range(slack_i_jnc.shape[0]):
                for t in range(slack_i_jnc.shape[1]):
                    slack_cost += slack_i_jnc[j, t]**2
            for ic in range(n_vehs_cntrld):
                for jnc in range(slack_ic_jnc[ic].shape[0]):
                    for t in range(slack_ic_jnc[ic].shape[1]):
                        slack_cost += slack_ic_jnc[ic][jnc, t]**2

        if n_vehs_cntrld > 0:
            for jc in range(slack_i_jc.shape[0]):
                for t in range(slack_i_jc.shape[1]):
                    slack_cost += slack_i_jc[jc, t]**2

            for ic in range(n_vehs_cntrld):
                for jc in range(slack_ic_jc[ic].shape[0]):
                    for t in range(slack_ic_jc[ic].shape[1]):
                        slack_cost += slack_ic_jc[ic][jc, t]**2

        return slack_cost

    def get_solver_name(self):
        solver_name_prefix = "mpc_%02d_%02d" % (self.n_vehs_cntrld, self.n_other_vehicle)

        return solver_name_prefix

    def add_state_constraints_g(self, X, U, ego_veh, strict_wall_y_constraint=False):
        ''' Construct vehicle specific constraints that only rely on
        the ego vehicle's own state '''
        for k in range(X.shape[1]):
            if strict_wall_y_constraint:
                self.add_bounded_constraint(ego_veh.min_y, X[1, k], ego_veh.max_y)
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

    def solve(self, uctrld_warm, uother):
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

    def compute_wall_slack_costs(self, N, n_vehs_cntrld, top_wall_slack, bottom_wall_slack, top_wall_slack_c,
                                 bottom_wall_slack_c, params):
        slack_cost = 0
        for k in range(N + 1):
            slack_cost += (top_wall_slack[0, k]**2 + bottom_wall_slack[0, k]**2)

        for ic in range(n_vehs_cntrld):
            for k in range(N + 1):
                slack_cost += (top_wall_slack_c[ic][0, k]**2 + bottom_wall_slack_c[ic][0, k]**2)

        return slack_cost

    def compute_collision_avoidance_costs(self, N, n_other_vehicle, n_vehs_cntrld, x0_ego, x0_allother, x0_cntrld,
                                          p_ego, p_other_vehicle_list, p_cntrld_list, x_ego, x_other, x_ctrld, params,
                                          k_ca2, k_CA_power):
        ''' n_other_vehicles
                x0_all_other
                x0_ego
            
            '''
        collision_cost = 0
        for k in range(N + 1):
            # Compute response vehicles collision center points
            for i in range(n_other_vehicle):
                initial_displacement = x0_allother[i] - x0_ego
                initial_xy_distance = cas.sqrt(initial_displacement[0]**2 + initial_displacement[1]**2)
                within_collision_distance = (initial_xy_distance <= params["collision_avoidance_checking_distance"])

                dist = minkowski_ellipse_collision_distance(p_ego, p_other_vehicle_list[i], x_ego[0, k], x_ego[1, k],
                                                            x_ego[2, k], x_other[i][0, k], x_other[i][1, k],
                                                            x_other[i][2, k])

                # Only keep track if within_collision_distance == 1  (i.e. if outside, don't add cost or constraint)
                distance_clipped = cas.fmax(dist, k_ca2+0.001)  # This can be a smaller distance if we'd like
                distance_clipped = cas.fmin(distance_clipped, 1.25)
                collision_cost += (within_collision_distance * 1 / (distance_clipped - k_ca2)**k_CA_power)
            for j in range(n_vehs_cntrld):
                dist = minkowski_ellipse_collision_distance(p_ego, p_cntrld_list[j], x_ego[0, k], x_ego[1, k], x_ego[2,
                                                                                                                     k],
                                                            x_ctrld[j][0, k], x_ctrld[j][1, k], x_ctrld[j][2, k])
                distance_clipped = cas.fmax(dist, k_ca2+0.001)  # This can be a smaller distance if we'd like
                distance_clipped = cas.fmin(distance_clipped, 1.25)
                collision_cost += (1 / (distance_clipped - k_ca2)**k_CA_power)

        for ic in range(n_vehs_cntrld):
            for k in range(N + 1):
                # Genereate collision circles for cntrld vehicles and other car
                for j in range(n_other_vehicle):
                    initial_displacement = x0_allother[j] - x0_cntrld[ic]
                    initial_xy_distance = cas.sqrt(initial_displacement[0]**2 + initial_displacement[1]**2)
                    within_collision_distance = (initial_xy_distance <= params["collision_avoidance_checking_distance"])
                    dist = minkowski_ellipse_collision_distance(p_cntrld_list[ic], p_other_vehicle_list[j],
                                                                x_ctrld[ic][0, k], x_ctrld[ic][1, k], x_ctrld[ic][2, k],
                                                                x_other[j][0, k], x_other[j][1, k], x_other[j][2, k])

                    distance_clipped = cas.fmax(dist, k_ca2 + 0.001)  # This can be a smaller distance if we'd like
                    distance_clipped = cas.fmin(distance_clipped, 1.25)
                    collision_cost += (within_collision_distance * 1 / (distance_clipped - k_ca2)**k_CA_power)
                for j in range(n_vehs_cntrld):
                    if j <= ic:
                        continue
                    else:
                        initial_displacement = x0_cntrld[j] - x0_cntrld[ic]
                        initial_xy_distance = cas.sqrt(initial_displacement[0]**2 + initial_displacement[1]**2)
                        within_collision_distance = (initial_xy_distance <=
                                                     params["collision_avoidance_checking_distance"])

                        dist = minkowski_ellipse_collision_distance(p_cntrld_list[ic], p_cntrld_list[j], x_ctrld[ic][0,
                                                                                                                     k],
                                                                    x_ctrld[ic][1, k], x_ctrld[ic][2, k],
                                                                    x_ctrld[j][0, k], x_ctrld[j][1, k], x_ctrld[j][2,
                                                                                                                   k])
                        distance_clipped = cas.fmax(dist, k_ca2 + 0.001)  # This can be a smaller distance if we'd like
                        distance_clipped = cas.fmin(distance_clipped, 1.25)                        
                        collision_cost += (within_collision_distance * 1 / (distance_clipped - k_ca2)**k_CA_power)

        return collision_cost

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

                        # self.opti.subject_to(delta_p_ij.T @ delta_v_ij / rel_dist_mag +
                        #                      cas.sqrt(2 * (alpha_i + alpha_j) *
                        #                               (rel_dist_mag - min_buffer_distance)) >= 0)
                        self.add_bounded_constraint(
                            0, delta_p_ij.T @ delta_v_ij / rel_dist_mag +
                            cas.sqrt(2 * (alpha_i + alpha_j) * (rel_dist_mag - min_buffer_distance)))

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
                ado_centers, ado_radius = get_vehicle_circles(xj)
                v_j = cas.vertcat(xj[4] * cas.cos(xj[2]), xj[4] * cas.sin(xj[2]))

                min_distance = min_buffer_distance + ego_radius + ado_radius
                for xy_i in ego_centers:
                    for xy_j in ado_centers:
                        delta_p_ij = xy_i - xy_j
                        delta_v_ij = v_i - v_j

                        rel_dist_mag = cas.sqrt(delta_p_ij.T @ delta_p_ij)
                        # added to help prevent large gradients in next constraint
                        # self.opti.subject_to(delta_p_ij.T @ delta_p_ij >= min_distance**2)
                        # stopping_distance = rel_dist_mag - min_distance
                        # make sure we are always positive

                        stopping_distance = cas.fmax(rel_dist_mag - min_distance, 0.00001)
                        self.add_bounded_constraint(min_distance**2, delta_p_ij.T @ delta_p_ij, None)
                        self.add_bounded_constraint(
                            0, delta_p_ij.T @ delta_v_ij / rel_dist_mag + cas.sqrt(2 * (alpha_i + alpha_j) *
                                                                                   (stopping_distance)), None)

                        # self.opti.subject_to(
                        #     delta_p_ij.T @ delta_v_ij / rel_dist_mag + cas.sqrt(2 * (alpha_i + alpha_j) *
                        #                                                         (rel_dist_mag - min_distance)) >= 0)

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
                # self.opti.subject_to(dot_product <= (dxegoamb**2 + dyegoamb**2) / (0.000001 + min_time_to_collision))
                self.add_bounded_constraint(0, (dxegoamb**2 + dyegoamb**2) / (0.000001 + min_time_to_collision) -
                                            dot_product)

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
                # self.opti.subject_to(dot_product <= (dxego**2 + dyego**2) / (0.000001 + min_time_to_collision))
                self.add_bounded_constraint(0, (dxego**2 + dyego**2) / (0.000001 + min_time_to_collision) - dot_product)

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
                        self.add_bounded_constraint(
                            0, -dot_product + (dxctrl**2 + dyctrl**2) / (0.00000001 + min_time_to_collision))
                        # self.opti.subject_to(dot_product <= (dxctrl**2 + dyctrl**2) /
                        #                      (0.00000001 + min_time_to_collision))

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
                # self.opti.subject_to(v_ego**2 <= v_max_constraint)
                self.add_bounded_constraint(0, v_max_constraint - v_ego**2)

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
                self.add_bounded_constraint(0, v_max_constraint - v_ego**2)

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
                    self.add_bounded_constraint(0, v_max_constraint - v_amb**2)


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


def mpcp_to_nlpp(x0_ego, p_ego, p_theta_i_ego, p_theta_i_c, p_theta_i_nc, x0_cntrld, p_cntrld_list, x0_allother,
                 p_other_vehicle_list, x_other_nc):
    ''' Converts 3 seperate parameter lists to a single long vector np X 1'''
    # Add ego parameters
    long_list = []
    long_list.append(x0_ego)
    long_list.append(p_ego.get_opti_params())
    long_list.append(p_theta_i_ego)
    long_list.append(p_theta_i_c)
    long_list.append(p_theta_i_nc)

    # Add cntrld vehicle parameters
    for i in range(len(x0_cntrld)):
        long_list.append(x0_cntrld[i])
        long_list.append(p_cntrld_list[i].get_opti_params())

    # Add non response parameters
    for i in range(len(x0_allother)):
        long_list.append(x0_allother[i])
        long_list.append(p_other_vehicle_list[i].get_opti_params())

    # Add non response trajectories used for collision avoidance
    for i in range(len(x_other_nc)):
        N = x_other_nc[i].shape[1] - 1
        # n_ctrl = u_other_nc[i].shape[0]
        n_state = x_other_nc[i].shape[0]
        # n_desired = xd_other_nc[i].shape[0]

        long_list.append(cas.reshape(x_other_nc[i], (n_state * (N + 1), 1)))
        # long_list.append(cas.reshape(u_other_nc[i], (n_ctrl * N, 1)))
        # long_list.append(cas.reshape(xd_other_nc[i], (n_desired * (N + 1), 1)))

    nlp_p = cas.vcat(long_list)
    return nlp_p


def nlpp_to_mpcp(nlp_p,
                 N=None,
                 n_cntrld_vehicles: int = 0,
                 n_other_vehicles: int = 0,
                 x0_size: int = 6,
                 p_size: int = None,
                 n_state=6,
                 n_ctrl=2,
                 n_desired=3):
    ''' Convert from tall np X 1 vector of all parameters to split up by individual agents '''
    if p_size is None:
        raise Exception("Parameter list size not inputed as argument. Make sure to provide p_size")

    # Get ego vehicle parameters
    idx = 0
    x0_ego = cas.reshape(nlp_p[idx:idx + x0_size], (x0_size, 1))
    idx += x0_size
    p_ego = nlp_p[idx:idx + p_size]
    idx += p_size
    p_theta_i_ego = nlp_p[idx:idx + 1]
    idx += 1
    p_theta_i_c = nlp_p[idx:idx + n_cntrld_vehicles]
    idx += n_cntrld_vehicles
    p_theta_i_nc = nlp_p[idx:idx + n_other_vehicles]
    idx += n_other_vehicles

    # Get Cntrld Vehicle Parameters
    x0_cntrld_list = []
    p_cntrld_list = []
    for _ in range(n_cntrld_vehicles):
        x0_cntrld_list.append(nlp_p[idx:idx + x0_size])
        idx += x0_size
        p_cntrld_list.append(nlp_p[idx:idx + p_size])
        idx += p_size

    # Get Non Response Vehicle Parameters
    x0_other_vehicle = []
    p_other_vehicle_list = []
    # u_other_nc = []
    # xd_other_nc = []

    # Get other vehicle x0 and parameters
    for _ in range(n_other_vehicles):
        x0_other_vehicle.append(nlp_p[idx:idx + x0_size])
        idx += x0_size
        p_other_vehicle_list.append(nlp_p[idx:idx + p_size])
        idx += p_size

    # Get other vehicle trajectories
    x_other_nc = []
    # u_other_nc = []
    # xd_other_nc = []
    for _ in range(n_other_vehicles):
        x_other_nc.append(cas.reshape(nlp_p[idx:idx + n_state * (N + 1)], (n_state, N + 1)))
        idx += n_state * (N + 1)
        # u_other_nc.append(cas.reshape(nlp_p[idx:idx + n_ctrl * N], (n_ctrl, N)))
        # idx += n_ctrl * N
        # xd_other_nc.append(cas.reshape(nlp_p[idx:idx + n_desired * (N + 1)], (n_desired, N + 1)))
        # idx += n_desired * (N + 1)

    return x0_ego, p_ego, p_theta_i_ego, p_theta_i_c, p_theta_i_nc, x0_cntrld_list, p_cntrld_list, x0_other_vehicle, p_other_vehicle_list, x_other_nc


def mpcx_to_nlpx(n_other: int, x_ego, u_ego, xd_ego, x_ctrl: List, u_ctrl: List, xd_ctrl: List, s_i_jnc, s_ic_jnc,
                 s_i_jc, s_ic_jc, s_top, s_bottom, s_c_top, s_c_bottom):
    ''' Converts the individual trajectories into a single descision variable (nx x 1)'''
    long_list = []

    N = u_ego.shape[1]
    n_ctrl = u_ego.shape[0]
    n_state = x_ego.shape[0]
    n_desired = xd_ego.shape[0]
    n_ctrld_vehicles = len(x_ctrl)

    long_list.append(cas.reshape(x_ego, (n_state * (N + 1), 1)))
    long_list.append(cas.reshape(u_ego, (n_ctrl * N, 1)))
    long_list.append(cas.reshape(xd_ego, (n_desired * (N + 1), 1)))

    for i in range(n_ctrld_vehicles):
        long_list.append(cas.reshape(x_ctrl[i], (n_state * (N + 1), 1)))
        long_list.append(cas.reshape(u_ctrl[i], (n_ctrl * N, 1)))
        long_list.append(cas.reshape(xd_ctrl[i], (n_desired * (N + 1), 1)))

    # Collision Slack Variables with Non-Planning Cars
    if n_other > 0:
        long_list.append(cas.reshape(s_i_jnc, (n_other * (N + 1), 1)))
        for i in range(n_ctrld_vehicles):
            long_list.append(cas.reshape(s_ic_jnc[i], (n_other * (N + 1), 1)))

    # Collision Slack Variables with Cntrld Planning Cars
    if n_ctrld_vehicles > 0:
        long_list.append(cas.reshape(s_i_jc, (n_ctrld_vehicles * (N + 1), 1)))
        for i in range(n_ctrld_vehicles):
            long_list.append(cas.reshape(s_ic_jc[i], (n_ctrld_vehicles * (N + 1), 1)))

    long_list.append(cas.reshape(s_top, (N + 1, 1)))
    long_list.append(cas.reshape(s_bottom, (N + 1, 1)))
    for i in range(n_ctrld_vehicles):
        long_list.append(cas.reshape(s_c_top[i], (N + 1, 1)))
        long_list.append(cas.reshape(s_c_bottom[i], (N + 1, 1)))

    nlp_x = cas.vcat(long_list)
    return nlp_x


def nlpx_to_mpcx(nlp_x, N: int = 0, n_ctrld_vehicles: int = 0, n_other: int = 0, n_state=6, n_ctrl=2, n_desired=3):
    ''' Splits the output of an nlp solver (nx X 1) into subsequent trajectories for ego and ado vehicles and slack variables '''

    idx = 0
    x_ego = cas.reshape(nlp_x[idx:idx + n_state * (N + 1)], (n_state, N + 1))
    idx += n_state * (N + 1)
    u_ego = cas.reshape(nlp_x[idx:idx + n_ctrl * N], (n_ctrl, N))
    idx += n_ctrl * N
    xd_ego = cas.reshape(nlp_x[idx:idx + n_desired * (N + 1)], (n_desired, N + 1))
    idx += n_desired * (N + 1)

    x_ctrl = []
    u_ctrl = []
    xd_ctrl = []
    for i in range(n_ctrld_vehicles):
        x_ctrl.append(cas.reshape(nlp_x[idx:idx + n_state * (N + 1)], (n_state, N + 1)))
        idx += n_state * (N + 1)
        u_ctrl.append(cas.reshape(nlp_x[idx:idx + n_ctrl * N], (n_ctrl, N)))
        idx += n_ctrl * N
        xd_ctrl.append(cas.reshape(nlp_x[idx:idx + n_desired * (N + 1)], (n_desired, N + 1)))
        idx += n_desired * (N + 1)

    s_i_jnc = 0
    s_ic_jnc = []
    if n_other > 0:
        s_i_jnc = cas.reshape(nlp_x[idx:idx + n_other * (N + 1)], (n_other, N + 1))
        idx += n_other * (N + 1)

        for i in range(n_ctrld_vehicles):
            s_ic_jnc.append(cas.reshape(nlp_x[idx:idx + n_other * (N + 1)], (n_other, (N + 1))))
            idx += n_other * (N + 1)

    s_i_jc = 0
    s_ic_jc = []
    if n_ctrld_vehicles > 0:
        s_i_jc = cas.reshape(nlp_x[idx:idx + n_ctrld_vehicles * (N + 1)], (n_ctrld_vehicles, N + 1))
        idx += n_ctrld_vehicles * (N + 1)

        for i in range(n_ctrld_vehicles):
            s_ic_jc.append(cas.reshape(nlp_x[idx:idx + n_ctrld_vehicles * (N + 1)], (n_ctrld_vehicles, (N + 1))))
            idx += n_ctrld_vehicles * (N + 1)

    s_top = cas.reshape(nlp_x[idx:idx + N + 1], (1, N + 1))
    idx += N + 1
    s_bottom = cas.reshape(nlp_x[idx:idx + N + 1], (1, N + 1))
    idx += N + 1

    s_c_top = []
    s_c_bottom = []
    for i in range(n_ctrld_vehicles):
        s_c_top.append(cas.reshape(nlp_x[idx:idx + N + 1], (1, N + 1)))
        idx += N + 1
        s_c_bottom.append(cas.reshape(nlp_x[idx:idx + N + 1], (1, N + 1)))
        idx += N + 1

    return x_ego, u_ego, xd_ego, x_ctrl, u_ctrl, xd_ctrl, s_i_jnc, s_ic_jnc, s_i_jc, s_ic_jc, s_top, s_bottom, s_c_top, s_c_bottom
