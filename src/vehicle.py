import numpy as np
import casadi as cas
import scipy.optimize as optimize
from typing import Tuple


class Vehicle(object):
    def __init__(self, dt):
        self.agent_id = 1
        self.dt = dt
        self.k_total = 1.0  # the overall weighting of the total costs
        self.theta_i = 0  # my theta wrt to ambulance
        self.theta_ij = {-1: np.pi / 4}
        self.L = 4.5
        self.W = 1.8
        self.n_circles = 2

        self.desired_lane = -1
        ## State Costs Constants

        self.k_x = 0
        self.k_y = 0
        self.k_phi = 0
        self.k_delta = 0

        self.k_v = 0
        self.k_s = -0.1

        ## Control Costs Constants
        self.k_u_delta = 10.0
        self.k_u_v = 1.0

        ## Derived State Costs Constants
        self.k_lat = 10.0
        self.k_lon = 1.0
        self.k_phi_error = 1.0
        self.k_phi_dot = 1.0

        self.k_x_dot = 0.0

        self.k_change_u_v = 1.0
        self.k_change_u_delta = 1.0

        self.k_final = 0

        ### These settings most recently gave good results

        self.k_u_v = 0.01
        self.k_u_delta = .00001
        self.k_change_u_v = 0.01
        self.k_change_u_delta = 0

        self.k_s = 0
        self.k_x = 0
        self.k_x_dot = -1.0 / 100.0
        self.k_lat = 0.001
        self.k_lon = 0.0

        self.k_phi_error = 0.001
        self.k_phi_dot = 0.0

        self.k_on_grass = 0.1
        ####

        # Constraints
        self.max_steering_rate = 5  # deg/sec
        self.max_delta_u = 5 * np.pi / 180 * self.dt  # rad (change in steering angle)

        self.max_acceleration = 4  #m/s^2
        self.max_v_u = self.max_acceleration * self.dt  # m/s (change in velocity)

        self.max_deceleration = 4  #m/s^2
        self.min_v_u = -self.max_deceleration * self.dt  # m/s  (change in velocity)

        # Speed limit
        self.max_v = 25 * 0.447  # m/s
        self.min_v = 0.0

        # Spatial constraints
        self.max_y = np.infty
        self.min_y = -np.infty

        self.grass_max_y = np.infty
        self.grass_min_y = np.infty

        self.max_X_dev = np.infty
        self.max_Y_dev = np.infty

        # Initialize vehicle dynamics
        self.f = self.gen_f_vehicle_dynamics()
        self.fd = None

        # Distance used for collision avoidance
        self.circle_radius = np.sqrt(2) * self.W / 2.0
        self.min_dist = 2 * self.circle_radius  # 2 times the radius of 1.5
        self.radius = None

        self.ax, self.by = self.get_ellipse(self.L, self.W)  # if you change L, W after construction
        
        self.theta_i_ego = 0
        self.theta_i_jc = [0 for j in range(10)] #TODO:  Change this
        self.theta_i_jnc = [0 for j in range(10)]
        # then it will need to be recalculated

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

    def generate_phidot_cost(self, X):
        ''' Yaw rate cost computed by the dynamics of the vehicle '''
        phid = X[4, :] * cas.tan(X[3, :]) / self.L
        phid_cost = cas.sumsqr(phid)

        return phid_cost

    def generate_costs(self, X, U, X_desired):
        ''' Compute the all the vehicle specific costs corresponding to performance
            of the vehicle as it traverse a desired  trajectory
        '''
        self.u_delta_cost = cas.sumsqr(U[0, :])
        self.u_v_cost = cas.sumsqr(U[1, :])

        self.lon_cost = self.generate_longitudinal_cost(X, X_desired)

        self.phi_error_cost = cas.sumsqr(X_desired[2, :] - X[2, :])
        X_ONLY = False
        if X_ONLY:
            self.s_cost = cas.sumsqr(X[0, -1])
            self.lat_cost = cas.sumsqr(X[1, :])
        else:
            self.lat_cost = self.generate_lateral_cost(X, X_desired)
            self.s_cost = cas.sumsqr(X[5, -1])
        # self.final_costs = self.generate_lateral_cost(X[:,-5:],X_desired[:,-5:]) + cas.sumsqr(X_desired[2,-5:]-X[2,-5:])
        self.final_costs = 0  # for now I've diactivated this
        self.v_cost = cas.sumsqr(X[4, :])
        self.phidot_cost = self.generate_phidot_cost(X)
        N = U.shape[1]
        self.change_u_delta = cas.sumsqr(U[0, 1:N - 1] - U[0, 0:N - 2])
        self.change_u_v = cas.sumsqr(U[1, 1:N - 1] - U[1, 0:N - 2])
        self.x_cost = cas.sumsqr(X[0, :])
        # x_cost = 0
        # for k in range(X.shape[0]):
        #     x_cost += 1.05**k * X[0,k]**2
        # self.x_cost = x_cost
        self.x_dot_cost = cas.sumsqr(X[4, :] * cas.cos(X[2, :]))

    def total_cost(self):
        ''' Compute the total vehicle cost with cost constants and 
        return list of costs for debugging
        '''
        all_costs = [
            self.k_u_delta * self.u_delta_cost, self.k_u_v * self.u_v_cost, self.k_lat * self.lat_cost,
            self.k_lon * self.lon_cost, self.k_phi_error * self.phi_error_cost, self.k_phi_dot * self.phidot_cost,
            self.k_s * self.s_cost, self.k_v * self.v_cost, self.k_change_u_v * self.change_u_v,
            self.k_change_u_delta * self.change_u_delta, self.k_final * self.final_costs, self.k_x * self.x_cost,
            self.k_x_dot * self.x_dot_cost
        ]

        cost_titles = [
            "self.k_u_delta * self.u_delta_cost",
            "self.k_u_v * self.u_v_cost",
            "self.k_lat * self.lat_cost",
            "self.k_lon * self.lon_cost",
            "self.k_phi_error * self.phi_error_cost",
            "self.k_phi_dot * self.phidot_cost",
            "self.k_s * self.s_cost",
            "self.k_v * self.v_cost",
            "self.k_change_u_v * self.change_u_v",
            "self.k_change_u_delta * self.change_u_delta",
            "self.k_final * self.final_costs",
            "self.k_x * self.x_cost",
            "self.k_x_dot * self.x_dot_cost",
        ]

        all_costs = np.array(all_costs)
        total_cost = np.sum(all_costs)
        return total_cost, all_costs, cost_titles

    def add_state_constraints(self, opti, X, U):
        ''' Construct vehicle specific constraints that only rely on
        the ego vehicle's own state '''

        # if self.strict_wall_constraint:  #TODO, change this to when creating min_y and max_y
            # opti.subject_to(opti.bounded(self.min_y + self.W / 2.0, X[1, :], self.max_y - self.W / 2.0))

        opti.subject_to(opti.bounded(-np.pi / 2, X[2, :], np.pi / 2))  #no crazy angle
        opti.subject_to(opti.bounded(self.min_v, X[4, :], self.max_v))

        # constraints on the control inputs
        opti.subject_to(opti.bounded(-self.max_delta_u, U[0, :], self.max_delta_u))
        opti.subject_to(opti.bounded(self.min_v_u, U[1, :], self.max_v_u))  # 0-60 around 4 m/s^2

    def add_dynamics_constraints(self, opti, X, U, X_desired, x0):
        ''' Construct any dynamic constraints based on the kinematic bicycle model
             A Run-Kutta approximation is used to discretize the vehicle dynamics.
        '''
        if self.fd == None:
            raise Exception("No Desired Trajectory Defined")

        # State Dynamics
        N = U.shape[1]
        for k in range(N):
            opti.subject_to(X[:, k + 1] == self.F_kutta(self.f, X[:, k], U[:, k]))

        for k in range(N + 1):
            opti.subject_to(X_desired[:, k] == self.fd(X[-1, k]))  #This should be the trajectory dynamic constraint
        opti.subject_to(X[:, 0] == x0)

    def F_kutta(self, f, x_k, u_k):
        ''' Run-Kutta Approximation of a continuous dynamics f.
        Returns x_{k+1}= f(x_k, u_k) with timestep self.dt
        '''

        k1 = f(x_k, u_k)
        k2 = f(x_k + self.dt / 2 * k1, u_k)
        k3 = f(x_k + self.dt / 2 * k2, u_k)
        k4 = f(x_k + self.dt * k3, u_k)
        x_next = x_k + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return x_next

    def forward_simulate_all(self, x_0: np.array, u_all: np.array) -> Tuple[np.array, np.array]:
        ''' Take an an initial state (x_0) and control inputs
        u_all (of shape 2, N) and compute the state trajectory
        
        x0:  Initial state
        u_all:  Control trajectory (shape=(2,N)
        
        Returns:  state trajectory, x, and desired trajectory x_des)
        '''
        N = u_all.shape[1]
        x = np.zeros(shape=(6, N + 1))
        x[:, 0:1] = x_0.reshape(6, 1)
        for k in range(N):
            u_k = u_all[:, k]
            x_k = x[:, k]
            x_knext = self.F_kutta(self.f, x_k, u_k)
            x[:, k + 1:k + 2] = x_knext

        x_des = np.zeros(shape=(3, N + 1))
        for k in range(N + 1):
            x_des[:, k:k + 1] = self.fd(x[-1, k])

        return x, x_des

    def gen_f_vehicle_dynamics(self, model: str = "kinematic_bicycle"):
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

            ode = cas.vertcat(V * cas.cos(Phi), V * cas.sin(Phi), V * cas.tan(Delta) / self.L, delta_u, v_u, V)

            f = cas.Function('f', [x, u], [ode], ['x', 'u'], ['ode'])
        else:
            raise Exception("Have not implemented non-kinematic bicycle: %s" % model)
        return f

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

    def get_ellipse(self, L, W):
        '''Solve for the minimal inscribing ellipse.
        Inputs:  
            L: Length of vehicle
            W: Width of vehicle
        Returns:
            ax [float]: half length of the major axis  
            by [float]: half length of the minor axis
        '''
        min_elipse_a = lambda a: (1 - L**2 / (2 * a)**2 - W**2 / (2 * a + W - L)**2)

        ax = optimize.fsolve(min_elipse_a, L / 2.0)
        by = ax + .5 * (W - L)

        return ax, by

    def get_collision_ellipse(self, r, L_other=None, W_other=None):
        '''Generate a vehicle's collision ellipse wrt another circle with radius r
        
        Any point xy in collision_ellipse corresponds to an equivalent circle 
        of radius intersection with ellipse with original axis a and b.

        Inputs:  
            r: Radius of vehicle circle
            L_other:   Length of other vehicle
            W_other:   Width of other vehicle

        Outputs:
            a_new:  Major (horizontal) axis of collision ellipse
            b_new: Minor (vertical) axis of collision ellipse
        '''
        #By default, we assume that the dimensions of other vehicle is the
        # the same as the ego vehicle
        if L_other is None:
            L_other = self.L
        if W_other is None:
            W_other = self.W
        a, b = self.get_ellipse(L_other, W_other)  #Generate an ellipse of other vehilce

        #Extend that ellipse to include collision circle using eqn from Alonso-Moro
        minimal_positive_root = lambda delta: (2 * (delta + r)**2 * (2 * a * b + a * (delta + r) + b * (delta + r))) / (
            (a + b) * (a + b + 2 * delta + 2 * r)) - r**2
        delta = optimize.fsolve(minimal_positive_root, r)
        a_new = a + delta + r
        b_new = b + delta + r

        return a_new, b_new, delta, a, b

    def get_theta_ij(self, j):
        if j in self.theta_ij:
            return self.theta_ij[j]
        else:
            return self.theta_i

    def update_desired_lane(self, world, lane_number, right_direction=True):
        self.fd = self.gen_f_desired_lane(world, lane_number, right_direction)

    # Old code:  used to use when we modeled cars as circles
    def get_car_circles(self, X):
        ''' Compuute circles that cover the vehicle area to be used for 
        faster collision checking.
        The radius and center offst (dx) are pre-computed for L=4.5, W=1.8 
        for n_circles = 2 or 3

        Inputs:  Curent state variable
        Returns: a list of xy coordinates and radius for n_circles
         '''

        if self.n_circles == 2:
            r, dx = 1.728843542029462, 0.7738902428000489  #solved using notebook for minimizing radius
            r, dx = 1.75, 0.77389
            x_circle_front = X[0:2, :] + dx * cas.vertcat(cas.cos(X[2, :]), cas.sin(X[2, :]))
            x_circle_rear = X[0:2, :] - dx * cas.vertcat(cas.cos(X[2, :]), cas.sin(X[2, :]))
            radius = 1.5
            min_dist = 2 * radius
            radius = r
            centers = [x_circle_rear, x_circle_front]
        elif self.n_circles == 3:
            r, dx = 1.464421812899125, 1.0947808598616502
            r, dx = 1.47, 1.15
            x_circle_mid = X[0:2, :]
            x_circle_rear = X[0:2, :] - dx * cas.vertcat(cas.cos(X[2, :]), cas.sin(X[2, :]))
            x_circle_front = X[0:2, :] + dx * cas.vertcat(cas.cos(X[2, :]), cas.sin(X[2, :]))

            centers = (x_circle_rear, x_circle_mid, x_circle_front)
            self.radius = r
        return centers, r

    def get_car_circles_np(self, X):
        ''' Same as get_car_circles() however compatible with numpy floats for plotting
        
        Inputs:  Car state as np array
        Outputs:  Circle centers (xy) in numpy array and radius
        '''

        if self.n_circles == 2:
            r, dx = 1.75, 0.77389
            x_circle_front = X[0:2, :] + dx * np.array([np.cos(X[2, :]), np.sin(X[2, :])])
            x_circle_rear = X[0:2, :] - dx * np.array([np.cos(X[2, :]), np.sin(X[2, :])])
            radius = r
            # min_dist = 2*radius
            centers = [x_circle_rear, x_circle_front]
        elif self.n_circles == 3:
            r, dx = 1.47, 1.15
            x_circle_mid = X[0:2, :]
            x_circle_rear = X[0:2, :] - dx * np.array([np.cos(X[2, :]), np.sin(X[2, :])])
            x_circle_front = X[0:2, :] + dx * np.array([np.cos(X[2, :]), np.sin(X[2, :])])

            centers = [x_circle_rear, x_circle_mid, x_circle_front]
            radius = r
            # min_dist = 2*radius
        elif self.n_circles == 1:
            radius = self.L / 2.0
            x_circle = X[0:2, :]
            centers = [x_circle]
        else:
            raise Exception("self.n_circles was not set with a correct number")

        return centers, radius