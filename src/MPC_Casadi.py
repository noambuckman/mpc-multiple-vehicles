import numpy as np
import casadi as cas

class MPC:
    def __init__(self, dt, L=1.0):
        self.dt = dt
        self.k_total = 1.0 # the overall weighting of the total costs
        self.theta_iamb = np.pi/4 # my theta wrt to ambulance
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
        self.k_phi_error = 10.0
        self.k_phi_dot = 5.0

        self.k_change_u = 0.0
        
        # Constraints
        self.max_delta_u = 5 * np.pi/180
        self.max_acceleration = 4
        self.max_v_u = self.max_acceleration * self.dt

        self.max_v =  25 * 0.447 # m/s
        self.max_y = 10
        self.min_y = -1

        self.max_X_dev = np.infty
        self.max_Y_dev = 10.0        

        self.f = self.gen_f_vehicle_dynamics(L)
        self.fd = None

    def generate_lateral_cost(self, X, X_desired):
        lateral_cost = np.sum([(-cas.sin(X_desired[2,k]) * (X[0,k]-X_desired[0,k]) + 
            cas.cos(X_desired[2,k]) * (X[1,k]-X_desired[1,k]))**2
           for k in range(X.shape[1])])
        
        return lateral_cost
    
    def generate_longitudinal_cost(self, X, X_desired):
        longitudinal_cost = np.sum([(cas.cos(X_desired[2,k]) * (X[0,k]-X_desired[0,k]) + 
            cas.sin(X_desired[2,k]) * (X[1,k]-X_desired[1,k]))**2
           for k in range(X.shape[1])]) 
        
        return longitudinal_cost
    
    def generate_phidot_cost(self, X, L = 1.0):
        phid = X[4,:] * cas.tan(X[3,:]) / L
        phid_cost = cas.sumsqr(phid)    
        
        return phid_cost
    
    def generate_costs(self, X, U, X_desired):
        self.u_delta_cost = self.k_u_delta * cas.sumsqr(U[0,:])
        self.u_v_cost = cas.sumsqr(U[1,:])

        self.lon_cost = self.generate_longitudinal_cost(X, X_desired)
        
        self.phi_error_cost = cas.sumsqr(X_desired[2,:]-X[2,:]) 
        X_ONLY = False
        if X_ONLY:
            self.s_cost = cas.sumsqr(X[0,-1])
            self.lat_cost = cas.sumsqr(X[1,:])
        else:
            self.lat_cost = self.generate_lateral_cost(X, X_desired)
            self.s_cost = cas.sumsqr(X[5,-1])   
        self.phidot_cost = self.generate_phidot_cost(X)
        N = U.shape[1] 
        self.change_u = cas.sumsqr(U[:,1:N-1] - U[:,0:N-2])

    def total_cost(self):
        total_cost = (
            self.k_u_delta * self.u_delta_cost + 
            self.k_u_v * self.u_v_cost + 
            self.k_lat * self.lat_cost + 
            self.k_lon * self.lon_cost + 
            self.k_phi_error * self.phi_error_cost + 
            self.k_phi_dot * self.phidot_cost +
            self.k_s * self.s_cost + 
            self.k_change_u * self.change_u
            )
        return total_cost 

    def add_state_constraints(self, opti, X, U, X_desired, T):

        opti.subject_to( opti.bounded(-1, X[0,:], self.max_v * T) ) #Constraints on X, Y
        opti.subject_to( opti.bounded(self.min_y, X[1,:], self.max_y) )
        opti.subject_to( opti.bounded(-np.pi/2, X[2,:], np.pi/2) ) #no crazy angle

        opti.subject_to(opti.bounded(-self.max_delta_u, U[0,:], self.max_delta_u))            
        opti.subject_to(opti.bounded(-self.max_v_u, U[1,:], self.max_v_u)) # 0-60 around 4 m/s^2
        opti.subject_to(opti.bounded(0, X[4,:], self.max_v))    

        # Lane Deviations
        if self.max_X_dev < np.infty:
            opti.subject_to( opti.bounded(-self.max_X_dev, X[0,:] - X_desired[0,:], self.max_X_dev))
        opti.subject_to( opti.bounded(-self.max_Y_dev, X[1,:] - X_desired[1,:], self.max_Y_dev))


    def add_dynamics_constraints(self, opti, X, U, X_desired, x0):
        if self.fd == None:
            raise Exception("No Desired Trajectory Defined")

        # State Dynamics
        N = U.shape[1]
        for k in range(N):
            # opti.subject_to( X[:, k+1] == F(X[:, k], U[:, k], dt))
            opti.subject_to( X[:, k+1] == self.F_kutta(self.f, X[:, k], U[:, k]))
        
        for k in range(N+1):
            opti.subject_to( X_desired[:, k] == self.fd(X[-1, k]) ) #This should be the trajectory dynamic constraint             

        opti.subject_to(X[:,0] == x0)

    def F_kutta(self, f, x_k, u_k):
        k1 = f(x_k, u_k)
        k2 = f(x_k+self.dt/2*k1, u_k)
        k3 = f(x_k+self.dt/2*k2, u_k)
        k4 = f(x_k+self.dt*k3,   u_k)
        x_next = x_k + self.dt/6*(k1+2*k2+2*k3+k4) 
        return x_next
    
    def gen_f_vehicle_dynamics(self, L=1.0):
        X = cas.MX.sym('X')
        Y = cas.MX.sym('Y')
        Phi = cas.MX.sym('Phi')
        Delta = cas.MX.sym('Delta')
        V = cas.MX.sym('V')
        s = cas.MX.sym('s')

        delta_u = cas.MX.sym('delta_u')
        v_u = cas.MX.sym('v_u')
        x = cas.vertcat(X, Y, Phi, Delta, V, s)
        u = cas.vertcat(delta_u, v_u)

        ode = cas.vertcat(V * cas.cos(Phi),
                        V * cas.sin(Phi),
                        V * cas.tan(Delta) / L,
                        delta_u,
                        v_u,
                        V)

        f = cas.Function('f',[x,u],[ode],['x','u'],['ode'])
        return f

class OptimizationMPC():
    def __init__(self, car1MPC, car2MPC,  ambMPC):
        self.car1MPC = car1MPC
        self.car2MPC = car2MPC   
        self.ambMPC = ambMPC
        self.opti = cas.Opti()

    def generate_optimization(self, N, min_dist, fd, T, x0, x0_2, x0_amb, print_level=5):

        t_amb_goal = self.opti.variable()
        n_state, n_ctrl = 6, 2
        #Variables
        self.x_opt, self.x2_opt, self.xamb_opt = self.opti.variable(n_state, N+1), self.opti.variable(n_state, N+1), self.opti.variable(n_state, N+1)
        self.u_opt, self.u2_opt, self.uamb_opt = self.opti.variable(n_ctrl, N), self.opti.variable(n_ctrl, N), self.opti.variable(n_ctrl, N)
        self.x_desired, self.x2_desired, self.xamb_desired = self.opti.variable(3, N+1), self.opti.variable(3, N+1), self.opti.variable(3, N+1)
        p, p2, pamb = self.opti.parameter(n_state, 1), self.opti.parameter(n_state, 1), self.opti.parameter(n_state, 1)

        #### Costs
        self.car1MPC.generate_costs(self.x_opt, self.u_opt, self.x_desired)
        car1_costs = self.car1MPC.total_cost()

        self.car2MPC.generate_costs(self.x2_opt, self.u2_opt, self.x2_desired)
        car2_costs = self.car2MPC.total_cost()



        self.ambMPC.generate_costs(self.xamb_opt, self.uamb_opt, self.xamb_desired)
        amb_costs = self.ambMPC.total_cost()

        theta_1 = np.pi/4
        theta_2 = np.pi/4
        theta_amb = 0
        ######## optimization  ##################################
        self.opti.minimize(np.cos(theta_amb)*amb_costs + np.sin(theta_amb)*(car1_costs + car2_costs) + 
                    (np.cos(theta_1)*car1_costs + np.sin(theta_1)*amb_costs) + 
                    (np.cos(theta_2)*car2_costs + np.sin(theta_2)*amb_costs)
                    )    
        ##########################################################

        #constraints
        self.car1MPC.add_dynamics_constraints(self.opti, self.x_opt, self.u_opt, self.x_desired, p)
        self.car1MPC.add_state_constraints(self.opti, self.x_opt, self.u_opt, self.x_desired, T )

        self.car2MPC.add_state_constraints(self.opti, self.x2_opt, self.u2_opt, self.x2_desired, T )
        self.car2MPC.add_dynamics_constraints(self.opti, self.x2_opt, self.u2_opt, self.x2_desired, p2)


        self.ambMPC.add_state_constraints(self.opti, self.xamb_opt, self.uamb_opt, self.xamb_desired, T)
        self.ambMPC.add_dynamics_constraints(self.opti, self.xamb_opt, self.uamb_opt, self.xamb_desired, pamb)

        # Collision Avoidance
        for k in range(N+1):
            self.opti.subject_to( cas.sumsqr(self.x_opt[0:2,k] - self.x2_opt[0:2,k]) > min_dist**2 )
            self.opti.subject_to( cas.sumsqr(self.x_opt[0:2,k] - self.xamb_opt[0:2,k]) > min_dist**2 )
            self.opti.subject_to( cas.sumsqr(self.x2_opt[0:2,k] - self.xamb_opt[0:2,k]) > min_dist**2 )

        self.opti.set_value(p, x0)
        self.opti.set_value(p2, x0_2)
        self.opti.set_value(pamb, x0_amb) 
        self.opti.solver('ipopt',{'warn_initial_bounds':True},{'print_level':print_level})

    def warm_start(self, u1, u2, uamb, x0, x0_2, x0_amb):
        N = u1.shape[1]

        x_warm = np.zeros((6,N+1))
        x2_warm = np.zeros((6,N+1))
        xamb_warm = np.zeros((6,N+1))


        x_warm[:,0] = x0
        x2_warm[:,0] = x0_2
        xamb_warm[:,0] = x0_amb
        for k in range(N):
            x_warm[:,k+1:k+2] = self.car1MPC.F_kutta(self.car1MPC.f, x_warm[:, k], u1[:, k])
            x2_warm[:,k+1:k+2] = self.car2MPC.F_kutta(self.car2MPC.f, x2_warm[:, k], u2[:, k])
            xamb_warm[:,k+1:k+2] = self.ambMPC.F_kutta(self.ambMPC.f, xamb_warm[:, k], uamb[:, k])

        self.opti.set_initial(self.u_opt, u1)
        self.opti.set_initial(self.u2_opt, u2)
        self.opti.set_initial(self.uamb_opt, uamb)

        self.opti.set_initial(self.x_opt, x_warm)
        self.opti.set_initial(self.x2_opt, x2_warm)
        self.opti.set_initial(self.xamb_opt, xamb_warm)

        return x_warm, x2_warm, xamb_warm




    def get_solution(self):
        self.solution = self.opti.solve()
        x1 = self.solution.value(self.x_opt)
        u1 = self.solution.value(self.u_opt)

        x2 = self.solution.value(self.x2_opt)
        u2 = self.solution.value(self.u2_opt)

        xamb = self.solution.value(self.xamb_opt)
        uamb = self.solution.value(self.uamb_opt)

        x1_des = self.solution.value(self.x_desired)
        x2_des = self.solution.value(self.x2_desired)
        xamb_des = self.solution.value(self.xamb_desired)

        return x1, u1, x1_des, x2, u2, x2_des, xamb, uamb, xamb_des



class IterativeBestResponseMPC(OptimizationMPC):
    ### We always assume that car1 is the one being optimized
    def __init__(self, car1MPC, car2MPC, ambMPC):
        OptimizationMPC.__init__(self, car1MPC, car2MPC, ambMPC)            

    def generate_optimization(self, N, min_dist, fd, T, x0, x0_2, x0_amb, print_level=5):

        t_amb_goal = self.opti.variable()
        n_state, n_ctrl = 6, 2
        #Variables
        self.x_opt = self.opti.variable(n_state, N+1)
        self.u_opt  = self.opti.variable(n_ctrl, N)
        self.x_desired  = self.opti.variable(3, N+1)
        p = self.opti.parameter(n_state, 1)

        # Presume to be given...and we will initialize soon
        self.x2_opt, self.xamb_opt = self.opti.parameter(n_state, N+1), self.opti.parameter(n_state, N+1)
        self.u2_opt, self.uamb_opt = self.opti.parameter(n_ctrl, N), self.opti.parameter(n_ctrl, N)
        self.x2_desired, self.xamb_desired = self.opti.parameter(3, N+1), self.opti.parameter(3, N+1)
        p2, pamb = self.opti.parameter(n_state, 1), self.opti.parameter(n_state, 1)

        #### Costs
        self.car1MPC.generate_costs(self.x_opt, self.u_opt, self.x_desired)
        car1_costs = self.car1MPC.total_cost()

        self.ambMPC.generate_costs(self.xamb_opt, self.uamb_opt, self.xamb_desired)
        amb_costs = self.ambMPC.total_cost()

        # theta_1 = np.pi/4
        # theta_2 = np.pi/4
        # theta_amb = 0
        ######## optimization  ##################################
        self.opti.minimize(
                    (np.cos(self.car1MPC.theta_iamb)*car1_costs + np.sin(self.car1MPC.theta_iamb)*amb_costs) 
                    )    
        ##########################################################

        #constraints
        self.car1MPC.add_dynamics_constraints(self.opti, self.x_opt, self.u_opt, self.x_desired, p)
        self.car1MPC.add_state_constraints(self.opti, self.x_opt, self.u_opt, self.x_desired, T)

        # Collision Avoidance
        for k in range(N+1):
            self.opti.subject_to( cas.sumsqr(self.x_opt[0:2,k] - self.x2_opt[0:2,k]) > min_dist**2 )
            self.opti.subject_to( cas.sumsqr(self.x_opt[0:2,k] - self.xamb_opt[0:2,k]) > min_dist**2 )

        self.opti.set_value(p, x0)
        self.opti.set_value(p2, x0_2)
        self.opti.set_value(pamb, x0_amb) 
        self.opti.solver('ipopt',{'warn_initial_bounds':True},{'print_level':print_level})


    def get_solution(self, x2, u2, x2_desired, xamb, uamb, xamb_desired):

        self.opti.set_value(self.x2_opt, x2)
        self.opti.set_value(self.xamb_opt, xamb)
        self.opti.set_value(self.u2_opt, u2) 
        self.opti.set_value(self.uamb_opt, uamb) 
        self.opti.set_value(self.x2_desired, x2_desired)
        self.opti.set_value(self.xamb_desired, xamb_desired)

        self.solution = self.opti.solve()
        x1 = self.solution.value(self.x_opt)
        u1 = self.solution.value(self.u_opt)
        x1_des = self.solution.value(self.x_desired)


        return x1, u1, x1_des



