import numpy as np
import casadi as cas
import src.MPC_Casadi as mpc




class IterativeBestResponseMPCMultiple:
    ### We always assume that car1 is the one being optimized
    def __init__(self, responseMPC, ambulanceMPC, otherMPClist):

        self.responseMPC = responseMPC
        self.otherMPClist = otherMPClist   
        self.ambMPC = ambulanceMPC

        self.opti = cas.Opti()
        self.min_dist = 2 * 1.5   # 2 times the radius of 1.5
        self.k_slack = 99999
        self.k_CA = 0
        self.collision_cost = 0


    def generate_optimization(self, N, T, x0, x0_amb, x0_other, print_level=5, slack=True):
        
        # t_amb_goal = self.opti.variable()
        n_state, n_ctrl = 6, 2
        #Variables
        self.x_opt = self.opti.variable(n_state, N+1)
        # self.x_opt = cas.sym('x_opt',n_state, N+1))
        
        self.u_opt  = self.opti.variable(n_ctrl, N)
        self.x_desired  = self.opti.variable(3, N+1)
        # self.x_desired = cas.MX.sym('x_des', 3, N+1)
        p = self.opti.parameter(n_state, 1)

        # Presume to be given...and we will initialize soon
        if self.ambMPC:
            self.xamb_opt = self.opti.variable(n_state, N+1)
            self.uamb_opt = self.opti.parameter(n_ctrl, N)
            self.xamb_desired = self.opti.variable(3, N+1)
            pamb = self.opti.parameter(n_state, 1)

        self.allother_x_opt = [self.opti.variable(n_state, N+1) for i in self.otherMPClist] 
        self.allother_u_opt = [self.opti.parameter(n_ctrl, N) for i in self.otherMPClist] 
        self.allother_x_desired = [self.opti.variable(3, N+1) for i in self.otherMPClist] 
        # self.allother_x_desired = [cas.MX.sym('x_des', 3, N+1) for i in self.otherMPClist] 
        self.allother_p = [self.opti.parameter(n_state, 1) for i in self.otherMPClist]

        #### Costs
            
        self.responseMPC.generate_costs(self.x_opt, self.u_opt, self.x_desired)
        self.car1_costs, self.car1_costs_list = self.responseMPC.total_cost()

        if self.ambMPC:
            self.ambMPC.generate_costs(self.xamb_opt, self.uamb_opt, self.xamb_desired)
            self.amb_costs, self.amb_costs_list = self.ambMPC.total_cost()
        else:
            self.amb_costs, self.amb_costs_list = 0, []

        ## We don't need the costs for the other vehicles

        # self.slack1, self.slack2, self.slack3 = self.generate_slack_variables(slack, N)

        # We will do collision avoidance for ego vehicle with all other vehicles
        self.slack_vars_list = self.generate_slack_variables(slack, N, len(self.otherMPClist), n_circles = self.responseMPC.n_circles)
        
        if self.ambMPC:    
            self.slack_amb = self.generate_slack_variables(slack, N, 1)[0]

        self.slack_cost = 0
        for slack_var in self.slack_vars_list:
            self.slack_cost += cas.sumsqr(slack_var)
        
        if self.ambMPC:    
            self.slack_cost += cas.sumsqr(self.slack_amb)


        self.response_svo_cost = np.cos(self.responseMPC.theta_iamb)*self.car1_costs
        self.other_svo_cost = np.sin(self.responseMPC.theta_iamb)*self.amb_costs
        self.total_svo_cost = self.response_svo_cost + self.other_svo_cost + self.k_slack * self.slack_cost + self.k_CA * self.collision_cost
        ######## optimization  ##################################
        
        self.opti.minimize(self.total_svo_cost)    
        ##########################################################

        #constraints
        self.responseMPC.add_dynamics_constraints(self.opti, self.x_opt, self.u_opt, self.x_desired, p)
        self.responseMPC.add_state_constraints(self.opti, self.x_opt, self.u_opt, self.x_desired, T, x0)
        
        if self.ambMPC:        
            self.ambMPC.add_dynamics_constraints(self.opti, self.xamb_opt, self.uamb_opt, self.xamb_desired, pamb)
        
        for i in range(len(self.otherMPClist)):
            self.otherMPClist[i].add_dynamics_constraints(self.opti, 
                                                        self.allother_x_opt[i], self.allother_u_opt[i], self.allother_x_desired[i], 
                                                        self.allother_p[i])

        ## Generate the circles
        


        # Proxy for the collision avoidance points on each vehicle
        self.c1_vars = [self.opti.variable(2, N+1) for c in range(self.responseMPC.n_circles)]
        if self.ambMPC:  
            ca_vars = [self.opti.variable(2, N+1) for c in range(self.responseMPC.n_circles)]  

        self.other_circles = [[self.opti.variable(2, N+1) for c in range(self.responseMPC.n_circles)] for i in range(len(self.allother_x_opt))]
        self.collision_cost = 0

        # Before we start, compute the alpha beta geometry parameters for the other vehicles
        alphas = []
        betas = []
        centers, response_radius = self.responseMPC.get_car_circles(self.x_opt[:,0]) 

        for i in range(len(self.otherMPClist)):
            a_other, b_other, delta, a, b = self.otherMPClist[i].get_collision_ellipse(response_radius)
            alphas += [a_other]
            betas += [b_other]
        
        if self.ambMPC:
            a_amb, b_amb, delta, a, b = self.ambMPC.get_collision_ellipse(r)


        # Collision Avoidance
        for k in range(N+1):
            # center_offset
            centers, response_radius = self.responseMPC.get_car_circles(self.x_opt[:,k]) 
            for c1_circle in centers:
                for i in range(len(self.allother_x_opt)):
                    initial_distance = cas.sqrt(cas.sumsqr(x0_other[i] - x0))
                    if initial_distance <= 20: #collision avoidance distance for other cars
                        # print("CA:  Car %d, t%d %.04f"%(i, k, initial_distance))
                        r = self.responseMPC.radius ### THIS NEEDS TO BE ACCURATE
                        
                        self.generate_collision_ellipse(c1_circle[0], c1_circle[1], 
                                                        self.allother_x_opt[i][0,k], self.allother_x_opt[i][1,k], self.allother_x_opt[i][2,k],
                                                        alphas[i], betas[i], None)
                        CIRCLE = False
                        if CIRCLE:
                            other_centers, other_radius = self.otherMPClist[i].get_car_circles(self.allother_x_opt[i][:,k])    
                            for ci in range(len(other_centers)):
                                dist_sqr = cas.sumsqr(c1_circle - other_centers[ci])
                                dist_btw_object = cas.sqrt(dist_sqr) - (response_radius + other_radius)
                                self.opti.subject_to(dist_btw_object > 0 - self.slack_vars_list[i][ci,k])
                                distance_clipped = cas.fmax(dist_btw_object, 0.001)
                                # dist_btw_object = cas.fmax(cas.sqrt(dist_sqr) - 1.1*(response_radius + other_radius), 0.00001)

                                self.collision_cost += 10/distance_clipped**2
                # Don't forget the ambulance
                if self.ambMPC:    
                    amb_circles, amb_radius = self.ambMPC.get_car_circles(self.xamb_opt[:,k])
                    CIRCLE = False
                    if CIRCLE:    
                        for ci in range(len(amb_circles)):                    
                            self.opti.subject_to(cas.sumsqr(c1_circle - amb_circles[ci]) > (response_radius + amb_radius)**2 - self.slack_amb[ci,k])
                            dist_btw_object = cas.fmax(cas.sqrt(cas.sumsqr(c1_circle - amb_circles[ci])) - 1.1*(response_radius + amb_radius), 0.00001)
                            self.collision_cost += 10/dist_btw_object**2
                    else:
                        self.generate_collision_ellipse(c1_circle[0], c1_circle[1], 
                                                                                self.xamb_opt[0,k], self.xamb_opt[1,k], self.xamb_opt[2,k],
                                                                                a_amb, b_amb, None)                        
                
                WALL_CA = True
                if WALL_CA:
                    dist_btw_wall_bottom = c1_circle[1] - (self.responseMPC.min_y + self.responseMPC.W/2.0)
                    dist_btw_wall_top = c1_circle[1] - (self.responseMPC.max_y - self.responseMPC.W/2.0)
                    self.collision_cost += 0.1 * 1/dist_btw_wall_bottom**2
                    self.collision_cost += 0.1 * 1/dist_btw_wall_top**2

  

        self.opti.set_value(p, x0)

        for i in range(len(self.allother_p)):
            self.opti.set_value(self.allother_p[i], x0_other[i])

        if self.ambMPC:    
            self.opti.set_value(pamb, x0_amb) 

        self.opti.solver('ipopt',{'warn_initial_bounds':True},
        {'print_level':print_level, 'max_iter':10000, 'constr_viol_tol':0.0001})


    def solve(self, uamb, uother):

        if self.ambMPC:    
            self.opti.set_value(self.uamb_opt, uamb)

        for i in range(len(self.allother_u_opt)):
            self.opti.set_value(self.allother_u_opt[i], uother[i])
        # self.opti.set_value(self.u2_opt, u2) 

        self.solution = self.opti.solve()         

    def get_bestresponse_solution(self):
        x1, u1, x1_des, = self.solution.value(self.x_opt), self.solution.value(self.u_opt), self.solution.value(self.x_desired) 

        return x1, u1, x1_des

    def get_solution(self):
        x1, u1, x1_des, = self.solution.value(self.x_opt), self.solution.value(self.u_opt), self.solution.value(self.x_desired) 
        if self.ambMPC:    
            xamb, uamb, xamb_des, = self.solution.value(self.xamb_opt), self.solution.value(self.uamb_opt), self.solution.value(self.xamb_desired) 
        else:
            xamb, uamb, xamb_des, = None, None, None

        other_x = [self.solution.value(self.allother_x_opt[i]) for i in range(len(self.allother_x_opt))]
        other_u = [self.solution.value(self.allother_u_opt[i]) for i in range(len(self.allother_u_opt))]
        other_des = [self.solution.value(self.allother_x_desired[i]) for i in range(len(self.allother_x_desired))]
        return x1, u1, x1_des, xamb, uamb, xamb_des, other_x, other_u, other_des

    def generate_slack_variables(self, slack, N, number_slack_vars=3, n_circles=2):
        if slack == True:
            slack_vars = [self.opti.variable(n_circles, N+1) for i in range(number_slack_vars)]
            for slack in slack_vars:
                self.opti.subject_to(cas.vec(slack)>=0)
                # self.opti.subject_to(slack<=1.0)
        else:
            slack_vars = [self.opti.parameter(n_circles, N+1) for i in range(number_slack_vars)]
            for slack in slack_vars:
                self.opti.set_value(slack, np.zeros((n_circles,N+1)))
        return slack_vars

    def generate_collision_ellipse(self, x_e, y_e, x_o, y_o, phi_o, alpha_o, beta_o, slack):
        dx = x_o - x_e
        dy = y_o - y_o
        if slack is None:
            slack = 0
        R_o = np.array([[cas.cos(phi_o), 0],[0, cas.sin(phi_o)]])

        dX = cas.vertcat(dx, dy)
        M_ellipse = np.array([[1/alpha_o**2, 0],[0, 1/beta_o**2]])      
        self.opti.subject_to(cas.mtimes([dX.T, R_o.T, M_ellipse, R_o, dX]) > (1 - slack))


def load_state(file_name, n_others):
    xamb = np.load(file_name + "xamb.npy",allow_pickle=False)
    uamb = np.load(file_name + "uamb.npy",allow_pickle=False)
    xamb_des = np.load(file_name + "xamb_des.npy",allow_pickle=False)


    xothers, uothers, xothers_des = [], [], []
    for i in range(n_others):
        x = np.load(file_name + "x%0d.npy"%i, allow_pickle=False)
        u = np.load(file_name + "u%0d.npy"%i, allow_pickle=False)
        x_des = np.load(file_name + "x_des%0d.npy"%i, allow_pickle=False)
        xothers += [x]
        uothers += [u]
        xothers_des += [x_des]

    return xamb, uamb, xamb_des, xothers, uothers, xothers_des


def save_state(file_name, xamb, uamb, xamb_des, xothers, uothers, xothers_des):
    np.save(file_name + "xamb", xamb,allow_pickle=False)
    np.save(file_name + "uamb", uamb,allow_pickle=False)
    np.save(file_name + "xamb_des", xamb_des, allow_pickle=False)

    for i in range(len(xothers)):
        x, u, x_des = xothers[i], uothers[i], xothers_des[i]    
        np.save(file_name + "x%0d"%i, x, allow_pickle=False)
        np.save(file_name + "u%0d"%i, u, allow_pickle=False)
        np.save(file_name + "x_des%0d"%i, x_des, allow_pickle=False)
    
    return file_name

def save_costs(file_name, ibr):
    ### Get the value for each cost variable
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

    car1_costs_list = np.load(file_name + "car1_costs_list.npy", allow_pickle=False)
    amb_costs_list = np.load(file_name + "amb_costs_list.npy", allow_pickle=False)
    svo_cost = np.load(file_name + "svo_cost.npy", allow_pickle=False)
    other_svo_cost = np.load(file_name + "other_svo_cost.npy", allow_pickle=False)
    total_svo_cost = np.load(file_name + "total_svo_cost.npy", allow_pickle=False)


    return car1_costs_list, amb_costs_list, svo_cost, other_svo_cost , total_svo_cost


def load_costs_int(i):

    car1_costs_list = np.load("%03dcar1_costs_list.npy"%i, allow_pickle=False)
    amb_costs_list = np.load("%03damb_costs_list.npy"%i, allow_pickle=False)
    svo_cost = np.load("%03dsvo_cost.npy"%i, allow_pickle=False)
    other_svo_cost = np.load("%03dother_svo_cost.npy"%i, allow_pickle=False)
    total_svo_cost = np.load("%03dtotal_svo_cost.npy"%i, allow_pickle=False)


    return car1_costs_list, amb_costs_list, svo_cost, other_svo_cost , total_svo_cost


    