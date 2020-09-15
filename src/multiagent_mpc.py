import numpy as np
import casadi as cas
import matplotlib.pyplot as plt

import src.vehicle as vehicle
import src.car_plotting_multiple as cmplot
import src.traffic_world as tw

class MultiMPC(object):
    ### We always assume that car1 is the one being optimized
    def __init__(self, 
                responseMPC : vehicle.Vehicle, 
                ambulanceMPC : vehicle.Vehicle, 
                otherMPClist,
                world, 
                solver_params = None):

        self.responseMPC = responseMPC
        self.otherMPClist = otherMPClist   
        self.ambMPC = ambulanceMPC
        if solver_params is None:
            solver_params = {}
            self.k_slack = 99999
            self.k_CA = 0
            self.k_CA_power = 8
            self.collision_cost = 0      
            self.WALL_CA = False
        else:
            self.k_slack = solver_params['k_slack']
            self.k_CA = solver_params['k_CA']
            self.k_CA_power = solver_params['k_CA_power']
            self.WALL_CA = solver_params['wall_CA']
        
        self.world = world

        self.opti = cas.Opti()
        self.min_dist = 2 * 1.5   # 2 times the radius of 1.5




    def generate_optimization(self, N, T, x0, x0_amb, x0_other, print_level=5, slack=True, solve_amb=False, params = None, ipopt_params=None):

        n_state, n_ctrl, n_desired = 6, 2, 3
        #Response (planning) Vehicle Variables
        self.x_opt = self.opti.variable(n_state, N+1)        
        self.u_opt  = self.opti.variable(n_ctrl, N)
        self.x_desired  = self.opti.variable(n_desired, N+1)
        p = self.opti.parameter(n_state, 1)

        # Ambulance Variables
        if self.ambMPC:
            if solve_amb:
                self.xamb_opt = self.opti.variable(n_state, N+1)
                self.uamb_opt = self.opti.variable(n_ctrl, N)
                self.xamb_desired = self.opti.variable(n_desired, N+1)                
            else:
                self.xamb_opt = self.opti.parameter(n_state, N+1)
                self.uamb_opt = self.opti.parameter(n_ctrl, N)
                self.xamb_desired = self.opti.parameter(n_desired, N+1)
            pamb = self.opti.parameter(n_state, 1)

        ### Variables of surrounding vehicles assumed fixed (TODO: This could probably be np arrays)
        if len(self.otherMPClist) > 0:
            self.allother_x_opt = [self.opti.parameter(n_state, N+1) for i in self.otherMPClist] 
            self.allother_u_opt = [self.opti.parameter(n_ctrl, N) for i in self.otherMPClist] 
            self.allother_x_desired = [self.opti.parameter(3, N+1) for i in self.otherMPClist]
            self.allother_p = [self.opti.parameter(n_state, 1) for i in self.otherMPClist]

        #### Costs
        self.responseMPC.generate_costs(self.x_opt, self.u_opt, self.x_desired)
        self.car1_costs, self.car1_costs_list, self.car1_cost_titles = self.responseMPC.total_cost()

        if self.ambMPC:
            self.ambMPC.generate_costs(self.xamb_opt, self.uamb_opt, self.xamb_desired)
            self.amb_costs, self.amb_costs_list, self.amb_cost = self.ambMPC.total_cost()
        else:
            self.amb_costs, self.amb_costs_list = 0, []

        ## Generate Slack Variables used as part of collision avoidance
        if len(self.otherMPClist) > 0:
            self.slack_vars_list = self.generate_slack_variables(slack, N, len(self.otherMPClist), n_ego_circles = self.responseMPC.n_circles)
        else:
            self.slack_vars_list = []
        
        self.slack_cost = 0       
        for agent_i in range(len(self.slack_vars_list)):
            for ci in range(self.slack_vars_list[agent_i].shape[0]):
                for t in range(self.slack_vars_list[agent_i].shape[1]):
                    self.slack_cost += self.slack_vars_list[agent_i][ci,t]**2

        if self.ambMPC:    
            self.slack_amb = self.generate_slack_variables(slack, N, 1, n_ego_circles = self.responseMPC.n_circles)[0]
            self.slack_cost += cas.sumsqr(self.slack_amb)            

        if solve_amb and self.ambMPC:    
            if len(self.otherMPClist) > 0:
                self.slack_amb_other = self.generate_slack_variables(slack, N, len(self.otherMPClist), n_ego_circles = self.responseMPC.n_circles)
            else:
                self.slack_amb_other = []
            for slack_var in self.slack_amb_other:
                for i in range(slack_var.shape[0]):
                    for j in range(slack_var.shape[1]):
                        self.slack_cost += slack_var[i,j]**2
        
        self.response_svo_cost = np.cos(self.responseMPC.theta_iamb)*self.car1_costs
        self.other_svo_cost = np.sin(self.responseMPC.theta_iamb)*self.amb_costs

        ##### Add Constraints
        self.responseMPC.add_dynamics_constraints(self.opti, self.x_opt, self.u_opt, self.x_desired, p)
        self.responseMPC.add_state_constraints(self.opti, self.x_opt, self.u_opt, self.x_desired, T)

        
        if solve_amb and self.ambMPC: # only add constraints if we're solving ampMPC
            self.ambMPC.add_dynamics_constraints(self.opti, self.xamb_opt, self.uamb_opt, self.xamb_desired, pamb)
            self.ambMPC.add_state_constraints(self.opti, self.xamb_opt, self.uamb_opt, self.xamb_desired, T)

        ######################### Compute Collision Avoidance #########################

        # Proxy for the collision avoidance points on each vehicle
        
        # Compute the ellipse alpha beta geometry parameters for the other vehicles #
        alphas = []
        betas = []
        rv, response_radius = self.responseMPC.get_car_circles(self.x_opt[:,0]) 
        for i in range(len(self.otherMPClist)):
            a_other, b_other, delta, a, b = self.otherMPClist[i].get_collision_ellipse(response_radius)
            alphas += [a_other]
            betas += [b_other]
        if self.ambMPC:
            a_amb, b_amb, delta, a, b = self.ambMPC.get_collision_ellipse(response_radius)

        self.pairwise_distances = [] #keep track of all the distances between ego and ado vehicles
        self.collision_cost = 0
        # Collision Avoidance
        
        for k in range(N+1):
            # Compute response vehicles collision center points
            response_centers, response_radius = self.responseMPC.get_car_circles(self.x_opt[:,k]) 
            for center_i in range(len(response_centers)):
                response_circle_xy = response_centers[center_i]
                for i in range(len(self.allother_x_opt)):
                    initial_displacement = x0_other[i] - x0
                    initial_xy_distance = cas.sqrt(initial_displacement[0]**2 + initial_displacement[1]**2)
                    if initial_xy_distance <= params["collision_avoidance_checking_distance"]: #collision avoidance distance for other cars 
                        # buffer_distance, dist = self.generate_collision_ellipse(response_circle_xy[0], response_circle_xy[1], 
                        #                                 self.allother_x_opt[i][0,k], self.allother_x_opt[i][1,k], self.allother_x_opt[i][2,k],
                        #                                 alphas[i], betas[i], None)

                        dist = self.minkowski_ellipse_collision_distance(self.responseMPC, self.otherMPClist[i], 
                                                                                self.x_opt[0, k], self.x_opt[1,k], self.x_opt[2,k], 
                                                                                self.allother_x_opt[i][0,k], self.allother_x_opt[i][1,k], self.allother_x_opt[i][2,k])
                        
                        self.pairwise_distances += [dist]
                        self.opti.subject_to(dist >= (1 - self.slack_vars_list[i][center_i, k]))
                        distance_clipped = cas.fmax(dist, 0.00001) ## This can be a smaller distance if we'd like
                        self.collision_cost += 1/distance_clipped**self.k_CA_power      
               
                if self.ambMPC:  # Don't forget the ambulance collision avoidance   
                    dist = self.minkowski_ellipse_collision_distance(self.responseMPC, self.ambMPC, 
                                                                        self.x_opt[0, k], self.x_opt[1,k], self.x_opt[2,k], 
                                                                        self.xamb_opt[0,k], self.xamb_opt[1,k], self.xamb_opt[2,k])
                        
                    self.opti.subject_to(dist >= 1 - self.slack_amb[center_i, k] )         
                    self.pairwise_distances += [dist]
                    distance_clipped = cas.fmax(dist, 0.00001)
                    self.collision_cost += 1/distance_clipped**self.k_CA_power  
                
                if self.WALL_CA: #Add a collision cost related to distance from wall
                    dist_btw_wall_bottom =  response_circle_xy[1] - (self.responseMPC.min_y + self.responseMPC.W/2.0) 
                    dist_btw_wall_top = (self.responseMPC.max_y - self.responseMPC.W/2.0) - response_circle_xy[1]

                    self.collision_cost += 0.1 * 1/(cas.fmax(dist_btw_wall_bottom, 0.0001)**self.k_CA_power)
                    self.collision_cost += 0.1 * 1/(cas.fmax(dist_btw_wall_top, 0.0001)**self.k_CA_power)
        
        if solve_amb and self.ambMPC:  
            self.ca_vars = [self.opti.variable(2, N+1) for c in range(self.responseMPC.n_circles)]  
            for k in range(N+1):
                ## Genereate collision circles for ambulance and do collision avoidance
                if solve_amb and self.ambMPC:
                    for ci in range(len(self.ca_vars)):
                        ca_circle_xy = self.ca_vars[ci]
                        for i in range(len(self.allother_x_opt)):
                            initial_displacement = x0_other[i] - x0_amb
                            initial_xy_distance = cas.sqrt(initial_displacement[0]**2 + initial_displacement[1]**2)
                            if initial_xy_distance <= params["collision_avoidance_checking_distance"]: #collision avoidance distance for other cars                        
                                # buffer_distance, dist = self.generate_collision_ellipse(ca_circle_xy[0], ca_circle_xy[1], 
                                #                                 self.allother_x_opt[i][0,k], self.allother_x_opt[i][1,k], self.allother_x_opt[i][2,k],
                                #                                 alphas[i], betas[i], None)                      
                                dist = self.minkowski_ellipse_collision_distance(self.ambMPC, self.otherMPClist[i], 
                                                                                self.xamb_opt[0, k], self.xamb_opt[1,k], self.xamb_opt[2,k], 
                                                                                self.allother_x_opt[i][0,k], self.allother_x_opt[i][1,k], self.allother_x_opt[i][2,k])
                                
                                self.opti.subject_to(dist >= (1 - self.slack_amb_other[i][ci, k]))
                                distance_clipped = cas.fmax(dist, 0.0001) # could be buffered if we'd like
                                self.collision_cost += 1/distance_clipped**self.k_CA_power     
                        if self.WALL_CA: # Compute CA cost of ambulance and wall
                            dist_btw_wall_bottom =  ca_circle_xy[1] - (self.responseMPC.min_y + self.responseMPC.W/2.0) 
                            dist_btw_wall_top = (self.responseMPC.max_y - self.responseMPC.W/2.0) - ca_circle_xy[1]

                            self.collision_cost += 0.1 * 1/(cas.fmax(dist_btw_wall_bottom, 0.0001)**self.k_CA_power)
                            self.collision_cost += 0.1 * 1/(cas.fmax(dist_btw_wall_top, 0.0001)**self.k_CA_power)                
            
        ###### Collision Braking Avoidance
        k_min_ttc = 1.0 ## Vehicle must have a time to collision greater than this number
        if self.ambMPC:
            self.generate_stopping_constraint_ttc(self.x_opt, self.xamb_opt, self.allother_x_opt, solve_amb, min_time_to_collision = k_min_ttc) ###
        else:
            self.generate_stopping_constraint_ttc(self.x_opt, None, self.allother_x_opt, False, min_time_to_collision = k_min_ttc) ###

        ######## optimization  ##################################
        self.total_svo_cost = self.response_svo_cost + self.other_svo_cost + self.k_slack * self.slack_cost + self.k_CA * self.collision_cost
        self.opti.minimize(self.total_svo_cost)    
        ##########################################################

        # Set the initial conditions
        self.opti.set_value(p, x0)
        for i in range(len(self.allother_p)):
            self.opti.set_value(self.allother_p[i], x0_other[i])
        if self.ambMPC:    
            self.opti.set_value(pamb, x0_amb) 

        # Set the solver conditions
        if ipopt_params is None:
            ipopt_params = {}
            ipopt_params['print_level'] = print_level
            # ipopt_params['max_iter'] = 0
        self.opti.solver('ipopt',{}, ipopt_params)


    def solve(self, uamb, uother, solve_amb=False):

        if self.ambMPC:
            if solve_amb:
                self.opti.set_initial(self.uamb_opt, uamb)
            else:
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

    def generate_slack_variables(self, slack_bool, N_time_steps, number_other_vehicles, n_ego_circles):
        if slack_bool:
            slack_vars = []
            for i in range(number_other_vehicles):
                slack_vars += [self.opti.variable(n_ego_circles, N_time_steps+1)]
            for slack_v in slack_vars:
                self.opti.subject_to(cas.vec(slack_v)>=0)
                # self.opti.subject_to(slack<=1.0)
        else:
            slack_vars = []
            for i in range(number_other_vehicles):
                slack_vars += [self.opti.parameter(n_ego_circles, N_time_steps+1)]
            for slack_v in slack_vars:
                self.opti.set_value(slack_v, np.zeros((n_ego_circles,N_time_steps+1)))
        return slack_vars

    def generate_collision_ellipse(self, x_e, y_e, x_o, y_o, phi_o, alpha_o, beta_o, slack_var):
        ''' alpha_o:  major axis of length'''
        dx = x_e - x_o
        dy = y_e - y_o

        R_o = cas.vertcat(cas.horzcat(cas.cos(phi_o), cas.sin(phi_o)), cas.horzcat(-cas.sin(phi_o), cas.cos(phi_o)))
        M = cas.vertcat(cas.horzcat(1/alpha_o**2, 0), cas.horzcat(0, 1/beta_o**2))
        dX = cas.vertcat(dx, dy)
        prod =    cas.mtimes([dX.T, R_o.T, M, R_o, dX])

        M_smaller = cas.vertcat(cas.horzcat(1/(0.5*alpha_o)**2, 0), cas.horzcat(0, 1/(.5*beta_o)**2))
        dist_prod =    cas.mtimes([dX.T, R_o.T, M_smaller, R_o, dX])
        # dist = dist_prod - 1

        # euc_dist = dx**2 + dy**2
        return dist_prod, prod

    # def feasible_callback(self, i):
    #     self.opti.debug.
    #     self.intermediate_solution = self.opti.debug.value(self.x_opt)

        

    def debug_callback(self, i, plot_range=[], file_name = False):
        xothers_plot = [self.opti.debug.value(xo) for xo in self.allother_x_opt]
        xamb_plot = self.opti.debug.value(self.x_opt)
        if self.ambMPC:
            xothers_plot += [self.opti.debug.value(self.xamb_opt)]
        max_x = np.max(xamb_plot[0,:])
        # plot_range = range(xamb_plot.shape[1])
        # plot_range = [xamb_plot.shape[1] - 1]
        if file_name:
            uamb_plot = self.opti.debug.value(self.u_opt)
            uothers_plot = [self.opti.debug.value(xo) for xo in self.allother_u_opt]
            save_state(file_name, xamb_plot, uamb_plot, None, xothers_plot, uothers_plot, None)

        if len(plot_range)>0:

            cmplot.plot_cars(self.world, self.responseMPC, xamb_plot, xothers_plot, 
                 None, "ellipse", False, 0)
            plt.show()            
            # CIRCLES=True
            # for k in plot_range:
            #     cmplot.plot_multiple_cars( k, self.responseMPC, xothers_plot, xamb_plot, CIRCLES, None, None, None, self.world, 0)     
            #     plt.plot(xamb_plot[0,:], xamb_plot[1,:],'o')
            #     plt.show()
            plt.plot(xamb_plot[4,:],'--')
            plt.plot(xamb_plot[4,:] * np.cos(xamb_plot[2,:]))
            plt.ylabel("Velocity / Vx")
            plt.hlines(35*0.447,0,xamb_plot.shape[1])
            plt.show()
        print("%d Total Cost %.03f J_i %.03f,  J_j %.03f, Slack %.03f, CA  %.03f"%
                (i, self.opti.debug.value(self.total_svo_cost), self.opti.debug.value(self.response_svo_cost), self.opti.debug.value(self.other_svo_cost), self.opti.debug.value(self.k_slack*self.slack_cost), self.opti.debug.value(self.k_CA*self.collision_cost)))
        for i in range(len(self.car1_costs_list)):
            print(" %.04f : %s"%(self.opti.debug.value(self.car1_costs_list[i]),self.car1_cost_titles[i]))        

    def plot_collision_slack_cost(self):
        '''Make a function that plots the contours for collision and slack
        '''
        x_e = np.array([0, 100]) # these need to be corrected from world
        y_e = np.array([-5, 5]) #these need to be corrected from world
        X, Y = np.meshgrid(x_e, y_e)
        for i in range(len(self.allother_x_opt)):
            x_o, y_o, phi_o, alpha_o, beta_o = (2, 1, 0, 1, 1) ### This is a test pose, we need to get it from our optimization
                        # R_o = cas.vertcat(cas.horzcat(cas.cos(phi_o), cas.sin(phi_o)), cas.horzcat(-cas.sin(phi_o), cas.cos(phi_o)))
            R_o = np.array([[np.cos(phi_o), np.sin(phi_o)],[-np.sin(phi_o), np.cos(phi_o)]])
                        # M = cas.vertcat(cas.horzcat(1/alpha_o**2, 0), cas.horzcat(0, 1/beta_o**2))
            M = np.array([[1/alpha_o**2, 0],[0, 1/beta_o**2]])
            dx = X - x_o
            dy = Y - y_o          
            dX = np.stack((dx, dy), axis=2)
            prod =    cas.mtimes([dX.T, R_o.T, M, R_o, dX])


    def generate_stopping_constraint_ttc(self, x_opt, xamb_opt, xothers_opt, solve_amb, min_time_to_collision = 3.0):
        ''' Add a velocity constraint on the ego vehicle so it doesn't go too fast behind a lead vehicle.
            Constrains the vehicle so that the time to collision is always less than time_to_collision seconds
            assuming that both ego and ado vehicle mantain constant velocity
        '''

        N = x_opt.shape[1]
        car_length = self.responseMPC.L
        car_width = self.responseMPC.W        
        for k in range(N):
            x_ego = x_opt[0, k]
            y_ego = x_opt[1, k]
            phi_ego = x_opt[2, k]
            v_ego = x_opt[4, k]
            v_ego_components = (v_ego * cas.cos(phi_ego), v_ego * cas.sin(phi_ego))

            if xamb_opt is not None:
                x_amb = xamb_opt[0, k]
                y_amb = xamb_opt[1, k]
                phi_amb = xamb_opt[2, k]
                v_amb = xamb_opt[4, k]    

                v_amb_components = (v_amb * cas.cos(phi_amb), v_amb * cas.sin(phi_amb))              
                
                dxegoamb = (x_amb - x_ego) - car_length
                dyegoamb = (y_amb - y_ego) - car_width
                dot_product = (v_ego_components[0] - v_amb_components[0])*dxegoamb + (v_ego_components[1] - v_amb_components[1])*dyegoamb
                self.opti.subject_to(dot_product <= (dxegoamb**2 + dyegoamb**2) / (0.000001 + min_time_to_collision))
                # dist_egoamb = cas.sqrt(dxegoamb**2 + dyegoamb**2)                
                
                # egobehind_amb = cas.fmax(-dxegoamb, 0) ## 0 if ego is beind, else should be >0

                # v_max_constraint = cas.fmax((v_amb**2 - 2*a_maxbraking * (dist_egoamb - safety_buffer)), 999*egobehind_amb)
                # self.opti.subject_to(v_ego**2 <= v_max_constraint)
            
            for j in range(len(xothers_opt)):
                x_j = xothers_opt[j][0, k]
                y_j = xothers_opt[j][1, k]
                phi_j = xothers_opt[j][2, k]
                v_j = xothers_opt[j][4, k]       

                #### Add constraint between ego and j
                dxego = (x_j - x_ego) - car_length
                dyego = (y_j - y_ego) - car_width
                
                v_j_components = (v_j * cas.cos(phi_j), v_j * cas.sin(phi_j))
                dot_product = (v_ego_components[0] - v_j_components[0])*dxego + (v_ego_components[1] - v_j_components[1])*dyego
                
                self.opti.subject_to(dot_product <= (dxego**2 + dyego**2) / (0.000001 + min_time_to_collision))
                
                # dist_ego = cas.sqrt(dxego**2 + dyego**2)
                # ego_behind = cas.fmax(-dxego, 0) ###how many meters behind or 0 if ahead/same
                # v_max_constraint = cas.fmax(v_j**2 - 2*a_maxbraking * (dist_ego - safety_buffer), 999*ego_behind)
                # self.opti.subject_to(v_ego**2 <= v_max_constraint)

                #### Add constraint betweem amb and j
                if xamb_opt is not None and solve_amb:
                    dxamb = (x_j - x_amb) - car_length
                    dyamb = (y_j - y_amb) - car_width
                    
                    dot_product = (v_amb_components[0] - v_j_components[0])*dxamb + (v_amb_components[1] - v_j_components[1])*dyamb
                    self.opti.subject_to(dot_product <= (dxamb**2 + dyamb**2) / (0.00000001 + min_time_to_collision))
                    
                    # dist_amb = cas.sqrt(dxamb**2 + dyamb**2)

                    # amb_behind = cas.fmax(-dxamb, 0) ## 0 if ambulance is beind, else should be >0
                    # v_max_constraint = cas.fmax((v_j**2 - 2*a_maxbraking * (dist_amb - safety_buffer)), 999*amb_behind)
                    # self.opti.subject_to(v_amb**2 <= v_max_constraint)

    def generate_stopping_constraint(self, x_opt, xamb_opt, xothers_opt, solve_amb, safety_buffer=0.50):
        ''' Add a velocity constraint on the ego vehicle so it doesn't go too fast behind a lead vehicle.
        Constrains ego velocity so that ego vehicle can brake (at u_v_max) to the same velocity of the lead vehicle
        within the distance to the lead vehicle.  We add a buffer distance to ensure doesn't collide.

        safety_buffer:  Shortened distance for the braking
        '''
        u_v_maxbraking = self.responseMPC.min_v_u #this is max change in V, in discrete steps
        a_maxbraking = u_v_maxbraking / self.responseMPC.dt ### find max acceleration
        N = x_opt.shape[1]
        car_length = self.responseMPC.L
        car_width = self.responseMPC.W
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
                egobehind_amb = cas.fmax(-dxegoamb, 0) ## 0 if ego is beind, else should be >0
                
                v_max_constraint = cas.fmax((v_amb**2 - 2*a_maxbraking * (dist_egoamb - safety_buffer)), 999*egobehind_amb)
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
                ego_behind = cas.fmax(-dxego, 0) ###how many meters behind or 0 if ahead/same
                v_max_constraint = cas.fmax(v_j**2 - 2*a_maxbraking * (dist_ego - safety_buffer), 999*ego_behind)
                self.opti.subject_to(v_ego**2 <= v_max_constraint)

                #### Add constraint betweem amb and j
                if xamb_opt is not None and solve_amb:
                    dxamb = (x_j - x_amb) - car_length
                    dyamb = (y_j - y_amb) - car_width
                    dist_amb = cas.sqrt(dxamb**2 + dyamb**2)

                    amb_behind = cas.fmax(-dxamb, 0) ## 0 if ambulance is beind, else should be >0
                    v_max_constraint = cas.fmax((v_j**2 - 2*a_maxbraking * (dist_amb - safety_buffer)), 999*amb_behind)
                    self.opti.subject_to(v_amb**2 <= v_max_constraint)

    def minkowski_ellipse_collision_distance(self, ego_veh, ado_veh, x_ego, y_ego, phi_ego, x_ado, y_ado, phi_ado):
        ''' Return the squared distance between the ego vehicle and ado vehicle
        for collision avoidance '''
        # if not numpy:
        shape_matrix_ego = np.array([[float(ego_veh.ax), 0.0],[0.0, float(ego_veh.by)]])
        shape_matrix_ado = np.array([[float(ado_veh.ax), 0.0],[0.0, float(ado_veh.by)]])

        rotation_matrix_ego = cas.vertcat(cas.horzcat(cas.cos(phi_ego), cas.sin(phi_ego)), cas.horzcat(-cas.sin(phi_ego), cas.cos(phi_ego)))
        rotation_matrix_ado = cas.vertcat(cas.horzcat(cas.cos(phi_ado), cas.sin(phi_ado)), cas.horzcat(-cas.sin(phi_ado), cas.cos(phi_ado)))

        # Compute the Minkowski Sum
        M_e_curr = cas.mtimes([rotation_matrix_ego, shape_matrix_ego])
        Q1 = cas.mtimes([M_e_curr, cas.transpose(M_e_curr)])

        M_a_curr = cas.mtimes([rotation_matrix_ado, shape_matrix_ado])
        Q2 = cas.mtimes([M_a_curr, cas.transpose(M_a_curr)])

        beta = cas.sqrt(cas.trace(Q1) / cas.trace(Q2))
        Q_minkowski = (1+1.0/beta) * Q1 + (1.0+beta) * Q2

        X_ego = cas.vertcat(x_ego, y_ego)
        X_ado = cas.vertcat(x_ado, y_ado)
        dist_squared = cas.mtimes([cas.transpose(X_ado - X_ego), cas.inv(Q_minkowski), (X_ado - X_ego)])

        return dist_squared
    # else:
    #     shape_matrix_ego = np.array([[float(ego_veh.ax), 0.0],[0.0, float(ego_veh.by)]])
    #     shape_matrix_ado = np.array([[float(ado_veh.ax), 0.0],[0.0, float(ado_veh.by)]])
    #     rotation_matrix_ego = np.array([[np.cos(phi_ego), np.sin(phi_ego)],[-np.sin(phi_ego), np.cos(phi_ego)]])
    #     rotation_matrix_ado = np.array([[np.cos(phi_ado), np.sin(phi_ado)],[-np.sin(phi_ado), np.cos(phi_ado)]])
    #     # Compute the Minkowski Sum
    #     M_e_curr = rotation_matrix_ego @ shape_matrix_ego
    #     M_a_curr = rotation_matrix_ado @ shape_matrix_ado
    #     Q1 = M_e_curr @ M_e_curr.T
    #     Q2 = M_a_curr @ M_a_curr.T
    #     beta = np.sqrt(np.trace(Q1) / np.trace(Q2))
    #     Q_minkowski = (1+1/beta) * Q1 + (1+beta) * Q2

    #     X_ego = np.array([[x_ego],[y_ego]])
    #     X_ado = np.array([[x_ado],[y_ado]])
    #     dist_squared = (X_ado - X_ego).T @ np.linalg.inv(Q_minkowski) @ (X_ado - X_ego)

    #     return dist_squared        

        ### For collision free:  dist_squared >= 1



def load_state(file_name, n_others, ignore_des=False):
    xamb = np.load(file_name + "xamb.npy",allow_pickle=False)
    uamb = np.load(file_name + "uamb.npy",allow_pickle=False)
    if not ignore_des:
        xamb_des = np.load(file_name + "xamb_des.npy",allow_pickle=False)
    else:
        xamb_des = None

    xothers, uothers, xothers_des = [], [], []
    for i in range(n_others):
        x = np.load(file_name + "x%0d.npy"%i, allow_pickle=False)
        u = np.load(file_name + "u%0d.npy"%i, allow_pickle=False)
        xothers += [x]
        uothers += [u]
        if not ignore_des:
            x_des = np.load(file_name + "x_des%0d.npy"%i, allow_pickle=False)
            xothers_des += [x_des]

    return xamb, uamb, xamb_des, xothers, uothers, xothers_des


def save_state(file_name, xamb, uamb, xamb_des, xothers, uothers, xothers_des, end_t=None):
    #TODO: Move xamb_des to an optional aparamter
    if end_t is None:
        end_t = xamb.shape[1]

    np.save(file_name + "xamb", xamb[:, :end_t], allow_pickle=False)
    np.save(file_name + "uamb", uamb[:, :end_t], allow_pickle=False)
    if xamb_des is not None:
        np.save(file_name + "xamb_des", xamb_des[:, :end_t], allow_pickle=False)

    for i in range(len(xothers)):
        x, u = xothers[i], uothers[i]   
        np.save(file_name + "x%0d"%i, x[:, :end_t], allow_pickle=False)
        np.save(file_name + "u%0d"%i, u[:, :end_t], allow_pickle=False)
        if xothers_des is not None:
            x_des = xothers_des[i]
            np.save(file_name + "x_des%0d"%i, x_des[:, :end_t], allow_pickle=False)
    
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

def generate_warm_x(car_mpc, world, x0, average_v=None): 
    ''' Warm starts that return a trajectory in x (control) -space
    N:  Number of control points
    car_mpc:  Vehicle instance
    car_x0:  Initial position
    
    Return:  x_warm_profiles [dict]
                keys: label of warm start [str]
                values: 6xn state vector

            ux_warm_profiles [dict]
                keys: label of warm start [str]
                values:  control vector (initialized as zero), state vector, x desired vector
    '''


    x_warm_profiles = {}
    N = car_mpc.N
    lane_width = world.lane_width
    if average_v is None:
        constant_v = car_mpc.max_v
    else:
        constant_v = average_v
    t_array = np.arange(0, car_mpc.dt*(N+1) - 0.000001, car_mpc.dt)
    x = x0[0] + t_array * constant_v
    y0 = x0[1]
    x_warm_default = np.repeat(x0.reshape(6,1), N+1, 1)
    x_warm_default[0,:] = x
    x_warm_default[1,:] = y0
    x_warm_default[2,:] = np.zeros((1, N+1))
    x_warm_default[3,:] = np.zeros((1, N+1))
    x_warm_default[4,:] = constant_v
    x_warm_default[5,:] = t_array * constant_v
    x_warm_profiles["0constant v"] = x_warm_default
    # lane change up
    y_up = y0 + lane_width
    for percent_change in [0.00, 0.5, 0.75]:
        key = "0up %d"%(int(100*percent_change))
        x_up = np.copy(x_warm_default)
        ti_lane_change = int(percent_change * (N+1))
        y = y_up * np.ones((1, N+1))
        y[:,:ti_lane_change] = x0[1] * np.ones((1,ti_lane_change))
        x_up[1,:] = y
        x_warm_profiles[key] = x_up

    y_down = y0 - lane_width
    for percent_change in [0.00, 0.5, 0.75]:
        key = "0down %d"%(int(100*percent_change))
        x_up = np.copy(x_warm_default)
        ti_lane_change = int(percent_change * (N+1))
        y = y_down * np.ones((1, N+1))
        y[:,:ti_lane_change] = x0[1] * np.ones((1,ti_lane_change))
        x_up[1,:] = y
        x_warm_profiles[key] = x_up
    
    ux_warm_profiles = {}
    for k_warm in x_warm_profiles.keys():
        u_warm = np.zeros((2, N))
        x_warm = x_warm_profiles[k_warm]
        x_des_warm = np.zeros(shape=(3, N + 1))        
        for k in range(N + 1):
            x_des_warm[:, k:k+1] = car_mpc.fd(x_warm[-1,k])
        ux_warm_profiles[k_warm] = [u_warm, x_warm, x_des_warm]
    
    return x_warm_profiles, ux_warm_profiles


def centerline_following(N, car_mpc, car_x0):
    y_follow = car_x0[1]

    u_warm = np.zeros((2,N))
    u_warm[1,:] = np.zeros(shape=(1,N)) ### No acceleration
    
    x = np.zeros(shape=(6, N + 1))
    x[:,0:1] = car_x0.reshape(6,1)
    for k in range(N):
        k_u = .1
        u_turn = -k_u * (x[1,k] - y_follow)
        u_turn = np.clip(u_turn, -car_mpc.max_delta_u, car_mpc.max_delta_u)
        x_k = x[:, k]
        u_warm[0, k:k+1] = u_turn
        x_knext = car_mpc.F_kutta(car_mpc.f, x_k, u_warm[:,k])
        x[:, k+1:k+2]  = x_knext
    
    x_des = np.zeros(shape=(3, N + 1))        
    for k in range(N + 1):
        x_des[:, k:k+1] = car_mpc.fd(x[-1,k])

    return [u_warm, x, x_des]


def generate_warm_u(N, car_mpc, car_x0):
    ''' Warm starts that return a trajectory in u (control) -space
    N:  Number of control points
    car_mpc:  Vehicle instance
    car_x0:  Initial position
    
    Return:  u_warm_profiles [dict]
                keys: label of warm start [str]
                values: 2xn control vector

            ux_warm_profiles [dict]
                keys: label of warm start [str]
                values:  control vector, state vector, x desired vector
    '''

    u0_warm_profiles = {}
    u1_warm_profiles = {}
    ## braking
    u_warm = np.zeros((2,N))
    u_warm[0,:] = np.zeros(shape=(1,N))
    u_warm[1,:] = np.ones(shape=(1,N)) * car_mpc.min_v_u
    u1_warm_profiles["braking"] = u_warm

    ## accelerate
    u_warm = np.zeros((2,N))
    u_warm[0,:] = np.zeros(shape=(1,N))
    u_warm[1,:] = np.ones(shape=(1,N)) * car_mpc.max_v_u
    u1_warm_profiles["accelerating"] = u_warm

    u_warm = np.zeros((2,N))
    u_warm[0,:] = np.zeros(shape=(1,N))
    t_half = int(N)
    u_warm[1,:t_half] = np.ones(shape=(1,t_half)) * car_mpc.max_v_u / 3.0
    # u1_warm_profiles["halfaccel"] = u_warm 
    ## no accelerate
    u_warm = np.zeros((2,N))
    u1_warm_profiles["none"] = u_warm
    ##############################

    ## lane change left
    u_warm = np.zeros((2,N))
    u_l1 = 0
    u_r1 = int(N/3)
    u_l2 = int(2*N/3)
    # u_r2 = int(3*N/4)
    u_warm[0,u_l1] = - 0.5 * car_mpc.max_delta_u
    u_warm[0,u_r1] = car_mpc.max_delta_u
    u_warm[0,u_l2] = - 0.5 * car_mpc.max_delta_u

    u_warm[1,:] = np.zeros(shape=(1,N)) 
    u0_warm_profiles["lane_change_right"] = u_warm

    u0_warm_profiles["lane_change_left"] = - u0_warm_profiles["lane_change_right"]
    u0_warm_profiles["none"] = np.zeros(shape=(2,N))

    u_warm_profiles = {}
    for u0_k in u0_warm_profiles.keys():
        for u1_k in u1_warm_profiles.keys():
            u_k = u0_k + " " + u1_k
            u_warm_profiles[u_k] = u0_warm_profiles[u0_k] + u1_warm_profiles[u1_k]

    # Generate x, x_des from the u_warm profiles
    ux_warm_profiles = {}
    for k_warm in u_warm_profiles.keys():
        u_warm = u_warm_profiles[k_warm]
        x_warm, x_des_warm = car_mpc.forward_simulate_all(car_x0.reshape(6,1), u_warm)
        ux_warm_profiles[k_warm] = [u_warm, x_warm, x_des_warm]


    ### Generate some line following examples
    u_warm, x_warm, x_des_warm = centerline_following(N, car_mpc, car_x0)
    k_warm = "line_following"
    u_warm_profiles[k_warm] = u_warm
    ux_warm_profiles[k_warm] = [u_warm, x_warm, x_des_warm]

    return u_warm_profiles, ux_warm_profiles



