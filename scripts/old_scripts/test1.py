import time, datetime, argparse
import os, sys
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt
import copy as cp
import pickle

PROJECT_PATH = '/home/nbuckman/Dropbox (MIT)/DRL/2020_01_cooperative_mpc/mpc-multiple-vehicles/'
sys.path.append(PROJECT_PATH)

import casadi as cas
import src.MPC_Casadi as mpc
import src.TrafficWorld as tw
import src.IterativeBestResponseMPCMultiple as mibr
import src.car_plotting_multiple as cmplot


##########################################################
svo_theta = np.pi/4.0
# random_seed = args.random_seed[0]
random_seed = 3
NEW = True
if NEW:
    optional_suffix = "ellipses"
    subdir_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + optional_suffix
    folder = "results/" + subdir_name + "/"
    os.makedirs(folder)
    os.makedirs(folder+"imgs/")
    os.makedirs(folder+"data/")
    os.makedirs(folder+"vids/")
    os.makedirs(folder+"plots/")
else:
    subdir_name = "20200224-103456_real_dim_CA"
    folder = "results/" + subdir_name + "/"
print(folder)
if random_seed > 0:
    np.random.seed(random_seed)

#######################################################################
T = 3  # MPC Planning Horizon
dt = 0.3
N = int(T/dt) #Number of control intervals in MPC
n_rounds_mpc = 6
percent_mpc_executed = 0.5 ## This is the percent of MPC that is executed
number_ctrl_pts_executed =  int(np.floor(N*percent_mpc_executed))
XAMB_ONLY = False

n_other = 2
n_rounds_ibr = 2

world = tw.TrafficWorld(2, 0, 1000)
    # large_world = tw.TrafficWorld(2, 0, 1000, 5.0)

#########################################################################
actual_xamb = np.zeros((6, n_rounds_mpc*number_ctrl_pts_executed + 1))
actual_uamb = np.zeros((2, n_rounds_mpc*number_ctrl_pts_executed))
actual_xothers = [np.zeros((6, n_rounds_mpc*number_ctrl_pts_executed + 1)) for i in range(n_other)]
actual_uothers = [np.zeros((2, n_rounds_mpc*number_ctrl_pts_executed)) for i in range(n_other)]
actual_all_other_x0 = [np.zeros((6, 2*N)) for i in range(n_other)]


xamb = np.zeros(shape=(6, N+1))
t_start_time = time.time()
####################################################
## Create the Cars in this Problem
all_other_x0 = []
all_other_u = []
all_other_MPC = []
all_other_x = [np.zeros(shape=(6, N+1)) for i in range(n_other)]
next_x0 = 0
for i in range(n_other):
    x1_MPC = mpc.MPC(dt)
    x1_MPC.n_circles = 3
    x1_MPC.theta_iamb =  svo_theta
    x1_MPC.N = N
    x1_MPC.k_change_u_v = 0.001
    x1_MPC.max_delta_u = 50 * np.pi/180 * x1_MPC.dt
    x1_MPC.k_u_v = 0.01
    x1_MPC.k_u_delta = .00001
    x1_MPC.k_change_u_v = 0.01
    x1_MPC.k_change_u_delta = 0.001

    x1_MPC.k_s = 0
    x1_MPC.k_x = 0
    x1_MPC.k_x_dot = -1.0 / 100.0
    x1_MPC.k_lat = 0.001
    x1_MPC.k_lon = 0.0

    x1_MPC.k_phi_error = 0.001
    x1_MPC.k_phi_dot = 0.01    
    
    ####Vehicle Initial Conditions
    if i%2 == 0: 
        lane_number = 0
        next_x0 += x1_MPC.L + 2*x1_MPC.min_dist
    else:
        lane_number = 1
    initial_speed = 0.75*x1_MPC.max_v
    traffic_world = world
    x1_MPC.fd = x1_MPC.gen_f_desired_lane(traffic_world, lane_number, True)
    x0 = np.array([next_x0, traffic_world.get_lane_centerline_y(lane_number), 0, 0, initial_speed, 0]).T

    ## Set the initial control of the other vehicles
    u1 = np.zeros((2,N))
    # u1[0,:] = np.clip(np.pi/180 *np.random.normal(size=(1,N)), -2 * np.pi/180, 2 * np.pi/180)
    SAME_SIDE = False
    if lane_number == 1 or SAME_SIDE:
        u1[0,0] = 2 * np.pi/180
    else:
        u1[0,0] = -2 * np.pi/180
    u1[0,0] = 0 

    all_other_MPC += [x1_MPC]
    all_other_x0 += [x0]
    all_other_u += [u1]    

# Settings for Ambulance
amb_MPC = cp.deepcopy(x1_MPC)
amb_MPC.theta_iamb = 0.0

amb_MPC.k_u_v = 0.0000
amb_MPC.k_u_delta = .01
amb_MPC.k_change_u_v = 0.0000
amb_MPC.k_change_u_delta = 0

amb_MPC.k_s = 0
amb_MPC.k_x = 0
amb_MPC.k_x_dot = -1.0 / 100.0

amb_MPC.k_x = -1.0/100
amb_MPC.k_x_dot = 0

amb_MPC.k_lat = 0.00001
amb_MPC.k_lon = 0.0
# amb_MPC.min_v = 0.8*initial_speed
amb_MPC.max_v = 35 * 0.447 # m/s
amb_MPC.k_phi_error = 0.1
amb_MPC.k_phi_dot = 0.01

NO_GRASS = False
amb_MPC.min_y = world.y_min        
amb_MPC.max_y = world.y_max
if NO_GRASS:
    amb_MPC.min_y += world.grass_width
    amb_MPC.max_y -= world.grass_width
amb_MPC.fd = amb_MPC.gen_f_desired_lane(world, 0, True)
x0_amb = np.array([0, 0, 0, 0, initial_speed , 0]).T
pickle.dump(x1_MPC, open(folder + "data/"+"mpc%d"%i + ".p",'wb'))
pickle.dump(amb_MPC, open(folder + "data/"+"mpcamb" + ".p",'wb'))

########################################################################
#### SOLVE THE MPC #####################################################
for i_mpc in range(n_rounds_mpc):
    min_slack = np.infty
    actual_t = i_mpc * number_ctrl_pts_executed
    ###### Update the initial conditions for all vehicles
    if i_mpc > 0:
        x0_amb = xamb[:, number_ctrl_pts_executed] 
        for i in range(len(all_other_x0)):
            all_other_x0[i] = all_other_x[i][:, number_ctrl_pts_executed]        

        

    ###### Initial guess for the other u.  This will be updated once the other vehicles
    ###### solve the best response to the ambulance. Initial guess just looks at the last solution. This could also be a lange change
    # Obtain a simulated trajectory from other vehicle control inputs
    all_other_x = [np.zeros(shape=(6, N+1)) for i in range(n_other)]
    all_other_x_des = [np.zeros(shape=(3, N+1)) for i in range(n_other)]
    for i in range(n_other):
        if i_mpc == 0:
            all_other_u[i] = np.zeros(shape=(6,N))
        else:
            all_other_u[i] = np.concatenate((all_other_u[i][:, number_ctrl_pts_executed:], np.tile(all_other_u[i][:,-1:],(1, number_ctrl_pts_executed))),axis=1) ##   
        x_mpci, u_all_i, x_0_i = all_other_MPC[i], all_other_u[i], all_other_x0[i]
        all_other_x[i], all_other_x_des[i] = x_mpci.forward_simulate_all(x_0_i, u_all_i)

    for i_rounds_ibr in range(n_rounds_ibr):  
        ########## Solve the Ambulance MPC ##########
        response_MPC = amb_MPC
        response_x0 = x0_amb
        
        nonresponse_MPC_list = all_other_MPC
        nonresponse_x0_list = all_other_x0
        nonresponse_u_list = all_other_u
        nonresponse_x_list = all_other_x
        nonresponse_xd_list = all_other_x_des
        ################# Generate the warm starts ###############################

        u_warm_profiles = mibr.generate_warm_u(N, response_MPC)
        ### Ambulance Warm Start
        if i_rounds_ibr > 0:   # warm start with the solution from the last IBR round
            u_warm_profiles["previous"] = uamb    
        else:
            # take the control inputs of the last MPC and continue the ctrl
            if i_mpc > 0:
                u_warm_profiles["previous"] = np.concatenate((uamb[:, number_ctrl_pts_executed:], np.tile(uamb[:,-1:],(1, number_ctrl_pts_executed))),axis=1) ##    
        #######################################################################
        min_response_cost = 99999999
        for k_warm in u_warm_profiles.keys():
            u_warm = u_warm_profiles[k_warm]
            x_warm, x_des_warm = response_MPC.forward_simulate_all(response_x0.reshape(6,1), u_warm)
            bri = mibr.IterativeBestResponseMPCMultiple(response_MPC, None, nonresponse_MPC_list )
            k_slack = 10000.0
            k_CA = 0.000000000000000
            k_CA_power = 4
            wall_CA = True

            bri.k_slack = k_slack
            bri.k_CA = k_CA
            bri.k_CA_power = k_CA_power
            bri.world = world
            bri.wall_CA = wall_CA
            
            # for slack_var in bri.slack_vars_list: ## Added to constrain slacks
            #     bri.opti.subject_to(cas.vec(slack_var) <= 1.0)
            INFEASIBLE = True            
            bri.generate_optimization(N, T, response_x0, None, nonresponse_x0_list,  1, slack=False)
            bri.opti.set_initial(bri.u_opt, u_warm)            
            bri.opti.set_initial(bri.x_opt, x_warm)
            bri.opti.set_initial(bri.x_desired, x_des_warm)   
        
            ### Set the trajectories of the nonresponse vehicles (as given)        
            for i in range(n_other):
                bri.opti.set_value(bri.allother_x_opt[i], nonresponse_x_list[i])
                bri.opti.set_value(bri.allother_x_desired[i], nonresponse_xd_list[i])
            
            ### Solve the Optimization
            # Debugging
            # plot_range = [N]
            # bri.opti.callback(lambda i: bri.debug_callback(i, plot_range))
            # bri.opti.callback(lambda i: print("J_i %.03f,  J_j %.03f, Slack %.03f, CA  %.03f"%(bri.solution.value(bri.response_svo_cost), bri.solution.value(bri.other_svo_cost), bri.solution.value(bri.k_slack*bri.slack_cost), bri.solution.value(bri.k_CA*bri.collision_cost))))
            try:
                bri.solve(None, nonresponse_u_list)
                x1, u1, x1_des, _, _, _, _, _, _ = bri.get_solution()

                print("i_mpc %d n_round %d i %02d Cost %.02f Slack %.02f "%(i_mpc, i_rounds_ibr, i, bri.solution.value(bri.total_svo_cost), bri.solution.value(bri.slack_cost)))
                print("J_i %.03f,  J_j %.03f, Slack %.03f, CA  %.03f"%(bri.solution.value(bri.response_svo_cost), bri.solution.value(bri.other_svo_cost), bri.solution.value(bri.k_slack*bri.slack_cost), bri.solution.value(bri.k_CA*bri.collision_cost)))
                print("Dir:", subdir_name)
                print("k_warm", k_warm)
                INFEASIBLE = False
                if bri.solution.value(bri.slack_cost) < min_slack:
                    current_cost = bri.solution.value(bri.total_svo_cost)
                    if current_cost < min_response_cost:
                        uamb = u1
                        xamb = x1
                        xamb_des = x1_des
                        min_response_cost = current_cost
                        min_response_warm = k_warm
                        min_bri = bri
                        # file_name = folder + "data/"+'%03d'%ibr_sub_it
                        # mibr.save_state(file_name, xamb, uamb, xamb_des, all_other_x, all_other_u, all_other_x_des)
                        # mibr.save_costs(file_name, bri)                  
            except RuntimeError:
                print("Infeasibility: k_warm %s"%k_warm)
                # ibr_sub_it +=1

        ########### SOLVE FOR THE OTHER VEHICLES ON THE ROAD
        if not XAMB_ONLY:
            for i in range(len(all_other_MPC)):
                response_MPC = all_other_MPC[i]
                response_x0 = all_other_x0[i]

                nonresponse_MPC_list = all_other_MPC[:i] + all_other_MPC[i+1:]
                nonresponse_x0_list = all_other_x0[:i] + all_other_x0[i+1:]
                nonresponse_u_list = all_other_u[:i] + all_other_u[i+1:]
                nonresponse_x_list = all_other_x[:i] + all_other_x[i+1:]
                nonresponse_xd_list = all_other_x_des[:i] + all_other_x_des[i+1:]
                ################  Warm Start
                u_warm_profiles = mibr.generate_warm_u(N, response_MPC)
                if i_rounds_ibr > 0:   # warm start with the solution from the last IBR round
                    u_warm_profiles["previous"] = all_other_u[i] 
                else:
                    # take the control inputs of the last MPC and continue the ctrl
                    if i_mpc > 0:
                        u_warm_profiles["previous"] = np.concatenate((all_other_u[i][:, number_ctrl_pts_executed:], np.tile(all_other_u[i][:,-1:],(1, number_ctrl_pts_executed))),axis=1) ##    

                min_response_cost = 99999999
                for k_warm in u_warm_profiles.keys():
                    u_warm = u_warm_profiles[k_warm]
                    x_warm, x_des_warm = response_MPC.forward_simulate_all(response_x0.reshape(6,1), u_warm)
                    
                    bri = mibr.IterativeBestResponseMPCMultiple(response_MPC, amb_MPC, nonresponse_MPC_list)
                    bri.k_slack = k_slack
                    bri.k_CA = k_CA
                    bri.k_CA_power = k_CA_power
                    bri.world = world
                    bri.wall_CA = wall_CA

                    INFEASIBLE = True            
                    bri.generate_optimization(N, T, response_x0, x0_amb, nonresponse_x0_list,  1, slack=False)
                    # for slack_var in bri.slack_vars_list: ## Added to constrain slacks
                    #     bri.opti.subject_to(cas.vec(slack_var) <= 1.0)                    
                    bri.opti.set_initial(bri.u_opt, u_warm)            
                    bri.opti.set_initial(bri.x_opt, x_warm)
                    bri.opti.set_initial(bri.x_desired, x_des_warm)   

                    ### Set the trajectories of the nonresponse vehicles (as given)        
                    bri.opti.set_value(bri.xamb_opt, xamb)
                    for i in range(len(nonresponse_x_list)):
                        bri.opti.set_value(bri.allother_x_opt[i], nonresponse_x_list[i])
                        bri.opti.set_value(bri.allother_x_desired[i], nonresponse_xd_list[i])
                    
                    # Debugging
                    # bri.opti.callback(lambda i: bri.debug_callback(i, [N]))
                    # bri.opti.callback(lambda i: print("J_i %.03f,  J_j %.03f, Slack %.03f, CA  %.03f"%(bri.solution.value(bri.response_svo_cost), bri.solution.value(bri.other_svo_cost), bri.solution.value(bri.k_slack*bri.slack_cost), bri.solution.value(bri.k_CA*bri.collision_cost))))
                    try:                     ### Solve the Optimization
                        bri.solve(uamb, nonresponse_u_list)
                        x1_nr, u1_nr, x1_des_nr, _, _, _, _, _, _ = bri.get_solution()

                        print("  i_mpc %d n_round %d i %02d Cost %.02f Slack %.02f "%(i_mpc, i_rounds_ibr, i, bri.solution.value(bri.total_svo_cost), bri.solution.value(bri.slack_cost)))
                        print("  J_i %.03f,  J_j %.03f, Slack %.03f, CA  %.03f"%(bri.solution.value(bri.response_svo_cost), bri.solution.value(bri.other_svo_cost), bri.solution.value(bri.k_slack*bri.slack_cost), bri.solution.value(bri.k_CA*bri.collision_cost)))
                        print("  Dir:", subdir_name)
                        print("  k_warm", k_warm)
                        INFEASIBLE = False
                        if bri.solution.value(bri.slack_cost) < min_slack:
                            current_cost = bri.solution.value(bri.total_svo_cost)
                            if current_cost < min_response_cost:
                                all_other_u[i] = u1_nr
                                all_other_x = all_other_x[:i] + [x1_nr] + all_other_x[i:]
                                all_other_u = all_other_u[:i] + [u1_nr] + all_other_u[i:]
                                all_other_x_des = all_other_x_des[:i] + [x1_des_nr] + all_other_x_des[i:]

                                min_response_cost = current_cost
                                min_response_warm = k_warm
                                min_bri = bri
                                # file_name = folder + "data/"+'%03d'%ibr_sub_it
                                # mibr.save_state(file_name, xamb, uamb, xamb_des, all_other_x, all_other_u, all_other_x_des)
                                # mibr.save_costs(file_name, bri)
                    except RuntimeError:
                        print("  Infeasibility: k_warm %s"%k_warm)
                        # ibr_sub_it +=1     
                        # 
        print("  IBR Done:  Rd %02d / %02d"%(i_rounds_ibr, n_rounds_ibr))

    file_name = folder + "data/"+'r%02d%03d'%(i_mpc, i_rounds_ibr)
    if not INFEASIBLE:
        mibr.save_state(file_name, xamb, uamb, xamb_des, xothers, uothers, xothers_des)
        mibr.save_costs(file_name, bri)
        actual_t = i_mpc * number_ctrl_pts_executed
        actual_xamb[:,actual_t:actual_t+number_ctrl_pts_executed+1]  = xamb[:,:number_ctrl_pts_executed+1]
        print(" MPC Done:  Rd %02d / %02d"%(i_mpc, n_rounds_mpc))
        print(" Full MPC Solution", xamb[0:2,:])
        print(" Executed MPC", xamb[0:2,:number_ctrl_pts_executed+1])
        print(" Solution Costs...")
        for cost in bri.car1_costs_list:
            print("%.04f"%bri.solution.value(cost))
        print(min_bri.solution.value(min_bri.k_CA * min_bri.collision_cost), min_bri.solution.value(min_bri.collision_cost))
        print(min_bri.solution.value(min_bri.k_slack * min_bri.slack_cost), min_bri.solution.value(min_bri.slack_cost))
        print(" Save to...", file_name)
        actual_uamb[:,actual_t:actual_t+number_ctrl_pts_executed] = uamb[:,:number_ctrl_pts_executed]
        plot_range = range(N+1)
        for k in plot_range:
            cmplot.plot_multiple_cars( k, min_bri.responseMPC, xothers, xamb, True, None, None, None, min_bri.world, 0)     
            plt.show()
        plt.plot(xamb[4,:],'--')
        plt.plot(xamb[4,:] * np.cos(xamb[2,:]))
        plt.ylabel("Velocity / Vx")
        plt.hlines(35*0.447,0,xamb.shape[1])
        plt.show()
        plt.plot(uamb[1,:],'o')
        plt.hlines(amb_MPC.max_v_u,0,xamb.shape[1])
        plt.ylabel("delta_u_v")
        plt.show()
        for i in range(len(xothers)):
            actual_xothers[i][:,actual_t:actual_t+number_ctrl_pts_executed+1] = xothers[i][:,:number_ctrl_pts_executed+1]
            actual_uothers[i][:,actual_t:actual_t+number_ctrl_pts_executed] = uothers[i][:,:number_ctrl_pts_executed]
#             all_other_u[i] = np.concatenate((uothers[i][:, number_ctrl_pts_executed:],uothers[i][:,:number_ctrl_pts_executed]),axis=1)
    else:
        raise Exception("Xamb is None", i_mpc, i_rounds_ibr, "slack cost", bri.solution.value(bri.slack_cost))

print("Solver Done!  Runtime: %.1d"%(time.time()-t_start_time))
