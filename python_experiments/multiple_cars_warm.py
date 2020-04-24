import time, datetime, argparse
import os, sys
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt
import copy as cp
import pickle

PROJECT_PATH = '/home/nbuckman/Dropbox (MIT)/DRL/2020_01_cooperative_mpc/mpc-multiple-vehicles/'
# PROJECT_PATH = '/Users/noambuckman/mpc-multiple-vehicles/'
sys.path.append(PROJECT_PATH)

import casadi as cas
import src.MPC_Casadi as mpc
import src.TrafficWorld as tw
import src.IterativeBestResponseMPCMultiple as mibr
import src.car_plotting_multiple as cmplot


##########################################################
svo_theta = np.pi/3.0
# random_seed = args.random_seed[0]
random_seed = 3
NEW = True
if NEW:
    optional_suffix = "4cars"
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


def solve_for_warm_start_amb():
    if k_warm[0] == "0":
        ###THESE were generate by x_warm
        u_warm = np.zeros((2, N))
        x_warm = u_warm_profiles[k_warm]
        x_des_warm = np.zeros(shape=(3, N + 1))        
        for k in range(N + 1):
            x_des_warm[:, k:k+1] = response_MPC.fd(x_warm[-1,k])
    else:
        u_warm = u_warm_profiles[k_warm]
        x_warm, x_des_warm = response_MPC.forward_simulate_all(response_x0.reshape(6,1), u_warm)

    bri = mibr.IterativeBestResponseMPCMultiple(response_MPC, None, nonresponse_MPC_list )
    k_slack, k_CA, k_CA_power, wall_CA = 10000.0, 0.001, 4, True
    bri.k_slack, bri.k_CA, bri.k_CA_power, bri.world, bri.wall_CA = k_slack, k_CA, k_CA_power, world, wall_CA
    # for slack_var in bri.slack_vars_list: ## Added to constrain slacks
    #     bri.opti.subject_to(cas.vec(slack_var) <= 1.0)
    solve_amb = True
    if i_rounds_ibr == 0:
        slack = True
    else:
        slack = False
    bri.generate_optimization(N, T, response_x0, None, nonresponse_x0_list,  0, slack=slack, solve_amb=solve_amb)
    bri.opti.set_initial(bri.u_opt, u_warm)            
    bri.opti.set_initial(bri.x_opt, x_warm)
    bri.opti.set_initial(bri.x_desired, x_des_warm)   
    ### Set the trajectories of the nonresponse vehicles (as given)        
    for j in range(len(nonresponse_x_list)):
        bri.opti.set_value(bri.allother_x_opt[j], nonresponse_x_list[j])
        bri.opti.set_value(bri.allother_x_desired[j], nonresponse_xd_list[j])
    try:
        bri.solve(None, nonresponse_u_list, solve_amb)
        x1, u1, x1_des, _, _, _, _, _, _ = bri.get_solution()
        if bri.solution.value(bri.slack_cost) < min_slack:
            current_cost = bri.solution.value(bri.total_svo_cost)
            min_response_cost = 9999999999
            if current_cost < min_response_cost:                # Update the solution for response vehicle at this iteration
                uamb_ibr = u1
                xamb_ibr = x1
                xamb_des_ibr = x1_des
                min_response_cost = current_cost
                min_response_warm_ibr = k_warm
                min_bri_ibr = bri
                amb_solved_flag = True  
                print("Current Min Response Key: %s"%k_warm)      
                # file_name = folder + "data/"+'%03d'%ibr_sub_it
                # mibr.save_state(file_name, xamb, uamb, xamb_des, all_other_x, all_other_u, all_other_x_des)
                # mibr.save_costs(file_name, bri)                  
    except RuntimeError:
        print("Infeasibility: k_warm %s"%k_warm)
        # ibr_sub_it +=1   
    return amb_solved_flag, min_bri_ibr, xamb_ibr, k_slack, k_CA, k_CA_power, wall_CA, xamb_des_ibr, uamb_ibr

#######################################################################
T = 2  # MPC Planning Horizon
dt = 0.2
N = int(T/dt) #Number of control intervals in MPC
n_rounds_mpc = 3
percent_mpc_executed = .80 ## This is the percent of MPC that is executed
number_ctrl_pts_executed =  int(np.floor(N*percent_mpc_executed))
print("number ctrl pts:  %d"%number_ctrl_pts_executed)
XAMB_ONLY = False
PLOT_FLAG, SAVE_FLAG, PRINT_FLAG = False, False, False
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
next_x0_0 = 0
next_x0_1 = 0
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
    
    x1_MPC.min_y = world.y_min        
    x1_MPC.max_y = world.y_max    
    x1_MPC.strict_wall_constraint = True

    

    ####Vehicle Initial Conditions
    lane_offset = np.random.uniform(0, 1) * x1_MPC.L
    if i%2 == 0: 
        lane_number = 0
        next_x0_0 += x1_MPC.L + 2*x1_MPC.min_dist + lane_offset
        next_x0 = next_x0_0
    else:
        lane_number = 1
        next_x0_1 += x1_MPC.L + 2*x1_MPC.min_dist + lane_offset
        next_x0 = next_x0_1
        
        
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
    # u1[0,0] = 0 

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
if SAVE_FLAG:
    pickle.dump(x1_MPC, open(folder + "data/"+"mpc%d"%i + ".p",'wb'))
    pickle.dump(amb_MPC, open(folder + "data/"+"mpcamb" + ".p",'wb'))
########################################################################
#### SOLVE THE MPC #####################################################
xamb_executed, all_other_x_executed = None, [] #This gets updated after each round of MPC
uamb_mpc, all_other_u_mpc = None, []

i_mpc = 0
i_rounds_ibr = 0
uamb_ibr = None
xamb_ibr = None
xamb_des_ibr = None
all_other_u_ibr = [None for i in range(n_other)]    
all_other_x_ibr = [None for i in range(n_other)]
all_other_x_des_ibr = [None for i in range(n_other)]
amb_solved_flag = False
other_solved_flag = [False for i in range(n_other)]

print("Initial conditions MPC_i: %d IBR_i: %d x: %0.1f y:%0.1f"%(i_mpc, i_rounds_ibr, x0_amb[0], x0_amb[1]))
###### Initial guess for the other u.  This will be updated once the other vehicles
###### solve the best response to the ambulance. Initial guess just looks at the last solution. This could also be a lange change
for i in range(n_other):
    all_other_u_ibr[i] = np.zeros(shape=(2, N)) ## All vehicles constant velocity
    if i%2==0:
        all_other_u_ibr[i][0,0] = -2 * np.pi/180  # This is a hack and should be explicit that it's lane change
    else:
        all_other_u_ibr[i][0,0] = 2 * np.pi/180  # This is a hack and should be explicit that it's lane change                   
    all_other_x_ibr[i], all_other_x_des_ibr[i] = all_other_MPC[i].forward_simulate_all(all_other_x0[i].reshape(6,1), all_other_u_ibr[i])

########## Solve the Ambulance MPC ##########
response_MPC = amb_MPC
response_x0 = x0_amb
nonresponse_MPC_list = all_other_MPC
nonresponse_x0_list = all_other_x0
nonresponse_u_list = all_other_u_ibr
nonresponse_x_list = all_other_x_ibr
nonresponse_xd_list = all_other_x_des_ibr
################# Generate the warm starts ###############################
u_warm_profiles = mibr.generate_warm_u(N, response_MPC)
other_velocity = np.median([x[4] for x in nonresponse_x0_list])
x_warm_profiles = mibr.generate_warm_x(response_MPC, traffic_world,  response_x0, other_velocity)
u_warm_profiles.update(x_warm_profiles) # combine into one
#######################################################################
min_response_cost = 99999999
for k_warm in u_warm_profiles.keys():
    if k_warm[0] == "0":
        ###THESE were generate by x_warm
        u_warm = np.zeros((2, N))
        x_warm = u_warm_profiles[k_warm]
        x_des_warm = np.zeros(shape=(3, N + 1))        
        for k in range(N + 1):
            x_des_warm[:, k:k+1] = response_MPC.fd(x_warm[-1,k])
    else:
        u_warm = u_warm_profiles[k_warm]
        x_warm, x_des_warm = response_MPC.forward_simulate_all(response_x0.reshape(6,1), u_warm)

    bri = mibr.IterativeBestResponseMPCMultiple(response_MPC, None, nonresponse_MPC_list )
    k_slack, k_CA, k_CA_power, wall_CA = 10000.0, 0.001, 4, True
    bri.k_slack, bri.k_CA, bri.k_CA_power, bri.world, bri.wall_CA = k_slack, k_CA, k_CA_power, world, wall_CA
    # for slack_var in bri.slack_vars_list: ## Added to constrain slacks
    #     bri.opti.subject_to(cas.vec(slack_var) <= 1.0)
    solve_amb = True
    if i_rounds_ibr == 0:
        slack = True
    else:
        slack = False
    bri.generate_optimization(N, T, response_x0, None, nonresponse_x0_list,  0, slack=slack, solve_amb=solve_amb)
    bri.opti.set_initial(bri.u_opt, u_warm)            
    bri.opti.set_initial(bri.x_opt, x_warm)
    bri.opti.set_initial(bri.x_desired, x_des_warm)   
    ### Set the trajectories of the nonresponse vehicles (as given)        
    for j in range(len(nonresponse_x_list)):
        bri.opti.set_value(bri.allother_x_opt[j], nonresponse_x_list[j])
        bri.opti.set_value(bri.allother_x_desired[j], nonresponse_xd_list[j])    
    break
cmplot.plot_single_frame(world, response_MPC, x_warm, nonresponse_x_list, None, "Both", False, 0, [1])   
# plt.show()


i_mpc_start = 0



for i_mpc in range(i_mpc_start, n_rounds_mpc):
    min_slack = np.infty
    actual_t = i_mpc * number_ctrl_pts_executed
    ###### Update the initial conditions for all vehicles
    if i_mpc > 0:
        x0_amb = xamb_executed[:, number_ctrl_pts_executed] 
        for i in range(len(all_other_x0)):
            all_other_x0[i] = all_other_x_executed[i][:, number_ctrl_pts_executed]        

    uamb_ibr = None
    xamb_ibr = None
    xamb_des_ibr = None
    all_other_u_ibr = [None for i in range(n_other)]    
    all_other_x_ibr = [None for i in range(n_other)]
    all_other_x_des_ibr = [None for i in range(n_other)]
    amb_solved_flag = False
    other_solved_flag = [False for i in range(n_other)]

    for i_rounds_ibr in range(n_rounds_ibr):
        print("Initial conditions MPC_i: %d IBR_i: %d x: %0.1f y:%0.1f"%(i_mpc, i_rounds_ibr, x0_amb[0], x0_amb[1]))
        if i_rounds_ibr == 0:
            ###### Initial guess for the other u.  This will be updated once the other vehicles
            ###### solve the best response to the ambulance. Initial guess just looks at the last solution. This could also be a lange change
            for i in range(n_other):
                if i_mpc >= 0:
                    all_other_u_ibr[i] = np.zeros(shape=(2, N)) ## All vehicles constant velocity
                    if i%2==0:
                        all_other_u_ibr[i][0,0] = -2 * np.pi/180  # This is a hack and should be explicit that it's lane change
                    else:
                        all_other_u_ibr[i][0,0] = 2 * np.pi/180  # This is a hack and should be explicit that it's lane change                   
                else:
#                     all_other_u_ibr[i] = np.concatenate((all_other_u_mpc[i][:, number_ctrl_pts_executed:], np.tile(all_other_u_mpc[i][:,-1:],(1, number_ctrl_pts_executed))),axis=1) ##   
                    all_other_u_ibr[i] = np.concatenate((all_other_u_mpc[i][:, number_ctrl_pts_executed:], np.tile(np.zeros(shape=(2,1)),(1, number_ctrl_pts_executed))),axis=1) ##   

                all_other_x_ibr[i], all_other_x_des_ibr[i] = all_other_MPC[i].forward_simulate_all(all_other_x0[i].reshape(6,1), all_other_u_ibr[i])
        ########## Solve the Ambulance MPC ##########
        response_MPC = amb_MPC
        response_x0 = x0_amb
        nonresponse_MPC_list = all_other_MPC
        nonresponse_x0_list = all_other_x0
        nonresponse_u_list = all_other_u_ibr
        nonresponse_x_list = all_other_x_ibr
        nonresponse_xd_list = all_other_x_des_ibr
        ################# Generate the warm starts ###############################
        u_warm_profiles = mibr.generate_warm_u(N, response_MPC)
        if i_rounds_ibr > 0:            # warm start with the solution from the last IBR round
            u_warm_profiles["previous"] = uamb_ibr    
        else:                           # take the control inputs of the last MPC and continue the ctrl
            if i_mpc > 0:
                u_warm_profiles["previous"] = np.concatenate((uamb_mpc[:, number_ctrl_pts_executed:], np.tile(uamb_mpc[:,-1:],(1, number_ctrl_pts_executed))),axis=1) ##    
        
        other_velocity = np.median([x[4] for x in nonresponse_x0_list])
        x_warm_profiles = mibr.generate_warm_x(response_MPC, traffic_world,  response_x0, other_velocity)
        u_warm_profiles.update(x_warm_profiles) # combine into one
        #######################################################################
        min_response_cost = 99999999      
        
        for k_warm in u_warm_profiles.keys():
            amb_solved_flag, min_bri_ibr, xamb_ibr, k_slack, k_CA, k_CA_power, wall_CA, xamb_des_ibr, uamb_ibr = solve_for_warm_start_amb()
                # ibr_sub_it +=1   
        if not amb_solved_flag:
            raise Exception("Ambulance did not converge to a solution")
        print("Ambulance Solution:  mpc_i %d  ibr_round %d"%(i_mpc, i_rounds_ibr))    
        cmplot.plot_single_frame(world, min_bri_ibr.responseMPC, xamb_ibr, nonresponse_x_list, None, xamb_desired=None, xothers_desired=None,  
                CIRCLES=True, parallelize=True, camera_speed = 0)
        plt.show()
                                        
        ########### SOLVE FOR THE OTHER VEHICLES ON THE ROAD
        if not XAMB_ONLY:
            for i in range(len(all_other_MPC)):
                print("SOLVING AGENT %d"%i)
                response_MPC = all_other_MPC[i]
                response_x0 = all_other_x0[i]
                nonresponse_MPC_list = all_other_MPC[:i] + all_other_MPC[i+1:]
                nonresponse_x0_list = all_other_x0[:i] + all_other_x0[i+1:]
                nonresponse_u_list = all_other_u_ibr[:i] + all_other_u_ibr[i+1:]
                nonresponse_x_list = all_other_x_ibr[:i] + all_other_x_ibr[i+1:]
                nonresponse_xd_list = all_other_x_des_ibr[:i] + all_other_x_des_ibr[i+1:]

                ################  Warm Start
                u_warm_profiles = mibr.generate_warm_u(N, response_MPC)
                if i_rounds_ibr > 0:   # warm start with the solution from the last IBR round
                    u_warm_profiles["previous"] = all_other_u_ibr[i] 
                else:
                    # take the control inputs of the last MPC and continue the ctrl
                    if i_mpc > 0:
                        u_warm_profiles["previous"] = np.concatenate((all_other_u_mpc[i][:, number_ctrl_pts_executed:], np.tile(all_other_u_mpc[i][:,-1:],(1, number_ctrl_pts_executed))),axis=1) ##    
                other_velocity = np.median([x[4] for x in nonresponse_x0_list])
                x_warm_profiles = mibr.generate_warm_x(response_MPC, traffic_world, response_x0, other_velocity)
                u_warm_profiles.update(x_warm_profiles) # combine into one
                
                min_response_cost = 99999999
                for k_warm in u_warm_profiles.keys():
                    #### OPTIMZATION SETTINGS
                    bri = mibr.IterativeBestResponseMPCMultiple(response_MPC, amb_MPC, nonresponse_MPC_list)
                    bri.k_slack, bri.k_CA, bri.k_CA_power, bri.world, bri.wall_CA = k_slack, k_CA, k_CA_power, world, wall_CA

                    if np.sqrt((response_x0[0] - x0_amb[0])**2 + (response_x0[1] - x0_amb[1])**2) > 30:
                        solve_amb = False
                    elif i_rounds_ibr >= 2: #in the future
                        solve_amb = False
                    else:
                        solve_amb = True  

                    bri.generate_optimization(N, T, response_x0, x0_amb, nonresponse_x0_list,  0, slack=False, solve_amb=solve_amb)
                    # bri.opti.callback(lambda i: print("J_i %.03f,  J_j %.03f, Slack %.03f, CA  %.03f"%(bri.solution.value(bri.response_svo_cost), bri.solution.value(bri.other_svo_cost), bri.solution.value(bri.k_slack*bri.slack_cost), bri.solution.value(bri.k_CA*bri.collision_cost))))
                    # bri.opti.callback(lambda i: bri.debug_callback(i, range(N)))
                    ### Set the trajectories of the nonresponse vehicles (as given)  
                    # 
                    #    
                    if solve_amb:
                        bri.opti.set_initial(bri.xamb_opt, xamb_ibr)
                        bri.opti.set_initial(bri.xamb_desired, xamb_des_ibr)                        
                    else:
                        bri.opti.set_value(bri.xamb_opt, xamb_ibr)
                        bri.opti.set_value(bri.xamb_desired, xamb_des_ibr)
                    for j in range(len(nonresponse_x_list)):
                        bri.opti.set_value(bri.allother_x_opt[j], nonresponse_x_list[j])
                        bri.opti.set_value(bri.allother_x_desired[j], nonresponse_xd_list[j])
                    # for slack_var in bri.slack_vars_list: ## Added to constrain slacks
                    #     bri.opti.subject_to(cas.vec(slack_var) <= 1.0)                    
                    
                    if k_warm[0] == "0":
                        ###THESE were generate by x_warm
                        u_warm = np.zeros((2, N))
                        x_warm = u_warm_profiles[k_warm]
                        x_des_warm = np.zeros(shape=(3, N + 1))        
                        for k in range(N + 1):
                            x_des_warm[:, k:k+1] = response_MPC.fd(x_warm[-1,k])
                    else:
                        u_warm = u_warm_profiles[k_warm]
                        x_warm, x_des_warm = response_MPC.forward_simulate_all(response_x0.reshape(6,1), u_warm)                    
                    bri.opti.set_initial(bri.u_opt, u_warm)            
                    bri.opti.set_initial(bri.x_opt, x_warm)
                    bri.opti.set_initial(bri.x_desired, x_des_warm)   

                    # Debugging
                    try:                     ### Solve the Optimization
                        bri.solve(uamb_ibr, nonresponse_u_list, solve_amb)
                        x1_ibr_k, u1_ibr_k, x1_des_ibr_k, _, _, _, _, _, _ = bri.get_solution()
                        if PRINT_FLAG:
                            print("  i_mpc %d n_round %d i %02d Cost %.02f Slack %.02f "%(i_mpc, i_rounds_ibr, i, bri.solution.value(bri.total_svo_cost), bri.solution.value(bri.slack_cost)))
                            print("  J_i %.03f,  J_j %.03f, Slack %.03f, CA  %.03f"%(bri.solution.value(bri.response_svo_cost), bri.solution.value(bri.other_svo_cost), bri.solution.value(bri.k_slack*bri.slack_cost), bri.solution.value(bri.k_CA*bri.collision_cost)))
                            print("  Dir:", subdir_name)
                        if bri.solution.value(bri.slack_cost) < min_slack:
                            current_cost = bri.solution.value(bri.total_svo_cost)
                            if current_cost < min_response_cost:
                                all_other_x_ibr[i] = x1_ibr_k
                                all_other_u_ibr[i] = u1_ibr_k
                                all_other_x_des_ibr[i] = x1_des_ibr_k
                                other_solved_flag[i] = True
                                min_response_cost = current_cost
                                min_response_warm = k_warm
                                min_bri = bri

                                if SAVE_FLAG:
                                    file_name = folder + "data/"+'%03d'%i_rounds_ibr
                                    mibr.save_state(file_name, xamb_ibr, uamb_ibr, xamb_des_ibr, all_other_x_ibr, all_other_u_ibr, all_other_x_des_ibr)
                                    mibr.save_costs(file_name, bri)
                    except RuntimeError:
                        print("  Infeasibility: k_warm %s"%k_warm)
                if not other_solved_flag[i]:
                    raise Exception("i did not converge to a solution")
                print("Vehicle i=%d Solution:  mpc_i %d  ibr_round %d"%(i, i_mpc, i_rounds_ibr))    
                cmplot.plot_single_frame(world, min_bri_ibr.responseMPC, all_other_x_ibr[i], [xamb_ibr] + nonresponse_x_list, None, xamb_desired=None, xothers_desired=None,  
                        CIRCLES=True, parallelize=True, camera_speed = 0)
                plt.show()                    

    xamb_mpc = xamb_ibr
    xamb_des_mpc = xamb_des_ibr
    uamb_mpc = uamb_ibr
    all_other_u_mpc = all_other_u_ibr
    all_other_x_mpc = all_other_x_ibr
    all_other_x_des_mpc = all_other_x_des_ibr
    bri_mpc = min_bri
    file_name = folder + "data/"+'r%02d%03d'%(i_mpc, i_rounds_ibr)

    all_other_x_executed = [np.zeros(shape=(6,number_ctrl_pts_executed+1)) for i in range(n_other)]
    all_other_u_executed = [np.zeros(shape=(2,number_ctrl_pts_executed)) for i in range(n_other)]
    if (not amb_solved_flag) or (False in other_solved_flag):
        print("MPC Error: One of the vehicles did not converge to a solution.")
        print(xamb_mpc)
        print(all_other_x_mpc)
        raise Exception
    else:
        ### SAVE EXECUTED MPC SOLUTION TO HISTORY
        actual_t = i_mpc * number_ctrl_pts_executed
        xamb_executed, uamb_executed = xamb_mpc[:,:number_ctrl_pts_executed+1], uamb_mpc[:,:number_ctrl_pts_executed]
        actual_xamb[:,actual_t:actual_t+number_ctrl_pts_executed+1], actual_uamb[:,actual_t:actual_t+number_ctrl_pts_executed] = xamb_executed, uamb_executed
        for i in range(len(all_other_x)):
            all_other_x_executed[i], all_other_u_executed[i] = all_other_x_mpc[i][:,:number_ctrl_pts_executed+1], all_other_u_mpc[i][:,:number_ctrl_pts_executed]
            actual_xothers[i][:,actual_t:actual_t+number_ctrl_pts_executed+1], actual_uothers[i][:,actual_t:actual_t+number_ctrl_pts_executed] = all_other_x_executed[i], all_other_u_executed[i]

        ### SAVE STATES AND PLOT
        if SAVE_FLAG:
            mibr.save_state(file_name, xamb_mpc, uamb_mpc, xamb_des_mpc, all_other_x_mpc, all_other_u_mpc, all_other_x_des_mpc)
            mibr.save_costs(file_name, bri_mpc)        
        print(" MPC Done:  Rd %02d / %02d"%(i_mpc, n_rounds_mpc))
        print(" Full MPC Solution", xamb_mpc[0:2,:])
        print(" Executed MPC", xamb_mpc[0:2,:number_ctrl_pts_executed+1])
        print(" Solution Costs...")
        for cost in bri_mpc.car1_costs_list:
            print("%.04f"%bri_mpc.solution.value(cost))
        print(bri_mpc.solution.value(bri_mpc.k_CA * bri_mpc.collision_cost), bri_mpc.solution.value(bri_mpc.collision_cost))
        print(bri_mpc.solution.value(bri_mpc.k_slack * bri_mpc.slack_cost), bri_mpc.solution.value(bri_mpc.slack_cost))
        print(" Save to...", file_name)
        
#         plot_range = range(number_ctrl_pts_executed+1)
        
#         for k in range(xamb_executed.shape[1]):
#             cmplot.plot_multiple_cars( k, bri_mpc.responseMPC, all_other_x_executed, xamb_executed, True, None, None, None, bri_mpc.world, 0)     
            
#             plt.show()


        mean_amb_v = np.mean(xamb_executed[4,:])
        im_dir = folder + '%02d/'%i_mpc
        os.makedirs(im_dir+"imgs/")
        cmplot.plot_cars(bri_mpc.world, bri_mpc.responseMPC, xamb_executed, all_other_x_executed, im_dir, None, None, False, False, mean_amb_v)
        plt.plot(xamb_mpc[4,:],'--')
        plt.plot(xamb_mpc[4,:] * np.cos(xamb_mpc[2,:]))
        plt.ylabel("Velocity / Vx (full mpc)")
        plt.hlines(35*0.447,0,xamb_mpc.shape[1])
        plt.show()
        plt.plot(uamb_mpc[1,:],'o')
        plt.hlines(amb_MPC.max_v_u,0,uamb_mpc.shape[1])
        plt.hlines(amb_MPC.min_v_u,0,uamb_mpc.shape[1])
        plt.ylabel("delta_u_v")
        plt.show()
print("Solver Done!  Runtime: %.1d"%(time.time()-t_start_time))

final_t = actual_t+number_ctrl_pts_executed+1
actual_xamb = actual_xamb[:,:actual_t+number_ctrl_pts_executed+1]
for i in range(len(all_other_x)):
    actual_xothers[i] = actual_xothers[i][:,:actual_t+number_ctrl_pts_executed+1]

mean_amb_v = np.median(actual_xamb[4,:])
min_amb_v = np.min(actual_xamb[4,:])
print("Min Speed", min_amb_v, mean_amb_v)
print("Plotting all")
cmplot.plot_cars(bri_mpc.world, bri_mpc.responseMPC, actual_xamb, actual_xothers, folder, None, None, False, False, min_amb_v)

file_name = folder + "data/"+'a%02d%03d'%(i_mpc, i_rounds_ibr)
print("Saving to...  ", file_name)
mibr.save_state(file_name, actual_xamb, uamb_mpc, xamb_des_mpc, actual_xothers, all_other_u_mpc, all_other_x_des_mpc)
