import time, datetime, argparse
import os, sys
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt
import copy as cp
import pickle

PROJECT_PATHS = ['/home/nbuckman/Dropbox (MIT)/DRL/2020_01_cooperative_mpc/mpc-multiple-vehicles/', '/Users/noambuckman/mpc-multiple-vehicles/']
for p in PROJECT_PATHS:
    sys.path.append(p)

import casadi as cas
import src.MPC_Casadi as mpc
import src.TrafficWorld as tw
import src.IterativeBestResponseMPCMultiple as mibr
import src.car_plotting_multiple as cmplot
import src.solver_helper as helper

##########################################################
svo_theta = np.pi/3.0
# random_seed = args.random_seed[0]
random_seed = 3
NEW = True
if NEW:
    optional_suffix = "6cars"
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
T = 5  # MPC Planning Horizon
dt = 0.2
N = int(T/dt) #Number of control intervals in MPC
n_rounds_mpc = 8
percent_mpc_executed = .50 ## This is the percent of MPC that is executed
number_ctrl_pts_executed =  int(np.floor(N*percent_mpc_executed))
print("number ctrl pts:  %d"%number_ctrl_pts_executed)
XAMB_ONLY = False
PLOT_FLAG, SAVE_FLAG, PRINT_FLAG = False, False, False
n_other = 5
n_rounds_ibr = 3
world = tw.TrafficWorld(2, 0, 1000)
    # large_world = tw.TrafficWorld(2, 0, 1000, 5.0)
#########################################################################

all_other_x0, all_other_MPC, amb_MPC, x0_amb = helper.initialize_cars(n_other, N, dt, world, svo_theta)
if SAVE_FLAG:
    pickle.dump(all_other_MPC[0], open(folder + "data/"+"mpc%d"%i + ".p",'wb'))
    pickle.dump(amb_MPC, open(folder + "data/"+"mpcamb" + ".p",'wb'))

t_start_time = time.time()
actual_xamb = np.zeros((6, n_rounds_mpc*number_ctrl_pts_executed + 1))
actual_uamb = np.zeros((2, n_rounds_mpc*number_ctrl_pts_executed))
actual_xothers = [np.zeros((6, n_rounds_mpc*number_ctrl_pts_executed + 1)) for i in range(n_other)]
actual_uothers = [np.zeros((2, n_rounds_mpc*number_ctrl_pts_executed)) for i in range(n_other)]
    
xamb_executed, all_other_x_executed = None, [] #This gets updated after each round of MPC
uamb_mpc, all_other_u_mpc = None, []

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
        ############# Generate (if needed) the control inputs of other vehicles
        if i_rounds_ibr == 0:
            if i_mpc == 0:
                all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr = helper.pullover_guess(N, all_other_MPC, all_other_x0)  # This is a hack and should be explicit that it's lane change                   
            else:
                all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr = helper.extend_last_mpc_ctrl(all_other_u_mpc, number_ctrl_pts_executed, all_other_MPC, all_other_x0)  # This is a hack and should be explicit that it's lane change                   
        ########## Solve the Response MPC ##########
        response_MPC, response_x0 = amb_MPC, x0_amb
        nonresponse_MPC_list, nonresponse_x0_list = all_other_MPC, all_other_x0
        nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list = all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr

        ################# Generate the warm starts ###############################
        u_warm_profiles, ux_warm_profiles = mibr.generate_warm_u(N, response_MPC, response_x0)
        if i_rounds_ibr == 0 and i_mpc > 0:
            u_warm_profiles["previous"] = np.concatenate((uamb_mpc[:, number_ctrl_pts_executed:], np.tile(uamb_mpc[:,-1:],(1, number_ctrl_pts_executed))),axis=1) ##    
        if i_rounds_ibr > 0:
            u_warm_profiles["previous"] = uamb_ibr    
        if (i_rounds_ibr == 0 and i_mpc > 0) or i_rounds_ibr > 0 :
            x_warm, x_des_warm = response_MPC.forward_simulate_all(response_x0.reshape(6,1), u_warm_profiles["previous"])
            ux_warm_profiles["previous"] = [u_warm_profiles["previous"], x_warm, x_des_warm]
        x_warm_profiles, x_ux_warm_profiles = mibr.generate_warm_x(response_MPC, world,  response_x0, np.median([x[4] for x in nonresponse_x0_list]))
        ux_warm_profiles.update(x_ux_warm_profiles) # combine into one
        ################# Solve the Best Response ############################
        k_slack, k_CA, k_CA_power, wall_CA = 1000000.0, 0.001, 4, True
        k_max_slack = 0.01
        if i_rounds_ibr == 0:
            slack = True
        else:
            slack = False 
        solve_again, solve_number = True, 0
        while solve_again and solve_number < 4:
            min_response_cost = np.infty            
            k_CA_power *= 10
            k_slack *= 10
            for k_warm in u_warm_profiles.keys():
                u_warm, x_warm, x_des_warm = ux_warm_profiles[k_warm]
                solve_amb, a_MPC = False, None

                amb_solved_flag, current_cost, max_slack, bri, xamb, xamb_des, uamb = helper.solve_best_response(response_MPC, a_MPC, nonresponse_MPC_list, k_slack, k_CA, k_CA_power, world, wall_CA, N, T, response_x0, x0_amb, nonresponse_x0_list, slack, solve_amb, k_warm, u_warm, x_warm, x_des_warm, nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list, )
                if current_cost <= min_response_cost:
                    min_response_cost = current_cost
                    xamb_ibr, xamb_des_ibr, uamb_ibr = xamb, xamb_des, uamb
                    max_slack_ibr = max_slack
                    min_bri_ibr = bri

            if max_slack_ibr > k_max_slack:
                print("Max Slack is too large %.05f > thresh %.05f"%(max_slack_ibr, k_max_slack))
                solve_again = True
                solve_number += 1
            else:
                solve_again = False
        if not amb_solved_flag:
            raise Exception("Ambulance did not converge to a solution")
        if solve_again:
            raise Exception("Slack variable is too high")
        # print("Ambulance Solution:  mpc_i %d  ibr_round %d"%(i_mpc, i_rounds_ibr))    
        # cmplot.plot_single_frame(world, min_bri_ibr.responseMPC, xamb_ibr, nonresponse_x_list, None, CIRCLES="Ellipse", parallelize=True, camera_speed = None, plot_range = range(N+1)[:int(N/2)], car_ids = None, xamb_desired=None, xothers_desired=None)        
        # plt.show()
        # cmplot.plot_single_frame(world, min_bri_ibr.responseMPC, xamb_ibr, nonresponse_x_list, None, CIRCLES="Ellipse", parallelize=True, camera_speed = None, plot_range = range(N+1)[int(N/2):], car_ids = None, xamb_desired=None, xothers_desired=None)        
        # plt.show()
                                        
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

                ################# Generate the warm starts ###############################
                u_warm_profiles, ux_warm_profiles = mibr.generate_warm_u(N, response_MPC, response_x0)
                if i_rounds_ibr == 0 and i_mpc > 0:
                    u_warm_profiles["previous"] = np.concatenate((all_other_u_mpc[i][:, number_ctrl_pts_executed:], np.tile(all_other_u_mpc[i][:,-1:],(1, number_ctrl_pts_executed))),axis=1) ##    
                if i_rounds_ibr > 0:
                    u_warm_profiles["previous"] = all_other_u_ibr[i]     
                if (i_rounds_ibr == 0 and i_mpc > 0) or i_rounds_ibr > 0 :
                    x_warm, x_des_warm = response_MPC.forward_simulate_all(response_x0.reshape(6,1), u_warm)
                    ux_warm_profiles["previous"] = [u_warm_profiles["previous"], x_warm, x_des_warm]
                x_warm_profiles, x_ux_warm_profiles = mibr.generate_warm_x(response_MPC, world,  response_x0, np.median([x[4] for x in nonresponse_x0_list]))
                ux_warm_profiles.update(x_ux_warm_profiles) # combine into one

                k_solve_amb_min_distance, k_solve_amb_max_ibr = 30, 2
                initial_distance_to_ambulance = np.sqrt((response_x0[0] - x0_amb[0])**2 + (response_x0[1] - x0_amb[1])**2)
                if i_rounds_ibr < k_solve_amb_max_ibr and (initial_distance_to_ambulance < k_solve_amb_max_ibr):
                    solve_amb = False
                else:
                    solve_amb = True  

                min_response_cost = np.infty
                k_slack, k_CA, k_CA_power, wall_CA = 1000.0, 0.0001, 4, True
                solve_again = True
                while solve_again:
                    k_slack *= 10
                    k_CA *= 10
                    for k_warm in u_warm_profiles.keys():
                        u_warm, x_warm, x_des_warm = ux_warm_profiles[k_warm]              
                        solved_flag, current_cost, max_slack, bri, x, x_des, u = helper.solve_best_response(response_MPC, amb_MPC, nonresponse_MPC_list, k_slack, k_CA, k_CA_power, world, wall_CA, N, T, response_x0, x0_amb, nonresponse_x0_list, slack, solve_amb, k_warm, u_warm, x_warm, x_des_warm, nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list, uamb_ibr, xamb_ibr, xamb_des_ibr, )

                                # print("  i_mpc %d n_round %d i %02d Cost %.02f Slack %.02f "%(i_mpc, i_rounds_ibr, i, bri.solution.value(bri.total_svo_cost), bri.solution.value(bri.slack_cost)))
                                # print("  J_i %.03f,  J_j %.03f, Slack %.03f, CA  %.03f"%(bri.solution.value(bri.response_svo_cost), bri.solution.value(bri.other_svo_cost), bri.solution.value(bri.k_slack*bri.slack_cost), bri.solution.value(bri.k_CA*bri.collision_cost)))
                                # print("  Dir:", subdir_name)
                        if current_cost < min_response_cost:
                            all_other_u_ibr[i], all_other_x_ibr[i], all_other_x_des_ibr[i]  = u, x, x_des
                            other_solved_flag[i] = True
                            min_response_cost = current_cost
                            min_response_warm = k_warm
                            min_bri = bri

                    if SAVE_FLAG:
                        file_name = folder + "data/"+'%03d'%i_rounds_ibr
                        mibr.save_state(file_name, xamb_ibr, uamb_ibr, xamb_des_ibr, all_other_x_ibr, all_other_u_ibr, all_other_x_des_ibr)
                        mibr.save_costs(file_name, bri)
                        
                    if max_slack >= k_max_slack:
                        print("Max Slack is too large %.05f > thresh %.05f"%(max_slack, k_max_slack))
                        solve_number += 1
                    else:
                        solve_again = False

                if not other_solved_flag[i]:
                    raise Exception("i did not converge to a solution")
                if solve_again:
                    raise Exception("Slack variable is too high")     

                print("Vehicle i=%d Solution:  mpc_i %d  ibr_round %d"%(i, i_mpc, i_rounds_ibr))    
                cmplot.plot_single_frame(world, min_bri_ibr.responseMPC, all_other_x_ibr[i], [xamb_ibr] + nonresponse_x_list, None, CIRCLES="Ellipse", parallelize=True, camera_speed = None, plot_range = None, car_ids = None, xamb_desired=None, xothers_desired=None)
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
        for i in range(len(all_other_x_mpc)):
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
for i in range(len(actual_xothers)):
    actual_xothers[i] = actual_xothers[i][:,:actual_t+number_ctrl_pts_executed+1]

mean_amb_v = np.median(actual_xamb[4,:])
min_amb_v = np.min(actual_xamb[4,:])
print("Min Speed", min_amb_v, mean_amb_v)
print("Plotting all")
cmplot.plot_cars(bri_mpc.world, bri_mpc.responseMPC, actual_xamb, actual_xothers, folder, None, None, False, False, min_amb_v)

file_name = folder + "data/"+'a%02d%03d'%(i_mpc, i_rounds_ibr)
print("Saving to...  ", file_name)
mibr.save_state(file_name, actual_xamb, uamb_mpc, xamb_des_mpc, actual_xothers, all_other_u_mpc, all_other_x_des_mpc)
