import time, datetime, argparse, os, sys, pickle, psutil
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt
import casadi as cas
import copy as cp
PROJECT_PATHS = ['/home/nbuckman/Dropbox (MIT)/DRL/2020_01_cooperative_mpc/mpc-multiple-vehicles/', '/Users/noambuckman/mpc-multiple-vehicles/']
for p in PROJECT_PATHS:
    sys.path.append(p)
import src.vehicle as vehicle
import src.traffic_world as tw
import src.multiagent_mpc as mpc
import src.car_plotting_multiple as cmplot
import src.solver_helper as helper

##########################################################
svo_theta = np.pi/3.0
# random_seed = args.random_seed[0]
random_seed = 9
if random_seed > 0:
    np.random.seed(random_seed)
#######################################################################
T = 5  # MPC Planning Horizon
dt = 0.2
N = int(T/dt) #Number of control intervals in MPC
n_rounds_mpc = 75
percent_mpc_executed = .10 ## This is the percent of MPC that is executed
number_ctrl_pts_executed =  int(np.floor(N*percent_mpc_executed))
print("number ctrl pts:  %d"%number_ctrl_pts_executed)
PLOT_FLAG, SAVE_FLAG, PRINT_FLAG = False, True, False
n_other = 13
n_rounds_ibr = 4
world = tw.TrafficWorld(2, 0, 1000)
n_processors = 16
    # large_world = tw.TrafficWorld(2, 0, 1000, 5.0)
#########################################################################
optional_suffix = str(n_other) + "nograss"
subdir_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + optional_suffix
folder = "results/" + subdir_name + "/"
for f in [folder, folder+"imgs/", folder+"data/", folder+"vids/", folder+"plots/"]:
    os.makedirs(f)
print(folder)

position_list = [
    (0, 20),
    (0, 35),
    (1, 40),
    (0, 60),
    (1, 60),
    (0, 80),
    (1, 85),
    (1, 100),
    (1, 123),
    (0, 120),
    (0, 145),
    (0, 160),
    (0, 200)
]
if len(position_list) != n_other:
    raise Exception("number of vehicles don't match n_other")


all_other_x0, all_other_MPC, amb_MPC, amb_x0 = helper.initialize_cars(n_other, N, dt, world, svo_theta, True, False, False, position_list)

t_start_time = time.time()
actual_t = 0    
actual_xamb, actual_uamb = np.zeros((6, n_rounds_mpc*number_ctrl_pts_executed + 1)), np.zeros((2, n_rounds_mpc*number_ctrl_pts_executed)) 
actual_xothers = [np.zeros((6, n_rounds_mpc*number_ctrl_pts_executed + 1)) for i in range(n_other)]
actual_uothers = [np.zeros((2, n_rounds_mpc*number_ctrl_pts_executed)) for i in range(n_other)]
    
xamb_executed, all_other_x_executed = None, [] #This gets updated after each round of MPC
uamb_mpc, all_other_u_mpc = None, []
if SAVE_FLAG:
    pass
    # pickle.dump(all_other_MPC[0], open(folder + "data/"+"mpcother" + ".p",'wb'))
    # pickle.dump(amb_MPC, open(folder + "data/"+"mpcamb" + ".p",'wb'))

preloaded_data = True

i_mpc_start = 0                 
preloaded_data = False
for i_mpc in range(i_mpc_start, n_rounds_mpc):
    min_slack = np.infty
#     actual_t = i_mpc * number_ctrl_pts_executed
    ###### Update the initial conditions for all vehicles
    if i_mpc > 0 and not preloaded_data:
        amb_x0 = xamb_executed[:, number_ctrl_pts_executed] 
        for i in range(len(all_other_x0)):
            all_other_x0[i] = all_other_x_executed[i][:, number_ctrl_pts_executed]
    preloaded_data = False
    vehicles_index_in_mpc = []
    vehicles_index_constant_v = []
    for i in range(n_other):
        initial_distance = np.sqrt((amb_x0[0]-all_other_x0[i][0])**2 + (amb_x0[1]-all_other_x0[i][1])**2)
        if initial_distance <= 150:
            vehicles_index_in_mpc += [i]
        else:
            vehicles_index_constant_v += [i]    
    vehicles_index_in_mpc = [j for j in range(n_other)] ## Default

    uamb_ibr, xamb_ibr, xamb_des_ibr = None, None, None
    all_other_u_ibr = [None for i in range(n_other)]    
    all_other_x_ibr = [None for i in range(n_other)]
    all_other_x_des_ibr = [None for i in range(n_other)]
    for j in vehicles_index_constant_v:
        all_other_u_ibr[j] = np.zeros((2,N)) #constant v control inputs
        all_other_x_ibr[j], all_other_x_des_ibr[j] = all_other_MPC[j].forward_simulate_all(all_other_x0[j].reshape(6,1), all_other_u_ibr[j]) 
        
    for i_rounds_ibr in range(n_rounds_ibr):
        ############# Generate (if needed) the control inputs of other vehicles        
        if i_rounds_ibr == 0:
            if i_mpc == 0:
                all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr = helper.pullover_guess(N, all_other_MPC, all_other_x0)  # This is a hack and should be explicit that it's lane change                   
            else:
                all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr = helper.extend_last_mpc_ctrl(all_other_u_mpc, number_ctrl_pts_executed, N, all_other_MPC, all_other_x0)  # This is a hack and should be explicit that it's lane change                   
        ########## Solve the Response MPC ##########
        response_MPC, response_x0 = amb_MPC, amb_x0
        nonresponse_MPC_list, nonresponse_x0_list = all_other_MPC, all_other_x0
        nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list = all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr

        fake_amb_i = helper.get_min_dist_i(amb_x0, all_other_x0, True)
        nonresponse_MPC_list = all_other_MPC[:fake_amb_i] + all_other_MPC[fake_amb_i+1:]
        nonresponse_x0_list = all_other_x0[:fake_amb_i] + all_other_x0[fake_amb_i+1:]
        nonresponse_u_list = all_other_u_ibr[:fake_amb_i] + all_other_u_ibr[fake_amb_i+1:]
        nonresponse_x_list = all_other_x_ibr[:fake_amb_i] + all_other_x_ibr[fake_amb_i+1:]
        nonresponse_xd_list = all_other_x_des_ibr[:fake_amb_i] + all_other_x_des_ibr[fake_amb_i+1:]
        fake_amb_MPC = all_other_MPC[fake_amb_i]     
        fake_amb_x0 = all_other_x0[fake_amb_i]
        fake_amb_u = all_other_u_ibr[fake_amb_i]
        fake_amb_x= all_other_x_ibr[fake_amb_i] 
        fake_amb_xd = all_other_x_des_ibr[fake_amb_i]    
        solve_amb = True  
        i = -1
        ################# Generate the warm starts ###############################
        u_warm_profiles, ux_warm_profiles = mpc.generate_warm_u(N, response_MPC, response_x0)
        if i_mpc > 0:
            u_warm_profiles["previous_mpc"] = np.concatenate((uamb_mpc[:, number_ctrl_pts_executed:], np.tile(uamb_mpc[:,-1:],(1, number_ctrl_pts_executed))),axis=1) ##    
            x_warm, x_des_warm = response_MPC.forward_simulate_all(response_x0.reshape(6,1), u_warm_profiles["previous_mpc"]) 
            ux_warm_profiles["previous_mpc"] = [u_warm_profiles["previous_mpc"], x_warm, x_des_warm]
        if i_rounds_ibr > 0:
            u_warm_profiles["previous_ibr"] = uamb_ibr    
            x_warm, x_des_warm = response_MPC.forward_simulate_all(response_x0.reshape(6,1), u_warm_profiles["previous_ibr"]) 
            ux_warm_profiles["previous_ibr"] = [u_warm_profiles["previous_ibr"], x_warm, x_des_warm]            
            
        x_warm_profiles, x_ux_warm_profiles = mpc.generate_warm_x(response_MPC, world,  response_x0, np.median([x[4] for x in nonresponse_x0_list]))
        ux_warm_profiles.update(x_ux_warm_profiles) # combine into one
        ################# Solve the Ambulance Best Response ############################
        k_slack_d, k_CA_d, k_CA_power_d, wall_CA_d = 100000, 0.1, 4, True
        
        k_max_slack = 0.01
        k_solve_amb_max_ibr = 3
        k_max_solve_number = 3
        k_max_round_with_slack = np.infty
        slack = True if i_rounds_ibr <= k_max_round_with_slack else False
        solve_amb = True if i_rounds_ibr < k_solve_amb_max_ibr or fake_amb_i>-1 else False
        solve_again, solve_number, max_slack_ibr, debug_flag = True, 0, np.infty, False
        while solve_again and solve_number < k_max_solve_number:
            print("SOLVING AMBULANCE:  Attempt %d / %d"%(solve_number+1, k_max_solve_number)) 
            k_slack = k_slack_d * 10**solve_number
            k_CA = k_CA_d * 10**solve_number
            if solve_number > 2 or True:
                debug_flag = True
            if psutil.virtual_memory().percent >= 90.0:
                raise Exception("Virtual Memory is too high, exiting to save computer")
            solved, min_cost_ibr, max_slack_ibr, x_ibr, x_des_ibr, u_ibr, debug_list = helper.solve_warm_starts(n_processors, ux_warm_profiles, response_MPC, fake_amb_MPC, nonresponse_MPC_list, k_slack, k_CA, k_CA_power_d, world, wall_CA_d, N, T, response_x0, fake_amb_x0, nonresponse_x0_list, slack, solve_amb, nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list, uamb=fake_amb_u, xamb=fake_amb_x, xamb_des=fake_amb_xd, debug_flag=debug_flag)
            if max_slack_ibr <= k_max_slack:
                xamb_ibr, xamb_des_ibr, uamb_ibr = x_ibr, x_des_ibr, u_ibr
                solve_again = False
            else:
                print("Max Slack is too large %.05f > thresh %.05f"%(max_slack_ibr, k_max_slack))
                solve_again = True
                solve_number += 1
#             raise Exception("Test")
        if solve_again:
            raise Exception("Slack variable is too high or infeasible.  MaxS = %.05f > thresh %.05f"%(max_slack_ibr, k_max_slack))
        print("Ambulance Plot Max Slack %0.04f"%max_slack_ibr)
        end_frame = actual_t+number_ctrl_pts_executed+1
        # cmplot.plot_cars(world, amb_MPC, xamb_ibr[:,:], [x[:,:] for x in all_other_x_ibr], None, "ellipse", False, 0)                        
        # plt.show()                                              
        ################# SOLVE BEST RESPONSE FOR THE OTHER VEHICLES ON THE ROAD ############################
        for i in vehicles_index_in_mpc:
            response_MPC, response_x0 = all_other_MPC[i], all_other_x0[i]
            nonresponse_MPC_list = all_other_MPC[:i] + all_other_MPC[i+1:]
            nonresponse_x0_list = all_other_x0[:i] + all_other_x0[i+1:]
            nonresponse_u_list = all_other_u_ibr[:i] + all_other_u_ibr[i+1:]
            nonresponse_x_list = all_other_x_ibr[:i] + all_other_x_ibr[i+1:]
            nonresponse_xd_list = all_other_x_des_ibr[:i] + all_other_x_des_ibr[i+1:]

            ################# Generate the warm starts ###############################
            u_warm_profiles, ux_warm_profiles = mpc.generate_warm_u(N, response_MPC, response_x0)            
            if i_mpc > 0:
                u_warm_profiles["previous_mpc"] = np.concatenate((all_other_u_mpc[i][:, number_ctrl_pts_executed:], np.tile(all_other_u_mpc[i][:,-1:],(1, number_ctrl_pts_executed))),axis=1) ##    
                x_warm, x_des_warm = response_MPC.forward_simulate_all(response_x0.reshape(6,1), u_warm_profiles["previous_mpc"])
                ux_warm_profiles["previous_mpc"] = [u_warm_profiles["previous_mpc"], x_warm, x_des_warm]            
            
            if i_rounds_ibr > 0:
                u_warm_profiles["previous_ibr"] = all_other_u_ibr[i]  
                x_warm, x_des_warm = response_MPC.forward_simulate_all(response_x0.reshape(6,1), u_warm_profiles["previous_ibr"])
                ux_warm_profiles["previous_ibr"] = [u_warm_profiles["previous_ibr"], x_warm, x_des_warm]
            
            x_warm_profiles, x_ux_warm_profiles = mpc.generate_warm_x(response_MPC, world,  response_x0, np.median([x[4] for x in nonresponse_x0_list]))
            ux_warm_profiles.update(x_ux_warm_profiles) # combine into one
       
            solve_again, solve_number, max_slack_ibr, debug_flag, = True, 0, np.infty, False,
            
            k_solve_amb_min_distance = 30
            initial_distance_to_ambulance = np.sqrt((response_x0[0] - amb_x0[0])**2 + (response_x0[1] - amb_x0[1])**2)       
            solve_amb = True if (i_rounds_ibr < k_solve_amb_max_ibr and 
                                  initial_distance_to_ambulance < k_solve_amb_min_distance and 
                                 response_x0[0]>amb_x0[0]-0) else False     
            slack = True if i_rounds_ibr <= k_max_round_with_slack else False
            while solve_again and solve_number < k_max_solve_number:
                print("SOLVING Agent %d:  Attempt %d / %d"%(i, solve_number+1, k_max_solve_number))        
                k_slack = k_slack_d * 10**solve_number
                k_CA = k_CA_d * 10**solve_number                
                if solve_number > 2:
                    debug_flag = True
                if psutil.virtual_memory().percent >= 90.0:
                    raise Exception("Virtual Memory is too high, exiting to save computer")                
                solved, min_cost_ibr, max_slack_ibr, x_ibr, x_des_ibr, u_ibr, debug_list = helper.solve_warm_starts(n_processors, ux_warm_profiles, response_MPC, amb_MPC, nonresponse_MPC_list, k_slack, k_CA, k_CA_power_d, world, wall_CA_d, N, T, response_x0, amb_x0, nonresponse_x0_list, slack, solve_amb, nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list, uamb_ibr, xamb_ibr, xamb_des_ibr, debug_flag)
                if max_slack_ibr <= k_max_slack:
                    all_other_x_ibr[i], all_other_x_des_ibr[i], all_other_u_ibr[i] = x_ibr, x_des_ibr, u_ibr
                    solve_again = False
                else:
                    print("Max Slack is too large %.05f > thresh %.05f"%(max_slack_ibr, k_max_slack))
                    solve_again = True
                    solve_number += 1
                    if (solve_number == k_max_solve_number) and solve_amb:
                        print("Re-solving without ambulance")
                        solve_number = 0   
                        solve_amb = False
            if solve_again:
                raise Exception("Slack variable is too high. Max Slack = %0.04f"%max_slack_ibr)    
            # if SAVE_FLAG:
            #     file_name = folder + "data/"+'%03d'%i_rounds_ibr
            #     mpc.save_state(file_name, xamb_ibr, uamb_ibr, xamb_des_ibr, all_other_x_ibr, all_other_u_ibr, all_other_x_des_ibr)
            #     # mpc.save_costs(file_name, bri)             

            print("Vehicle i=%d Solution:  mpc_i %d  ibr_round %d"%(i, i_mpc, i_rounds_ibr)) 
            print("Dir: %s"%folder)   
            end_frame = actual_t+number_ctrl_pts_executed+1
            start_frame = max(0, end_frame - 20)
#             cmplot.plot_cars(world, amb_MPC, xamb_ibr[:,:], [x[:,:] for x in all_other_x_ibr], None, "ellipse", False, 0)                        
            # cmplot.plot_cars(world, amb_MPC,  all_other_x_ibr[i], [xamb_ibr] + nonresponse_x_list, None, "ellipse", False, 0)                        
            # plt.show()
            # cmplot.plot_single_frame(world, response_MPC, all_other_x_ibr[i], [xamb_ibr] + nonresponse_x_list, None, "Ellipse", parallelize=True, camera_speed = None, plot_range = None, car_ids = None, xamb_desired=None, xothers_desired=None)
            # plt.show()                    

    ################ SAVE THE BEST RESPONSE SOLUTION FOR THE CURRENT PLANNING HORIZONG/MPC ITERATION ###########################

    xamb_mpc, xamb_des_mpc, uamb_mpc = xamb_ibr, xamb_des_ibr, uamb_ibr
    all_other_u_mpc, all_other_x_mpc, all_other_x_des_mpc = all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr
    bri_mpc = None

    all_other_x_executed = [np.zeros(shape=(6,number_ctrl_pts_executed+1)) for i in range(n_other)]
    all_other_u_executed = [np.zeros(shape=(2,number_ctrl_pts_executed)) for i in range(n_other)]

    ### SAVE EXECUTED MPC SOLUTION TO HISTORY
#     actual_t = i_mpc * number_ctrl_pts_executed
    xamb_executed, uamb_executed = xamb_mpc[:,:number_ctrl_pts_executed+1], uamb_mpc[:,:number_ctrl_pts_executed]
    actual_xamb[:,actual_t:actual_t+number_ctrl_pts_executed+1], actual_uamb[:,actual_t:actual_t+number_ctrl_pts_executed] = xamb_executed, uamb_executed
    for i in range(len(all_other_x_mpc)):
        all_other_x_executed[i], all_other_u_executed[i] = all_other_x_mpc[i][:,:number_ctrl_pts_executed+1], all_other_u_mpc[i][:,:number_ctrl_pts_executed]
        actual_xothers[i][:,actual_t:actual_t+number_ctrl_pts_executed+1], actual_uothers[i][:,actual_t:actual_t+number_ctrl_pts_executed] = all_other_x_executed[i], all_other_u_executed[i]
    actual_t += number_ctrl_pts_executed

    ### SAVE STATES AND PLOT
    file_name = folder + "data/"+'mpc_%02d'%(i_mpc)
    print(" MPC Done:  Rd %02d / %02d"%(i_mpc, n_rounds_mpc))
    if SAVE_FLAG:
        mpc.save_state(file_name, xamb_mpc, uamb_mpc, xamb_des_mpc, all_other_x_mpc, all_other_u_mpc, all_other_x_des_mpc)
        print("MPC Solution Save to...", file_name)
        # mpc.save_costs(file_name, bri_mpc)        

      
    mean_amb_v = np.mean(xamb_executed[4,:])
    im_dir = folder + '%02d/'%i_mpc
    os.makedirs(im_dir+"imgs/")    
    end_frame = actual_t+number_ctrl_pts_executed+1
    start_frame = max(0, end_frame - 20)
    cmplot.plot_cars(world, amb_MPC, actual_xamb[:,start_frame:end_frame], [x[:,start_frame:end_frame] for x in actual_xothers], im_dir, "ellipse", True, 0)                        
     
print("Solver Done!  Runtime: %.1d"%(time.time()-t_start_time))
######################## SAVE THE FINAL STATE OF THE VEHICLES
final_t = actual_t+number_ctrl_pts_executed+1
actual_xamb = actual_xamb[:,:actual_t+number_ctrl_pts_executed+1]
for i in range(len(actual_xothers)):
    actual_xothers[i] = actual_xothers[i][:,:actual_t+number_ctrl_pts_executed+1]

print("Plotting all")
cmplot.plot_cars(world, response_MPC, actual_xamb, actual_xothers, folder, None, None, False, False, 0)

file_name = folder + "data/"+'all%02d'%(i_mpc)
print("Saving Actual Positions to...  ", file_name)
mpc.save_state(file_name, actual_xamb, uamb_mpc, xamb_des_mpc, actual_xothers, all_other_u_mpc, all_other_x_des_mpc)







            # for k_warm in u_warm_profiles.keys():
            #     u_warm, x_warm, x_des_warm = ux_warm_profiles[k_warm]
            #     temp, current_cost, max_slack, bri, xamb, xamb_des, uamb = helper.solve_best_response(response_MPC, None, nonresponse_MPC_list, k_slack, k_CA, k_CA_power, world, wall_CA, N, T, response_x0, x0_amb, nonresponse_x0_list, slack, False, k_warm, u_warm, x_warm, x_des_warm, nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list, )                
                
            #     print("MPC_i: %d IBR_i: %d Veh: Amb k_warm %s"%(i_mpc, i_rounds_ibr, k_warm))        
            #     if current_cost >= np.infty:
            #         print("Failed:  Converged to Locally Infeasible Solution")
            #     else:
            #         if current_cost < min_response_cost:
            #             min_response_cost = current_cost
            #             xamb_ibr, xamb_des_ibr, uamb_ibr = xamb, xamb_des, uamb
            #             max_slack_ibr = max_slack
            #             min_bri_ibr = bri
            #             amb_solved_flag = True
            #             print("Converged:  Saved min cost response: k_warm %s  min_cost: %0.03f  max_slack %.03f" % (k_warm, min_response_cost, max_slack_ibr))
            #         else:
            #             print("Converged:  Cost (%0.03f) was higher than other warm (Current Min:  %0.03f)"%(current_cost, min_response_cost))
            #     if solve_number > 2:
            #         print("Debug:  Plotting Warm Start:  %s"%k_warm)
            #         for k in range(N+1):
            #             cmplot.plot_single_frame(world, response_MPC, x_warm, nonresponse_x_list, None, "Ellipse", parallelize=False, camera_speed = 0, plot_range = [k])                
            #             plt.show()
            #         if current_cost >= np.infty:
            #             x_plot = bri.opti.debug.value(bri.x_opt)
            #         else:
            #             x_plot = xamb
            #         print("Debug:  Plotting Solution/Debug")
            #         for k in range(N+1):
            #             cmplot.plot_single_frame(world, response_MPC, x_plot, nonresponse_x_list, None, "Ellipse", parallelize=False, camera_speed = 0, plot_range = [k])                
            #             plt.show()                     



                    # print("Ambulance Solution:  mpc_i %d  ibr_round %d"%(i_mpc, i_rounds_ibr))    
        # cmplot.plot_single_frame(world, min_bri_ibr.responseMPC, xamb_ibr, nonresponse_x_list, None, CIRCLES="Ellipse", parallelize=True, camera_speed = None, plot_range = range(N+1)[:int(N/2)], car_ids = None, xamb_desired=None, xothers_desired=None)        
        # plt.show()
        # cmplot.plot_single_frame(world, min_bri_ibr.responseMPC, xamb_ibr, nonresponse_x_list, None, CIRCLES="Ellipse", parallelize=True, camera_speed = None, plot_range = range(N+1)[int(N/2):], car_ids = None, xamb_desired=None, xothers_desired=None)        
        # plt.show()

        #               
# 
# 
# for k_warm in u_warm_profiles.keys():
#                     u_warm, x_warm, x_des_warm = ux_warm_profiles[k_warm]     
#                     temp_solved_flag, current_cost, max_slack, bri, x, x_des, u = helper.solve_best_response(response_MPC, amb_MPC, nonresponse_MPC_list, k_slack, k_CA, k_CA_power, world, wall_CA, N, T, response_x0, x0_amb, nonresponse_x0_list, slack, solve_amb, k_warm, u_warm, x_warm, x_des_warm, nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list, uamb_ibr, xamb_ibr, xamb_des_ibr, )
#                     print("MPC_i: %d IBR_i: %d Veh: %i k_warm %s"%(i_mpc, i_rounds_ibr, i, k_warm))        

#                             # print("  i_mpc %d n_round %d i %02d Cost %.02f Slack %.02f "%(i_mpc, i_rounds_ibr, i, bri.solution.value(bri.total_svo_cost), bri.solution.value(bri.slack_cost)))
#                             # print("  J_i %.03f,  J_j %.03f, Slack %.03f, CA  %.03f"%(bri.solution.value(bri.response_svo_cost), bri.solution.value(bri.other_svo_cost), bri.solution.value(bri.k_slack*bri.slack_cost), bri.solution.value(bri.k_CA*bri.collision_cost)))
#                             # print("  Dir:", subdir_name)
#                     if current_cost >= np.infty: #Infeasible solution
#                         print("Failed: Convered to Locally Infeasible Solution")
#                     elif current_cost < min_response_cost:
#                         all_other_u_ibr[i], all_other_x_ibr[i], all_other_x_des_ibr[i]  = u, x, x_des
#                         other_solved_flag[i] = True
#                         min_response_cost = current_cost
#                         min_response_warm = k_warm
#                         min_bri = bri
#                         max_slack_ibr = max_slack
#                         print("Converged:  Saved min cost response: k_warm %s  min_cost: %0.03f  max_slack %.03f" % (k_warm, min_response_cost, max_slack_ibr))
#                     else:
#                         print("Converged:  Cost (%0.03f) was higher than other warm (Current Min:  %0.03f)"%(current_cost, min_response_cost))

#                     if solve_number > 2:
#                         print("Debug:  Plotting Warm Start:  %s"%k_warm)
#                         for k in range(N+1):
#                             cmplot.plot_single_frame(world, response_MPC, x_warm, [xamb_ibr] + nonresponse_x_list, None, "Ellipse", parallelize=False, camera_speed = 0, plot_range = [k])                
#                             plt.show()
#                         if current_cost >= np.infty:
#                             x_plot = bri.opti.debug.value(bri.x_opt)
#                         else:
#                             x_plot = x
#                         print("Debug:  Plotting Solution/Debug")
#                         for k in range(N+1):
#                             cmplot.plot_single_frame(world, response_MPC, x_plot, [xamb_ibr] + nonresponse_x_list, None, "Ellipse", parallelize=False, camera_speed = 0, plot_range = [k])                
#                             plt.show() 
#                         print("Settings:  k_slack %.04f  k_CA %.04f  solve_amb? %d slack? %d"%(k_slack, k_CA, solve_amb, slack))
#                         if current_cost >= np.infty:
#                             print("Costs: Total Cost %.04f Vehicle-Only Cost:  %.04f Collision Cost %0.04f  Slack Cost %0.04f"%((current_cost, bri.opti.debug.value(bri.response_svo_cost), bri.opti.debug.value(bri.k_CA*bri.collision_cost), bri.opti.debug.value(bri.k_slack*bri.slack_cost))))                 
#                         else:
#                             print("Costs: Total Cost %.04f Vehicle-Only Cost:  %.04f Collision Cost %0.04f  Slack Cost %0.04f"%((current_cost, bri.solution.value(bri.response_svo_cost), bri.solution.value(bri.k_CA*bri.collision_cost), bri.solution.value(bri.k_slack*bri.slack_cost))))                 

    # plt.show()
    # plt.plot(xamb_mpc[4,:],'--')
    # plt.plot(xamb_mpc[4,:] * np.cos(xamb_mpc[2,:]))
    # plt.ylabel("Velocity / Vx (full mpc)")
    # plt.hlines(35*0.447,0,xamb_mpc.shape[1])
    # plt.show()
    # plt.plot(uamb_mpc[1,:],'o')
    # plt.hlines(amb_MPC.max_v_u,0,uamb_mpc.shape[1])
    # plt.hlines(amb_MPC.min_v_u,0,uamb_mpc.shape[1])
    # plt.ylabel("delta_u_v")
    # plt.show()

    # for cost in bri_mpc.car1_costs_list:
    #     print("%.04f"%bri_mpc.solution.value(cost))
    # print(bri_mpc.solution.value(bri_mpc.k_CA * bri_mpc.collision_cost), bri_mpc.solution.value(bri_mpc.collision_cost))
    # print(bri_mpc.solution.value(bri_mpc.k_slack * bri_mpc.slack_cost), bri_mpc.solution.value(bri_mpc.slack_cost))
        
    # plot_range = range(number_ctrl_pts_executed+1)
    
    # for k in range(xamb_executed.shape[1]):
    #     cmplot.plot_multiple_cars( k, bri_mpc.responseMPC, all_other_x_executed, xamb_executed, True, None, None, None, bri_mpc.world, 0)     
    #     plt.show() 
