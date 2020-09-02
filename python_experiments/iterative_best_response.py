import time, datetime, argparse, os, sys, pickle, psutil
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt
import casadi as cas
import copy as cp
PROJECT_PATHS = ['/home/nbuckman/Dropbox (MIT)/DRL/2020_01_cooperative_mpc/mpc-multiple-vehicles/', '/Users/noambuckman/mpc-multiple-vehicles/']
for p in PROJECT_PATHS:
    sys.path.append(p)
import src.traffic_world as tw
import src.multiagent_mpc as mpc
import src.car_plotting_multiple as cmplot
import src.solver_helper as helper

def nonresponse_slice(i, all_other_x0, all_other_MPC, all_other_u, all_other_x, all_other_x_des):
    ''' Returns all vehicles except i'''
    nonresponse_x0_list = all_other_x0[:i] + all_other_x0[i+1:]
    nonresponse_MPC_list = all_other_MPC[:i] + all_other_MPC[i+1:]
    nonresponse_u_list = all_other_u[:i] + all_other_u[i+1:]
    nonresponse_x_list = all_other_x[:i] + all_other_x[i+1:]
    nonresponse_xd_list = all_other_x_des[:i] + all_other_x_des[i+1:]
    
    return nonresponse_x0_list, nonresponse_MPC_list, nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list


##########################################################
svo_theta = np.pi/3.0
# random_seed = args.random_seed[0]
random_seed = 9
if random_seed > 0:
    np.random.seed(random_seed)
#######################################################################

parser = argparse.ArgumentParser(description='Run iterative best response with SVO')
parser.add_argument('--T', type=int, default=5)
parser.add_argument('--dt', type=float, default=0.2)
parser.add_argument('--n-rounds-mpc', type=int, default=300)
parser.add_argument('--percent-mpc-executed', type=float, default=0.1)

parser.add_argument('--plot-flag', action='store_true')
parser.add_argument('--save-flag', action='store_true') #Make the default to safe
parser.add_argument('--print-flag', action='store_true')

parser.add_argument('--n-other', type=int, default=17)
parser.add_argument('--n-rounds-ibr', type=int, default=3)
parser.add_argument('--n-processors', type=int, default=15)

parser.add_argument('--k-max-slack', type=float, default=0.01)
parser.add_argument('--k-solve-amb-max-ibr', type=int, default=2)
parser.add_argument('--k-max-solve-number', type=int, default=3)
parser.add_argument('--k_max_round_with_slack', type=int, default=np.infty)    


parser.add_argument('--k-slack-d', type=float, default=1000)
parser.add_argument('--k_CA-d', type=float, default=100)
parser.add_argument('--k-CA-power', type=float, default=4)
parser.add_argument('--wall-CA', action='store_false')

parser.add_argument('--plan_fake_ambulance', action='store_true')

args = parser.parse_args()
params = vars(args)

T = params['T']  # MPC Planning Horizon
dt = params['dt']
n_rounds_mpc = params['n_rounds_mpc']
percent_mpc_executed = params['percent_mpc_executed'] ## This is the percent of MPC that is executed
PLOT_FLAG, SAVE_FLAG, PRINT_FLAG = params['plot_flag'], params['save_flag'], params['print_flag']
# n_other = params['n_other']


N = int(T/dt) #Number of control intervals in MPC
number_ctrl_pts_executed =  int(np.floor(N*percent_mpc_executed))
params['N'] = N
params['number_ctrl_pts_executed'] = number_ctrl_pts_executed

print("number ctrl pts:  %d"%number_ctrl_pts_executed)


world = tw.TrafficWorld(2, 0, 999999)
    # large_world = tw.TrafficWorld(2, 0, 1000, 5.0)
#########################################################################
#TODO: change this to date/random string of letters
optional_suffix = str(params['n_other']) + "nograss"
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
    # (1, 60),
    # (0, 80),
    # (1, 85),
    # (1, 100),
    # (1, 123),
    # (0, 120),
    # (0, 145),
    # (0, 160),
    # (0, 200),
    # (1, 250),
    # (1, 300),
    # (0, 320),
    # (1, 400),
]
if len(position_list) != params['n_other']:
    raise Exception("number of vehicles don't match n_other, %d, %d"%(len(position_list), params['n_other']))

#TODO: fix this line
all_other_x0, all_other_MPC, amb_MPC, amb_x0 = helper.initialize_cars(params['n_other'], params['N'], params['dt'], world, svo_theta, True, False, False, position_list)

t_start_time = time.time()
actual_t = 0    

actual_xamb, actual_uamb = np.zeros((6, n_rounds_mpc*number_ctrl_pts_executed + 1)), np.zeros((2, n_rounds_mpc*number_ctrl_pts_executed)) 
actual_xothers = [np.zeros((6, n_rounds_mpc*number_ctrl_pts_executed + 1)) for i in range(params['n_other'])]
actual_uothers = [np.zeros((2, n_rounds_mpc*number_ctrl_pts_executed)) for i in range(params['n_other'])]    
xamb_executed, all_other_x_executed = None, [] #This gets updated after each round of MPC
uamb_mpc, all_other_u_mpc = None, []

if params['save_flag']:
    pickle.dump(all_other_MPC[0], open(folder + "data/"+"mpcother" + ".p",'wb'))
    pickle.dump(amb_MPC, open(folder + "data/"+"mpcamb" + ".p",'wb'))

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
    for i in range(params['n_other']):
        initial_distance = np.sqrt((amb_x0[0]-all_other_x0[i][0])**2 + (amb_x0[1]-all_other_x0[i][1])**2)
        if initial_distance <= np.infty: #RIGHT NOW THIS ISN'T REALLY WORKING
            vehicles_index_in_mpc += [i]
        else:
            vehicles_index_constant_v += [i]    

    for j in vehicles_index_constant_v:
        all_other_u_ibr[j] = np.zeros((2,N)) #constant v control inputs
        all_other_x_ibr[j], all_other_x_des_ibr[j] = all_other_MPC[j].forward_simulate_all(all_other_x0[j].reshape(6,1), all_other_u_ibr[j]) 
        
    ############# Generate (if needed) the control inputs of other vehicles        
    if i_mpc == 0:
        all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr = helper.pullover_guess(N, all_other_MPC, all_other_x0)  # This is a hack and should be explicit that it's lane change                   
    else:
        all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr = helper.extend_last_mpc_ctrl(all_other_u_mpc, number_ctrl_pts_executed, N, all_other_MPC, all_other_x0)  # This is a hack and should be explicit that it's lane change                   
    
    for i_rounds_ibr in range(params['n_rounds_ibr']):    
        ########## Solve the Response MPC ##########
        response_MPC, response_x0 = amb_MPC, amb_x0
        nonresponse_MPC_list, nonresponse_x0_list = all_other_MPC, all_other_x0
        nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list = all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr
        
        if params['plan_fake_ambulance']:
            fake_amb_i = helper.get_min_dist_i(amb_x0, all_other_x0, restrict_greater=True)
            nonresponse_x0_list, nonresponse_MPC_list, nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list = nonresponse_slice(fake_amb_i, all_other_x0, all_other_MPC, all_other_u_ibr, other_x_ibr, all_other_x_des_ibr)
            fake_amb_MPC, fake_amb_x0, fake_amb_u, fake_amb_x, fake_amb_xd = all_other_MPC[fake_amb_i], all_other_x0[fake_amb_i], all_other_u_ibr[fake_amb_i], all_other_x_ibr[fake_amb_i], all_other_x_des_ibr[fake_amb_i]    
        else:
            fake_amb_i = -1
            nonresponse_MPC_list, nonresponse_x0_list = all_other_MPC, all_other_x0
            nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list = all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr
            fake_amb_MPC, fake_amb_x0, fake_amb_u, fake_amb_x, fake_amb_xd = None, None, None, None, None
        
        ################# Generate the warm starts ###############################
        u_warm_profiles, ux_warm_profiles = mpc.generate_warm_u(N, response_MPC, response_x0)
        if i_mpc > 0: #TODO: What is going on here?
            u_warm_profiles["previous_mpc"] = np.concatenate((uamb_mpc[:, number_ctrl_pts_executed:], np.tile(uamb_mpc[:,-1:],(1, number_ctrl_pts_executed))), axis=1) ##    
            x_warm, x_des_warm = response_MPC.forward_simulate_all(response_x0.reshape(6,1), u_warm_profiles["previous_mpc"]) 
            ux_warm_profiles["previous_mpc"] = [u_warm_profiles["previous_mpc"], x_warm, x_des_warm]
        if i_rounds_ibr > 0: #TODO: What is going on here?
            u_warm_profiles["previous_ibr"] = uamb_ibr    
            x_warm, x_des_warm = response_MPC.forward_simulate_all(response_x0.reshape(6,1), u_warm_profiles["previous_ibr"]) 
            ux_warm_profiles["previous_ibr"] = [u_warm_profiles["previous_ibr"], x_warm, x_des_warm]            
            
        x_warm_profiles, x_ux_warm_profiles = mpc.generate_warm_x(response_MPC, world,  response_x0, np.median([x[4] for x in nonresponse_x0_list]))
        ux_warm_profiles.update(x_ux_warm_profiles) # combine into one
        ################# Solve the Ambulance Best Response ############################
        
        solver_params = {}
        solver_params['slack'] = True if i_rounds_ibr <= params['k_max_round_with_slack'] else False
        solver_params['solve_amb'] = False if fake_amb_i == -1 or i_rounds_ibr >= params['k_solve_amb_max_ibr'] else True

        solve_number, solve_again, max_slack_ibr, debug_flag = 0, True, np.infty, False
        while solve_again and solve_number < params['k_max_solve_number']:
            print("SOLVING AMBULANCE:  Attempt %d / %d"%(solve_number+1, params['k_max_solve_number'])) 
            solver_params['k_slack'] = params['k_slack_d'] * 10**solve_number
            solver_params['k_CA'] = params['k_CA_d'] * 2**solve_number
            solver_params['k_CA_power'] = params['k_CA_power']
            solver_params['wall_CA'] = params['wall_CA']
            if solve_number > 2:
                debug_flag = True
            if psutil.virtual_memory().percent >= 90.0:
                raise Exception("Virtual Memory is too high, exiting to save computer")
            solved, min_cost_ibr, max_slack_ibr, x_ibr, x_des_ibr, u_ibr, key_ibr, debug_list = helper.solve_warm_starts(ux_warm_profiles, 
                                                                                                                response_MPC, fake_amb_MPC, nonresponse_MPC_list, 
                                                                                                                response_x0, fake_amb_x0, nonresponse_x0_list, 
                                                                                                                world, solver_params, params, 
                                                                                                                nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list, 
                                                                                                                uamb=fake_amb_u, xamb=fake_amb_x, xamb_des=fake_amb_xd, 
                                                                                                                debug_flag=debug_flag)
            if max_slack_ibr <= params['k_max_slack']:
                xamb_ibr, xamb_des_ibr, uamb_ibr = x_ibr, x_des_ibr, u_ibr
                solve_again = False
            else:
                print("Max Slack is too large %.05f > thresh %.05f"%(max_slack_ibr, params['k_max_slack']))
                solve_again = True
                solve_number += 1
#             raise Exception("Test")
        if solve_again:
            raise Exception("Slack variable is too high or infeasible.  MaxS = %.05f > thresh %.05f"%(max_slack_ibr, params['k_max_slack']))
        print("Ambulance Plot Max Slack %0.04f"%max_slack_ibr)
        end_frame = actual_t+number_ctrl_pts_executed+1
        # cmplot.plot_cars(world, amb_MPC, xamb_ibr[:,:], [x[:,:] for x in all_other_x_ibr], None, "ellipse", False, 0)                        
        # plt.show()                                              
        ################# SOLVE BEST RESPONSE FOR THE OTHER VEHICLES ON THE ROAD ############################
        for i in vehicles_index_in_mpc:
            response_MPC, response_x0 = all_other_MPC[i], all_other_x0[i]
            #TODO:  Limit the nonresponse list?
            nonresponse_x0_list, nonresponse_MPC_list, nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list = nonresponse_slice(i, 
                                                                                                all_other_x0, all_other_MPC, all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr)

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
            
            params['k_solve_amb_min_distance'] = 50
            initial_distance_to_ambulance = np.sqrt((response_x0[0] - amb_x0[0])**2 + (response_x0[1] - amb_x0[1])**2)   

            solver_params = {}    

            solver_params['solve_amb'] = True if (i_rounds_ibr < params['k_solve_amb_max_ibr'] and 
                                  initial_distance_to_ambulance < params['k_solve_amb_min_distance'] and 
                                 response_x0[0]>amb_x0[0]-0) else False     
            solver_params['slack'] = True if i_rounds_ibr <= params['k_max_round_with_slack'] else False
            while solve_again and solve_number < params['k_max_solve_number']:
                print("SOLVING Agent %d:  Attempt %d / %d"%(i, solve_number+1, params['k_max_solve_number']))    
                solver_params['k_slack'] = params['k_slack_d'] * 10**solve_number
                solver_params['k_CA'] = params['k_CA_d'] * 2**solve_number    
                solver_params['k_CA_power'] = params['k_CA_power']
                solver_params['wall_CA'] = params['wall_CA']

                if solve_number > 2:
                    debug_flag = True
                if psutil.virtual_memory().percent >= 90.0:
                    raise Exception("Virtual Memory is too high, exiting to save computer")                
                solved, min_cost_ibr, max_slack_ibr, x_ibr, x_des_ibr, u_ibr, key_ibr, debug_list = helper.solve_warm_starts(ux_warm_profiles, 
                                                                                                        response_MPC, amb_MPC, nonresponse_MPC_list, 
                                                                                                        response_x0, amb_x0, nonresponse_x0_list, 
                                                                                                        world, solver_params, params, 
                                                                                                        nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list, 
                                                                                                        uamb_ibr, xamb_ibr, xamb_des_ibr, debug_flag)
                if max_slack_ibr <= params['k_max_slack']:
                    all_other_x_ibr[i], all_other_x_des_ibr[i], all_other_u_ibr[i] = x_ibr, x_des_ibr, u_ibr
                    solve_again = False
                else:
                    print("Max Slack is too large %.05f > thresh %.05f"%(max_slack_ibr, params['k_max_slack']))
                    solve_again = True
                    solve_number += 1
                    if (solve_number == params['k_max_solve_number']) and solver_params['solve_amb']:
                        print("Re-solving without ambulance")
                        solve_number = 0   
                        solver_params['solve_amb'] = False
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
    ## CLEANUP:  x_mpc, x_actual, x_ibr, x_executed
    xamb_mpc, xamb_des_mpc, uamb_mpc = xamb_ibr, xamb_des_ibr, uamb_ibr
    all_other_u_mpc, all_other_x_mpc, all_other_x_des_mpc = all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr
    bri_mpc = None

    all_other_x_executed = [np.zeros(shape=(6,number_ctrl_pts_executed+1)) for i in range(params['n_other'])]
    all_other_u_executed = [np.zeros(shape=(2,number_ctrl_pts_executed)) for i in range(params['n_other'])]

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



