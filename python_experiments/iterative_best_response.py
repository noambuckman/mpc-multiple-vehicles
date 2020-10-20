import time, datetime, argparse, os, sys, pickle, psutil, logging
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

import json
import string, random

import io
from contextlib import redirect_stdout

def nonresponse_subset(list_of_veh_idxs, all_other_x0, all_other_MPC, all_other_u, all_other_x, all_other_x_des):
    ''' Returns subsets of previous lists '''
    nonresponse_x0_list = [all_other_x0[i] for i in list_of_veh_idxs] 
    nonresponse_MPC_list = [all_other_MPC[i] for i in list_of_veh_idxs] 
    nonresponse_u_list = [all_other_u[i] for i in list_of_veh_idxs] 
    nonresponse_x_list = [all_other_x[i] for i in list_of_veh_idxs] 
    nonresponse_xd_list = [all_other_x_des[i] for i in list_of_veh_idxs]

    return nonresponse_x0_list, nonresponse_MPC_list, nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list

def nonresponse_slice(i, all_other_x0, all_other_MPC, all_other_u, all_other_x, all_other_x_des):
    ''' Returns all vehicles except i'''
    nonresponse_x0_list = all_other_x0[:i] + all_other_x0[i+1:]
    nonresponse_MPC_list = all_other_MPC[:i] + all_other_MPC[i+1:]
    nonresponse_u_list = all_other_u[:i] + all_other_u[i+1:]
    nonresponse_x_list = all_other_x[:i] + all_other_x[i+1:]
    nonresponse_xd_list = all_other_x_des[:i] + all_other_x_des[i+1:]
    
    return nonresponse_x0_list, nonresponse_MPC_list, nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list

def warm_profiles_subset(n_warm_keys, ux_warm_profiles):
    '''choose randomly n_warm_keys keys from ux_warm_profiles and return the subset'''
    
    warm_keys = list(ux_warm_profiles.keys())
    random.shuffle(warm_keys)
    warm_subset_keys = warm_keys[:n_warm_keys]
    ux_warm_profiles_subset = dict((k, ux_warm_profiles[k]) for k in warm_subset_keys)
    
    return ux_warm_profiles_subset
##########################################################
svo_theta = np.pi/3.0
# svo_theta = 0.0
# random_seed = args.random_seed[0]
# random_seed = 9
# if random_seed > 0:
#     np.random.seed(random_seed)
#######################################################################
default_position_list = [
    (0, 20),
    (0, 35),
    (2, 35),
    (2, 60),
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
    (0, 200),
    (1, 250),
    (1, 300),
    (0, 320),
    (1, 400),
]



parser = argparse.ArgumentParser(description='Run iterative best response with SVO')
parser.add_argument('--load-log-dir',type=str, default=None, help="Load log")
parser.add_argument('--mpc-start-iteration', type=int, default=0, help="At which mpc iteration should the simulation start")
parser.add_argument('--save-solver-input', action='store_true')

parser.add_argument('--T', type=int, default=5)
parser.add_argument('--dt', type=float, default=0.2)
parser.add_argument('--p-exec', type=float, default=0.4, help="Percent of MPC points executed")
parser.add_argument('--car-density', type=int, default=5000, help="Car density across all lanes, cars per hour")
parser.add_argument('--plot-flag', action='store_true')
# parser.add_argument('--save-flag', action='store_true') #Make the default to safe
parser.add_argument('--print-flag', action='store_true')

parser.add_argument('--n-other', type=int, default=10, help="Number of ado vehicles")
parser.add_argument('--n-mpc', type=int, default=300)
parser.add_argument('--n-ibr', type=int, default=3, help="Number of rounds of iterative best response before excuting mpc")
parser.add_argument('--n-processors', type=int, default=15, help="Number of processors used when solving a single mpc")
parser.add_argument('--n-lanes', type=int, default=2, help="Number of lanes in the right direction")
parser.add_argument('--n-cntrld', type=int, default=0, help="How many cars does the response control in planning")

parser.add_argument('--rnds-shrd-cntrl', type=int, default=2)
parser.add_argument('--k-max-slack', type=float, default=0.01, help="Maximum allowed collision slack/overlap between vehicles")
parser.add_argument('--k-solve-amb-max-ibr', type=int, default=2, help="Max number iterations where ado solves for ambulance controls, afterwards only ado")
parser.add_argument('--k-max-solve-number', type=int, default=3, help = "Max re-solve attempts of the mpc for an individual vehicle")
parser.add_argument('--k-max-round-with-slack', type=int, default=np.infty, help = "Max rounds of ibr with slack variable used")    


parser.add_argument('--k-slack-d', type=float, default=1000)
parser.add_argument('--k-CA-d', type=float, default=0.05)
parser.add_argument('--k-CA-power', type=float, default=1.0)
parser.add_argument('--wall-CA', action='store_true')


parser.add_argument('--default-n-warm-starts', type=int, default=15)
parser.add_argument('--random-svo', type=int, default = 1, help="Randomly assign svo to other vehicles")
parser.add_argument('--plan-fake-ambulance', action='store_true')
parser.add_argument('--default-positions', action='store_true')
parser.add_argument('--save-ibr', type=int, default=1, help="Save the IBR control inputs, 1=True, 0=False")
parser.add_argument('--print-level', type=int, default=0)
args = parser.parse_args()
params = vars(args)


if args.load_log_dir is None:
    params["start_time_string"] = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    alpha_num = string.ascii_lowercase[:8] + string.digits
    subdir_name = ''.join(random.choice(alpha_num) for j in range(4)) + '-' + ''.join(random.choice(alpha_num) for j in range(4)) + "-" + params["start_time_string"]
    folder = "/home/nbuckman/mpc_results/" + subdir_name + "/"
    for f in [folder+"imgs/", folder+"data/", folder+"vids/", folder+"plots/"]:
        os.makedirs(f, exist_ok = True)
    i_mpc_start = 0

    params['N'] = max(1, int(params["T"]/params["dt"]))
    params['number_ctrl_pts_executed'] = max(1, int(np.floor(params['N']*params['p_exec'])))
    params["svo_theta"] = svo_theta #TODO: make this an input arg

    ### THIS NEEDS TO BE ADDED TO LOADING
    world = tw.TrafficWorld(params["n_lanes"], 0, 999999)
    if params['default_positions']:
        position_list = default_position_list[:params['n_other']]
        amb_MPC, amb_x0, all_other_MPC, all_other_x0  = helper.initialize_cars(params['n_other'], params['N'], params['dt'], world, svo_theta, True, False, False, position_list)
    else:
        # This should be replaced with random placement
        MAX_VELOCITY = 25 * 0.447 # m/s
        VEHICLE_LENGTH = 4.5 #m
        time_duration_s = (params["n_other"] * 3600.0 / params["car_density"] ) * 10 # amount of time to generate traffic
        initial_vehicle_positions = helper.poission_positions(params["car_density"], int(time_duration_s), params["n_lanes"] , MAX_VELOCITY, VEHICLE_LENGTH)
        position_list = initial_vehicle_positions[:params["n_other"]]
        if params['random_svo'] == 1:
            list_of_svo = [np.random.choice([0, np.pi/4.0, np.pi/2.01]) for i in range(params["n_other"])]
        else:
            list_of_svo = [params["svo_theta"] for i in range(params["n_other"])]
        amb_MPC, amb_x0, all_other_MPC, all_other_x0 = helper.initialize_cars_from_positions(params["N"], params["dt"], world,  
                                                                    True, 
                                                                    position_list, list_of_svo)    
    
    for i in range(len(all_other_MPC)):
        pickle.dump(all_other_MPC[i], open(folder + "data/mpcother%02d.p"%i,'wb'))
    pickle.dump(amb_MPC, open(folder + "data/mpcamb.p",'wb'))
    pickle.dump(world, open(folder + "data/world.p",'wb'))
    params['save_flag'] = True

    print(folder)
else:
    print("Preloading settings from log %s"%args.load_log_dir)
    folder = args.load_log_dir
    with open(args.load_log_dir + "params.json",'rb') as fp:
        params = json.load(fp)
    i_mpc_start = args.mpc_start_iteration

    amb_MPC = pickle.load(open(folder + "data/mpcamb.p",'rb'))
    all_other_MPC = [pickle.load(open(folder + "data/mpcother%02d.p"%i,'rb')) for i in range(params["n_other"])]
    world = pickle.load(open(folder + "data/world.p",'rb'))
    params["pid"] = os.getpid()

T = params['T']  # MPC Planning Horizon
dt = params['dt']
PLOT_FLAG, SAVE_FLAG, PRINT_FLAG = params['plot_flag'], params['save_flag'], params['print_flag']
N = params['N'] #Number of control intervals in MPC
number_ctrl_pts_executed =  params['number_ctrl_pts_executed']

params["pid"] = os.getpid()

if params['n_other'] != len(all_other_MPC):
    raise Exception("n other larger than default position list")

with open(folder + 'params.json', 'w') as fp:
    json.dump(params, fp, indent=2)


t_start_time = time.time()
actual_t = 0    
xamb_actual, uamb_actual = np.zeros((6, params['n_mpc']*number_ctrl_pts_executed + 1)), np.zeros((2, params['n_mpc']*number_ctrl_pts_executed)) 
xothers_actual = [np.zeros((6, params['n_mpc']*number_ctrl_pts_executed + 1)) for i in range(params['n_other'])]
uothers_actual = [np.zeros((2, params['n_mpc']*number_ctrl_pts_executed)) for i in range(params['n_other'])]    



xamb_executed, all_other_x_executed, uamb_mpc, all_other_u_mpc = None, [], None, [] #This gets updated after each round of MPC


f = open(folder + 'out.txt',"w")
# f = io.StringIO()
sys.stdout = f

for i_mpc in range(i_mpc_start, params['n_mpc']):
    min_slack = np.infty

    ###### Update the initial conditions for all vehicles
    if args.load_log_dir: #TODO: Should be moved above
        previous_mpc_file = folder + 'data/mpc_%02d'%(i_mpc_start - 1)
        xamb_executed, uamb_executed, _, all_other_x_executed, all_other_u_executed, _,  = mpc.load_state(previous_mpc_file, params['n_other'])
        all_other_u_mpc = all_other_u_executed
        uamb_mpc = uamb_executed
        print("Loaded initial positions from %s"%(previous_mpc_file))
        previous_all_file = folder + 'data/all_%02d'%(i_mpc_start -1)
        xamb_actual_prev, uamb_actual_prev, _, xothers_actual_prev, uothers_actual_prev, _ = mpc.load_state(previous_all_file, params['n_other'], ignore_des=True)
        t_end = xamb_actual_prev.shape[1]
        xamb_actual[:, :t_end] = xamb_actual_prev[:, :t_end]
        uamb_actual[:, :t_end] = uamb_actual_prev[:, :t_end]
        for i in range(len(xothers_actual_prev)):
            xothers_actual[i][:, :t_end] = xothers_actual_prev[i][:, :t_end]
            uothers_actual[i][:, :t_end] = uothers_actual_prev[i][:, :t_end]
        actual_t = i_mpc * number_ctrl_pts_executed
        
    if i_mpc > 0:
        amb_x0_g = cp.deepcopy(xamb_executed[:, number_ctrl_pts_executed])
        all_other_x0_g = [cp.deepcopy(all_other_x_executed[i][:, number_ctrl_pts_executed]) for i in range(len(all_other_x_executed))]
        ## Amb_x0 is localized in renference to ambulance
        amb_x0 = cp.deepcopy(amb_x0_g)
        amb_x0[0] -= amb_x0_g[0]
        amb_x0[-1] = 0
        all_other_x0 = cp.deepcopy(all_other_x0_g)
        for j in range(len(all_other_x0_g)):
            all_other_x0[j][0] -= amb_x0_g[0]
            all_other_x0[j][-1] = 0
    else:
        amb_x0_g = cp.deepcopy(amb_x0) # for use later on
    
    ############ Select which vehicles will be solved in IBR
    vehicles_index_best_responders = [j for j in range(params['n_other']) if (-10*amb_MPC.L <= all_other_x0[j][0] - amb_x0[0] <= 10*amb_MPC.L)]    
    vehicles_index_constant_v = [j for j in range(params['n_other']) if j not in vehicles_index_best_responders]

    ############# Generate (if needed) the control inputs of other vehicles        
    if i_mpc == 0:
        # all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr = helper.pullover_guess(N, all_other_MPC, all_other_x0)  # This is a hack and should be explicit that it's lane change     
        previous_all_other_u_mpc = [np.zeros((2,params["N"])) for i in range(len(all_other_MPC))]
        all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr = helper.extend_last_mpc_and_follow(previous_all_other_u_mpc, params["N"]-1, N, all_other_MPC, all_other_x0, params, world)  
    else:
        # all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr = helper.extend_last_mpc_ctrl(all_other_u_mpc, number_ctrl_pts_executed, N, all_other_MPC, all_other_x0)  
        all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr = helper.extend_last_mpc_and_follow(all_other_u_mpc, number_ctrl_pts_executed, N, all_other_MPC, all_other_x0, params, world)  
    
    # for j in range(len(all_other_MPC)):
    #     if j in vehicles_index_constant_v:
    #         # Solve but constrain u_v = 0, allow for small changes in steering
    #         response_MPC = all_other_MPC[j]
    #         response_x0 = all_other_x0[j]
    #         all_other_u_ibr[j], all_other_x_ibr[j], all_other_x_des_ibr[j] = mpc.centerline_following(params["N"], response_MPC, response_x0)

    for i_rounds_ibr in range(params['n_ibr']):    
        print("MPC %d, IBR %d / %d"%(i_mpc, i_rounds_ibr, params['n_ibr'] - 1))
        ########## Solve the Response MPC ##########
        response_MPC, response_x0 = amb_MPC, amb_x0

        # Select which vehicles should be included in the ambulance's MPC (3 cars ahead and 2 car behind)
        veh_idxs_in_amb_mpc = [i for i in range(len(all_other_x0)) if (-20*amb_MPC.L <= all_other_x0[i][0] - amb_x0[0] <= 20*amb_MPC.L)]
        
        fake_amb_i = -1
        fake_amb_MPC, fake_amb_x0, fake_amb_u, fake_amb_x, fake_amb_xd = None, None, None, None, None
        cntrld_vehicles, cntrld_x0, cntrld_u, cntrld_x, cntrld_xd = [], [], [], [], []
        if params['plan_fake_ambulance']:
            fake_amb_i = helper.get_min_dist_i(amb_x0, all_other_x0, restrict_greater=True)
            fake_amb_MPC, fake_amb_x0, fake_amb_u, fake_amb_x, fake_amb_xd = all_other_MPC[fake_amb_i], all_other_x0[fake_amb_i], all_other_u_ibr[fake_amb_i], all_other_x_ibr[fake_amb_i], all_other_x_des_ibr[fake_amb_i]    
            veh_idxs_in_amb_mpc = [i for i in veh_idxs_in_amb_mpc if i != fake_amb_i]

            cntrld_vehicles = [fake_amb_MPC]
            cntrld_x0 = [fake_amb_x0]
            cntrld_u = [fake_amb_u]
            cntrld_x = [fake_amb_x]
            cntrld_xd = [fake_amb_xd]
        
        nonresponse_x0_list, nonresponse_MPC_list, nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list = nonresponse_subset(veh_idxs_in_amb_mpc, 
                                                                                all_other_x0, all_other_MPC, all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr)

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
        
        if len(nonresponse_x0_list) > 0:
            warm_velocity = np.median([x[4] for x in nonresponse_x0_list])
        else:
            warm_velocity = response_x0[4]
        x_warm_profiles, x_ux_warm_profiles = mpc.generate_warm_x(response_MPC, world,  response_x0, warm_velocity)
        ux_warm_profiles.update(x_ux_warm_profiles) # combine into one
        ################# Solve the Ambulance Best Response ############################
        
        solver_params = {}
        solver_params['slack'] = True if i_rounds_ibr <= params['k_max_round_with_slack'] else False
        solver_params['solve_amb'] = False if len(cntrld_vehicles) == 0 or i_rounds_ibr >= params['k_solve_amb_max_ibr'] else True
        solver_params['n_warm_starts'] = params['default_n_warm_starts']
        solve_number, solve_again, max_slack_ibr, debug_flag = 0, True, np.infty, False
        print("...Amb Solver:")
        ipopt_params = {'print_level': params["print_level"]}
        while solve_again and solve_number < params['k_max_solve_number']:
            # print("...Attempt %d / %d"%(solve_number, params['k_max_solve_number'] - 1)) 
            solver_params['k_slack'] = params['k_slack_d'] * 10**solve_number
            # solver_params['k_CA'] = params['k_CA_d'] * 2**solve_number
            solver_params['k_CA'] = params['k_CA_d']
            solver_params['k_CA_power'] = params['k_CA_power']
            solver_params['wall_CA'] = params['wall_CA']
            if solve_number > 2:
                debug_flag = True

            solver_params['n_warm_starts'] = solver_params['n_warm_starts'] + 5 * solve_number
            ux_warm_profiles_subset = warm_profiles_subset(solver_params['n_warm_starts'], ux_warm_profiles)
            if psutil.virtual_memory().percent >= 90.0:
                raise Exception("Virtual Memory is too high, exiting to save computer")
            start_ipopt_time = time.time()
            with redirect_stdout(f):
                if args.save_solver_input:
                    with open(folder + 'data/inputs_amb_mpc_%d_ibr_%d_s_%d.p'%(i_mpc, i_rounds_ibr, solve_number), 'wb') as fp:
                        list_of_inputs = [ux_warm_profiles_subset, response_MPC, cntrld_vehicles, nonresponse_MPC_list, response_x0, cntrld_x0, nonresponse_x0_list, world, solver_params, params, nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list, cntrld_u, cntrld_x, cntrld_xd, debug_flag]
                        pickle.dump(list_of_inputs, fp)
                print("....# Vehicles in Amb's Best Response: %d "%(len(nonresponse_MPC_list)))
                solved, min_cost_ibr, max_slack_ibr, x_ibr, x_des_ibr, u_ibr, key_ibr, debug_list = helper.solve_warm_starts(ux_warm_profiles_subset, 
                                                                                                                    response_MPC, cntrld_vehicles, nonresponse_MPC_list, 
                                                                                                                    response_x0, cntrld_x0, nonresponse_x0_list, 
                                                                                                                    world, solver_params, params, ipopt_params,
                                                                                                                    nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list, 
                                                                                                                    cntrld_u, cntrld_x, cntrld_xd, 
                                                                                                                    debug_flag=debug_flag)
            end_ipopt_time = time.time()
            if max_slack_ibr <= params['k_max_slack']:
                print("......Solver converged.  Solver time: %0.1f s"%(end_ipopt_time - start_ipopt_time))            
                xamb_ibr, xamb_des_ibr, uamb_ibr = x_ibr, x_des_ibr, u_ibr
                solve_again = False                    
            else:
                # print("ipopt solved in %0.1f s"%(end_ipopt_time - start_ipopt_time))            
                print("......Re-solve %d/%d:  Slack too large: %.05f > Max Threshold (%.05f).  Solver time: %0.1f s"%(solve_number+1, params['k_max_solve_number'], max_slack_ibr, params['k_max_slack'], end_ipopt_time - start_ipopt_time))
                solve_again = True
                solve_number += 1
#             raise Exception("Test")
        if solve_again:
            raise Exception("Reached maximum number of re-solves @ Veh %s MPC Rd %d IBR %d.   Max Slack = %.05f > thresh %.05f"%("Amb", i_mpc, i_rounds_ibr, 
                                                                                                                                max_slack_ibr, params['k_max_slack']))
        if params["save_ibr"] == 1:
            file_name = folder + "data/"+'ibr_m%03di%03damb'%(i_mpc, i_rounds_ibr)
            xamb_ibr_g = cp.deepcopy(xamb_ibr)
            xamb_ibr_g[0,:] += amb_x0_g[0]
            all_other_x_ibr_g = [x for x in cp.deepcopy(all_other_x_ibr)]
            for j in range(len(all_other_x_ibr_g)):
                all_other_x_ibr_g[j][0,:] += amb_x0_g[0]
            mpc.save_state(file_name, xamb_ibr_g, uamb_ibr, xamb_des_ibr, all_other_x_ibr_g, all_other_u_ibr, all_other_x_des_ibr)       
        # out = f.getvalue()                            
        ################# SOLVE BEST RESPONSE FOR THE OTHER VEHICLES ON THE ROAD ############################
        for response_i in vehicles_index_best_responders:
            response_MPC, response_x0 = all_other_MPC[response_i], all_other_x0[response_i]

            # Select which vehicles should be included in the ambulance's MPC (3 cars ahead and 2 car behind)
            veh_idxs_in_mpc = [j for j in range(len(all_other_x0)) if j != response_i and (-20*response_MPC.L <= (all_other_x0[j][0] - response_x0[0]) <= 20*response_MPC.L)]
            cntrld_x0, cntrld_vehicles, cntrld_u, cntrld_x, cntrld_xd = [], [], [], [], []
            if params["n_cntrld"] > 0 and i_rounds_ibr < (params["rnds_shrd_cntrl"]): # For now ambulance is always included
                delta_x = [response_x0[0]-x[0] for x in all_other_x0] 
                sorted_i = [i for i in np.argsort(delta_x) if (delta_x[i] > 0 and i in veh_idxs_in_mpc and i in vehicles_index_best_responders)] #This necessary but could limit fringe best response interactions with outside best response
                cntrld_i = sorted_i[:params["n_cntrld"]-1]
                if len(cntrld_i) > 0:
                    cntrld_x0, cntrld_vehicles, cntrld_u, cntrld_x, cntrld_xd = nonresponse_subset(cntrld_i, all_other_x0, all_other_MPC, all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr)
                cntrld_x0 += [amb_x0]
                cntrld_vehicles += [amb_MPC]
                cntrld_x += [xamb_ibr]
                cntrld_u += [uamb_ibr]
                cntrld_xd += [xamb_des_ibr]
                veh_idxs_in_mpc = [idx for idx in veh_idxs_in_mpc if idx not in cntrld_i]
            nonresponse_x0_list, nonresponse_MPC_list, nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list = nonresponse_subset(veh_idxs_in_mpc, 
                                                                                                all_other_x0, all_other_MPC, all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr)
            if len(cntrld_vehicles) == 0:
                nonresponse_x0_list += [amb_x0]
                nonresponse_MPC_list += [amb_MPC]
                nonresponse_u_list += [uamb_ibr]
                nonresponse_x_list += [xamb_ibr]
                nonresponse_xd_list += [xamb_des_ibr]
            ################# Generate the warm starts ###############################
            # Generate control specified warm starts #TODO: Clean this code so it's fewer lines of code
            u_warm_profiles, ux_warm_profiles = mpc.generate_warm_u(N, response_MPC, response_x0)            
            if i_mpc > 0:
                u_warm_profiles["previous_mpc"] = np.concatenate((all_other_u_mpc[response_i][:, number_ctrl_pts_executed:], np.tile(all_other_u_mpc[response_i][:,-1:],(1, number_ctrl_pts_executed))),axis=1) ##    
                x_warm, x_des_warm = response_MPC.forward_simulate_all(response_x0.reshape(6,1), u_warm_profiles["previous_mpc"])
                ux_warm_profiles["previous_mpc"] = [u_warm_profiles["previous_mpc"], x_warm, x_des_warm]            
            if i_rounds_ibr > 0:
                u_warm_profiles["previous_ibr"] = all_other_u_ibr[response_i]  
                x_warm, x_des_warm = response_MPC.forward_simulate_all(response_x0.reshape(6,1), u_warm_profiles["previous_ibr"])
                ux_warm_profiles["previous_ibr"] = [u_warm_profiles["previous_ibr"], x_warm, x_des_warm]
            # Generate spatial warm starts
            warm_velocity = np.median([x[4] for x in nonresponse_x0_list] + [response_x0[4]])
            x_warm_profiles, x_ux_warm_profiles = mpc.generate_warm_x(response_MPC, world,  response_x0, warm_velocity)
            ux_warm_profiles.update(x_ux_warm_profiles) # combine into one
            #######


            solve_again, solve_number, max_slack_ibr, debug_flag, = True, 0, np.infty, False,     
            params['k_solve_amb_min_distance'] = 50
            initial_distance_to_ambulance = np.sqrt((response_x0[0] - amb_x0[0])**2 + (response_x0[1] - amb_x0[1])**2)   

            solver_params = {}    

            solver_params['solve_amb'] = True if (i_rounds_ibr < params['k_solve_amb_max_ibr'] and 
                                  initial_distance_to_ambulance < params['k_solve_amb_min_distance'] and 
                                 response_x0[0]>amb_x0[0]-0) else False     
            solver_params['slack'] = True if i_rounds_ibr <= params['k_max_round_with_slack'] else False
            solver_params['n_warm_starts'] = params['default_n_warm_starts']
            if (response_i not in veh_idxs_in_amb_mpc) and (response_i != fake_amb_i):
                solver_params['constant_v'] = True
            print("...Veh %02d Solver:"%response_i)
            while solve_again and solve_number < params['k_max_solve_number']:
                # print("SOLVING Agent %d:  Attempt %d / %d"%(response_i, solve_number+1, params['k_max_solve_number']))    
                solver_params['k_slack'] = params['k_slack_d'] * 10**solve_number
                # solver_params['k_CA'] = params['k_CA_d'] * 2**solve_number    
                solver_params['k_CA'] = params['k_CA_d']
                solver_params['k_CA_power'] = params['k_CA_power']
                solver_params['wall_CA'] = params['wall_CA']
                solver_params['n_warm_starts'] = solver_params['n_warm_starts'] + 5 * solve_number
                if solve_number > 2:
                    debug_flag = True
                if psutil.virtual_memory().percent >= 90.0:
                    raise Exception("Virtual Memory is too high, exiting to save computer")                

                ux_warm_profiles_subset = warm_profiles_subset(solver_params['n_warm_starts'], ux_warm_profiles)

                start_ipopt_time = time.time()

                ## TODO:  Also solve amb in the nonresponse MPC list
                with redirect_stdout(f):
                    
                    if args.save_solver_input:
                        with open(folder + 'data/inputs_v%02d_mpc_%03d_ibr_%01d_s_%01d.p'%(response_i, i_mpc, i_rounds_ibr, solve_number), 'wb') as fp:
                            #TODO: Add ipopt_params, make this non-repetative
                            list_of_inputs = [ux_warm_profiles_subset, response_MPC, cntrld_vehicles, nonresponse_MPC_list,response_x0, cntrld_x0, nonresponse_x0_list, world, solver_params, params, nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list, cntrld_u, cntrld_x, cntrld_xd, debug_flag]
                            pickle.dump(list_of_inputs, fp)
                    print("....# Cntrld Vehicles: %d # Non-Response: %d "%(len(cntrld_vehicles), len(nonresponse_MPC_list)))
                    solved, min_cost_ibr, max_slack_ibr, x_ibr, x_des_ibr, u_ibr, key_ibr, debug_list = helper.solve_warm_starts(ux_warm_profiles_subset, 
                                                                                                            response_MPC, cntrld_vehicles, nonresponse_MPC_list, 
                                                                                                            response_x0, cntrld_x0, nonresponse_x0_list, 
                                                                                                            world, solver_params, params, ipopt_params,
                                                                                                            nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list, 
                                                                                                            cntrld_u, cntrld_x, cntrld_xd, debug_flag)
                end_ipopt_time = time.time()
                if max_slack_ibr <= params['k_max_slack']:
                    all_other_x_ibr[response_i], all_other_x_des_ibr[response_i], all_other_u_ibr[response_i] = x_ibr, x_des_ibr, u_ibr
                    solve_again = False
                    print("......Solved.  veh: %02d | mpc: %d | ibr: %d | solver time: %0.1f s"%(response_i, i_mpc, i_rounds_ibr, end_ipopt_time - start_ipopt_time))                  
                else:
                    print("......Re-solve. veh: %02d | mpc: %d | ibr: %d | Slack %.05f > thresh %.05f  | solver time: %0.1f"%(response_i, i_mpc, i_rounds_ibr, max_slack_ibr, params['k_max_slack'], end_ipopt_time - start_ipopt_time))
                    solve_again = True
                    solve_number += 1
                    if (solve_number == params['k_max_solve_number']) and solver_params['solve_amb']:
                        print("Re-solving without ambulance")
                        solve_number = 0   
                        solver_params['solve_amb'] = False
            if solve_again:
                raise Exception("Reached maximum number of re-solves @ Veh %02d MPC Rd %d IBR %d.   Max Slack = %.05f > thresh %.05f"%(response_i, i_mpc, i_rounds_ibr, 
                                                                                                                           max_slack_ibr, params['k_max_slack']))
            if params["save_ibr"] == 1:
                file_name = folder + "data/"+'ibr_m%03di%03da%03d'%(i_mpc, i_rounds_ibr, response_i)
                mpc.save_state(file_name, xamb_ibr, uamb_ibr, xamb_des_ibr, all_other_x_ibr, all_other_u_ibr, all_other_x_des_ibr)                                                                                                                           
        if params["save_ibr"] == 1:
            file_name = folder + "data/"+'ibr_m%03di%03d'%(i_mpc, i_rounds_ibr)
            xamb_ibr_g = cp.deepcopy(xamb_ibr)
            xamb_ibr_g[0,:] += amb_x0_g[0]
            all_other_x_ibr_g = [x for x in cp.deepcopy(all_other_x_ibr)]
            for j in range(len(all_other_x_ibr_g)):
                all_other_x_ibr_g[j][0,:] += amb_x0_g[0]
            mpc.save_state(file_name, xamb_ibr_g, uamb_ibr, xamb_des_ibr, all_other_x_ibr_g, all_other_u_ibr, all_other_x_des_ibr)

    ################ SAVE THE BEST RESPONSE SOLUTION FOR THE CURRENT PLANNING HORIZONG/MPC ITERATION ###########################
    ## CLEANUP:  x_mpc, x_actual, x_ibr, x_executed
    xamb_mpc, xamb_des_mpc, uamb_mpc = xamb_ibr_g, xamb_des_ibr, uamb_ibr
    all_other_u_mpc, all_other_x_mpc, all_other_x_des_mpc = all_other_u_ibr, all_other_x_ibr_g, all_other_x_des_ibr
    bri_mpc = None

    all_other_x_executed = [np.zeros(shape=(6, number_ctrl_pts_executed+1)) for i in range(params['n_other'])]
    all_other_u_executed = [np.zeros(shape=(2, number_ctrl_pts_executed)) for i in range(params['n_other'])]

    ### SAVE EXECUTED MPC SOLUTION TO HISTORY
    xamb_executed, uamb_executed = xamb_mpc[:,:number_ctrl_pts_executed+1], uamb_mpc[:,:number_ctrl_pts_executed]
    xamb_actual[:, actual_t:actual_t+number_ctrl_pts_executed+1], uamb_actual[:, actual_t:actual_t+number_ctrl_pts_executed] = xamb_executed, uamb_executed
    for i in range(len(all_other_x_mpc)):
        all_other_x_executed[i], all_other_u_executed[i] = all_other_x_mpc[i][:,:number_ctrl_pts_executed+1], all_other_u_mpc[i][:,:number_ctrl_pts_executed]
        xothers_actual[i][:,actual_t:actual_t+number_ctrl_pts_executed+1], uothers_actual[i][:,actual_t:actual_t+number_ctrl_pts_executed] = all_other_x_executed[i], all_other_u_executed[i]

    ### SAVE STATES AND PLOT
    file_name = folder + "data/"+'mpc_%02d'%(i_mpc)
    if SAVE_FLAG:
        mpc.save_state(file_name, xamb_mpc, uamb_mpc, xamb_des_mpc, all_other_x_mpc, all_other_u_mpc, all_other_x_des_mpc)
        print("Saving MPC Rd %02d / %02d to ... %s" % (i_mpc, params['n_mpc']-1, file_name))
        # mpc.save_costs(file_name, bri_mpc) 
        file_name = folder + "data/"+'all_%02d'%(i_mpc)        
        mpc.save_state(file_name, xamb_actual, uamb_actual, None, xothers_actual, uothers_actual, None, end_t = actual_t+number_ctrl_pts_executed+1)
      
    mean_amb_v = np.mean(xamb_executed[4,:])
    im_dir = folder + '%02d/'%i_mpc
    os.makedirs(im_dir+"imgs/", exist_ok=True)    
    end_frame = actual_t + number_ctrl_pts_executed + 1
    start_frame = max(0, end_frame - 20)
    # cmplot.plot_cars(world, amb_MPC, xamb_actual[:,start_frame:end_frame], [x[:,start_frame:end_frame] for x in xothers_actual], im_dir, "ellipse", True, 0)                        
    # cmplot.concat_imgs(im_dir)
    actual_t += number_ctrl_pts_executed
    
f.close()
print("Solver Done!  Runtime: %.1d"%(time.time()-t_start_time))
######################## SAVE THE FINAL STATE OF THE VEHICLES
final_t = actual_t+number_ctrl_pts_executed+1
xamb_actual = xamb_actual[:,:actual_t+number_ctrl_pts_executed+1]
for i in range(len(xothers_actual)):
    xothers_actual[i] = xothers_actual[i][:,:actual_t+number_ctrl_pts_executed+1]

print("Plotting all")
cmplot.plot_cars(world, response_MPC, xamb_actual, xothers_actual, folder, None, None, False, False, 0)

file_name = folder + "data/"+'all%02d'%(i_mpc)
print("Saving Actual Positions to...  ", file_name)
mpc.save_state(file_name, xamb_actual, uamb_mpc, xamb_des_mpc, xothers_actual, all_other_u_mpc, all_other_x_des_mpc)



