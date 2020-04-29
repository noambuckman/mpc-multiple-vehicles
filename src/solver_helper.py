import numpy as np
import copy as cp
import multiprocessing, functools

import src.IterativeBestResponseMPCMultiple as mibr
import src.MPC_Casadi as mpc


# def warm_solve_subroutine(k_warm, ux_warm_profiles, response_MPC, amb_MPC, nonresponse_MPC_list, k_slack, k_CA, k_CA_power, world, wall_CA, N, T, response_x0, amb_x0, nonresponse_x0_list, slack, solve_amb, nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list, uamb=None, xamb=None, xamb_des=None):
#     u_warm, x_warm, x_des_warm = ux_warm_profiles[k_warm]
#     temp, current_cost, max_slack, bri, x, x_des, u = solve_best_response(response_MPC, amb_MPC, nonresponse_MPC_list, k_slack, k_CA, k_CA_power, world, wall_CA, N, T, response_x0, amb_x0, nonresponse_x0_list, slack, solve_amb, k_warm, u_warm, x_warm, x_des_warm, nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list, uamb, xamb, xamb_des)                
#     return (temp, current_cost, max_slack, x, x_des, u)






def extend_last_mpc_ctrl(all_other_u_mpc, number_ctrl_pts_executed, N, all_other_MPC, all_other_x0):
    '''Copy the previous mpc and extend the last values'''
    all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr = [np.zeros(shape=(2, N)) for i in range(len(all_other_MPC))], [np.zeros(shape=(6, N+1)) for i in range(len(all_other_MPC))], [np.zeros(shape=(3, N+1)) for i in range(len(all_other_MPC))]
    for i in range(len(all_other_MPC)):
        all_other_u_ibr[i] = np.concatenate((all_other_u_mpc[i][:, number_ctrl_pts_executed:], np.tile(np.zeros(shape=(2,1)),(1, number_ctrl_pts_executed))),axis=1) ##   
        all_other_x_ibr[i], all_other_x_des_ibr[i] = all_other_MPC[i].forward_simulate_all(all_other_x0[i].reshape(6,1), all_other_u_ibr[i])
    return all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr


def pullover_guess(N, all_other_MPC, all_other_x0):
    ''' Provide a 2deg turn for all the other vehicles on the road'''

    all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr = [np.zeros(shape=(2, N)) for i in range(len(all_other_MPC))], [np.zeros(shape=(6, N+1)) for i in range(len(all_other_MPC))], [np.zeros(shape=(3, N+1)) for i in range(len(all_other_MPC))]
    for i in range(len(all_other_MPC)):
        if i%2==0:
            all_other_u_ibr[i][0,0] = -2 * np.pi/180  # This is a hack and should be explicit that it's lane change
        else:
            all_other_u_ibr[i][0,0] = 2 * np.pi/180  # This is a hack and should be explicit that it's lane change  
        all_other_x_ibr[i], all_other_x_des_ibr[i] = all_other_MPC[i].forward_simulate_all(all_other_x0[i].reshape(6,1), all_other_u_ibr[i])
    return all_other_u_ibr, all_other_x_ibr, all_other_x_des_ibr

def solve_best_response(u_warm, x_warm, x_des_warm, ux_warm_profiles, response_MPC, amb_MPC, nonresponse_MPC_list, k_slack, k_CA, k_CA_power, world, wall_CA, N, T, response_x0, amb_x0, nonresponse_x0_list, slack, solve_amb, k_warm, u_warm, x_warm, x_des_warm, nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list, uamb=None, xamb=None, xamb_des=None):
    '''Create the iterative best response object and solve.  Assumes that it receives warm start profiles.
    This really should only require a u_warm, x_warm, x_des_warm and then one level above we generate those values'''
    
    bri = mibr.IterativeBestResponseMPCMultiple(response_MPC, amb_MPC, nonresponse_MPC_list )
    bri.k_slack, bri.k_CA, bri.k_CA_power, bri.world, bri.wall_CA = k_slack, k_CA, k_CA_power, world, wall_CA
    bri.generate_optimization(N, T, response_x0, amb_x0, nonresponse_x0_list,  0, slack=slack, solve_amb=solve_amb)

    # u_warm, x_warm, x_des_warm = ux_warm_profiles[k_warm]
    bri.opti.set_initial(bri.u_opt, u_warm)            
    bri.opti.set_initial(bri.x_opt, x_warm)
    bri.opti.set_initial(bri.x_desired, x_des_warm)   

    if amb_MPC:
        if solve_amb:
            bri.opti.set_initial(bri.xamb_opt, xamb)
            bri.opti.set_initial(bri.xamb_desired, xamb_des)                        
        else:
            bri.opti.set_value(bri.xamb_opt, xamb)
            bri.opti.set_value(bri.xamb_desired, xamb_des)

    for j in range(len(nonresponse_x_list)):
        bri.opti.set_value(bri.allother_x_opt[j], nonresponse_x_list[j])
        bri.opti.set_value(bri.allother_x_desired[j], nonresponse_xd_list[j])
    try:
        bri.solve(uamb, nonresponse_u_list, solve_amb)
        x1, u1, x1_des, _, _, _, _, _, _ = bri.get_solution()
        min_slack = np.infty
        if bri.solution.value(bri.slack_cost) < min_slack:
            current_cost = bri.solution.value(bri.total_svo_cost)
            max_slack = np.max([np.max(bri.solution.value(s)) for s in bri.slack_vars_list])                                                                         
            uamb_ibr = u1
            xamb_ibr = x1
            xamb_des_ibr = x1_des
            min_response_cost = current_cost
            min_response_warm_ibr = k_warm
            min_bri_ibr = bri
            amb_solved_flag = True     
    except RuntimeError:
        # print("Infeasibility: k_warm %s"%k_warm)
        return False, np.infty, np.infty, None, None, None
        # ibr_sub_it +=1  
    return amb_solved_flag, current_cost, max_slack, xamb_ibr, xamb_des_ibr, uamb_ibr 


def solve_warm_starts(parallelize, ux_warm_profiles, response_MPC, amb_MPC, nonresponse_MPC_list, k_slack, k_CA, k_CA_power, world, wall_CA, N, T, response_x0, amb_x0, nonresponse_x0_list, slack, solve_amb, nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list, uamb=None, xamb=None, xamb_des=None):
    warm_solve_partial  = functools.partial(solve_best_response, ux_warm_profiles=ux_warm_profiles, response_MPC=response_MPC, amb_MPC=amb_MPC, nonresponse_MPC_list=nonresponse_MPC_list, k_slack=k_slack, k_CA=k_CA, k_CA_power=k_CA_power, world=world, wall_CA=wall_CA, N=N, T=T, response_x0=response_x0, amb_x0=amb_x0, nonresponse_x0_list=nonresponse_x0_list, slack=slack, solve_amb=solve_amb, nonresponse_u_list=nonresponse_u_list, nonresponse_x_list=nonresponse_x_list, nonresponse_xd_list=nonresponse_xd_list, uamb=uamb, xamb=xamb, xamb_des=xamb_des)
    
    if parallelize:
        pool = multiprocessing.Pool(processes=8)
        solve_costs_solutions  =  pool.starmap(warm_solve_partial, ux_warm_profiles.values()) #will apply k=1...N to plot_partial
    else:
        solve_costs_solutions = []
        for k_warm in ux_warm_profiles.keys():
            solve_costs_solutions += warm_solve_partial(*ux_warm_profiles[k_warm])

    min_cost_solution = min(solve_costs_solutions, key=lambda r:r[1])  

    return min_cost_solution  

            # print(" i_mpc %d n_round %d Amb Cost %.02f Slack %.02f "%(i_mpc, i_rounds_ibr, bri.solution.value(bri.total_svo_cost), bri.solution.value(bri.slack_cost)))
            # print(" J_i %.03f,  J_j %.03f, Slack %.03f, CA  %.03f"%(bri.solution.value(bri.response_svo_cost), bri.solution.value(bri.other_svo_cost), bri.solution.value(bri.k_slack*bri.slack_cost), bri.solution.value(bri.k_CA*bri.collision_cost)))
            # print(" Dir:", subdir_name)            
            # print("  WARM START: %s, C: %0.02f"%(k_warm, current_cost)) 
            # for k in range(N):
            #     cmplot.plot_multiple_cars( k, bri.responseMPC, nonresponse_x_list, x_warm,  True, None, None, None, bri.world, 0)                                                     
            #     plt.show()     
            # print("Current Min Response Key: %s"%k_warm)      
                # file_name = folder + "data/"+'%03d'%ibr_sub_it
                # mibr.save_state(file_name, xamb, uamb, xamb_des, all_other_x, all_other_u, all_other_x_des)
                # mibr.save_costs(file_name, bri)                           




def initialize_cars(n_other, N, dt, world, svo_theta):
    ## Create the Cars in this Problem
    all_other_x0 = []
    all_other_u = []
    all_other_MPC = []
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

        initial_speed = 0.75 * x1_MPC.max_v
        x1_MPC.fd = x1_MPC.gen_f_desired_lane(world, lane_number, True)
        x0 = np.array([next_x0, world.get_lane_centerline_y(lane_number), 0, 0, initial_speed, 0]).T
        all_other_MPC += [x1_MPC]
        all_other_x0 += [x0]

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
    
    return all_other_x0, all_other_MPC, amb_MPC, x0_amb
