import numpy as np
import copy as cp
import multiprocessing, functools

import src.multiagent_mpc as mpc
import src.vehicle as vehicle


# def warm_solve_subroutine(k_warm, ux_warm_profiles, response_MPC, amb_MPC, nonresponse_MPC_list, k_slack, k_CA, k_CA_power, world, wall_CA, N, T, response_x0, amb_x0, nonresponse_x0_list, slack, solve_amb, nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list, uamb=None, xamb=None, xamb_des=None):
#     u_warm, x_warm, x_des_warm = ux_warm_profiles[k_warm]
#     temp, current_cost, max_slack, bri, x, x_des, u = solve_best_response(response_MPC, amb_MPC, nonresponse_MPC_list, k_slack, k_CA, k_CA_power, world, wall_CA, N, T, response_x0, amb_x0, nonresponse_x0_list, slack, solve_amb, k_warm, u_warm, x_warm, x_des_warm, nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list, uamb, xamb, xamb_des)                
#     return (temp, current_cost, max_slack, x, x_des, u)



def get_min_dist_i(ambulance_x0, all_other_x0, restrict_greater=False):
    all_dist_sqrd = [(ambulance_x0[0]-x0[0])**2 + (ambulance_x0[1]-x0[1])**2 for x0 in all_other_x0]
    
    if restrict_greater:
        for i in range(len(all_other_x0)):
            if all_other_x0[i][0] < ambulance_x0[0]:
                all_dist_sqrd[i] = 999999999
    
    if len(all_dist_sqrd) == 0:
        return []
    else:
        return np.argmin(all_dist_sqrd)


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

def solve_best_response(warm_key, warm_trajectory, 
                        response_MPC, amb_MPC, nonresponse_MPC_list, 
                        response_x0, amb_x0, nonresponse_x0_list,
                        world, solver_params, params, nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list,
                        uamb=None, xamb=None, xamb_des=None, return_bri=False):
    '''Create the iterative best response object and solve.  Assumes that it receives warm start profiles.
    This really should only require a u_warm, x_warm, x_des_warm and then one level above we generate those values'''
    
    u_warm, x_warm, x_des_warm = warm_trajectory
    bri = mpc.MultiMPC(response_MPC, amb_MPC, nonresponse_MPC_list, world, solver_params)
    params["collision_avoidance_checking_distance"] = 100
    bri.generate_optimization(params["N"], params["T"], response_x0, amb_x0, nonresponse_x0_list,  print_level=params["print_level"], slack=solver_params['slack'], solve_amb=solver_params['solve_amb'], params=params)
    print("Succesffully generated optimzation %d %d %d"%(len(nonresponse_MPC_list), len(nonresponse_x_list), len(nonresponse_u_list)))
    # u_warm, x_warm, x_des_warm = ux_warm_profiles[k_warm]
    bri.opti.set_initial(bri.u_opt, u_warm)            
    bri.opti.set_initial(bri.x_opt, x_warm)
    bri.opti.set_initial(bri.x_desired, x_des_warm)   

    if amb_MPC:
        if solver_params['solve_amb']:
            bri.opti.set_initial(bri.xamb_opt, xamb)
            bri.opti.set_initial(bri.xamb_desired, xamb_des)                        
        else:
            bri.opti.set_value(bri.xamb_opt, xamb)
            bri.opti.set_value(bri.xamb_desired, xamb_des)

    for j in range(len(nonresponse_x_list)):
        bri.opti.set_value(bri.allother_x_opt[j], nonresponse_x_list[j])
        bri.opti.set_value(bri.allother_x_desired[j], nonresponse_xd_list[j])
    
    try:
        if "constant_v" in solver_params and solver_params["constant_v"]:
            bri.opti.subject_to(bri.u_opt[1,:] == 0)
        bri.solve(uamb, nonresponse_u_list, solver_params['solve_amb'])
        x_ibr, u_ibr, x_des_ibr, _, _, _, _, _, _ = bri.get_solution()
        current_cost = bri.solution.value(bri.total_svo_cost)
        all_slack_vars = [bri.solution.value(s) for s in bri.slack_vars_list]
        if amb_MPC:
             all_slack_vars += [bri.solution.value(bri.slack_amb)]
        max_slack = np.max([np.max(s) for s in all_slack_vars] + [0.000000000000])
        # max_slack = np.max([np.max(bri.solution.value(s)) for s in bri.slack_vars_list])                                                                         
        min_response_warm_ibr = None #<This used to return k_warm
        

        debug_list = [current_cost, bri.solution.value(bri.response_svo_cost), bri.solution.value(bri.k_CA*bri.collision_cost), 
                        bri.solution.value(bri.k_slack*bri.slack_cost), all_slack_vars]
        solved = True
        if return_bri:
            debug_list += [bri]
        return solved, current_cost, max_slack, x_ibr, x_des_ibr, u_ibr, warm_key, debug_list
    except RuntimeError:
        # print("Infeasibility: k_warm %s"%k_warm)
        if return_bri:
            return bri
        return False, np.infty, np.infty, None, None, None, None, []
        # ibr_sub_it +=1  

def solve_warm_starts(ux_warm_profiles, 
                    response_MPC, amb_MPC, nonresponse_MPC_list, 
                    response_x0, amb_x0, nonresponse_x0_list, 
                    world, solver_params, params,
                    nonresponse_u_list, nonresponse_x_list, nonresponse_xd_list, 
                    uamb=None, xamb=None, xamb_des=None, 
                    debug_flag=False):
    warm_solve_partial  = functools.partial(solve_best_response, 
                        response_MPC=response_MPC, amb_MPC=amb_MPC, nonresponse_MPC_list=nonresponse_MPC_list, 
                        response_x0=response_x0, amb_x0=amb_x0, nonresponse_x0_list=nonresponse_x0_list, 
                        world=world, solver_params = solver_params, params=params,
                        nonresponse_u_list=nonresponse_u_list, nonresponse_x_list=nonresponse_x_list, nonresponse_xd_list=nonresponse_xd_list, 
                        uamb=uamb, xamb=xamb, xamb_des=xamb_des, return_bri=False)
    
    if params['n_processors']>1:
        pool = multiprocessing.Pool(processes=params['n_processors'])
        solve_costs_solutions  =  pool.starmap(warm_solve_partial, ux_warm_profiles.items()) #will apply k=1...N to plot_partial
        pool.terminate()
    else:
        solve_costs_solutions = []
        for k_warm in ux_warm_profiles:
            solve_costs_solutions += [warm_solve_partial(k_warm, ux_warm_profiles[k_warm])]

    if debug_flag:
        for ki in range(len(solve_costs_solutions)):
            debug_list = solve_costs_solutions[ki][7]
            print(debug_list)
            if len(debug_list) == 0:
                print("Infeasible")
            else:
                # print("Feasible")
                print("Costs: Total Cost %.04f Vehicle-Only Cost:  %.04f Collision Cost %0.04f  Slack Cost %0.04f"%(tuple(debug_list[0:4])))                 


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




def initialize_cars(n_other, N, dt, world, svo_theta, no_grass = False, random_lane=False, x_variance = 1.0, list_of_positions = None):
    '''x_variance is in terms of number of min_dist'''
    ## Create the Cars in this Problem
    all_other_x0 = []
    all_other_u = []
    all_other_MPC = []
    next_x0_0 = 0
    next_x0_1 = 0
    for i in range(n_other):
        x1_MPC = vehicle.Vehicle(dt)
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
        if no_grass:
                x1_MPC.min_y += world.grass_width
                x1_MPC.max_y -= world.grass_width        
        x1_MPC.strict_wall_constraint = True

        ####Vehicle Initial Conditions
        lane_offset = np.random.uniform(0, 1) * x_variance * 2 * x1_MPC.min_dist
        if random_lane:
            lane_number = np.random.randint(2)
        else:
            lane_number = 0 if i%2==0 else 1
        if lane_number == 0: 
            next_x0_0 += x1_MPC.L + 2*x1_MPC.min_dist + lane_offset
            next_x0 = next_x0_0
        else:
            next_x0_1 += x1_MPC.L + 2*x1_MPC.min_dist + lane_offset
            next_x0 = next_x0_1

        if list_of_positions: #list of positions overrides everything
            lane_number, next_x0 = list_of_positions[i]
            
        initial_speed = 0.99 * x1_MPC.max_v
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
    amb_MPC.max_v = 30 * 0.447 # m/s
    amb_MPC.k_phi_error = 0.1
    amb_MPC.k_phi_dot = 0.01
    amb_MPC.min_y = world.y_min        
    amb_MPC.max_y = world.y_max
    if no_grass:
        amb_MPC.min_y += world.grass_width
        amb_MPC.max_y -= world.grass_width
    amb_MPC.fd = amb_MPC.gen_f_desired_lane(world, 0, True)
    x0_amb = np.array([0, 0, 0, 0, initial_speed , 0]).T
    
    return amb_MPC, x0_amb, all_other_MPC, all_other_x0


def initialize_cars_from_positions(N, dt, world, svo_theta, no_grass = False, list_of_positions = None):
    '''x_variance is in terms of number of min_dist'''
    ## Create the Cars in this Problem
    all_other_x0 = []
    all_other_u = []
    all_other_MPC = []
    next_x0_0 = 0
    next_x0_1 = 0
    for i in range(len(list_of_positions)):
        x1_MPC = vehicle.Vehicle(dt)
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
        if no_grass:
                x1_MPC.min_y += world.grass_width
                x1_MPC.max_y -= world.grass_width        
        x1_MPC.strict_wall_constraint = True

        lane_number, next_x0 = list_of_positions[i]
            
        initial_speed = 0.99 * x1_MPC.max_v
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
    amb_MPC.max_v = 30 * 0.447 # m/s
    amb_MPC.k_phi_error = 0.1
    amb_MPC.k_phi_dot = 0.01
    amb_MPC.min_y = world.y_min        
    amb_MPC.max_y = world.y_max
    if no_grass:
        amb_MPC.min_y += world.grass_width
        amb_MPC.max_y -= world.grass_width
    amb_MPC.fd = amb_MPC.gen_f_desired_lane(world, 0, True)
    x0_amb = np.array([0, 0, 0, 0, initial_speed , 0]).T
    
    return amb_MPC, x0_amb, all_other_MPC, all_other_x0


def poission_positions(cars_per_hour, total_seconds, n_lanes, average_velocity, car_length):
    cars_per_second = cars_per_hour / 3600.0
    rng = np.random.default_rng()
    n_cars_per_second = rng.poisson(cars_per_second, total_seconds)
    
    vehicle_x_distance = [[average_velocity*(s + rng.uniform(0,1))]*n_cars_per_second[s] for s in range(len(n_cars_per_second))]
    
    all_vehicle_positions = []
    for s in range(len(vehicle_x_distance)):
        if len(vehicle_x_distance[s]) == 0:
            continue
        else:
            for j in range(len(vehicle_x_distance[s])):
                all_vehicle_positions += [(rng.integers(0, n_lanes), vehicle_x_distance[s][j])]

    # Remove cars that would have collided
    prev_car_lane = -9999 * np.ones(n_lanes)
    prev_car_lane[0] = 0.0
    initial_vehicle_positions = []
    for (lane, x) in all_vehicle_positions:
        if x > prev_car_lane[lane] + 1.1*car_length:
            initial_vehicle_positions += [(lane, float(x))]
            prev_car_lane[lane] = x

    return initial_vehicle_positions
