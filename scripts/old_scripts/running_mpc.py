import datetime
import os, sys
import numpy as np
import matplotlib.pyplot as plt

import casadi as cas
import pickle
import copy as cp

import argparse

PROJECT_PATH = '/home/nbuckman/Dropbox (MIT)/DRL/2020_01_cooperative_mpc/mpc-multiple-vehicles/'
sys.path.append(PROJECT_PATH)

import src.MPC_Casadi as mpc
import src.TrafficWorld as tw
import src.IterativeBestResponseMPCMultiple as mibr
import src.car_plotting_multiple as cmplot

np.set_printoptions(precision=2)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Using a running MPC to solve')
    parser.add_argument('theta_type', type=str, nargs=1,
                    help='An SVO type', default="p", choices=["p","e","a"])
    parser.add_argument('random_seed', default=0, nargs=1, type=int, help='specify the random seed number')

    args = parser.parse_args()

    if (args.theta_type[0]).lower() == "e":
        svo_theta = 0
    elif args.theta_type[0].lower() == "a":
        svo_theta = np.pi/2.05
    elif args.theta_type[0].lower() == "p":
        svo_theta = np.pi/4.0
    else:
        raise Exception("incorect svo arg")

    NEW = True
    if NEW:
        optional_suffix = "faster_pullover_warmstart"
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

    T = 3
    # dt = 0.2
    dt = 0.4

    N = int(T/dt) #Number of control intervals
    print(N)
    world = tw.TrafficWorld(2, 0, 1000)
    n_other = 4
    if args.random_seed[0] > 0:
        np.random.seed(args.random_seed[0])

    all_other_x0 = []
    all_other_u = []
    all_other_MPC = []
    next_x0 = 0
    for i in range(n_other):
        x1_MPC = mpc.MPC(dt)
        x1_MPC.n_circles = 3
        x1_MPC.theta_iamb =  svo_theta
        x1_MPC.N = N
        x1_MPC.k_final = 0.0
        x1_MPC.k_s = 0
        x1_MPC.k_x = 0
        x1_MPC.k_x_dot = -1.0 / 100.0    
        
        # x1_MPC.k_u_v = 1.0
        # x1_MPC.k_u_delta = 1.00
        # x1_MPC.k_lat = .25
        # x1_MPC.k_change_u_v = .50
        # x1_MPC.k_change_u_delta = .50    
        
        x1_MPC.k_u_v = .10
        x1_MPC.k_u_delta = .01
        x1_MPC.k_lat = 5.0
        x1_MPC.k_lon = 0.01
        x1_MPC.k_phi_error = 0.05 
        x1_MPC.k_change_u_v = .01
        x1_MPC.k_change_u_delta = .01        
        
        NO_GRASS = False
        x1_MPC.min_y = world.y_min        
        x1_MPC.max_y = world.y_max
        if NO_GRASS:
            x1_MPC.min_y += world.grass_width
            x1_MPC.max_y -= world.grass_width
        # x1_MPC.k_phi_error = 25

        if i%2 == 0:
            lane_number = 0
            next_x0 += x1_MPC.L + 2*x1_MPC.min_dist
        else:
            lane_number = 1
        
        initial_speed = 20 * 0.447 # m/s
        initial_speed = x1_MPC.max_v
        initial_speed = 0.75*x1_MPC.max_v

        # large_world = tw.TrafficWorld(2, 0, 1000, 5.0)
        traffic_world = tw.TrafficWorld(2, 0, 1000)


        x1_MPC.fd = x1_MPC.gen_f_desired_lane(traffic_world, lane_number, True)
        
        x0 = np.array([next_x0, traffic_world.get_lane_centerline_y(lane_number), 0, 0, initial_speed, 0]).T
        u1 = np.zeros((2,N))
        # u1[0,:] = np.clip(np.pi/180 *np.random.normal(size=(1,N)), -2 * np.pi/180, 2 * np.pi/180)
        SAME_SIDE = False
        if lane_number == 1 or SAME_SIDE:
            u1[0,0] = 2 * np.pi/180
        else:
            u1[0,0] = -2 * np.pi/180
        # u1[0,0] = 0 ###
        all_other_MPC += [x1_MPC]
        all_other_x0 += [x0]
        all_other_u += [u1]    
    pickle.dump(x1_MPC, open(folder + "data/"+"mpc%d"%i + ".p",'wb'))


    amb_MPC = cp.deepcopy(x1_MPC)
    amb_MPC.theta_iamb = 0.0

    amb_MPC.k_u_v = 0.01
    amb_MPC.k_u_delta = .01
    amb_MPC.k_change_u_v = 0.01
    amb_MPC.k_change_u_delta = 0.01

    amb_MPC.k_s = 0
    amb_MPC.k_x = 0
    amb_MPC.k_x_dot = -1.0 / 100.0
    amb_MPC.k_lat = 0.001
    amb_MPC.k_lon = 0.0
    amb_MPC.min_v = 0.8*initial_speed
    amb_MPC.max_v = 35 * 0.447 # m/s
    
    amb_MPC.fd = amb_MPC.gen_f_desired_lane(world, 0, True)
    pickle.dump(amb_MPC, open(folder + "data/"+"mpcamb" + ".p",'wb'))
    x0_amb = np.array([0, 0, 0, 0, initial_speed , 0]).T
    uamb = np.zeros((2,N))
    uamb[0,:] = np.clip(np.pi/180 * np.random.normal(size=(1,N)), -2 * np.pi/180, 2 * np.pi/180)
    # uamb[0,0] = 2 * np.pi/180


    n_rounds_mpc = 60 # seconds
    number_ctrl_pts_executed = 5  #dt = 0.2
    actual_xamb = np.zeros((6, n_rounds_mpc*number_ctrl_pts_executed + 1))
    actual_uamb = np.zeros((2, n_rounds_mpc*number_ctrl_pts_executed))
    actual_xothers = [np.zeros((6, n_rounds_mpc*number_ctrl_pts_executed + 1)) for i in range(n_other)]
    actual_uothers = [np.zeros((2, n_rounds_mpc*number_ctrl_pts_executed)) for i in range(n_other)]

    # actual_xamb[:,0] = x0_amb

    actual_all_other_x0 = [np.zeros((6, 2*N)) for i in range(n_other)]
    XAMB_ONLY = True
    xamb = None
    ibr_sub_it = 1
    for i_mpc in range(n_rounds_mpc):
        WARM = False
        runtimeerrors = 0
        min_slack = 99999.0
        # xamb = None
        if xamb is not None:
            xamb = np.concatenate((xamb[:,number_ctrl_pts_executed+1:] , xamb[:,:number_ctrl_pts_executed+1]),axis=1)        
        actual_t = i_mpc * number_ctrl_pts_executed
        print("x0amb", x0_amb)
        n_total_round = 1
        for n_round in range(n_total_round):
            # if n_round > 10:
            #     min_slack = 2.0
            # if n_round > 15:
            #     min_slack = 1.0
            # if n_round > 20:
            #     min_slack = 0.5
            # if n_round > 25:
            #     min_slack = 0.1        
            response_MPC = amb_MPC
            response_x0 = x0_amb
            
            nonresponse_MPC_list = all_other_MPC
            nonresponse_x0_list = all_other_x0
            nonresponse_u_list = all_other_u
            bri = mibr.IterativeBestResponseMPCMultiple(response_MPC, None, nonresponse_MPC_list )
            bri.k_slack = 100.0
            bri.k_CA = 100.0
            bri.generate_optimization(N, T, response_x0, None, nonresponse_x0_list,  5, slack=True)
            for slack_var in bri.slack_vars_list: ## Added to constrain slacks
                bri.opti.subject_to(cas.vec(slack_var) < 10.0)
            INFEASIBLE = True
            UWARM = True
            if WARM and (xamb is not None):
                bri.opti.set_initial(bri.x_opt, xamb)
            elif UWARM and (uamb is not None):
                bri.opti.set_initial(bri.u_opt, uamb)
            try:
                bri.solve(None, nonresponse_u_list)
                x1, u1, x1_des, _, _, _, other_x, other_u, other_des = bri.get_solution()
                xothers = other_x # initialize the x values of the other vehicles
                uothers = other_u
                xothers_des = other_des
                print("i_mpc %d n_round %d i %02d Cost %.02f Slack %.02f "%(i_mpc, n_round, i, bri.solution.value(bri.total_svo_cost), bri.solution.value(bri.slack_cost)))
                print("J_i %.03f,  J_j %.03f, Slack %.03f, CA  %.03f"%(bri.solution.value(bri.response_svo_cost), bri.solution.value(bri.other_svo_cost), bri.solution.value(bri.k_slack*bri.slack_cost), bri.solution.value(bri.k_CA*bri.collision_cost)))
                print("Dir:", subdir_name)
                INFEASIBLE = False
                if bri.solution.value(bri.slack_cost) <= min_slack:
                    uamb = u1
                    xamb = x1
                    xamb_des = x1_des
                    # all_other_u[i] = u1
                    file_name = folder + "data/"+'%03d'%ibr_sub_it
                    mibr.save_state(file_name, x1, u1, x1_des, other_x, other_u, other_des)
                    mibr.save_costs(file_name, bri)          
                else: 
                    print("Slack too large")
            except RuntimeError:
                print("Max Iterations or Infeasible")
                INFEASIBLE = True
                runtimeerrors += 1                
            ibr_sub_it +=1
            if XAMB_ONLY:
                pass
            else:
                for i in range(len(all_other_MPC)):
                    response_MPC = all_other_MPC[i]
                    response_x0 = all_other_x0[i]

                    nonresponse_MPC_list = all_other_MPC[:i] + all_other_MPC[i+1:]
                    nonresponse_x0_list = all_other_x0[:i] + all_other_x0[i+1:]

                    # all_other_u changes over time
                    nonresponse_u_list = all_other_u[:i] + all_other_u[i+1:]

                    bri = mibr.IterativeBestResponseMPCMultiple(response_MPC, amb_MPC, nonresponse_MPC_list )
                    bri.k_slack = 9999

                    bri.generate_optimization(N, T, response_x0, x0_amb, nonresponse_x0_list,  5, slack=True)
                    
                    try:
                        if WARM:
                            bri.opti.set_initial(bri.x_opt, xothers[i])
                            bri.opti.set_initial(bri.u_opt, uothers[i])

                        bri.solve(uamb, nonresponse_u_list)
                        x1, u1, x1_des, xamb, uamb, xamb_des, other_x, other_u, other_des = bri.get_solution()
                        print("i_mpc %d n_round %d  i %02d Cost %.02f Slack %.02f "%(i_mpc, n_round, i, bri.solution.value(bri.total_svo_cost), bri.solution.value(bri.slack_cost)))
                        print("Dir:", subdir_name)

                        if bri.solution.value(bri.slack_cost) <= min_slack:
                            # Update the responder
                            all_other_u[i] = u1
                            
                            #for saving
                            xothers = other_x[:i] + [x1] + other_x[i:]
                            uothers = other_u[:i] + [u1] + other_u[i:]
                            xothers_des = other_des[:i] + [x1_des] + other_des[i:]

                            file_name = folder + "data/"+'%03d'%ibr_sub_it
                            mibr.save_state(file_name, xamb, uamb, xamb_des, xothers, uothers, xothers_des)
                            mibr.save_costs(file_name, bri)
                        else: 
                            print("Slack too large")    
                    except RuntimeError:
                        print("Max Iterations or Infeasible")
                        runtimeerrors += 1   
                    ibr_sub_it+=1

        file_name = folder + "data/"+'r%02d%03d'%(i_mpc, n_round)
        if not INFEASIBLE:
            mibr.save_state(file_name, xamb, uamb, xamb_des, xothers, uothers, xothers_des)
            mibr.save_costs(file_name, bri)
            actual_t = i_mpc * number_ctrl_pts_executed
            actual_xamb[:,actual_t:actual_t+number_ctrl_pts_executed+1]  = xamb[:,:number_ctrl_pts_executed+1]
            print(i_mpc, xamb[0:2,:number_ctrl_pts_executed+1])
            
            actual_uamb[:,actual_t:actual_t+number_ctrl_pts_executed] = uamb[:,:number_ctrl_pts_executed]
            
            for i in range(len(xothers)):
                actual_xothers[i][:,actual_t:actual_t+number_ctrl_pts_executed+1] = xothers[i][:,:number_ctrl_pts_executed+1]
                actual_uothers[i][:,actual_t:actual_t+number_ctrl_pts_executed] = uothers[i][:,:number_ctrl_pts_executed]

            x0_amb = xamb[:,number_ctrl_pts_executed] # NEW INITIAL STATE
            uamb_warm = np.concatenate((uamb[:, number_ctrl_pts_executed:], 0*uamb[:, :number_ctrl_pts_executed]),axis=1) ## <--- This is not a kosher operation
            
            for i in range(len(all_other_x0)):
                all_other_x0[i] = xothers[i][:,number_ctrl_pts_executed]
                all_other_u[i] = np.concatenate((uothers[i][:, number_ctrl_pts_executed:],uothers[i][:,:number_ctrl_pts_executed]),axis=1)
        else:
            raise Exception("Xamb is None", i_mpc, n_round, "slack cost", bri.solution.value(bri.slack_cost))