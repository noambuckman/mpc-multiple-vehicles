import datetime
import os, sys
import numpy as np
import matplotlib.pyplot as plt

import casadi as cas
import pickle
import copy as cp

PROJECT_PATH = '/home/nbuckman/Dropbox (MIT)/DRL/2020_01_cooperative_mpc/mpc-multiple-vehicles/'
sys.path.append(PROJECT_PATH)

import src.MPC_Casadi as mpc
import src.TrafficWorld as tw
import src.IterativeBestResponseMPCMultiple as mibr
import src.car_plotting_multiple as cmplot

np.set_printoptions(precision=2)


NEW = True
if NEW:
    optional_suffix = "20200227_161307pi5degaltrunograss2"
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





T = 5
dt = 0.2

N = int(T/dt) #Number of control intervals
world = tw.TrafficWorld(2, 0, 1000)
n_other = 4


all_other_x0 = []
all_other_u = []
all_other_MPC = []
next_x0 = 0
for i in range(n_other):
    x1_MPC = mpc.MPC(dt)
    x1_MPC.n_circles = 3
    x1_MPC.theta_iamb =  5/180* np.pi
    x1_MPC.k_final = 1.0
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
    
    NO_GRASS = True
    if NO_GRASS:
        x1_MPC.min_y = world.y_min + world.grass_width
        x1_MPC.max_y = world.y_max - world.grass_width
    # x1_MPC.k_phi_error = 25

    if i%2 == 0:
        lane_number = 0
        next_x0 += x1_MPC.L/2.0 + 2*x1_MPC.min_dist
    else:
        lane_number = 1
    
    initial_speed = 20 * 0.447 # m/s
    initial_speed = x1_MPC.max_v
    x1_MPC.fd = x1_MPC.gen_f_desired_lane(world, lane_number, True)
    
    x0 = np.array([next_x0, world.get_lane_centerline_y(lane_number), 0, 0, initial_speed, 0]).T
    u1 = np.zeros((2,N))
    # u1[0,:] = np.clip(np.pi/180 *np.random.normal(size=(1,N)), -2 * np.pi/180, 2 * np.pi/180)
    SAME_SIDE = False
    if lane_number == 1 or SAME_SIDE:
        u1[0,0] = 2 * np.pi/180
    else:
        u1[0,0] = -2 * np.pi/180
    all_other_MPC += [x1_MPC]
    all_other_x0 += [x0]
    all_other_u += [u1]    
pickle.dump(x1_MPC, open(folder + "data/"+"mpc%d"%i + ".p",'wb'))


amb_MPC = cp.deepcopy(x1_MPC)
amb_MPC.theta_iamb = 0.0

amb_MPC.k_u_v = 0.10
amb_MPC.k_u_delta = .01
amb_MPC.k_change_u_v = 0.01
amb_MPC.k_change_u_delta = 0.01

amb_MPC.k_s = 0
amb_MPC.k_x = 0
amb_MPC.k_x_dot = -1.0 / 100.0

amb_MPC.min_v = 0.8*initial_speed
amb_MPC.max_v = 30 * 0.447 # m/s

amb_MPC.fd = amb_MPC.gen_f_desired_lane(world, 0, True)
pickle.dump(amb_MPC, open(folder + "data/"+"mpcamb" + ".p",'wb'))
x0_amb = np.array([0, 0, 0, 0, amb_MPC.max_v , 0]).T
uamb = np.zeros((2,N))
uamb[0,:] = np.clip(np.pi/180 * np.random.normal(size=(1,N)), -2 * np.pi/180, 2 * np.pi/180)

ibr_sub_it=1
actual_xamb = np.zeros((6, 5*N))
actual_xamb[:,0] = x0_amb

actual_all_other_x0 = [np.zeros((6, 5*N)) for i in range(n_other)]

for t_mpc in range(actual_xamb.shape[1]):
    x0_amb = actual_xamb[:,t_mpc]
    print("Initial",x0_amb)
    if t_mpc > 0:
        uamb[:,:-1] = uamb[:,1:]
    print(actual_xamb)
    for i in range(len(all_other_x0)):
        all_other_x0[i] = actual_all_other_x0[i][:,t_mpc]
        if t_mpc > 0:
            all_other_u[i][:,:-1] = all_other_u[i][:,1:]

    WARM = True
    n_total_round = 9
    ibr_sub_it = 1
    runtimeerrors = 0
    min_slack = 100000.0
    xamb = None

    for n_round in range(n_total_round):
        if n_round > 10:
            min_slack = 2.0
        if n_round > 15:
            min_slack = 1.0
        if n_round > 20:
            min_slack = 0.5
        if n_round > 25:
            min_slack = 0.1        
        response_MPC = amb_MPC
        response_x0 = x0_amb
        
        nonresponse_MPC_list = all_other_MPC
        nonresponse_x0_list = all_other_x0
        nonresponse_u_list = all_other_u
        bri = mibr.IterativeBestResponseMPCMultiple(response_MPC, None, nonresponse_MPC_list )
        bri.k_slack = 9999
        bri.generate_optimization(N, T, response_x0, None, nonresponse_x0_list,  1, slack=True)
        if WARM and (xamb is not None):
            bri.opti.set_initial(bri.x_opt, xamb)
        try:
            bri.solve(None, nonresponse_u_list)

            x1, u1, x1_des, _, _, _, other_x, other_u, other_des = bri.get_solution()
            xothers = other_x # initialize the x values of the other vehicles
            uothers = other_u
            
            print("n_round %d i %02d Cost %.02f Slack %.02f "%(n_round, i, bri.solution.value(bri.total_svo_cost), bri.solution.value(bri.slack_cost)))
            print("Dir:", subdir_name)
            if bri.solution.value(bri.slack_cost) <= min_slack:
                uamb = u1
                xamb = x1
                # all_other_u[i] = u1
                file_name = folder + "data/"+'%03d'%ibr_sub_it
                mibr.save_state(file_name, x1, u1, x1_des, other_x, other_u, other_des)
                mibr.save_costs(file_name, bri)          
            else: 
                print("Slack too large")
        except RuntimeError:
            print("Max Iterations or Infeasible")
            runtimeerrors += 1                
        ibr_sub_it +=1

        for i in range(len(all_other_MPC)):
            response_MPC = all_other_MPC[i]
            response_x0 = all_other_x0[i]

            nonresponse_MPC_list = all_other_MPC[:i] + all_other_MPC[i+1:]
            nonresponse_x0_list = all_other_x0[:i] + all_other_x0[i+1:]

            # all_other_u changes over time
            nonresponse_u_list = all_other_u[:i] + all_other_u[i+1:]

            bri = mibr.IterativeBestResponseMPCMultiple(response_MPC, amb_MPC, nonresponse_MPC_list )
            bri.k_slack = 9999

            bri.generate_optimization(N, T, response_x0, x0_amb, nonresponse_x0_list,  1, slack=True)
            
            try:
                if WARM:
                    bri.opti.set_initial(bri.x_opt, xothers[i])
                    bri.opti.set_initial(bri.u_opt, uothers[i])

                bri.solve(uamb, nonresponse_u_list)
                x1, u1, x1_des, xamb, uamb, xamb_des, other_x, other_u, other_des = bri.get_solution()
                print("n_round %d  i %02d Cost %.02f Slack %.02f "%(n_round, i, bri.solution.value(bri.total_svo_cost), bri.solution.value(bri.slack_cost)))
                print("Dir:", subdir_name)

                if bri.solution.value(bri.slack_cost) <= min_slack:
                    # Update the responder
                    all_other_u[i] = u1
                    
                    #for saving
                    xothers = other_x[:i] + [x1] + other_x[i:]
                    uothers = other_u[:i] + [u1] + other_u[i:]
                    xothers_des = other_des[:i] + [x1_des] + other_des[i:]

                    file_name = folder + "data/"+'%03d'%ibr_sub_it
                    mibr.save_state(file_name, xamb, xamb, xamb_des, xothers, uothers, xothers_des)
                    mibr.save_costs(file_name, bri)
                else: 
                    print("Slack too large")    
                ibr_sub_it+=1
            except RuntimeError:
                print("Max Iterations or Infeasible")
                runtimeerrors += 1         

        actual_xamb[:,t_mpc+1]  = xamb[:,1] # SAVE THE NEW INITIAL STATES
        for i in range(len(all_other_x0)):
            actual_all_other_x0[i][:,t_mpc+1] = xothers[i][:,1]                   
        ##I'm actually saving all the x at each time step
        # mibr.save_state("RunningMPC%03d"%t_mpc + file_name, x1, u1, x1_des, other_x, other_u, other_des)
        mibr.save_state(file_name + "RunningMPC%03d"%t_mpc, xamb, xamb, xamb_des, xothers, uothers, xothers_des)
        print("MPC RD", t_mpc,actual_xamb[:,t_mpc+1])
