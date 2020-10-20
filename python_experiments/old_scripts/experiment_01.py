import datetime
import os, sys
import numpy as np
import matplotlib.pyplot as plt

import casadi as cas
import pickle

##### For viewing the videos in Jupyter Notebook
import io
import base64
from IPython.display import HTML

# from ..</src> import car_plotting
# from .import src.car_plotting
PROJECT_PATH = '/home/nbuckman/Dropbox (MIT)/DRL/2020_01_cooperative_mpc/mpc-multiple-vehicles/'
sys.path.append(PROJECT_PATH)
import src.MPC_Casadi as mpc
import src.car_plotting as cplot
import src.TrafficWorld as tw
np.set_printoptions(precision=2)


NEW = True
if NEW:
    optional_suffix = "test"
    subdir_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + optional_suffix
    folder = "results/" + subdir_name + "/"
    os.makedirs(folder)
    os.makedirs(folder+"imgs/")
    os.makedirs(folder+"data/")
    os.makedirs(folder+"vids/")
else:
    subdir_name = "20200224-103456_real_dim_CA"
    folder = "results/" + subdir_name + "/"
print(folder)



T = 10 #numbr of time horizons
dt = 0.2
N = int(T/dt) #Number of control intervals


world = tw.TrafficWorld(2, 0, 1000)
# world.grass_width = world.lane_width/2.0
# Initial Conditions

x1_MPC = mpc.MPC(dt)
x2_MPC = mpc.MPC(dt)
amb_MPC = mpc.MPC(dt)

x1_MPC.theta_iamb = np.pi/4.0
x2_MPC.theta_iamb = np.pi/4.0
amb_MPC.theta_iamb = 0


x1_MPC.k_final = 1.0
x2_MPC.k_final = 1.0
amb_MPC.k_final = 1.0


x1_MPC.k_s = -2.0
x2_MPC.k_s = -2.0

amb_MPC.theta_iamb = 0.0
amb_MPC.k_u_v = 0.10
# amb_MPC.k_u_change = 1.0
amb_MPC.k_s = -2.0
amb_MPC.max_v = 40 * 0.447 # m/s
# amb_MPC.max_X_dev = 5.0

# x1_MPC.k_phi_error = 20
# x2_MPC.k_phi_error = 20
# amb_MPC.k_phi_error = 10
x1_MPC.k_u_delta = 10.0
x2_MPC.k_u_delta = 10.0
amb_MPC.k_u_delta = 10.0

# #### THIS SHOULD BE REPLACED
x1_MPC.min_y = world.y_min
x2_MPC.min_y = world.y_min
amb_MPC.min_y = world.y_min

x1_MPC.max_y = world.y_max
x2_MPC.max_y = world.y_max
amb_MPC.max_y = world.y_max

x1_MPC.fd = x1_MPC.gen_f_desired_lane(world, 0, True)
x2_MPC.fd = x2_MPC.gen_f_desired_lane(world, 1, True)
amb_MPC.fd = amb_MPC.gen_f_desired_lane(world, 0, True)

initial_speed = 20 * 0.447 # m/s
x0 = np.array([x1_MPC.L/2.0 + 2*x1_MPC.min_dist, 0, 0, 0, initial_speed, 0]).T
x0_2 = np.array([x1_MPC.L/2.0 + 2*x1_MPC.min_dist, world.get_lane_centerline_y(1), 0.0, 0, initial_speed, 0]).T
x0_amb = np.array([0, 0.0, 0, 0, 1.1*initial_speed , 0]).T



pickle.dump(x1_MPC, open(folder + "data/" + "x1.p",'wb'))
pickle.dump(x2_MPC, open(folder + "data/"  + "x2.p",'wb'))
pickle.dump(amb_MPC, open(folder + "data/" + "amb.p",'wb'))
brA = mpc.IterativeBestResponseMPC(amb_MPC, x1_MPC, x2_MPC)
brA.generate_optimization(N, T, x0_amb, x0, x0_2,  5, slack=True)
# pickle.dump(brA, open(folder + "data/" + "ibr.p",'wb'))


ibr_sub_it=1


uamb = np.zeros((2,N))
u1 = np.zeros((2,N))
u2 = np.zeros((2,N))
u1[1,:] = np.clip(np.random.normal(size=(1,N)), -np.pi/4, np.pi/4)
u2[1,:] = np.clip(np.random.normal(size=(1,N)), -np.pi/4, np.pi/4)

# ibr_sub_it=3
# br1 = mpc.IterativeBestResponseMPC(x1_MPC, x2_MPC, amb_MPC)
# br1.generate_optimization(N, fd, T,  x0, x0_2, x0_amb, 5, slack=True)
# x1, u1, x1_des, x2, u2, x2_des, xamb, uamb, xamb_des = br1.load_state(folder + "data/" + '%03d'%ibr_sub_it + "_1")
# ibr_sub_it+=1
WARM = True

n_total_round = 30
for n_round in range(n_total_round):
    brA = mpc.IterativeBestResponseMPC(amb_MPC, x1_MPC, x2_MPC)
    brA.generate_optimization(N, T, x0_amb, x0, x0_2,  5, slack=True)
    if WARM:
        brA.opti.set_initial(brA.u_opt, uamb)
        if n_round != 0:    
            brA.opti.set_initial(brA.x_opt, xamb)
            brA.opti.set_initial(brA.x_desired, xamb_des)
    brA.solve(u1, u2)
    print("A", brA.solution.value(brA.car1MPC.total_cost()))
    xamb, uamb, xamb_des, x1, u1, x1_des, x2, u2, x2_des = brA.get_solution()
    brA.save_state(folder + "data/"+'%03d'%ibr_sub_it + "_a", x1, u1, x1_des, x2, u2, x2_des, xamb, uamb, xamb_des)
    ibr_sub_it+=1

    br2 = mpc.IterativeBestResponseMPC(x2_MPC, x1_MPC, amb_MPC )
    br2.generate_optimization(N, T, x0_2, x0, x0_amb, 5, slack=True)
    if WARM:
        br2.opti.set_initial(br2.u_opt, u2) 
        br2.opti.set_initial(br2.x_opt, x2)           
        br2.opti.set_initial(br2.x_desired, x2_des)           
    br2.solve(u1, uamb)
    x2, u2, x2_des, x1, u1, x1_des, xamb, uamb, xamb_des = br2.get_solution()
    print(br2.solution.value(br2.slack_cost))
    print("2", br2.solution.value(br2.car1MPC.total_cost()))
    br2.save_state(folder + "data/" + '%03d'%ibr_sub_it + "_2", x1, u1, x1_des, x2, u2, x2_des, xamb, uamb, xamb_des)
    ibr_sub_it+=1

    br1 = mpc.IterativeBestResponseMPC(x1_MPC, x2_MPC, amb_MPC)
    br1.generate_optimization(N, T,  x0, x0_2, x0_amb, 5, slack=True)
    if WARM:
        br1.opti.set_initial(br1.u_opt, u1) 
        br1.opti.set_initial(br1.x_opt, x1)                 
        br1.opti.set_initial(br1.x_desired, x1_des)                 
    br1.solve(u2, uamb)
    x1, u1, x1_des, x2, u2, x2_des, xamb, uamb, xamb_des = br1.get_solution()
    print(br1.solution.value(br1.slack_cost))
    print("1", br1.solution.value(br1.car1MPC.total_cost()))
    br1.save_state(folder + "data/" + '%03d'%ibr_sub_it + "_1", x1, u1, x1_des, x2, u2, x2_des, xamb, uamb, xamb_des)
    ibr_sub_it+=1



