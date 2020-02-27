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

subdir_name = "20200226_201330shorter_window_running"
folder = "results/" + subdir_name + "/"
T = 5
# N = int(T/dt) #Number of control intervals

n_other = 4

ibr_sub_it=1


n_total_round = 6



# all_other_x0 = []
for time_step in range(5):
    for n_round in range(n_total_round):        
        ibr_sub_it +=1

        for i in range(n_other):            
            ibr_sub_it+=1


            suffixes = ["amb_costs_list", "car1_costs_list", "other_svo_cost", "svo_cost", "total_svo_cost", "u0", "u1", "u2", "u3",
                        "x0", "x1", "x2", "x3", "uamb", "xamb", "x_des0", "x_des1", "x_des2", "x_des3", "xamb_des" ]
            
            for s in suffixes:
                file_name = folder + "data/'%d %03d"%(time_step, ibr_sub_it) + s + ".npy'"
                new_file_name = folder + "data/%d_%03d"%(time_step, ibr_sub_it) + s + ".npy"

                cmd = "mv %s %s"%(file_name, new_file_name)     
                print(cmd)
                os.system(cmd)
            # raise Exception
    
        # if WARM:
        #     br1.opti.set_initial(br1.u_opt, u1) 
        #     br1.opti.set_initial(br1.x_opt, x1)                 
        #     br1.opti.set_initial(br1.x_desired, x1_des)                 


