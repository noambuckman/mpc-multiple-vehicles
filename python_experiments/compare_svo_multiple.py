import datetime
import os, sys
import numpy as np
import matplotlib.pyplot as plt

import casadi as cas

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
import src.IterativeBestResponseMPCMultiple as mibr
import pickle

SAVE = False
PLOT = False
rounds_ibr = 225
n_other_cars = 4
N = 50
###### LATEX Dimensions (Not currently Working)

fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height =fig_width*golden_mean       # height in inches
fig_size = [fig_width,fig_height]
fig_size = [6, 4]
#################33


def find_t_final(x, goal_x):
    i_upper = np.searchsorted(x[0,:], goal_x)
    i_lower = i_upper - 1
    dt = 0.2
    # if i_upper >= x.shape[1]:
    #     print(i_upper, x[0,i_lower])
    # print("Check: %.03f < %.03f"%(x[0,i_lower], goal_x))
    t_lower = i_lower*dt
    x_lower = x[0, i_lower]
    x_remaining = goal_x - x_lower

    v_x = np.cos(x[2, i_lower]) * x[4, i_lower]


    t_remaining = x_remaining / v_x
    t_final = t_lower + t_remaining
    # print("%.03f %.03f"%(t_lower, t_final))
    return t_final


#### STEP 1:  Sort all the files into the correct SVO

all_subdir = [
"20200301_215332random_ego",
"20200301_215346random_pro",
"20200301_215432random_altru",
"20200301_215520random_pro",
"20200301_215526random_altru",
"20200301_215537random_ego",
"20200301_215551random_pro",
"20200301_215602random_altru",
"20200301_215608random_ego",
"20200301_215623random_pro",
"20200301_215629random_altru",
"20200301_215636random_ego",
"20200301_215652random_pro",
"20200301_215658random_altru",
"20200301_215703random_ego",
"20200301_215713random_pro",
"20200301_215724random_altru",
"20200301_215742random_ego",
"20200301_215751random_pro",
"20200301_215757random_altru",
"20200301_215806random_ego",

"20200302_104840random_1p",
"20200302_104913random_2p",
"20200302_104916random_3p",
"20200302_104920random_4p",
"20200302_104926random_1e",
"20200302_104941random_2e",
"20200302_104946random_3e",
"20200302_105002random_4e",
"20200302_105059random_1a",
"20200302_105101random_2a",
"20200302_105104random_3a",
"20200302_105108random_4a",
"20200302_114834random_5e",
"20200302_114839random_6e",
"20200302_114841random_7e",
"20200302_114844random_8e",
"20200302_114853random_5p",
"20200302_114856random_6p",
"20200302_114859random_7p",
"20200302_114902random_8p",
"20200302_114909random_5a",
"20200302_114912random_6a",
"20200302_114914random_7a",
"20200302_114916random_8a",

"20200227_133704less_kxdotlarger",
"20200228_114359random_pro",
"20200228_114437random_pro",
"20200228_114440random_pro",
"20200228_114443random_pro",
"20200228_114448random_pro",
"20200228_114450random_pro",
"20200228_114913random_pro",
"20200228_114914random_pro",
"20200228_114916random_pro",
"20200228_114917random_pro",
"20200227_142916pi_01_ego",
"20200228_114517random_ego",
"20200228_114518random_ego",
"20200228_114528random_ego",
"20200228_114532random_ego",
"20200228_114547random_ego",
"20200228_114551random_ego",
"20200228_114803random_ego",
"20200228_114805random_ego",
"20200228_114806random_ego",
"20200227_141954pi2_5altru",
"20200228_114501random_altru",
"20200228_114503random_altru",
"20200228_114505random_altru",
"20200228_114506random_altru",
"20200228_114507random_altru",
"20200228_114509random_altru",
"20200228_114850random_altru",
"20200228_114851random_altru",
"20200228_114852random_altru",
]

subdir_name_prosocial_list = []
subdir_name_ego_list = []
subdir_name_altruistic_list = []
altr_theta = []
ego_theta = []
pro_theta = []
NO_GRASS = False
world = tw.TrafficWorld(2, 0, 1000)
for subdir in all_subdir:
    try:
        file_name = "results/" + subdir+"/data/"+"mpc3.p"
        mpc = pickle.load(open(file_name,'rb'))
        if mpc.min_y < -999999 or mpc.max_y > 9999999:
            print("Messed up ymin/max", file_name)
            continue
        elif mpc.min_y > world.y_min + 0.000001:
            print("Grass is NOT allowed!", file_name) 
            if not NO_GRASS:
                print("Too grass lmmited, ignored", file_name)
                continue
        elif mpc.min_y <= world.y_min + 0.00001:
            print("Grass is allowed!", file_name)
            if NO_GRASS:
                print("NO Grass, dataset ignored", file_name)
                continue
        if mpc.theta_iamb > np.pi/3:
            subdir_name_altruistic_list += [subdir]
            altr_theta += [mpc.theta_iamb]
        elif mpc.theta_iamb <= np.pi/6.0:
            subdir_name_ego_list += [subdir]
            ego_theta += [mpc.theta_iamb]
        else:
            subdir_name_prosocial_list += [subdir]
            pro_theta += [mpc.theta_iamb]
    except FileNotFoundError:
        print("Not found:", file_name)
print("Atruistic np.pi/2 = 1.5ish")
print(subdir_name_altruistic_list)
print(altr_theta)

print("Egoistic 0")
print(subdir_name_ego_list)
print(ego_theta)

print("Pro-Social", np.pi/2)
print(subdir_name_prosocial_list)
print(pro_theta)
# subdir_name_prosocial_list = [
# "20200227_133704less_kxdotlarger",
# "20200228_114359random_pro",
# "20200228_114437random_pro",
# "20200228_114440random_pro",
# "20200228_114443random_pro",
# "20200228_114448random_pro",
# "20200228_114450random_pro",
# "20200228_114913random_pro",
# "20200228_114914random_pro",
# "20200228_114916random_pro",
# "20200228_114917random_pro",
# ]



# subdir_name_prosocial = "20200227_133704less_kxdotlarger"    
# folder_prosocial = "results/" + subdir_name_prosocial + "/"

# subdir_name_ego_list = [
# "20200227_142916pi_01_ego",
# "20200228_114517random_ego",
# "20200228_114518random_ego",
# "20200228_114528random_ego",
# "20200228_114532random_ego",
# "20200228_114547random_ego",
# "20200228_114551random_ego",
# "20200228_114803random_ego",
# "20200228_114805random_ego",
# "20200228_114806random_ego",
# ]
# subdir_name_ego = "20200227_142916pi_01_ego"    
# folder_ego = "results/" + subdir_name_ego + "/"

# subdir_name_altruistic_list = [
# "20200227_141954pi2_5altru",
# "20200228_114501random_altru",
# "20200228_114503random_altru",
# "20200228_114505random_altru",
# "20200228_114506random_altru",
# "20200228_114507random_altru",
# "20200228_114509random_altru",
# "20200228_114850random_altru",
# "20200228_114851random_altru",
# "20200228_114852random_altru"]
# subdir_name_altruistic = "20200227_141954pi2_5altru"    
# folder_altruistic = "results/" + subdir_name_altruistic + "/"
################ Analyze Results
all_xamb_pro = []
all_uamb_pro = []
all_other_x_pro = []
all_other_u_pro = []
ibr_brounds_array_pro = [] 
all_xamb_ego = []
all_uamb_ego = []
all_other_x_ego = []
all_other_u_ego = []
ibr_brounds_array_ego = []
all_xamb_altru = []
all_uamb_altru = []
all_other_x_altru = []
all_other_u_altru = []
ibr_brounds_array_altru = []   
all_tfinalamb_pro = []
all_tfinalamb_ego = []
all_tfinalamb_altru = []

for sim_i in range(3):
    if sim_i==0:
        subdir_name_list = subdir_name_prosocial_list
    elif sim_i==1:
        subdir_name_list = subdir_name_ego_list
    else:
        subdir_name_list = subdir_name_altruistic_list
    for folder in subdir_name_list:
        n_full_rounds = 0  # rounods that the ambulance planned 
        n_all_rounds = 0

        all_xamb = np.zeros((6, N+1, rounds_ibr))
        all_uamb = np.zeros((2, N, rounds_ibr))
        all_xcost = np.zeros((3, rounds_ibr))
        all_tfinalamb = np.zeros((1, rounds_ibr))


        all_other_x = [np.zeros((6, N+1, rounds_ibr)) for i in range(n_other_cars)]
        all_other_u = [np.zeros((2, N, rounds_ibr)) for i in range(n_other_cars)]
        all_other_cost = [np.zeros((3, rounds_ibr)) for i in range(n_other_cars)]
        all_other_tfinal = [np.zeros((1, rounds_ibr)) for i in range(n_other_cars)]
        for amb_ibr_i in range(rounds_ibr):
            if (amb_ibr_i % (n_other_cars + 1) == 1) and amb_ibr_i>51: # We only look at sims when slack activated
                ibr_prefix =  '%03d'%amb_ibr_i
                try:
                    xamb, uamb, xamb_des, xothers, uothers, xothers_des = mibr.load_state("results/" + folder + "/" + "data/" + ibr_prefix, n_other_cars)
                    all_xamb[:,:,n_full_rounds] = xamb
                    all_uamb[:,:,n_full_rounds] = uamb
                    x_goal = 130
                    all_tfinalamb[:, n_full_rounds] = find_t_final(xamb, x_goal)
                    for i in range(n_other_cars):
                        all_other_x[i][:,:,n_full_rounds] = xothers[i] 
                        all_other_u[i][:,:,n_full_rounds] = uothers[i]
                        # all_other_tfinal[i][:,n_full_rounds] = find_t_final(xothers[i], 120)
                    n_full_rounds += 1
                except FileNotFoundError:
                    # print("amb_ibr_i %d missing"%amb_ibr_i)
                    pass
                n_all_rounds += 1

        ### Clip the extra dimension
        all_xamb = all_xamb[:,:,:n_full_rounds]
        all_uamb = all_uamb[:,:,:n_full_rounds]
        all_tfinalamb = all_tfinalamb[:,:n_full_rounds]
        for i in range(n_other_cars):
            all_other_x[i] = all_other_x[i][:,:,:n_full_rounds]
            all_other_u[i] = all_other_u[i][:,:,:n_full_rounds]
        ibr_brounds_array = np.array(range(1, n_full_rounds +1))

        if n_full_rounds > 0 : # only save those that meet slack requirement
            if sim_i==0: #prosocial directory
                all_xamb_pro += [all_xamb]
                all_uamb_pro += [all_uamb]
                all_other_x_pro += [all_other_x]
                all_other_u_pro += [all_other_u]
                ibr_brounds_array_pro += [ibr_brounds_array]  
                all_tfinalamb_pro += [all_tfinalamb]
            elif sim_i==1: #egoistic directory
                all_xamb_ego += [all_xamb]
                all_uamb_ego += [all_uamb]
                all_other_x_ego += [all_other_x]
                all_other_u_ego += [all_other_u]
                ibr_brounds_array_ego += [ibr_brounds_array]  
                all_tfinalamb_ego += [all_tfinalamb]
            else: #altruistic directory
                all_xamb_altru += [all_xamb]
                all_uamb_altru += [all_uamb]
                all_other_x_altru += [all_other_x]
                all_other_u_altru += [all_other_u]
                ibr_brounds_array_altru += [ibr_brounds_array]    
                all_tfinalamb_altru += [all_tfinalamb]
        else:
            print("No slack eligible", folder)



### SAVING IN PROSOCIAL'S DIRECTORy
folder = "random"   #<----

fig_trajectory, ax_trajectory = plt.subplots(1,1)
ax_trajectory.set_title("Ambulance Trajectories")
# fig_trajectory.set_figheight(fig_height)
# fig_trajectory.set_figwidth(fig_width)
fig_trajectory.set_size_inches((8,6))
print(len(all_xamb_pro))
print(all_xamb_pro[0].shape)
ax_trajectory.plot(all_xamb_pro[0][0,:,-1], all_xamb_pro[0][1,:,-1], '-o', label="Prosocial")
ax_trajectory.plot(all_xamb_ego[0][0,:,-1], all_xamb_ego[0][1,:,-1], '-o', label="Egoistic")
ax_trajectory.plot(all_xamb_altru[0][0,:,-1], all_xamb_altru[0][1,:,-1], '-o', label="Altruistic")

ax_trajectory.set_xlabel("X [m]")
ax_trajectory.set_ylabel("Y [m]")
if SAVE:
    fig_file_name = folder + 'plots/' + 'cfig1_amb_trajectory.eps'
    fig_trajectory.savefig(fig_file_name, dpi=95, format='eps')
    print("Save to....", fig_file_name)

##########################################333333

svo_labels = ["Egoistic", "Prosocial", "Altruistic"]

fig_uamb, ax_uamb = plt.subplots(3,1)
fig_uamb.set_size_inches((8,8))
fig_uamb.suptitle("Ambulance Control Input over IBR Iterations")
# ax_uamb[0].plot(ibr_brounds_array, np.sum(all_uamb[0,:,:] * all_uamb[0,:,:], axis=0), '-o')
ax_uamb[0].bar(range(3), [
                np.mean([np.sum(all_x[0,:,-1] * all_x[0,:,-1],axis=0) for all_x in all_uamb_ego]),
                np.mean([np.sum(all_x[0,:,-1] * all_x[0,:,-1],axis=0) for all_x in all_uamb_pro]),
                np.mean([np.sum(all_x[0,:,-1] * all_x[0,:,-1],axis=0) for all_x in all_uamb_altru])]
)

# ax_uamb[0].set_xlabel("IBR Iteration")
ax_uamb[0].set_ylabel(r"$\sum u_{\delta}^2$")
ax_uamb[0].set_xticks(range(3))
ax_uamb[0].set_xticklabels(svo_labels)    

ax_uamb[1].bar(range(3), [
                np.mean([np.sum(all_x[1,:,-1] * all_x[1,:,-1],axis=0) for all_x in all_uamb_ego]),
                np.mean([np.sum(all_x[1,:,-1] * all_x[1,:,-1],axis=0) for all_x in all_uamb_pro]),
                np.mean([np.sum(all_x[1,:,-1] * all_x[1,:,-1],axis=0) for all_x in all_uamb_altru])]
)
# ax_uamb[1].set_xlabel("IBR Iteration")
ax_uamb[1].set_ylabel(r"$\sum u_{v}^2$")
ax_uamb[1].set_xticks(range(3))
ax_uamb[1].set_xticklabels(svo_labels)    

# ax_uamb[2].bar(range(3), [
#                 np.sum(all_uamb_ego[0,:,-1] * all_uamb_ego[0,:,-1],axis=0) + np.sum(all_uamb_ego[1,:,-1] * all_uamb_ego[1,:,-1],axis=0),
#                 np.sum(all_uamb_pro[0,:,-1] * all_uamb_pro[1,:,-1], axis=0) + np.sum(all_uamb_pro[1,:,-1] * all_uamb_pro[1,:,-1], axis=0),
#                 np.sum(all_uamb_altru[0,:,-1] * all_uamb_altru[0,:,-1],axis=0) + np.sum(all_uamb_altru[1,:,-1] * all_uamb_altru[1,:,-1],axis=0)],)
# ax_uamb[2].set_xlabel("Vehicles' Social Value Orientation")
# ax_uamb[2].set_ylabel("$\sum ||u||^2$")   

ax_uamb[1].set_xticks(range(3))
ax_uamb[1].set_xticklabels(svo_labels)
if SAVE:
    fig_file_name = folder + 'plots/' + 'cfig2_amb_ctrl_iterations.eps'
    fig_uamb.savefig(fig_file_name, dpi=95, format='eps')
    print("Save to....", fig_file_name)
    

##########################################################
#### Convergence
#########################################################
fig_reluamb, ax_reluamb = plt.subplots(2,1)
# fig_reluamb.set_figheight(fig_height)
# fig_reluamb.set_figwidth(fig_width)
fig_reluamb.set_size_inches((8,6))
for sim_i in range(3):
    if sim_i==0: #prosocial directory
        all_uamb = all_uamb_ego
        label = "Egoistic"
        ibr_brounds_array = ibr_brounds_array_ego        
    elif sim_i==1: #egoistic directory
        all_uamb = all_uamb_pro 
        label = "Prosocial"
        ibr_brounds_array = ibr_brounds_array_pro 
    else: #altruistic directory
        all_uamb = all_uamb_altru
        all_other_u = all_other_u_altru
        label = "Altruistic"
        ibr_brounds_array = ibr_brounds_array_altru       
    ax_reluamb[0].plot(ibr_brounds_array[0][1:], np.sum((all_uamb[0][0,:,1:]-all_uamb[0][0,:,0:-1])*(all_uamb[0][0,:,1:]-all_uamb[0][0,:,0:-1]), axis=0), '-o', label=label)
    ax_reluamb[1].plot(ibr_brounds_array[0][1:], np.sum((all_uamb[0][1,:,1:]-all_uamb[0][1,:,0:-1])*(all_uamb[0][1,:,1:]-all_uamb[0][1,:,0:-1]), axis=0), '-o', label=label)

ax_reluamb[0].set_ylabel("$\sum (u_{v\delta,t}-u_{\delta,t-1})^2$")
ax_reluamb[1].set_xlabel("IBR Iteration")
ax_reluamb[1].set_ylabel("$\sum (u_{v,t}-u_{v,t-1})^2$")
ax_reluamb[0].legend()
ax_reluamb[1].legend()
fig_reluamb.suptitle("Change in Ambulance Control Input over IBR Iterations")

if SAVE:
    fig_file_name = folder + 'plots/' + 'cfig3_change_amb_ctrl_iterations.eps'
    fig_reluamb.savefig(fig_file_name, dpi=95, format='eps')
    print("Save to....", fig_file_name)

###################################################################3
##################################################################
fig_xfinal, ax_xfinal = plt.subplots(2,1)
fig_xfinal.suptitle("Final Ambulance State Over Iterations")
fig_xfinal.set_size_inches((8,6))
# fig_xfinal.set_figheight(fig_height)
# fig_xfinal.set_figwidth(fig_width)

for sim_i in range(3):
    if sim_i==0: #prosocial directory
        all_uamb = all_uamb_ego
        all_xamb = all_xamb_ego
        all_other_x = all_other_x_ego
        label = "Egoistic"
        ibr_brounds_array = ibr_brounds_array_ego        
    elif sim_i==1: #egoistic directory
        all_uamb = all_uamb_pro 
        all_xamb = all_xamb_pro
        all_other_x = all_other_x_pro
        label = "Prosocial"
        ibr_brounds_array = ibr_brounds_array_pro 
    else: #altruistic directory
        all_uamb = all_uamb_altru
        all_xamb = all_xamb_altru
        all_other_x = all_other_x_altru
        all_other_u = all_other_u_altru
        label = "Altruistic"        
        ibr_brounds_array = ibr_brounds_array_altru  
    
    ax_xfinal[0].plot(ibr_brounds_array[0], all_xamb[0][0,-1,:], '-o', label=label)
    ax_xfinal[1].plot(ibr_brounds_array[0], all_xamb[0][2,-1,:], '-o', label=label)

# ax_reluamb[0].set_xlabel("IBR Iteration")

ax_xfinal[0].set_ylabel("$x_{final}$")
ax_xfinal[0].legend()
ax_xfinal[1].set_xlabel("IBR Iteration")
ax_xfinal[1].set_ylabel(r"$\Theta_{final}$")
ax_xfinal[1].legend()
if SAVE:
    fig_file_name = folder + 'plots/' + 'cfig4_iterations_ambperformance.eps'
    fig_xfinal.savefig(fig_file_name, dpi=95, format='eps')
    print("Save to....", fig_file_name)

################################################################################
###################### NOW PLOTTING THE OTHER VEHICLES #########################

fig_xfinal_all, ax_xfinal_all = plt.subplots(3,1)
fig_xfinal_all.suptitle("Comparing Distance Travel for the Vehicles")
fig_xfinal_all.set_size_inches((8,8))
# fig_xfinal_all.set_figheight(fig_height)
# fig_xfinal_all.set_figwidth(fig_width)

for sim_i in range(3):
    if sim_i==0: #prosocial directory
        all_uamb = all_uamb_ego
        all_xamb = all_xamb_ego
        all_other_x = all_other_x_ego
        label = "Egoistic"
        ibr_brounds_array = ibr_brounds_array_ego        
    elif sim_i==1: #egoistic directory
        all_uamb = all_uamb_pro 
        all_xamb = all_xamb_pro
        all_other_x = all_other_x_pro
        label = "Prosocial"
        ibr_brounds_array = ibr_brounds_array_pro 
    else: #altruistic directory
        all_uamb = all_uamb_altru
        all_xamb = all_xamb_altru
        all_other_x = all_other_x_altru
        all_other_u = all_other_u_altru
        label = "Altruistic"        
        ibr_brounds_array = ibr_brounds_array_altru  

    bar_width = 0.5
    inter_car_width = 2*bar_width

    width_offset = bar_width*sim_i
    
    ticks = [width_offset + (2*bar_width + inter_car_width)*c for c in range(n_other_cars + 1)]


    # print(len(all_ither_x))

    # ax_xfinal_all[0].bar(ticks, 
    # [np.mean([all_x[0, -1, -1] - all_x[0, 0, -1] for all_x in all_xamb])] + [np.mean(all_o_x[i][0,-1,-1] - all_o_x[i][0,0,-1]) for i in range(n_other_cars) for all_o_x in all_other_x],
    # bar_width, label=label)    
    # ax_xfinal_all[0].set_xticks(range(n_other_cars + 1))
    # ax_xfinal_all[0].set_xticklabels(["A"] + [str(i) for i in range(1, n_other_cars+1)])

    # ax_xfinal_all[1].bar(ticks,  
    # [all_xamb[-1, -1, -1] - all_xamb[-1, 0, -1]] + [all_other_x[i][-1,-1,-1] - all_other_x[i][-1,0,-1] for i in range(n_other_cars)],
    # bar_width, label=label)
    # # ax_xfinal_all[1].set_xticks(range(n_other_cars + 1))
    # # ax_xfinal_all[1].set_xticklabels(["A"] + [str(i) for i in range(1, n_other_cars+1)])

    # ax_xfinal_all[2].bar(ticks,  
    # [np.sum(all_xamb[2,:,-1]*all_xamb[2,:,-1])] +  [np.sum(all_other_x[i][2,:,-1]*all_other_x[i][2,:,-1]) for i in range(n_other_cars)],
    # bar_width, label=label)

width_offset = bar_width*1
ticks = [width_offset + (2*bar_width + inter_car_width)*c for c in range(n_other_cars + 1)]

ax_xfinal_all[2].legend()
ax_xfinal_all[2].set_xticks(ticks)
ax_xfinal_all[2].set_xticklabels(["A"] + [str(i) for i in range(1, n_other_cars+1)])

ax_xfinal_all[0].set_ylabel("Horizontal Displacement $\Delta x$")
ax_xfinal_all[0].legend()
ax_xfinal_all[0].set_xticks(ticks)
ax_xfinal_all[0].set_xticklabels(["A"] + [str(i) for i in range(1, n_other_cars+1)])

ax_xfinal_all[1].set_ylabel("Total Distance $s_f - s_i$")
ax_xfinal_all[1].legend()
ax_xfinal_all[1].set_xticks(ticks)
ax_xfinal_all[1].set_xticklabels(["A"] + [str(i) for i in range(1, n_other_cars+1)])

ax_xfinal_all[2].set_ylabel("Angular Deviation $\sum_{t} \Theta_t^2$")

if SAVE:
    fig_file_name = folder + 'plots/' + 'cfig5_vehicles_comparison.eps'
    fig_xfinal_all.savefig(fig_file_name, dpi=95, format='eps')
    print("Save to....", fig_file_name)


#########################Let's Reproduce the Table ####################33
print("Amb X     Final   Avg.  Min.   Max.  ")

final_metric_ego = [all_x[0,-1,-1] for all_x in all_xamb_ego]
final_metric_pro = [all_x[0,-1,-1] for all_x in all_xamb_pro]
final_metric_altru = [all_x[0,-1,-1] for all_x in all_xamb_altru]

# print("Egoistic  & %.02f & %.02f & %.02f & %.02f"%(all_xamb_ego[0,-1,-1], np.mean(all_xamb_ego[0,-1,:]), np.min(all_xamb_ego[0,-1,:]), np.max(all_xamb_ego[0,-1,:])))
# print("Prosocial  & %.02f & %.02f & %.02f & %.02f"%(all_xamb_pro[0,-1,-1], np.mean(all_xamb_pro[0,-1,:]), np.min(all_xamb_pro[0,-1,:]), np.max(all_xamb_pro[0,-1,:])))
# print("Altruistic  & %.02f & %.02f & %.02f & %.02f"%(all_xamb_altru[0,-1,-1], np.mean(all_xamb_altru[0,-1,:]), np.min(all_xamb_altru[0,-1,:]), np.max(all_xamb_altru[0,-1,:])))
print("Egoistic  & %.02f (%.02f) & %.02f & %.02f"%(np.mean(final_metric_ego), np.std(final_metric_ego), np.min(final_metric_ego), np.max(final_metric_ego)))
print("Prosocial & %.02f (%.02f) & %.02f & %.02f"%(np.mean(final_metric_pro), np.std(final_metric_pro), np.min(final_metric_pro), np.max(final_metric_pro)))
print("Altruistic & %.02f (%.02f) & %.02f & %.02f"%(np.mean(final_metric_altru), np.std(final_metric_altru), np.min(final_metric_altru), np.max(final_metric_altru)))

final_metric_ego = [t_final[:,-1] for t_final in all_tfinalamb_ego]
final_metric_pro = [t_final[:,-1] for t_final in all_tfinalamb_pro]
final_metric_altru = [t_final[:,-1] for t_final in all_tfinalamb_altru]
# print(all_tfinalamb_ego[0].shape)
# print(final_metric_ego)
# print(final_metric_ego.shape)

# print("Egoistic  & %.02f & %.02f & %.02f & %.02f"%(all_xamb_ego[0,-1,-1], np.mean(all_xamb_ego[0,-1,:]), np.min(all_xamb_ego[0,-1,:]), np.max(all_xamb_ego[0,-1,:])))
# print("Prosocial  & %.02f & %.02f & %.02f & %.02f"%(all_xamb_pro[0,-1,-1], np.mean(all_xamb_pro[0,-1,:]), np.min(all_xamb_pro[0,-1,:]), np.max(all_xamb_pro[0,-1,:])))
# print("Altruistic  & %.02f & %.02f & %.02f & %.02f"%(all_xamb_altru[0,-1,-1], np.mean(all_xamb_altru[0,-1,:]), np.min(all_xamb_altru[0,-1,:]), np.max(all_xamb_altru[0,-1,:])))
print("Time To "+str(x_goal)+"m")
print("Egoistic  & %.02f (%.02f) & %.02f & %.02f %d"%(np.mean(final_metric_ego), np.std(final_metric_ego), np.min(final_metric_ego), np.max(final_metric_ego),len(final_metric_ego)))
print("Prosocial & %.02f (%.02f) & %.02f & %.02f %d"%(np.mean(final_metric_pro), np.std(final_metric_pro), np.min(final_metric_pro), np.max(final_metric_pro),len(final_metric_pro)))
print("Altruistic & %.02f (%.02f) & %.02f & %.02f %d"%(np.mean(final_metric_altru), np.std(final_metric_altru), np.min(final_metric_altru), np.max(final_metric_altru),len(final_metric_altru)))





print("Veh 1     Final   Avg.  Min.   Max.  ")
i = 0
veh_displace_ego = [all_other_x[i][0,-1,-1] - all_other_x[i][0,0,-1] for all_other_x in all_other_x_ego]
veh_displace_pro = [all_other_x[i][0,-1,-1] - all_other_x[i][0,0,-1] for all_other_x in all_other_x_pro]
veh_displace_altru = [all_other_x[i][0,-1,-1] - all_other_x[i][0,0,-1] for all_other_x in all_other_x_altru]

print(" ")
print("Egoistic  & %.02f (%.02f) & %.02f & %.02f"%(np.mean(veh_displace_ego), np.std(veh_displace_ego), np.min(veh_displace_ego), np.max(veh_displace_ego)))
print("Prosocial  & %.02f (%.02f) & %.02f  & %.02f "%(np.mean(veh_displace_pro),  np.std(veh_displace_pro), np.min(veh_displace_pro), np.max(veh_displace_pro)))
print("Altruistic  & %.02f (%.02f) & %.02f & %.02f "%( np.mean(veh_displace_altru),  np.std(veh_displace_altru), np.min(veh_displace_altru), np.max(veh_displace_altru)))


if PLOT:
    plt.show()


