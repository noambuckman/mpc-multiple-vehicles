#!/usr/bin/env python
# coding: utf-8


import numpy as np
np.set_printoptions(precision=5)
import matplotlib.pyplot as plt
import copy as cp
import sys, json, pickle
PROJECT_PATHS = ['/home/nbuckman/Dropbox (MIT)/DRL/2020_01_cooperative_mpc/mpc-multiple-vehicles/', '/Users/noambuckman/mpc-multiple-vehicles/']
for p in PROJECT_PATHS:
    sys.path.append(p)
import src.traffic_world as tw
import src.multiagent_mpc as mpc
import src.car_plotting_multiple as cmplot
import src.solver_helper as helper
import src.vehicle as vehicle
import tqdm
import argparse, glob


parser = argparse.ArgumentParser(description='Run iterative best response with SVO')
parser.add_argument('log_directory',type=str, default=None, help="Load log")
parser.add_argument('--end-mpc', type=int, default=-1, help="Number rounds of mpc data to plot")
parser.add_argument('--image-format',type=str, default="png", help="Options: svg, pdf, eps")
args = parser.parse_args()

log_directory = args.log_directory
if args.end_mpc == -1:
    # Find the last run.  We currently assumes that all runs start from 0 and increment by 1
    list_of_mpc_data = glob.glob(args.log_directory + 'data/all_*xamb.npy')
    rounds_mpc = len(list_of_mpc_data)
else:
    rounds_mpc = (args.end_mpc + 1)

# log_directory = "/home/nbuckman/mpc_results/0g5b-c3h6-20200924-145931/"
with open(log_directory + "params.json",'rb') as fp:
    params = json.load(fp)
for k in params:
    print(k,": ", params[k])


def load_ibr_results(i_mpc, i_ibr, log_directory, params, executed=True):
    file_name = log_directory + "data/"+'ibr_m%03di%03d'%(i_mpc, i_ibr)
    n_other = params["n_other"]
    xamb_ibr, uamb_ibr, xamb_des_ibr, all_other_x_ibr, all_other_u_ibr, all_other_x_des_ibr = mpc.load_state(file_name, n_other)
    if executed:
        n_exec = params["number_ctrl_pts_executed"]
        uamb_ibr = uamb_ibr[:, :n_exec]
        xamb_ibr = xamb_ibr[:, :n_exec+1]
        xamb_des_ibr = xamb_des_ibr[:, :n_exec+1]
        for j in range(len(all_other_x_ibr)):
            all_other_u_ibr[j] = all_other_u_ibr[j][:, :n_exec]
            all_other_x_ibr[j] = all_other_x_ibr[j][:, :n_exec+1]
            all_other_x_des_ibr[j] = all_other_x_des_ibr[j][:, :n_exec+1]
    return xamb_ibr, uamb_ibr, xamb_des_ibr, all_other_x_ibr, all_other_u_ibr, all_other_x_des_ibr

def u_tensor(rounds_mpc, log_directory, params, executed=True):
    '''Generate a tensor whose size is [2, N, mpc, ibr] for a given agent
    executed: True: Only return control inputs that are executed
              False:  Return all control inputs generated during IBR
    '''
    if executed:
        N = params["number_ctrl_pts_executed"]
    else:
        N = params["N"]
        
    U = np.zeros(shape=(2, N, rounds_mpc, params["n_ibr"]))
    Uother = np.zeros(shape=(2, N, rounds_mpc, params["n_ibr"], params["n_other"]))
    for i_mpc in tqdm.trange(rounds_mpc):
        for i_ibr in range(params["n_ibr"]):
            xamb_ibr, uamb_ibr, xamb_des_ibr, all_other_x_ibr, all_other_u_ibr, all_other_x_des_ibr = load_ibr_results(i_mpc, i_ibr, log_directory, params, executed)
            U[:, :, i_mpc, i_ibr] = uamb_ibr
            for j in range(params["n_other"]):
                Uother[:, :, i_mpc, i_ibr, j] = all_other_u_ibr[j]
    return U, Uother



# xamb_ibr, uamb_ibr, xamb_des_ibr, all_other_x_ibr, all_other_u_ibr, all_other_x_des_ibr = load_ibr_results(0, 0, log_directory, params)
U, Uother = u_tensor(rounds_mpc, log_directory, params)


# ## Defining Convergence Metric
# 
# $\delta(ibr) = \vec{u}_{mpc, ibr + 1} - \vec{u}_{mpc, ibr}$
# which has dimensions
# $ [2 x N] $
# 

# Let's first take a look at $\delta(ibr)[0]$ which is the convergence at the first control point


# delta_0 = np.diff(U[:,0, 0,:], axis=1)
# plt.plot(delta_0[0, :], label="Delta Steering")
# plt.plot(delta_0[1, :], label="Delta Velocity")
# plt.xlabel("Rounds Best Response")
# plt.ylabel('$\delta(u_i, u_{i+1})$')
# plt.legend()
# plt.title('Convergence of IBR, Ambulance, $i_{mpc}=0$')
# plt.savefig(log_directory + "plots/amb_conv_both_ctrl_mpc0")
# plt.close()


for j in range(params['n_other']):
    delta_0 = np.diff(Uother[:, 0, 0,:, j], axis=1)
    plt.plot(delta_0[0, :])
plt.xlabel("Rounds Best Response")
plt.ylabel('$\delta(u_i, u_{i+1})$')
plt.title('Convergence of All Agents, Steering, $i_{mpc}=0$')
# plt.show()
plt.savefig(log_directory + "plots/amb_conv_steering_i0" +"." + args.image_format)
plt.close()


# for j in range(params['n_other']):
#     delta_0 = np.diff(Uother[:, 1, 0,:,j], axis=1)
#     plt.plot(delta_0[0, :])
# plt.xlabel("Rounds Best Response")
# plt.ylabel('$\delta(u_i, u_{i+1})$')
# plt.title('Convergence of All Agents, Acceleration, $i_{mpc}=0$')
# plt.show()


i_mpc = 0
delta_U_mpc = np.diff(U[:, :, i_mpc, :], axis=2)
delta0_norm = np.linalg.norm(delta_U_mpc[0,:,:], axis=0)
delta1_norm = np.linalg.norm(delta_U_mpc[1,:,:], axis=0)
plt.plot(delta0_norm[:], label="Norm delta over N steer cntrl")
plt.plot(delta1_norm[:], label="Norm delta over N accel cntrl")
plt.title("Convergence for Ambulance at $i_{mpc}$=0")
plt.legend()
# plt.show()
plt.savefig(log_directory + "plots/amb_both_ctrl_i0" +"." + args.image_format)
plt.close()


for i_mpc in range(U.shape[2]):
    delta_U_mpc = np.diff(U[:, :, i_mpc, :], axis=2)
    delta0_norm = np.linalg.norm(delta_U_mpc[0,:,:], axis=0)
    plt.plot(delta0_norm[:])
plt.title("Convergence of Ambulance's Steering @ Each Round of MPC")
plt.xlabel("Rds of Best Response")
plt.ylabel("$|u^s_{t+1} - u^s_{t}|$")
plt.ylim([-.1, .1])
# plt.show()
plt.savefig(log_directory + "plots/amb_convergence_steering" +"." + args.image_format)
plt.close()

for i_mpc in range(U.shape[2]):
    delta_U_mpc = np.diff(U[:, :, i_mpc, :], axis=2)
    delta1_norm = np.linalg.norm(delta_U_mpc[1,:,:], axis=0)
    plt.plot(delta1_norm[:])
plt.ylim([-.1, .1])
plt.title("Convergence of Ambulance's Steering @ Each Round of MPC")
plt.xlabel("Rds of Best Response")
plt.ylabel("$|u^a_{t+1} - u^a_{t}|$")
# plt.show()
plt.savefig(log_directory + "plots/amb_convergence_acceleration" +"." + args.image_format)
plt.close()

grid_w = 3
grid_h = int(np.ceil(Uother.shape[4] / 3))
fig, axs = plt.subplots(grid_h, grid_w,sharex=True, sharey=True, figsize=(8,15))
fig.suptitle("Steering Convergence for All Agents")
for j in range(Uother.shape[4]):
    grid_i = j//grid_w
    grid_j = j%grid_w
#     print(grid_i, grid_j)
    axs[grid_i, grid_j].set_title("Ag %d"%j)
    for i_mpc in range(Uother.shape[2]):
        delta_U_mpc = np.diff(Uother[:, :, i_mpc, :, j], axis=2)
        delta0_norm = np.linalg.norm(delta_U_mpc[0,:,:], axis=0)
        axs[grid_i, grid_j].plot(delta0_norm[:])
# plt.title("Convergence of Ambulance's Steering @ Each Round of MPC")

# plt.xlabel("Rds of Best Response")
# plt.ylabel("$|u^s_{t+1} - u^s_{t}|$")
plt.tight_layout()
fig.text(0.5, 0.00, 'Rounds IBR', ha='center')
fig.text(0.0, 0.5, '$|u^s_{t+1} - u^s_{t}|$', va='center', rotation='vertical')
# plt.show()
plt.savefig(log_directory + "plots/" + "allagents_conv_steering" +"." + args.image_format)
plt.close()
