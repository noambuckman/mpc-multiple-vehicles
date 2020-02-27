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


subdir_name = "20200224-223223maxiter9999"
folder = "results/" + subdir_name + "/"
################ Analyze Results

fig, ax = plt.subplots(5,1)
fig.set_figheight(12)
fig.set_figwidth(12)
ax_t = ax[0]
ax_u0 = ax[1]
ax_u1 = ax[2]
ax_usum = ax[3]
ax_xmax = ax[4]

all_amb = np.zeros((6,101, 16))
all_x1 = np.zeros((6,101, 16))
all_x2 = np.zeros((6,101, 16))



initial = True
for ibr_sub_it in range(1, 47):
    if (ibr_sub_it % 3) == 0:
        response_car = "1"
    elif (ibr_sub_it % 3) == 1:
        response_car = "a"
    else:
        response_car = "2"
        
    if response_car == "a":
        ibr_prefix =  'data/' + '%03d'%ibr_sub_it + "_" + response_car
            
        x1, u1, x1_des, x2, u2, x2_des, xamb, uamb, xamb_des = mpc.load_state(folder + ibr_prefix)
        if initial:
            prev_u = uamb
            initial = False
        
        change_in_u = uamb - prev_u
        mag_change_in_u = (change_in_u**2).sum()
        
        ax_t.plot(xamb[0,:],xamb[1,:],'.')
        ax_u0.plot(uamb[0,:],'.')
        ax_u1.plot(uamb[1,:],'.')
        ax_usum.plot(ibr_sub_it, mag_change_in_u,'o')
        ax_xmax.plot(ibr_sub_it, xamb[0,-1],'p',color='red')
        ax_xmax.plot(ibr_sub_it, x1[0,-1],'p',color='green')
        ax_xmax.plot(ibr_sub_it, x2[0,-1],'p',color='blue')

        prev_u = uamb
        all_amb[:,:,int(ibr_sub_it/3)] = xamb        
        all_x1[:,:,int(ibr_sub_it/3)] = x1        
        all_x2[:,:,int(ibr_sub_it/3)] = x2        
        

ax_t.set_ylabel('y [m]')
ax_t.set_xlabel('x [m]')

ax_u0.set_ylabel('u0')
ax_u0.set_xlabel('t [100ms]')

ax_u1.set_ylabel('u1')
ax_u1.set_xlabel('t [100ms]')

ax_usum.set_ylabel(r'$\Delta u^2$')
ax_usum.set_xlabel('ibr iteration')

ax_xmax.set_ylabel(r'$x_{\max}$')
ax_xmax.set_xlabel('ibr iteration')
# ax_xmax.set_ylim([80,125])

if not os.path.exists(folder  + 'plots/'):
    os.mkdir(folder  + 'plots/')
# plt.show()
fig.savefig(folder  + 'plots/evolution_plot.png', dpi=96, transparent=False, bbox_inches="tight",pad_inches=0)


plt.figure()
plt.plot(all_x1[0, -1, 1:],'-o', color='green')
plt.plot(all_x2[0, -1, 1:],'-o', color='blue')
plt.plot(all_amb[0, -1, 1:],'-o', color='red')
plt.hlines(np.mean(all_x1[0, -1, 1:]),1, 16, linestyles='--', color='green')
plt.hlines(np.mean(all_x2[0, -1, 1:]),1, 16, linestyles='--', color='blue')
plt.hlines(np.mean(all_amb[0, -1, 1:]),1, 16, linestyles='--', color='red')
fig.savefig(folder  + 'plots/comparing_x_final.png', dpi=96, transparent=False, bbox_inches="tight",pad_inches=0)



plt.show()