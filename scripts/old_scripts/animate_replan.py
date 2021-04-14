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
import src.car_plotting_multiple as cmplot
import src.TrafficWorld as tw
np.set_printoptions(precision=2)

import src.IterativeBestResponseMPCMultiple as mibr

import gc

if __name__ == "__main__":
    subdir_name = "20200226_201330shorter_window_running"
    folder = "results/" + subdir_name + "/"
    PLOT_ALL = True
    CIRCLES = True
    dt = 0.1
    x_MPC = mpc.MPC(dt)
    x_MPC.n_circles = 3
    world = tw.TrafficWorld(2, 0, 1000)
    gc.enable()


    n_other_cars = 4

    time_step = 3
    ibr_sub_it_plot = 70
    # for time_step in range(5):
    #     for ibr_sub_it_plot in range(30,31): 
    gc.collect()
    # for ibr_sub_it in range(1, 40):
    if (ibr_sub_it_plot % (n_other_cars+1)) == 1:
        response_car = "a"
    else:
        response_car = "o"

    if response_car == "a" or PLOT_ALL:
        ibr_prefix = '%d_%03d'%(time_step,ibr_sub_it_plot) 

        if CIRCLES:
            vid_fname = folder + "vids/" + 'circle_' + subdir_name + ibr_prefix + '.mp4'
        else:
            vid_fname = folder + "vids/" + 'car_' + subdir_name + ibr_prefix +'.mp4'    
        if os.path.exists(vid_fname):
            # os.remove(vid_fname)      
            pass
        else:
            try:
                xamb, uamb, xamb_des, xothers, uothers, xothers_des = mibr.load_state(folder + "data/" + ibr_prefix, n_other_cars)
                cmplot.plot_cars(world, x_MPC, xamb, xothers, folder, 
                                None, None, CIRCLES, parallelize=True)


            
                cmd = 'ffmpeg -r 16 -f image2 -i {}imgs/%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {}'.format(folder, vid_fname)
                os.system(cmd)
                print('Saving video to: {}'.format(vid_fname))
            except FileNotFoundError:
                print("File", ibr_prefix, "Missing")
