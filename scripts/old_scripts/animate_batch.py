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

import gc

if __name__ == "__main__":
    subdir_name = "20200224-154959slack"
    folder = "results/" + subdir_name + "/"

    dt = 0.2
    x1_MPC = mpc.MPC(dt)

    world = tw.TrafficWorld(2, 0, 1000)
    gc.enable()
    for ibr_sub_it_plot in range(0, 13): 
        gc.collect()
        # for ibr_sub_it in range(1, 40):

        ibr_prefix = '%03d'%ibr_sub_it_plot
        x1, u1, x1_des, x2, u2, x2_des, xamb, uamb, xamb_des = mpc.load_state(folder + "data/" + ibr_prefix)
        CIRCLES = False
        cplot.plot_cars_multiproc(world, x1_MPC, x1, x2, xamb, folder)


        CIRCLES = False
        if CIRCLES:
            vid_fname = folder + "vids/" + 'circle_' + subdir_name + ibr_prefix + '.mp4'
        else:
            vid_fname = folder + "vids/" + 'car_' + subdir_name + ibr_prefix +'.mp4'    
        if os.path.exists(vid_fname):
            os.remove(vid_fname)
        cmd = 'ffmpeg -r 16 -f image2 -i {}imgs/%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {}'.format(folder, vid_fname)
        os.system(cmd)
        print('Saving video to: {}'.format(vid_fname))
