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



    subdirectory_list = [
        "20200227_133704less_kxdotlarger",
        "20200227_141954pi2_5altru",
        "20200227_142916pi_01_ego",
        "20200227_143225pi4_pulloversame",
        "20200227_143315pi2_5pulloversame",
        "20200227_145744pi2_5_nograss",
        "20200227_145800pi4_nograss",
        "20200227_145922pi_5dnograss",
        "20200227_161307pi4dnograss_slack",
        "20200227_16521120200227_161307pi4nograss2",
        "20200227_16523520200227_161307pi25altrunograss2",
        "20200227_16533320200227_161307pi5degaltrunograss2",
        # "20200227_165619constantV",
        # "20200227_170553constantHighSlack",
        # "20200227_171438constantVlargerLane",
        # "20200227_172912constantVNoSlack",
        # "20200227_173025constantVNoSlack",
        # "20200227_180757constantVSlackBigLane",
    ]
    for subdir_name in subdirectory_list:
        # subdir_name = "20200227_170553constantHighSlack"
        folder = "results/" + subdir_name + "/"
        PLOT_ALL = False
        CIRCLES = True
        dt = 0.1
        x_MPC = mpc.MPC(dt)
        x_MPC.n_circles = 3
        large_lane = 4.2
        world = tw.TrafficWorld(2, 0, 1000)
        gc.enable()


        n_other_cars = 4
        for ibr_sub_it_plot in range(1,200): 
            gc.collect()
            # for ibr_sub_it in range(1, 40):
            if (ibr_sub_it_plot % (n_other_cars+1)) == 1:
                response_car = "a"
            else:
                response_car = "o"

            if response_car == "a" or PLOT_ALL:
                ibr_prefix = '%03d'%ibr_sub_it_plot 

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
                        print(folder + "data/" + ibr_prefix)
                        print("File", ibr_prefix, "Missing")
                    # except RuntimeError:
                    #     print("R")
