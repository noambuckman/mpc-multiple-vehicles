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
# "20200228_114517random_ego",
# "20200228_114518random_ego",
# "20200228_114528random_ego",
# "20200228_114532random_ego",
# "20200228_114547random_ego",
# "20200228_114551random_ego",
# "20200228_114803random_ego",
# "20200228_114805random_ego",
# "20200228_114806random_ego",
# "20200228_114501random_altru",
# "20200228_114503random_altru",
# "20200228_114505random_altru",
# "20200228_114506random_altru",
# "20200228_114507random_altru",
# "20200228_114509random_altru",
# "20200228_114850random_altru",
# "20200228_114851random_altru",
# "20200228_114852random_altru",        
# "20200301_215332random_ego",
# "20200301_215346random_pro",
# "20200301_215432random_altru",
# "20200301_215520random_pro",
# "20200301_215526random_altru",
# "20200301_215537random_ego",
# "20200301_215551random_pro",
# "20200301_215602random_altru",
# "20200301_215608random_ego",
# "20200301_215623random_pro",
# "20200301_215629random_altru",
# "20200301_215636random_ego",
# "20200301_215652random_pro",
# "20200301_215658random_altru",
# "20200301_215703random_ego",
# "20200301_215713random_pro",
# "20200301_215724random_altru",
# "20200301_215742random_ego",
# "20200301_215751random_pro",
# "20200301_215757random_altru",
# "20200301_215806random_ego",
# "20200302_104840random_1p",
# "20200302_104913random_2p",
# "20200302_104916random_3p",
# "20200302_104920random_4p",
# "20200302_104926random_1e",
# "20200302_104941random_2e",
# "20200302_104946random_3e",
# "20200302_105002random_4e",
# "20200302_105059random_1a",
# "20200302_105101random_2a",
# "20200302_105104random_3a",
# "20200302_105108random_4a",
# "20200302_114834random_5e",
# "20200302_114839random_6e",
# "20200302_114841random_7e",
# "20200302_114844random_8e",
# "20200302_114853random_5p",
# "20200302_114856random_6p",
# "20200302_114859random_7p",
# "20200302_114902random_8p",
# "20200302_114909random_5a",
# "20200302_114912random_6a",
# "20200302_114914random_7a",
# "20200302_114916random_8a",
"20200303_182428running_ambonly",
    ]
    for subdir_name in subdirectory_list:
        # subdir_name = "20200227_170553constantHighSlack"
        folder = "results/" + subdir_name + "/"
        PLOT_ALL = False
        CIRCLES = True
        LAST = True
        dt = 0.2
        x_MPC = mpc.MPC(dt)
        x_MPC.n_circles = 3
        large_lane = 4.2
        world = tw.TrafficWorld(2, 0, 1000)
        gc.enable()

        largest_ibr = -1
        n_other_cars = 4
        for ibr_sub_it_plot in range(1,300): 
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
                try:
                    xamb, uamb, xamb_des, xothers, uothers, xothers_des = mibr.load_state(folder + "data/" + ibr_prefix, n_other_cars)
                    max_ibr = int(ibr_sub_it_plot)
                    print("Current Max", max_ibr)
                    if not LAST:
                        if os.path.exists(vid_fname):
                            # os.remove(vid_fname)      
                            pass     
                        else:                   
                            cmplot.plot_cars(world, x_MPC, xamb, xothers, folder, 
                                            None, None, CIRCLES, parallelize=True)
                            cmd = 'ffmpeg -r 16 -f image2 -i {}imgs/%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {}'.format(folder, vid_fname)
                            os.system(cmd)
                            print('Saving video to: {}'.format(vid_fname))
                except FileNotFoundError:
                    print("File", ibr_prefix, "Missing")
                    pass
                        # print(folder + "data/" + ibr_prefix)
                    # except RuntimeError:
                    #     print("R")
        if LAST:
            ibr_prefix = '%03d'%max_ibr
            print(max_ibr) 
            if CIRCLES:
                vid_fname = folder + "vids/" + 'circle_' + subdir_name + ibr_prefix + '.mp4'
            else:
                vid_fname = folder + "vids/" + 'car_' + subdir_name + ibr_prefix +'.mp4'    
            if os.path.exists(vid_fname):
                # os.remove(vid_fname)      
                pass
            else:
                xamb, uamb, xamb_des, xothers, uothers, xothers_des = mibr.load_state(folder + "data/" + ibr_prefix, n_other_cars)
                cmplot.plot_cars(world, x_MPC, xamb, xothers, folder, 
                                                None, None, CIRCLES, parallelize=True)
                cmd = 'ffmpeg -r 12 -f image2 -i {}imgs/%03d.png -vcodec libx264 -crf 23  -pix_fmt yuv420p {}'.format(folder, vid_fname)
                os.system(cmd)
            print('Saving video to: {}'.format(vid_fname))