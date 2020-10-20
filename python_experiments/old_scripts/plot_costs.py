import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)

import datetime
import os
import sys
import numpy as np
import scipy.misc
from scipy import ndimage

##### For viewing the videos in Jupyter Notebook
import io
import base64
from IPython.display import HTML

import multiprocessing
import functools


PROJECT_PATH = '/home/nbuckman/Dropbox (MIT)/DRL/2020_01_cooperative_mpc/mpc-multiple-vehicles/'
sys.path.append(PROJECT_PATH)
import src.TrafficWorld as tw
import src.IterativeBestResponseMPCMultiple as mibr


import argparse

def plot_costs(file_dir, number):


    file_name = file_dir + "data/%03d"%number


    car1_costs_list, amb_costs_list, svo_cost, other_svo_cost , total_svo_cost = mibr.load_costs(file_name)

    
    labels = ["self.u_delta_cost",
    "self.u_v_cost",
    "self.lat_cost", 
    "self.lon_cost", 
    "self.phi_error_cost",
    "self.phidot_cost",
    "self.s_cost",
    "self.v_cost",
    "self.change_u_v",
    "self.change_u_delta", 
    "self.final_costs",
    "self.x_cost",
    "self.x_dot_cost"]

    fig = plt.figure(1)
    plt.bar(range(len(car1_costs_list)), car1_costs_list)
    plt.xticks(range(len(car1_costs_list)), labels,rotation=45)
    plt.title('Car Response')
    plt.xlabel("Subcost")
    plt.ylabel("Cost Value")
    plt.show()

    fig.savefig(file_dir + "plots/response_cost%03d.png"%number)
    plt.close(fig)


    car1_costs_list = car1_costs_list

    fig = plt.figure(1)
    slack_cost = total_svo_cost - svo_cost - other_svo_cost


    plt.bar(range(3), [svo_cost, other_svo_cost , slack_cost])
    plt.xticks(range(3), ["Response", "Other", "Slack"],rotation=45)
    plt.title('SVO Slack Cost')
    plt.xlabel("Subcost")
    plt.ylabel("Cost Value")
    # plt.xlim([0, 10])
    plt.show()

    fig.savefig(file_dir + "plots/carresponse_cost%03d.png"%number)
    plt.close(fig)





if __name__ == "__main__":
    subdir_name = "20200226_184306increase_kphidotklat_warm"

    number = 6

    file_dir = "results/" + subdir_name + "/"     


    plot_costs(file_dir, number)



    
    # parser = argparse.ArgumentParser(description='Plot histograms')
    # parser.add_argument('file_dir', type=str)
    # parser.add_argument('number', type=int,
    #                 help='File name')
    # args = parser.parse_args()



    # plot_costs(args.file_dir, args.number)
    pass