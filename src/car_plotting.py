import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)

import datetime
import os
import numpy as np
import scipy.misc
from scipy import ndimage

##### For viewing the videos in Jupyter Notebook
import io
import base64
from IPython.display import HTML

PROJECT_PATH = '/home/nbuckman/Dropbox (MIT)/DRL/2020_01_cooperative_mpc/mpc-multiple-vehicles/'

def get_frame(x, ax=None, car_name="Car1", min_distance=-1, circle=False, L=1.0):
    '''Plots a car at a single state x.  Assumes red_car and ambulance.png'''
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,12))
    else:
        fig = ax.get_figure()
    X, Y, Phi, Delta, V, S = x.flatten()
    if car_name == "Car1":
        arr_img = plt.imread(PROJECT_PATH + 'images/red_car.png', format='png')
        car_width_px = 599
        car_height_px = 310        
    elif car_name == "Amb":
        arr_img = plt.imread(PROJECT_PATH + 'images/ambulance.png', format='png')
        car_width_px = 1280
        car_height_px = 640       
    elif car_name == "Car2": 
        arr_img = plt.imread(PROJECT_PATH + 'images/green_car.png', format='png')
        car_width_px = 599
        car_height_px = 310               
    degree = np.rad2deg(Phi)
    xy = (X, Y)
    rotated_img = ndimage.rotate(arr_img, degree)
    window_width = ax.get_xlim()[1] - ax.get_xlim()[0]
    window_height = ax.get_ylim()[1] - ax.get_ylim()[0]
    figwidth_in = fig.get_size_inches()[0]
    dpi = fig.get_dpi()
    if circle:
        circle_patch = patches.Circle(xy, radius=min_distance/2)
        ax.add_patch(circle_patch)
    else:
        if car_name == "Amb":
            zoom_ratio = L/car_width_px * (dpi*figwidth_in)/window_width  * 0.75 #0.8 is a hard coded correction            
        else:
            zoom_ratio = L/car_width_px * (dpi*figwidth_in)/window_width  * 0.75 #0.8 is a hard coded correction             
        imagebox = OffsetImage(rotated_img, zoom=zoom_ratio) #this zoom is to scale L=1            
        
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox, (X, Y),frameon=False)
        ax.add_artist(ab)        
    return fig, ax    

def plot_cars(x1_plot, x2_plot, xamb_plot, folder, x1_desired=None, x2_desired=None, xamb_desired=None, CIRCLES=False, min_dist=-1):
    N = x1_plot.shape[1]
    max_xplots =     max(np.hstack((x1_plot[0,:],x2_plot[0,:],xamb_plot[0,:]))) + 1
    max_yplots = max(np.hstack((x1_plot[1,:],x2_plot[1,:],xamb_plot[1,:],-x1_plot[1,:],-x2_plot[1,:],-xamb_plot[1,:])))
    xmin, xmax = -1, max_xplots
    ymax = max_yplots + 0.5
    ymin = -1 # Based on ymin that we give to MPC
    width = max_xplots/2.0
    axlim_minx = xmin
    axlim_maxx = xmin + width
    SLIDING_WINDOW = True
    if not SLIDING_WINDOW:
        axlim_minx = xmin
        axlim_maxx = xmax
    for k in range(N):
        figsize="LARGE"
        if figsize == "LARGE":
            figwidth_in=24.0
        else:
            figwidth_in=6.0
        fig, ax = plt.subplots(figsize=(figwidth_in,figwidth_in/2))
        ax.axis('square')
        ax.set_ylim((ymin, ymax))
        current_xmin, current_xmax = min([x1_plot[0,k], x2_plot[0,k], xamb_plot[0,k]]), max([x1_plot[0,k], x2_plot[0,k], xamb_plot[0,k]])
        if current_xmax > (axlim_maxx - 1) and SLIDING_WINDOW:
            # print("cmax", current_xmax, "axlim", axlim_maxx)
            axlim_minx = current_xmin - 5
            axlim_maxx = axlim_minx + width
        ax.set_xlim((axlim_minx , axlim_maxx))
        if CIRCLES:
            fig, ax = get_frame(x1_plot[:,k], ax, False,min_dist,True)
            fig, ax = get_frame(x2_plot[:,k], ax, False,min_dist,True)
            fig, ax = get_frame(xamb_plot[:,k], ax, False,min_dist,True)
        else:
            fig, ax = get_frame(x1_plot[:,k], ax,"Car1",min_dist,False)
            fig, ax = get_frame(x2_plot[:,k], ax,"Car2",min_dist,False)
            fig, ax = get_frame(xamb_plot[:,k], ax, "Amb",min_dist,False)
        
        if x1_desired is not None:
            ax.plot(x1_desired[0,:], x1_desired[1,:], '--',c='red')
        if x2_desired is not None:
            ax.plot(x2_desired[0,:], x2_desired[1,:], '--',c="green")
        if xamb_desired is not None:
            ax.plot(xamb_desired[0,:], xamb_desired[1,:], '--',c="red")

        fig.savefig(folder + 'imgs/' '{:03d}.png'.format(k))
        plt.close(fig)     

