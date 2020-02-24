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

import multiprocessing
import functools
import src.TrafficWorld as tw


PROJECT_PATH = '/home/nbuckman/Dropbox (MIT)/DRL/2020_01_cooperative_mpc/mpc-multiple-vehicles/'

def get_frame(x, ax=None, car_name="Car1", min_distance=-1, circle=False, alpha = 1.0, L=4.572):
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
    arr_img[:,:,3] = alpha * arr_img[:,:,3]
    circle=False
    if circle:        
        circle_patch = patches.Circle((X, Y), radius=min_distance/2)
        ax.add_patch(circle_patch)
        plt.annotate(car_name, (X-min_distance/2,Y), fontsize='xx-large', color='orange')
    else:
        degree = np.rad2deg(Phi)
        xy = (X, Y)
        rotated_img = ndimage.rotate(arr_img, degree)
        window_width = ax.get_xlim()[1] - ax.get_xlim()[0]
        window_height = ax.get_ylim()[1] - ax.get_ylim()[0]
        figwidth_in, figheight_in = fig.get_size_inches()
        dpi = fig.get_dpi()
        if dpi > 100:
            hard_coded_correction = 0.35
        else:
            hard_coded_correction = 0.75
        if car_name == "Amb":
            zoom_ratio = L/car_width_px * (dpi*figwidth_in)/window_width  * hard_coded_correction #0.8 is a hard coded correction  
        else:
            zoom_ratio = L/car_width_px * (dpi*figwidth_in)/window_width  * hard_coded_correction #0.8 is a hard coded correction             
        imagebox = OffsetImage(rotated_img, zoom=zoom_ratio) #this zoom is to scale L=1            
        
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox, (X, Y),frameon=False)
        ax.add_artist(ab)        
    return ax    



def plot_cars(world, x1_plot, x2_plot, xamb_plot, folder, x1_desired=None, x2_desired=None, xamb_desired=None, CIRCLES=False, min_dist=-1, SLIDING_WINDOW=False):
    N = x1_plot.shape[1]
    max_xplots =     max(np.hstack((x1_plot[0,:],x2_plot[0,:],xamb_plot[0,:]))) + 2
    min_xplots = min(np.hstack((x1_plot[0,:],x2_plot[0,:],xamb_plot[0,:]))) - 2
    max_yplots = max(np.hstack((x1_plot[1,:],x2_plot[1,:],xamb_plot[1,:])))
    xmin, xmax = min_xplots, max_xplots
    ymax = max_yplots + 0.5
    ymin = min(np.hstack((x1_plot[1,:],x2_plot[1,:],xamb_plot[1,:]))) - .5 # Based on ymin that we give to MPC
    width = max_xplots/2.0
    axlim_minx = min_xplots
    axlim_maxx = max_xplots
    if not SLIDING_WINDOW:
        axlim_minx = xmin
        axlim_maxx = xmax

    for k in range(N):
        plot_three_cars( k, ymax, ymin, axlim_maxx, axlim_minx, x1_plot,x2_plot, xamb_plot, SLIDING_WINDOW, width, min_dist, CIRCLES, x1_desired, x2_desired, xamb_desired, folder, world)     
    return None
            

def plot_cars_circles(world, x1_plot, x2_plot, xamb_plot, folder, x1_mpc, x2_mpc, xamb_mpc, ibr):
    N = x1_plot.shape[1]
    max_xplots =     max(np.hstack((x1_plot[0,:],x2_plot[0,:],xamb_plot[0,:]))) + 2
    min_xplots = min(np.hstack((x1_plot[0,:],x2_plot[0,:],xamb_plot[0,:]))) - 2
    max_yplots = max(np.hstack((x1_plot[1,:],x2_plot[1,:],xamb_plot[1,:])))
    xmin, xmax = min_xplots, max_xplots
    ymax = max_yplots + 0.5
    ymin = min(np.hstack((x1_plot[1,:],x2_plot[1,:],xamb_plot[1,:]))) - .5 # Based on ymin that we give to MPC
    width = max_xplots/2.0
    axlim_minx = min_xplots
    axlim_maxx = max_xplots


    for k in range(N):
        plot_three_circles( k, ymax, ymin, axlim_maxx, axlim_minx, x1_plot,x2_plot, xamb_plot, width, x1_mpc, x2_mpc, xamb_mpc, ibr, folder, world)

    return None


def plot_three_circles(k, ymax, ymin, axlim_maxx, axlim_minx, x1_plot, x2_plot, xamb_plot, width, x1_mpc, x2_mpc, xamb_mpc, ibr, folder, world):
    figsize="LARGE"
    if figsize == "LARGE":
        figwidth_in=12.0
    else:
        figwidth_in=6.0

    axlim_minx, axlim_maxx = xamb_plot[0,k] - 10, xamb_plot[0,k] + 10,    
    fig_height = np.ceil(1.1 * figwidth_in * (ymax - ymin) / (axlim_maxx - axlim_minx ))
    fig, ax = plt.subplots(figsize=(figwidth_in, fig_height), dpi=144)
    ax.axis('square')
    ax.set_ylim((ymin, ymax))
    # current_xmin, current_xmax = min([x1_plot[0,k], x2_plot[0,k], xamb_plot[0,k]]), max([x1_plot[0,k], x2_plot[0,k], xamb_plot[0,k]])
    # if current_xmax > (axlim_maxx - 1) and SLIDING_WINDOW:
    #     # print("cmax", current_xmax, "axlim", axlim_maxx)
    #     axlim_minx = current_xmin - 5
    #     axlim_maxx = axlim_minx + width
    ax.set_xlim((axlim_minx , axlim_maxx))
    # ax.set_xticks(np.arange(0, 20, 1))
 
    xy_f = ibr.opti.debug.value(ibr.c1_f[:,k])
    circle_patch_f = patches.Circle((xy_f[0], xy_f[1]), radius=ibr.min_dist/2)
    ax.add_patch(circle_patch_f)
        
    xy_r = ibr.opti.debug.value(ibr.c1_r[:,k])
    circle_patch_r = patches.Circle((xy_r[0], xy_r[1]), radius=ibr.min_dist/2)
    ax.add_patch(circle_patch_r)


    xy_f = ibr.opti.debug.value(ibr.c2_f[:,k])
    circle_patch_f = patches.Circle((xy_f[0], xy_f[1]), radius=ibr.min_dist/2)
    ax.add_patch(circle_patch_f)
        
    xy_r = ibr.opti.debug.value(ibr.c2_r[:,k])
    circle_patch_r = patches.Circle((xy_r[0], xy_r[1]), radius=ibr.min_dist/2)
    ax.add_patch(circle_patch_r)


    xy_f = ibr.opti.debug.value(ibr.ca_f[:,k])
    circle_patch_f = patches.Circle((xy_f[0], xy_f[1]), radius=ibr.min_dist/2)
    ax.add_patch(circle_patch_f)
        
    xy_r = ibr.opti.debug.value(ibr.ca_r[:,k])
    circle_patch_r = patches.Circle((xy_r[0], xy_r[1]), radius=ibr.min_dist/2)
    ax.add_patch(circle_patch_r)    



    add_lanes(ax, world)

    GRASS = True
    if GRASS:
        add_grass(ax, world, k)                
    # x1_desired, x2_desired, xamb_desired = None, None, None
    # if x1_desired is not None:
    #     ax.plot(x1_desired[0,:], x1_desired[1,:], '--',c='red')
    # if x2_desired is not None:
    #     ax.plot(x2_desired[0,:], x2_desired[1,:], '--',c="green")
    # if xamb_desired is not None:
    #     ax.plot(xamb_desired[0,:], xamb_desired[1,:], '--',c="red")
    fig = plt.gcf()
    fig.savefig(folder + 'imgs/' '{:03d}.png'.format(k))
    # plt.cla()
    # fig.clf()
    # ax.remove()
    plt.close(fig) 


















def plot_three_cars(k, ymax, ymin, axlim_maxx, axlim_minx, x1_plot, x2_plot, xamb_plot, SLIDING_WINDOW, width, min_dist, CIRCLES, x1_desired, x2_desired, xamb_desired, folder, world):
    figsize="LARGE"
    if figsize == "LARGE":
        figwidth_in=12.0
    else:
        figwidth_in=6.0

    axlim_minx, axlim_maxx = xamb_plot[0,k] - 10, xamb_plot[0,k] + 10,    
    fig_height = np.ceil(1.1 * figwidth_in * (ymax - ymin) / (axlim_maxx - axlim_minx ))
    fig, ax = plt.subplots(figsize=(figwidth_in, fig_height), dpi=144)
    ax.axis('square')
    ax.set_ylim((ymin, ymax))
    # current_xmin, current_xmax = min([x1_plot[0,k], x2_plot[0,k], xamb_plot[0,k]]), max([x1_plot[0,k], x2_plot[0,k], xamb_plot[0,k]])
    # if current_xmax > (axlim_maxx - 1) and SLIDING_WINDOW:
    #     # print("cmax", current_xmax, "axlim", axlim_maxx)
    #     axlim_minx = current_xmin - 5
    #     axlim_maxx = axlim_minx + width
    ax.set_xlim((axlim_minx , axlim_maxx))
    print("axlim", axlim_minx , axlim_maxx)
    # ax.set_xticks(np.arange(0, 20, 1))
    ax = get_frame(x1_plot[:,k], ax, "Car1", min_dist,CIRCLES)
    ax = get_frame(x2_plot[:,k], ax, "Car2", min_dist,CIRCLES)
    ax = get_frame(xamb_plot[:,k], ax, "Amb", min_dist,CIRCLES)

    add_lanes(ax, world)

    GRASS = True
    if GRASS:
        add_grass(ax, world, k)                

    if x1_desired is not None:
        ax.plot(x1_desired[0,:], x1_desired[1,:], '--',c='red')
    if x2_desired is not None:
        ax.plot(x2_desired[0,:], x2_desired[1,:], '--',c="green")
    if xamb_desired is not None:
        ax.plot(xamb_desired[0,:], xamb_desired[1,:], '--',c="red")
    fig = plt.gcf()
    fig.savefig(folder + 'imgs/' '{:03d}.png'.format(k))
    # plt.cla()
    # fig.clf()
    # ax.remove()
    plt.close(fig)    

def add_grass(ax, world, k):
    axlim_minx, axlim_maxx = ax.get_xlim()
    y_bottom_b, y_center_b, y_top_b = world.get_bottom_grass_y()
    width = axlim_maxx - axlim_minx
    left_x_grass = axlim_minx
    bottom_y_grass = y_bottom_b 
    bottom_grass = patches.Rectangle((left_x_grass,bottom_y_grass), width, y_top_b-y_bottom_b, facecolor='g', hatch='/') 
    ax.add_patch(bottom_grass)
    left_x_grass = axlim_maxx - k % (1.5*width)
    bottom_contrast_grass = patches.Rectangle((left_x_grass,bottom_y_grass), width/2.0, y_top_b-y_bottom_b, facecolor='g', hatch='x')                 
    ax.add_patch(bottom_contrast_grass)


    y_bottom_t, y_center_t, y_top_t = world.get_top_grass_y()
    left_x_grass = axlim_minx
    bottom_y_grass = y_bottom_t 
    top_grass = patches.Rectangle((left_x_grass,bottom_y_grass), width, y_top_t-y_bottom_t, facecolor='g', hatch='/') 
    ax.add_patch(top_grass) 

    left_x_grass = axlim_maxx - k % (1.5*width)
    top_contrast_grass = patches.Rectangle((left_x_grass,bottom_y_grass), width/2.0, y_top_t-y_bottom_t, facecolor='g', hatch='x') 
    ax.add_patch(top_contrast_grass)                

def add_lanes(ax, world):
    xmin1, xmax1 = ax.get_xlim()
    for lane_number in range(world.n_lanes_right):
        centerline_y = world.get_lane_centerline_y(lane_number, True)   
        topline_y = centerline_y + world.lane_width/2.0
        if lane_number == (world.n_lanes_right - 1):
            ax.plot([xmin1, xmax1], [topline_y, topline_y], dashes=[10,10],color='y')
        else:
            ax.plot([xmin1, xmax1], [topline_y, topline_y], dashes=[10,10], color='0.5')
    for lane_number in range(world.n_lanes_left):
        if lane_number == world.n_lanes_left - 1:
            pass
        else:
            centerline_y = world.get_lane_centerline_y(lane_number, False)   
            topline_y = centerline_y + world.lane_width/2.0       
            ax.plot([xmin1, xmax1], [topline_y, topline_y], dashes=[10,10],color='0.5')
    


def plot_cars_multiproc(world, x1_plot, x2_plot, xamb_plot, folder, x1_desired=None, x2_desired=None, xamb_desired=None, CIRCLES=False, min_dist=-1, SLIDING_WINDOW=True):
    N = x1_plot.shape[1]
    max_xplots =     max(np.hstack((x1_plot[0,:],x2_plot[0,:],xamb_plot[0,:]))) + 2
    min_xplots = min(np.hstack((x1_plot[0,:],x2_plot[0,:],xamb_plot[0,:]))) - 2
    max_yplots = max(np.hstack((x1_plot[1,:],x2_plot[1,:],xamb_plot[1,:])))
    xmin, xmax = min_xplots, max_xplots
    ymax = max_yplots + 0.5
    ymin = min(np.hstack((x1_plot[1,:],x2_plot[1,:],xamb_plot[1,:]))) - .5 # Based on ymin that we give to MPC
    width = max_xplots/2.0
    axlim_minx = min_xplots
    axlim_maxx = max_xplots
    if not SLIDING_WINDOW:
        axlim_minx = xmin
        axlim_maxx = xmax
    pool = multiprocessing.Pool(processes=8)

    plot_partial = functools.partial(plot_three_cars, world=world, ymax=ymax, ymin=ymin, axlim_maxx=axlim_maxx, axlim_minx=axlim_minx, x1_plot=x1_plot, x2_plot=x2_plot, xamb_plot=xamb_plot, SLIDING_WINDOW=False, width=width, min_dist=min_dist, CIRCLES=CIRCLES, 
                                    x1_desired=x1_desired, x2_desired=x2_desired, xamb_desired=xamb_desired, folder=folder)
        
    pool.map(plot_partial, range(N)) #will apply k=1...N to plot_partial