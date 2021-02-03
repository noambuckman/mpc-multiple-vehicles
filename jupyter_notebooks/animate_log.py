import src.multiagent_mpc as mpc
import src.traffic_world as tw
import src.car_plotting_multiple as cmplot
import glob
import argparse
import json
import pickle
import os
import numpy as np
# USAGE: python scripts/concat_vertical.py f509-425f-20200907-153800/04/
## Description: Converts a set of images in a folder into one tall image with all them

parser = argparse.ArgumentParser(description='Create a video')
parser.add_argument('log_directories', nargs='+', type=str)

parser.add_argument('--end-mpc', type=int, default = -1)
parser.add_argument('--camera-speed', type=int, default = -1)
parser.add_argument('--shape', type=str, default="ellipse")
parser.add_argument('--svo-colors', type=int, default=1)
parser.add_argument('--n-workers', type=int, default=8)
parser.add_argument('--fps',type=int,default=16, help="Frame rate in frames per second")
args = parser.parse_args()


for log_directory in args.log_directories:
    with open(log_directory + "params.json",'rb') as fp:
        params = json.load(fp)
    # filename = concat_imgs(args.folder) 
    print("Loading data...")
    if args.end_mpc == -1:
        # Find the last run.  We currently assumes that all runs start from 0 and increment by 1
        list_of_mpc_data = glob.glob(log_directory + 'data/all_*xamb.npy')
        n_mpc_runs = len(list_of_mpc_data)
        data_filename = log_directory + 'data/all_%03d'%(n_mpc_runs - 1)
        if os.path.isfile(data_filename + "xamb.npy"):
            pass
        else:
            data_filename = log_directory + 'data/all_%02d'%(n_mpc_runs - 1)
        print(data_filename)
    else:
        n_mpc_runs = (args.end_mpc + 1)
        data_filename = log_directory + 'data/all_%03d'%args.end_mpc
        if os.path.isfile(data_filename +"xamb.npy"):
            pass
        else:
            data_filename = log_directory + 'data/all_%02d'%args.end_mpc
    start_frame = 0
    end_frame = (n_mpc_runs + 1) * params['number_ctrl_pts_executed'] - 1 #Not exactly sure why we need minus 1

    xamb_actual, _, _, xothers_actual, _, _, = mpc.load_state(data_filename, params['n_other'], ignore_des=True)
    response_vehicle = pickle.load(open(log_directory + "data/" + "mpcamb.p",'rb'))
    if os.path.isfile(log_directory + "data/mpcother%03d.p"%0):
        other_vehicles = [pickle.load(open(log_directory + "data/mpcother%03d.p"%i,'rb')) for i in range(params["n_other"])]
    elif os.path.isfile(log_directory + "data/mpcother%02d.p"%0):
        other_vehicles = [pickle.load(open(log_directory + "data/mpcother%02d.p"%i,'rb')) for i in range(params["n_other"])]
    else:
        other_vehicles = [pickle.load(open(log_directory + "data/mpcother%d.p"%i,'rb')) for i in range(params["n_other"])]
    world = tw.TrafficWorld(params["n_lanes"], 0, 999999) 

    # Prep the image directory
    os.makedirs(log_directory+"imgs/", exist_ok=True)    
    filelist = glob.glob(os.path.join(log_directory+"imgs/", "*.png"))
    for f in filelist:
        os.remove(f)


    # For when we would return a large vector that included datapoints defaulted to zero
    if xamb_actual.shape[1] > end_frame:
        xamb_actual = xamb_actual[:, :end_frame]
        xothers_actual = [x[:, :end_frame] for x in xothers_actual]

    print("Saving photos to %s..."%(log_directory+"imgs/"))
    if args.camera_speed == -1:
        camera_speed = np.mean(xamb_actual[4,:])
    else:
        camera_speed = args.camera_speed
    if args.svo_colors == 1:
        svo = []
        for veh in other_vehicles:
            if hasattr(veh, "theta_ij"):
                svo += [veh.theta_ij[-1]]
            else:
                svo += [veh.theta_iamb]
        car_colors = ['r' for i in range(len(svo))]
        for i in range(len(svo)):
            if svo[i] < np.pi/8.0:
                car_colors[i] = 'red'
            elif np.pi/8 <= svo[i] <= 3*np.pi/8.0:
                car_colors[i] = 'purple'
            elif np.pi/8 <= svo[i] <= np.pi/2.0:
                car_colors[i] = 'blue'
    else:
        car_colors = None
    # if args.image:
    #     cmplot.plot_cars(world, response_vehicle, xamb_actual[:,start_frame:end_frame], [x[:,start_frame:end_frame] for x in xothers_actual], log_directory, "image", True, camera_speed)              
    # else:
    cmplot.plot_cars(world, response_vehicle, xamb_actual[:,start_frame:end_frame], [x[:,start_frame:end_frame] for x in xothers_actual], log_directory, args.shape, camera_speed, None, car_colors, args.n_workers)              
    # generate pictures for animation
    log_string = log_directory[:-1]
    log_string = log_string.replace("/","_")
    if args.shape == "both" or args.shape == "image":
        video_filename = log_directory + 'vids/' + log_string +'_%03d_%03di.mp4'%(start_frame, end_frame)
    else:
        video_filename = log_directory + 'vids/' + log_string +'_%03d_%03d.mp4'%(start_frame, end_frame)

    print(args.fps)
    outfile = cmplot.animate(log_directory, video_filename, args.fps)

    print("Saved video to: %s"%outfile)
    vid_directory = '/home/nbuckman/mpc-vids/'
    cmd = 'cp %s %s'%(video_filename, vid_directory)
    os.system(cmd)
    print("Saved video to: %s"%vid_directory)
