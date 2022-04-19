import os, pickle, json, glob, argparse
import numpy as np
import matplotlib.pyplot as plt

from src.utils.plotting.car_plotting import plot_cars, animate, plot_initial_positions


def post_process(args):
    ''' Load the directory and print the'''
    for logdir in args.logdir:
        print(logdir)
        try:
            params, vehicles, world, x0, x, u = load_data_from_logdir(logdir)    
        except FileNotFoundError as e:
            print(e)
            continue

        if args.show_initial_positions:
            plot_initial_positions(logdir, world, vehicles, x0, number_cars_included=10)

        if args.analyze_controls:
            analyze_controls(logdir, u, x)


        if args.animate:
            animate_cars(args, params, logdir, x, vehicles, world)


def load_data_from_logdir(logdir: str):
    ''' Loads experiment data from logdirectory string'''
    
    X0_PATH = os.path.join(logdir, "x0.p")
    x0 = pickle.load(open(X0_PATH, "rb"))

    U_PATH = os.path.join(logdir, "controls.npy")
    X_PATH = os.path.join(logdir, "trajectories.npy")
    u = np.load(U_PATH)
    x = np.load(X_PATH)

    vehicles_path = os.path.join(logdir, 'other_vehicles.p')
    world_path = os.path.join(logdir, 'world.p')

    vehicles = pickle.load(open(vehicles_path, 'rb'))
    world = pickle.load(open(world_path, 'rb'))    

    params = json.load(open(os.path.join(logdir, "params.json")))

    return params, vehicles, world, x0, x, u


def get_car_colors_from_svo(vehicles):
    ''' Get car color string for each car based on the car's SVO'''
    svo = []
    for veh in vehicles:
        if hasattr(veh, "theta_ij"):
            svo += [veh.theta_ij[-1]]
        else:
            svo += [veh.theta_iamb]
    
    car_colors = ['r' for i in range(len(svo))]
    for i in range(len(svo)):
        if svo[i] < np.pi / 8.0:
            car_colors[i] = 'red'
        elif np.pi / 8 <= svo[i] <= 3 * np.pi / 8.0:
            car_colors[i] = 'purple'
        elif np.pi / 8 <= svo[i] <= np.pi / 2.0:
            car_colors[i] = 'blue'
    
    return car_colors

def animate_cars(args, params, log_directory, trajectories, vehicles, world):
    ''' Create a video of the cars driving around'''


    vids_path = glob.glob(log_directory + "/*/*.mp4")
    if args.keep_old_vids and len(vids_path) > 0:
        print("Video already exists: %s" % log_directory)
        return None

    if args.end_mpc != -1:
        trajectories = trajectories[:, :, :args.end_mpc]

    # Prep the image directory
    img_dir = os.path.join(log_directory, "imgs")
    print("Img dir", img_dir)
    os.makedirs(img_dir, exist_ok=True)
    filelist = glob.glob(os.path.join(img_dir, "*.png"))
    for f in filelist:
        os.remove(f)

    # For when we would return a large vector that included datapoints defaulted to zero
    # if xamb_actual.shape[1] > end_frame:
    #     xamb_actual = xamb_actual[:, :end_frame]
    #     xothers_actual = [x[:, :end_frame] for x in xothers_actual]

    print("Saving photos to %s..." % img_dir)

    if args.svo_colors == 1:
        car_colors = get_car_colors_from_svo(vehicles)
    else:
        car_colors = None

    plot_cars(world,
              None,
              None, [trajectories[i, :, :] for i in range(params["n_other"])],
              img_dir + "/",
              args.shape,
              None,
              None,
              car_colors,
              args.n_workers,
              vid_track=args.vehicle_id_track,
              all_other_vehicles=vehicles)

    # log_string = log_directory.split('/')[-2]
    log_string = log_directory
    vid_dir = os.path.join(log_directory, 'vids/')
    os.makedirs(vid_dir, exist_ok=True)
    print("Video directory", vid_dir)
    if args.shape == "both" or args.shape == "image":
        video_filename = os.path.join(vid_dir, log_string + '.mp4')
    else:
        video_filename = os.path.join(vid_dir, log_string + 'ell.mp4')

    if os.path.exists(video_filename):
        os.remove(video_filename)
    outfile = animate(img_dir, video_filename, args.fps)
    print("Saved video to: %s" % outfile)
    vid_directory = '/home/nbuckman/mpc-vids/'
    cmd = 'cp %s %s' % (video_filename, vid_directory)
    os.system(cmd)
    print("Saved video to: %s" % vid_directory)

    return None



def analyze_controls(LOGDIR, u, x):    
    os.makedirs(os.path.join(LOGDIR, "plots/"), exist_ok=True)

    np.set_printoptions(3, suppress=True)
    for ag_i in range(x.shape[0]):
        print(ag_i)
        print(x[ag_i, 2, :])
        print(ag_i, "theta$")
        print(np.rad2deg(x[ag_i, 3, :]))


    print(u[4, :, :])

    plt.figure()
    plt.plot(u[4,0,:])
    plt.ylabel("u0_steering_change")
    plt.xlabel("Timestep")
    plt.savefig(os.path.join(LOGDIR, "plots/", "u0_4.png"))

    plt.figure()
    plt.plot(u[4,1,:])
    plt.ylabel("u1_vel_change")
    plt.xlabel("Timestep")
    plt.savefig(os.path.join(LOGDIR, "plots/", "u1_4.png"))
    return None



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", type=str, nargs="+", help="log directory to analyze")
    parser.add_argument("--show-initial-positions", action="store_true", help="load controls")
    parser.add_argument("--analyze-controls", action="store_true", help="load controls")
    parser.add_argument("--animate", action="store_true", help="Animate simulation")


    parser.add_argument('--end-mpc', type=int, default=-1)
    parser.add_argument('--vehicle-id-track', type=int, default=0)
    parser.add_argument('--camera-speed', type=int, default=-1)
    parser.add_argument('--shape', type=str, default="ellipse")
    parser.add_argument('--svo-colors', type=int, default=1)
    parser.add_argument('--n-workers', type=int, default=8)
    parser.add_argument('--keep-old-vids', type=bool, default=False)
    parser.add_argument('--fps', type=int, default=16, help="Frame rate in frames per second")



    args = parser.parse_args()

    post_process(args)
