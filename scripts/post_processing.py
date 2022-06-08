import os, pickle, json, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from src.multiagent_mpc import MultiMPC
from src.utils.plotting.car_plotting import plot_cars, animate, plot_initial_positions
from src.utils.sim_utils import get_closest_n_obstacle_vehs
from src.vehicle_mpc_information import VehicleMPCInformation
from src.best_response import generate_solver_params


def post_process(args):
    ''' Load the directory and print the'''
    for logdir in args.logdir:
    
        try:
            params, vehicles, world, x0, x, u = load_data_from_logdir(logdir)    
        except FileNotFoundError as e:
            print(e)
            continue

        if args.show_initial_positions:
            plot_initial_positions(logdir, world, vehicles, x0, number_cars_included=10)
        
        post_processing_dir_path = os.path.join(logdir, "postprocessing/")
        os.makedirs(post_processing_dir_path, exist_ok=True)    


        if args.animate:
            animate_cars(args, params, logdir, x, vehicles, world)

        if args.analyze_controls:
            analyze_controls(logdir, u, x)

        if args.analyze_costs:
            analyze_costs(vehicles, x, u, world, params, post_processing_dir_path)


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

    post_processing_dir_path = os.path.join(LOGDIR, "postprocessing/")

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
    plt.savefig(os.path.join(post_processing_dir_path, "u0_4.png"))

    plt.figure()
    plt.plot(u[4,1,:])
    plt.ylabel("u1_vel_change")
    plt.xlabel("Timestep")
    plt.savefig(os.path.join(post_processing_dir_path, "u1_4.png"))
    return None

def analyze_costs(vehicles, x, u, world, params, postprocessing_dir_path):
    '''Analyze the mpc costs'''


    create_cost_bar_chart(x, u, vehicles, world, params, postprocessing_dir_path)

    analyze_multicar_costs(vehicles, x, u, world, params, postprocessing_dir_path)

    

def analyze_multicar_costs(vehicles, x, u, world, params, postprocessing_dir_path):
    ''' Compare Response Costs and Collision Avoidance Costs'''
    mpc = MultiMPC(params["N"], params["dt"], world, 0, 10, params)
    cost_lists = []
    vehsinfo = []
    for i in range(len(vehicles)):
        xi = x[i, :, :]
        ui = u[i, :, :]

        # Regenerate the desired trajectory
        # vehicles[i].update_desired_lane_from_x0(world, x0[i])
        s = xi[-1,:]
        Fd = vehicles[i].fd.map(s.shape[0])
        xd = Fd(s)
        xd = np.array(xd)
        vehicle =  vehicles[i]
        x0 = xi[:, 0]

        vehsinfo.append(VehicleMPCInformation(vehicle, x0, ui, xi, xd))
    
    data = []
    for i in range(len(vehicles)):
        response_vehinfo = vehsinfo[i]
        obstacle_vehsinfo = [vehsinfo[j] for j in range(len(vehicles)) if j!=i]
        obstacle_vehsinfo = get_closest_n_obstacle_vehs(response_vehinfo, [], obstacle_vehsinfo, max_num_obstacles=10, min_num_obstacles_ego=3)

        x = response_vehinfo.x
        u = response_vehinfo.u
        xd = response_vehinfo.xd
        vehicle = response_vehinfo.vehicle 
        response_costs, _ = mpc.generate_veh_costs(x, u, xd, vehicle)

        response_svo_cost = np.cos(vehicle.theta_i) * response_costs
        other_svo_cost = 0

        # Generate Slack Variables used as part of collision avoidance
        # slack_cost = self.compute_quadratic_slack_cost(n_other_vehicle, n_vehs_cntrld, slack_i_jnc, slack_ic_jnc,
        #                                                slack_i_jc, slack_ic_jc) + self.compute_wall_slack_costs(self.N, n_vehs_cntrld, top_wall_slack, bottom_wall_slack,
        #                                             top_wall_slack_c, bottom_wall_slack_c)

        slack_cost = 0
        other_vehicles = [vinfo.vehicle for vinfo in obstacle_vehsinfo]
        x_other = [vinfo.x for vinfo in obstacle_vehsinfo]
        n_other = len(other_vehicles)

        solver_params = generate_solver_params(params, 0, 0)

        k_slack = solver_params["k_slack"]
        k_CA = solver_params["k_CA"]
        k_CA_power = solver_params["k_CA_power"]


        collision_cost = mpc.compute_collision_avoidance_costs(mpc.N, n_other, 0, vehicle,
                                                                other_vehicles, [], xi, x_other,
                                                                [], params, mpc.k_ca2, k_CA_power)


        collision_cost = float(collision_cost)
        weighted_collision_cost = k_CA * collision_cost

        data.append([i, response_costs, collision_cost, weighted_collision_cost])
    
    print(data)
    columns = ["agent_idx", "response_costs", "collision_cost", "weighted_collision_cost"]

    df = pd.DataFrame( data, columns=columns)
    fig, axs = plt.subplots(2, 1, figsize=(8,10))
    df0 = df[["agent_idx", "response_costs", "collision_cost"]]
    df0.plot(x="agent_idx", kind='bar', stacked=True, title="Cost w/Collision Cost", ax=axs[0], colormap='tab20')
    axs[0].legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
    
    df1 = df[["agent_idx", "response_costs", "weighted_collision_cost"]]
    df1.plot(x="agent_idx", kind='bar', stacked=True, title="Cost w/Weighted Cost", ax=axs[1], colormap='tab20')
    axs[1].legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
    
    
    fig_path = os.path.join(postprocessing_dir_path, "collision_cost_hist.png")
    fig.savefig(fig_path, bbox_inches='tight')



def create_cost_bar_chart(x, u, vehicles, world, params, postprocessing_dir_path):
    
    mpc = MultiMPC(params["N"], params["dt"], world, 1, 1, params)
    cost_lists = []
    for i in range(len(vehicles)):
        xi = x[i, :, :]
        ui = u[i, :, :]

        # Regenerate the desired trajectory
        # vehicles[i].update_desired_lane_from_x0(world, x0[i])
        s = xi[-1,:]
        Fd = vehicles[i].fd.map(s.shape[0])
        xd = Fd(s)
        xd = np.array(xd)

        total_costs, cost_list = mpc.generate_veh_costs(xi, ui, xd, vehicles[i])

        cost_lists.append(cost_list)



    n_agents = len(cost_lists)
    n_costs = len(cost_lists[0])

    cost_names = [
            "u_delta_cost", "u_v_cost", "lat_cost", "lon_cost",
            "phi_error_cost", "phidot_cost", "s_cost", "v_cost",
            "change_u_v",  "change_u_delta", "final_costs",
            "x_cost", "on_grass_cost", "x_dot_cost"
        ]

    columns = ["agent_idx"] + cost_names[:-1] #remove x_dot_cost for dataframe

    data = []
    speed_costs = []
    for idx in range(n_agents):

        row = [idx*1] + list(cost_lists[idx])
        speed_costs.append(row[-1])
        row = row[:-1] # Remove x_dot_cost
        data.append(row)

    positive_costs_df = pd.DataFrame(data, columns=columns)

    fig, axs = plt.subplots(2, 1, figsize=(8,10))
    positive_costs_df.plot(x="agent_idx", kind='bar', stacked=True, title="Stacked Bar Chart", ax=axs[0], colormap='tab20')
    axs[0].bar([idx*1 + 0 for idx in range(n_agents)], speed_costs, color='purple', label='x_dot_cost')
    axs[0].legend(bbox_to_anchor=(1.04,1), borderaxespad=0)

    positive_costs_df.plot(x="agent_idx", kind='bar', stacked=True, title="Stacked Bar Chart", ax=axs[1], colormap='tab20')
    axs[1].legend(bbox_to_anchor=(1.04,1), borderaxespad=0)

    fig_path = os.path.join(postprocessing_dir_path, "cost_hist.png")
    fig.savefig(fig_path, bbox_inches='tight')




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", type=str, nargs="+", help="log directory to analyze")
    parser.add_argument("--show-initial-positions", action="store_true", help="load controls")
    parser.add_argument("--analyze-controls", action="store_true", help="load controls")
    parser.add_argument("--animate", action="store_true", help="Animate simulation")
    parser.add_argument("--analyze-costs", action="store_true", help="Analyze and plot costs")

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
