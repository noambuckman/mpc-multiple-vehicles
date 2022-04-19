import os, pickle, json, glob, argparse
import numpy as np
import matplotlib.pyplot as plt


CONV_MS_TO_MPH = 2.23694


def post_process_experiments(args):
    ''' Load the directory and print the'''
    
    all_params, all_trajs, all_vehicles, all_dirs, experiment_params = load_experiment_results(args.experiment_dir)

    p_coop_array = np.array([all_params[exi]['p_cooperative'] for exi in range(len(all_params))])
    distances_array = np.array([all_trajs[exi][:,0, args.n_mpc_end] - all_trajs[exi][:,0,0] for exi in range(len(all_params))])
    p_distance_dict = {}
    p_passing_dict = {}
    for exi in range(distances_array.shape[0]):
        if p_coop_array[exi] not in p_distance_dict:
            p_distance_dict[p_coop_array[exi]] = []
        p_distance_dict[p_coop_array[exi]] += [distances_array[exi]]
        
        
    for p in p_distance_dict:
        p_distance_dict[p] = np.array(p_distance_dict[p])
    for p in p_passing_dict:
        p_passing_dict[p] = np.array(p_passing_dict[p])
        
    p_unique = np.array(list(p_distance_dict.keys()))
    p_unique = np.sort(p_unique)

    params = all_params[0]
    total_time = params["dt"] * args.n_mpc_end
    print("Total Sim Time", total_time)

    all_seeds = [param["seed"] for param in all_params]
    all_densities = [param["car_density"] for param in all_params]
    all_p_cooperative = [param["p_cooperative"] for param in all_params]

    summarize_sim_params(all_params, all_vehicles)

    total_collisions, total_experiments, percentage_collisions_experiments, collisions_per_km, collisions_per_bkm = get_collision_stats(all_trajs)
    print("N_collisions: %d,  N_exp: %d, P_collisions: %.2f,  Collisions/Km: %.3e, Collisions/BKm %.1e"%(
                    total_collisions, total_experiments, percentage_collisions_experiments, collisions_per_km, collisions_per_bkm)) 


    if args.analyze_collisions:
        for exp_i in range(len(all_trajs)):
            if check_collision_experiment(all_trajs[exp_i]):
                print(all_dirs[exp_i])
                print("P", all_params[exp_i]["p_cooperative"])
                print("Density",  all_params[exp_i]["car_density"])
                print("Collision Index", get_collision_index(all_trajs[exp_i]))
                print("Seed", all_params[exp_i]["seed"])    


        collision_params = get_collision_parameters(all_params, all_trajs)
        summarize_sim_params(collision_params, all_vehicles, "collision_params_hist.png")

    if args.animate_by_density:
        animate_by_density(experiment_params, all_dirs, all_params)

    if args.animate_by is not None:
        param_name = args.animate_by
        assert param_name in experiment_params, "%s not in experiment.json"%param_name
        assert param_name in all_params[0], "%s not in params.json"%param_name

        animate_by(param_name, experiment_params, all_dirs, all_params)


def load_experiment_results(experiment_dir):
    ''' Load experiment param and results'''
    sim_dirs = glob.glob(os.path.join(experiment_dir,'results/*-*'))
    print("# Sim Dirs: %d"%len(sim_dirs))

    # experiment_params_path = glob.glob('/home/nbuckman/mpc_results/txe_0719/experiment.json')
    experiment_params_path = os.path.join(experiment_dir, "experiment.json")
    experiment_params = json.load(open(experiment_params_path,'rb'))
    
    all_params = []
    all_trajs = []
    all_vehicles = []
    all_dirs = []

    for sim_dir in sim_dirs:
        params_path = glob.glob(sim_dir + '/params.json')
        trajs_path = glob.glob(sim_dir + '/trajectories.npy')
        vehicles_path = glob.glob(sim_dir + '/other_vehicles.p')
        
        if len(trajs_path) > 0:
            all_params += [json.load(open(params_path[0],'rb'))]
            all_trajs += [np.load(trajs_path[0])]
            all_vehicles += [pickle.load(open(vehicles_path[0],'rb'))]
            all_dirs += [sim_dir]
    print("# Simulations: %d"%len(all_dirs))

    return all_params, all_trajs, all_vehicles, all_dirs, experiment_params


def animate_by_density(experiment_params, all_dirs, all_params):
    
    for density in experiment_params['car_density']:
        vids_to_concat = []
        for i in range(len(all_dirs)):
            sim_dir = all_dirs[i]
            if all_params[i]['car_density'] == density:
                vid_path = glob.glob(sim_dir + '/vids/*.mp4')
                
                if len(vid_path) > 0:
                    vids_to_concat += [vid_path[0]]
        output_file_name = concat_vids(vids_to_concat, "all_density%d.mp4"%density)
        print(output_file_name)


def animate_by(param_name, experiment_params, all_dirs, all_params):
    for param_value in experiment_params[param_name]:
        vids_to_concat = []
        for i, sim_dir in enumerate(all_dirs):
            if all_params[i][param_name] == param_value:
                vid_path = glob.glob(sim_dir + '/vids/*.mp4')
                if len(vid_path) > 0:
                    vids_to_concat += [vid_path[0]]

        output_file_name = concat_vids(vids_to_concat, "%s_%s.mp4"%(param_name, param_value))
        print(output_file_name)




def get_collision_stats(all_trajs):
    ''' Compute collision/km'''

    distances_array = np.array([all_trajs[exi][:,0,args.n_mpc_end] - all_trajs[exi][:,0,0] for exi in range(len(all_trajs))])
    total_distance_meters = np.sum(distances_array)
    total_collisions = np.sum([check_collision_experiment(traj) for traj in all_trajs])
    total_experiments = len(all_trajs)
    percentage_collisions_experiments = float(total_collisions) / total_experiments
    collisions_per_km = float(total_collisions) / (total_distance_meters / 1000.0) 
    collisions_per_bkm = collisions_per_km * (1e9)


    return total_collisions, total_experiments, percentage_collisions_experiments, collisions_per_km, collisions_per_bkm


def get_collision_parameters(all_params, all_trajs):
    ''' Return list of parameters corresponding to experiments with collisions'''

    collision_params = [all_params[exp_i] for exp_i in range(len(all_trajs)) if check_collision_experiment(all_trajs[exp_i])]    

    return collision_params
    

def summarize_sim_params(all_params, all_vehicles, filename="param_summary_hist.png"):
    ''' Create a histogram of all the params'''
    
    fig, axs = plt.subplots(2,2)
    color=iter(plt.cm.rainbow(np.linspace(0,1,4)))
    counter = 0
    
    for param_key in ["seed", "car_density", "p_cooperative"]:
        row = counter%2
        col = counter // 2
        ax = axs[row, col]
        all_p_key = [param[param_key] for param in all_params]
        c = next(color)
        ax.hist(all_p_key, facecolor=c)
        ax.set_xlabel(param_key)
        ax.set_ylabel("# Exp")
        counter+=1
        
    c = next(color)    
    all_max_speeds = [veh.max_v for vehicles in all_vehicles for veh in vehicles]
    axs[1,1].hist(all_max_speeds)
    axs[1,1].set_xlabel('Max Speed')
    axs[1,1].set_ylabel('# Vehicles')
    fig.suptitle("Experiment Settings")

    fig.tight_layout()
    fig.show()

    plt.savefig(filename)


def get_collision_index(traj):
    ''' Check for collision in experiment
        If coollision occurs, the trajectory will be zeros(1,6)
    '''
    collision_array = np.all(traj[:,:,1:] == np.zeros((1, 6, 1)), axis=1) # vehs x time
    collision_array = np.any(collision_array, axis=0) # time
    return np.argmax(collision_array)

def check_collision_experiment(traj):
    ''' Check for collision in experiment
        If coollision occurs, the trajectory will be zeros(1,6)
    '''
    if np.any(np.all(traj[:,:,1:] == np.zeros((1, 6, 1)), axis=1)):
        return True
    return False


def concat_vids(list_vids_to_concat, output_file, dry_run=False, n_max_vids=16, n_columns=2):
    ''' Takes a list of videos to concat and adds them to a single video'''
    
    n_vids = min(len(list_vids_to_concat), n_max_vids)
    n_rows = int(np.ceil(n_vids / n_columns))


    input_files = list_vids_to_concat[:n_vids]
    temp_files = []


    if os.path.exists(output_file):
        os.remove(output_file)

    for input_file in input_files:
        # Crop
        temp_file = input_file[:-4] + "temp.mp4"
        crop_command = "ffmpeg -i %s -filter 'crop=iw*.9:ih*.3' %s"%(input_file, temp_file)
        if not dry_run:
            os.system(crop_command)
        temp_files += [temp_file]


    all_filter_commands = ""
    for vint in range(len(input_files)):
        row = vint//2
        column = vint%2

        if vint==0:
            filter_command = "[%d:v]pad=iw*%d:ih*%d[int%d];"%(vint,n_columns, n_rows, vint)
        else:
            if column == 1:
                overlay_x = "W/2"
            else:
                overlay_x = "0"

            overlay_y = "%d/%d*H"%((row), n_rows)

            filter_command = "[int%d][%d:v]overlay=%s:%s[int%d];"%(vint-1, vint, overlay_x, overlay_y, vint)
        all_filter_commands += filter_command


    concat_command = "ffmpeg "
    for vint in range(len(input_files)):
        concat_command += "-i %s "%temp_files[vint]
    concat_command += "-filter_complex '%s' "%all_filter_commands[:-1]
    concat_command += "-map '[int%d]' -c:v libx264 -crf 23 -preset veryfast %s"%(vint, output_file)


    # concat_command = "ffmpeg -i %s -i %s -i %s -i %s -filter_complex '[0:v]pad=iw*2:ih*3[int];[int][1:v]overlay=W/2:H/3[vid];[vid][2:v]overlay=0:H/3[vid2]' -map '[vid2]' -c:v libx264 -crf 23 -preset veryfast %s"%(
    #                     *temp_files, output_file)
    print(concat_command)
    if not dry_run:
        os.system(concat_command)
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)    
    
    return output_file


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=str, help="experiment directory to analyze")
    parser.add_argument("--n-mpc-end", type=int, default=500, help="experiment directory to analyze")

    parser.add_argument("--analyze-collisions", action="store_true", help="Print out parameters for every simulation with collisions")
    parser.add_argument("--animate-by-density", action="store_true", help="animate videos and concatenate by SVO")
    parser.add_argument("--animate-by", type=str, default=None, help="Paramater to organize animation grid")
    args = parser.parse_args()

    post_process_experiments(args)
