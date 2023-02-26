import os, json, glob, argparse
import numpy as np

########################################################
# Description: This code concatenates animations by seed
########################################################

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
        crop_command = "ffmpeg -i %s -filter 'crop=iw*.9:ih*.15' %s"%(input_file, temp_file)
        if not dry_run:
            os.system(crop_command)
        temp_files += [temp_file]


    all_filter_commands = ""
    for vint in range(len(input_files)):
        row = vint//n_columns
        column = vint%n_columns

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


def concat_seed(args, seed):
    
    

    
    all_sim_dirs = glob.glob(os.path.join(args.experiment_dir, "results", "*"))

    seed_dirs = []
    seed_params = []
    for sim_dir in all_sim_dirs:
        param_path = os.path.join(sim_dir, "params.json")
        with open(param_path, 'rb') as fp:
            params = json.load(fp)
            
            if params["seed"] == seed:
                seed_dirs.append(sim_dir)
                seed_params.append(params)
    
    if len(seed_dirs) == 0:
        raise Exception("No sims found with seed = %d"%seed)
    
    sorted_idx_by_p_coop = [i[0] for i in sorted(enumerate(seed_params), key=lambda x:x[1]["p_cooperative"])]
    sorted_dirs = [seed_dirs[idx] for idx in sorted_idx_by_p_coop]
    sorted_params = [seed_params[idx] for idx in sorted_idx_by_p_coop]

    list_vids_to_concat = []
    for idx, dir in enumerate(sorted_dirs):
        p_coop = sorted_params[idx]["p_cooperative"]
        vid_paths = glob.glob(os.path.join(dir, "vids", "*.mp4"))
        if len(vid_paths) == 0:
            vid_path = "/home/nbuckman/Downloads/na2.mp4"
            # raise Exception("Can't find an MP4 for dir %s"%dir)
        elif len(vid_paths) > 1:
            vid_path = None
            for vpath in vid_paths:
                if "ell" in vpath:
                    if args.shape == "ellipse":
                        vid_path = vpath
                    else:
                        continue
                else:
                    if args.shape == "ellipse":
                        continue
                    else:
                        vid_path = vpath
            
            if vid_path is None:
                vid_path = vpath            
        else:
            vid_path = vid_paths[0]
        list_vids_to_concat.append(vid_path)

    output_file = os.path.join(args.experiment_dir, "seed_%d.mp4"%seed)
    concat_vids(list_vids_to_concat, output_file, dry_run=False, n_max_vids=16, n_columns=1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=str, help="experiment directory to analyze")
    parser.add_argument("--seed", type=int, default=None, help="seed to concat")
    parser.add_argument("--shape", type=str, default="image", help="Which video should be used?")
    args = parser.parse_args()

    if args.experiment_dir == ".":
        args.experiment_dir = os.getcwd()


    if args.seed is None:
        all_dirs = glob.glob(os.path.join(args.experiment_dir, "results/*"))
        all_params = []
        for dir in all_dirs:
            param_path = os.path.join(dir, "params.json")
            with open(param_path, 'rb') as fp:
                params = json.load(fp)
                all_params.append(params)
        all_seeds = np.unique(np.array([all_params[exp_i]["seed"] for exp_i in range(len(all_params))]))
        for seed in all_seeds:
            concat_seed(args, seed)
    else:
        concat_seed(args, args.seed)
