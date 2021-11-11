import numpy as np
import json, pickle, glob

import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser("Add experiments")
parser.add_argument('sim_dirs', type=str, default=None, nargs='+', help="Load log")
args = parser.parse_args()

sim_dirs = args.sim_dirs
print(sim_dirs)

CONV_MS_TO_MPH = 2.23694
all_params = []
all_trajs = []
all_vehicles = []
all_dirs = []

for sim_dir in sim_dirs:
    params_path = glob.glob(sim_dir + '/params.json')
    trajs_path = glob.glob(sim_dir + '/trajectories.npy')
    vehicles_path = glob.glob(sim_dir + '/other_vehicles.p')

    if len(trajs_path) > 0:
        all_params += [json.load(open(params_path[0], 'rb'))]
        all_trajs += [np.load(trajs_path[0])]
        all_vehicles += [pickle.load(open(vehicles_path[0], 'rb'))]
        all_dirs += [sim_dir]
print("# Simulations: %d" % len(all_dirs))

n_mpc_end = all_trajs[0].shape[2] - 1

p_coop_array = np.array([all_params[exi]['p_cooperative'] for exi in range(len(all_params))])
distances_array = np.array(
    [all_trajs[exi][:, 0, n_mpc_end] - all_trajs[exi][:, 0, 0] for exi in range(len(all_params))])
p_distance_dict = {}
# p_passing_dict = {}
for exi in range(distances_array.shape[0]):
    if p_coop_array[exi] not in p_distance_dict:
        p_distance_dict[p_coop_array[exi]] = []
    p_distance_dict[p_coop_array[exi]] += [distances_array[exi]]

#     if p_coop_array[exi] not in p_passing_dict:
#         p_passing_dict[p_coop_array[exi]] = []
#     p_passing_dict[p_coop_array[exi]] += [passing_array[exi]]

for p in p_distance_dict:
    p_distance_dict[p] = np.array(p_distance_dict[p])
# for p in p_passing_dict:
#     p_passing_dict[p] = np.array(p_passing_dict[p])
p_unique = np.array(list(p_distance_dict.keys()))
p_unique = np.sort(p_unique)

params = all_params[0]
total_time = params["T"] * params['p_exec'] * params['n_mpc']
total_time = params["dt"] * n_mpc_end
print("Total Sim Time", total_time)

all_seeds = [param["seed"] for param in all_params]
all_densities = [param["car_density"] for param in all_params]
all_p_cooperative = [param["p_cooperative"] for param in all_params]

fig, axs = plt.subplots(2, 2)

color = iter(plt.cm.rainbow(np.linspace(0, 1, 4)))
counter = 0
for param_key in ["seed", "car_density", "p_cooperative"]:
    row = counter % 2
    col = counter // 2
    ax = axs[row, col]
    all_p_key = [param[param_key] for param in all_params]
    c = next(color)
    ax.hist(all_p_key, facecolor=c)
    ax.set_xlabel(param_key)
    ax.set_ylabel("# Exp")
    counter += 1

c = next(color)
all_max_speeds = [veh.max_v for vehicles in all_vehicles for veh in vehicles]
axs[1, 1].hist(all_max_speeds)
axs[1, 1].set_xlabel('Max Speed')
axs[1, 1].set_ylabel('# Vehicles')
fig.suptitle("Experiment Settings")

fig.tight_layout()
fig.show()


def check_collision_experiment(traj):
    ''' Check for collision in experiment
        If coollision occurs, the trajectory will be zeros(1,6)
    '''
    if np.any(np.all(traj[:, :, 1:] == np.zeros((1, 6, 1)), axis=1)):
        return True
    return False


total_distance_meters = np.sum(distances_array)
collision_bool_array = [check_collision_experiment(traj) for traj in all_trajs]
total_collisions = np.sum(collision_bool_array)
total_experiments = len(all_trajs)
percentage_collisions_experiments = float(total_collisions) / total_experiments
collisions_per_km = float(total_collisions) / (total_distance_meters / 1000.0)
collisions_per_bkm = collisions_per_km * (1e9)

print("N_collisions: %d,  N_exp: %d, P_collisions: %.2f,  Collisions/Km: %.3e, Collisions/BKm %.1e" %
      (total_collisions, total_experiments, percentage_collisions_experiments, collisions_per_km, collisions_per_bkm))

print("Logs with Collisions:")
logs_with_collisions = [all_dirs[i] for i in range(len(all_dirs)) if collision_bool_array[i]]
for log in logs_with_collisions:
    print(log)