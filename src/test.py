from src.multiagent_mpc import MultiMPC
from src.utils.ibr_argument_parser import IBRParser
import numpy as np
from src.traffic_world import TrafficWorld
from src.utils.solver_helper import poission_positions, initialize_cars_from_positions
from vehicle import Vehicle
from vehicle_parameters import VehicleParameters

parser = IBRParser()
args = parser.parse_args()
params = vars(args)
# args.load_log_dir = "/home/nbuckman/mpc_results/investigation/h777-4664-20211031-204805/"
# params["T"] = 1.0
params["n_other"] = 6
# Determine number of control points in the optimization
i_mpc_start = 0
params["N"] = max(1, int(params["T"] / params["dt"]))
params["number_ctrl_pts_executed"] = max(1, int(np.floor(params["N"] * params["p_exec"])))

# Create the world and vehicle objects
world = TrafficWorld(params["n_lanes"], 0, 999999)

# Create the vehicle placement based on a Poisson distribution
MAX_VELOCITY = 25 * 0.447  # m/s
VEHICLE_LENGTH = 4.5  # m
time_duration_s = (params["n_other"] * 3600.0 / params["car_density"]) * 10  # amount of time to generate traffic
initial_vehicle_positions = poission_positions(params["car_density"],
                                               params["n_other"] + 1,
                                               params["n_lanes"],
                                               MAX_VELOCITY,
                                               2 * VEHICLE_LENGTH,
                                               position_random_seed=params["seed"])
position_list = initial_vehicle_positions[:params["n_other"] + 1]

# Create the SVOs for each vehicle
if params["random_svo"] == 1:
    list_of_svo = [np.random.choice([0, np.pi / 4.0, np.pi / 2.01]) for i in range(params["n_other"])]
else:
    list_of_svo = [params["svo_theta"] for i in range(params["n_other"])]

(_, _, all_other_vehicles, all_other_x0) = initialize_cars_from_positions(params["N"], params["dt"], world, True,
                                                                          position_list, list_of_svo)

for vehicle in all_other_vehicles:
    # Set theta_ij randomly for all vehicles
    vehicle.theta_ij[-1] = np.random.uniform(0.001, np.pi / 2.01)
    for vehicle_j in all_other_vehicles:
        vehicle.theta_ij[vehicle_j.agent_id] = np.random.uniform(0.001, np.pi / 2.01)

if params["k_lat"]:
    for vehicle in all_other_vehicles:
        vehicle.k_lat = params["k_lat"]
# Save the vehicles and world for this simulation

# xothers_actual, uothers_actual = run_iterative_best_response(all_other_vehicles, world, all_other_x0, params,
#                                                              log_dir, args.load_log_dir, i_mpc_start)

# all_trajectories = np.array(xothers_actual)
# all_control = np.array(uothers_actual)

# np.save(open(log_dir + "/trajectories.npy", 'wb'), all_trajectories)
# np.save(open(log_dir + "/controls.npy", 'wb'), all_control)
solver_params = None
ipopt_params = {'ipopt':{"print_level": 5}}
params['slack'] = True

ego_idx = 0
c_idx = [1,2]
nc_idx = [3,4,5, 3,4,5, 3,4,5, 4 ]


nc = len(c_idx)
nnc = len(nc_idx)


x0 = all_other_x0[ego_idx]
x0_cntrld = [all_other_x0[j] for j in c_idx]
x0_nc = [all_other_x0[j] for j in nc_idx]

all_other_vehicle_params = []
for j in range(len(all_other_vehicles)):
    vp = VehicleParameters(nc, nnc)
    vp.set_param_values(all_other_vehicles[j])
    all_other_vehicle_params.append(vp)

p_ego = all_other_vehicle_params[ego_idx]
p_cntrld = [all_other_vehicle_params[j] for j in c_idx]
p_nc = [all_other_vehicle_params[j] for j in nc_idx]

theta_ego_i = np.pi/4
theta_ic = [np.pi/4 for j in range(nc)]
theta_i_nc = [np.pi/4 for j in range(nnc)]
params["safety_constraint"] = True

mpc = MultiMPC(params["N"], world, nc, nnc, solver_params, params, ipopt_params)

x0warm = np.zeros((mpc.nx))
p0 = mpc.mpcp_to_nlpp(x0, p_ego, theta_ego_i, theta_ic, theta_i_nc, x0_cntrld, p_cntrld, x0_nc, p_nc)
mpc._ubg_list = [float(mpc._ubg_list[i]) for i in range(mpc._ubg_list.shape[0])]
mpc._lbg_list = [float(mpc._lbg_list[i]) for i in range(mpc._lbg_list.shape[0])]

generate = True
if generate:
    from os import system
    mpc.solver.generate_dependencies('gen.c')
    # -O3 is the most optimized
    system('gcc -fPIC -shared -O2 gen.c -o gen.so')
else:
    from casadi import nlpsol
    solver_comp = nlpsol('solver', 'ipopt', './gen.so')
    S = solver_comp(x0=x0warm, p=p0, lbg=mpc._lbg_list, ubg=mpc._ubg_list)
    traj = S['x']

    traj_x = mpc.nlpx_to_mpcx(traj)
    print(traj_x)
# This calls the solver
# S = mpc.solver(x0=x0warm, p=p0, lbg=mpc._lbg_list, ubg=mpc._ubg_list)
# solution = S['x']

