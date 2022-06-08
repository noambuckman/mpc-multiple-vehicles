from src.multiagent_mpc import MultiMPC, mpcp_to_nlpp
from src.utils.ibr_argument_parser import IBRParser
import numpy as np
from src.traffic_world import TrafficWorld
from src.utils.solver_helper import poission_positions, initialize_cars_from_positions
from vehicle import Vehicle
from vehicle_parameters import VehicleParameters
from src.vehicle_mpc_information import VehicleMPCInformation, Trajectory
from src.best_response import solve_best_response_c, solve_warm_starts
from src.warm_starts import generate_warmstarts

parser = IBRParser()
args = parser.parse_args()
params = vars(args)
params["T"] = 1.0
# params["n_other"] = 6
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

(_, _, all_vehicles, all_x0) = initialize_cars_from_positions(params["N"], params["dt"], world, True, position_list,
                                                              list_of_svo)

for vehicle in all_vehicles:
    # Set theta_ij randomly for all vehicles
    vehicle.theta_ij[-1] = np.random.uniform(0.001, np.pi / 2.01)
    for vehicle_j in all_vehicles:
        vehicle.theta_ij[vehicle_j.agent_id] = np.random.uniform(0.001, np.pi / 2.01)

if params["k_lat"]:
    for vehicle in all_vehicles:
        vehicle.k_lat = params["k_lat"]

solver_params = {}
solver_params["solve_amb"] = True
solver_params["slack"] = True
solver_params["n_warm_starts"] = params["default_n_warm_starts"]
solver_params["k_CA"] = params["k_CA_d"]
solver_params["k_CA_power"] = params["k_CA_power"]
solver_params["k_slack"] = params["k_slack_d"]

ipopt_params = {"print_level": 2}
params['slack'] = True

ego_idx = 0
c_idx = []
nc_idx = [1, 2, 3]

nc = len(c_idx)
nnc = len(nc_idx)

params["safety_constraint"] = True

# At each round, we need to call this
## Testing Best Response

response_veh_info = VehicleMPCInformation(all_vehicles[ego_idx], all_x0[ego_idx])
vehs_ibr_info_predicted = [response_veh_info]
warm_starts = generate_warmstarts(response_veh_info.vehicle, world, response_veh_info.x0, vehs_ibr_info_predicted,
                                   params, None, None)

cntrld_vehicles = [all_vehicles[idx] for idx in c_idx]
nonresponse_vehicle_list = [all_vehicles[idx] for idx in nc_idx]

response_x0 = all_x0[ego_idx]
cntrld_x0 = [all_x0[idx] for idx in c_idx]
nonresponse_x0_list = [all_x0[idx] for idx in nc_idx]

other_u_initial = [None for i in range(len(nonresponse_vehicle_list))]
other_x_initial = [None for i in range(len(nonresponse_vehicle_list))]
other_xd_initial = [None for i in range(len(nonresponse_vehicle_list))]
for i in range(len(nonresponse_vehicle_list)):
    other_u_initial[i] = np.zeros((2, params["N"]))
    other_x_initial[i], other_xd_initial[i] = nonresponse_vehicle_list[i].forward_simulate_all(
        nonresponse_x0_list[i], other_u_initial[i])

ctrld_u_warm = [None for i in range(len(cntrld_vehicles))]
ctrld_x_warm = [None for i in range(len(cntrld_vehicles))]
ctrld_xd_warm = [None for i in range(len(cntrld_vehicles))]
for i in range(len(cntrld_vehicles)):
    ctrld_u_warm[i] = np.zeros((2, params["N"]))
    ctrld_x_warm[i], ctrld_xd_warm[i] = cntrld_vehicles[i].forward_simulate_all(cntrld_x0[i], ctrld_u_warm[i])

nonresponse_veh_info = [
    VehicleMPCInformation(nonresponse_vehicle_list[i], nonresponse_x0_list[i], other_u_initial[i], other_x_initial[i],
                          other_xd_initial[i]) for i in range(len(nonresponse_vehicle_list))
]

cntrl_veh_info = [
    VehicleMPCInformation(cntrld_vehicles[i], cntrld_x0[i], ctrld_u_warm[i], ctrld_x_warm[i], ctrld_xd_warm[i])
    for i in range(len(cntrld_vehicles))
]

min_cost_solution = solve_warm_starts(warm_starts, response_veh_info, world, solver_params, params, ipopt_params,
                                      nonresponse_veh_info, cntrl_veh_info)

print(min_cost_solution[2])
print("Debug here")
# # Testing just solve_best_response_
# results = {}
# for warm_key, warm_trajectory in warm_starts.items():
#     r = solve_best_response_c(warm_key,
#                               warm_trajectory,
#                               response_veh_info.vehicle,
#                               cntrld_vehicles,
#                               nonresponse_vehicle_list,
#                               response_x0,
#                               cntrld_x0,
#                               nonresponse_x0_list,
#                               world,
#                               solver_params,
#                               params,
#                               ipopt_params,
#                               other_u_initial,
#                               other_x_initial,
#                               other_xd_initial,
#                               cntrld_u_warm=None,
#                               cntrld_x_warm=None,
#                               cntrld_xd_warm=None,
#                               return_bri=False)

#     results[warm_key] = {"converged": r[0], "value": r[1], "max_slack": r[2]}

# for k in results:
#     print(k, results[k]["max_slack"])
generate = True
# if generate:
#     from os import system
#     mpc.solver.generate_dependencies('gen.c')
#     # -O3 is the most optimized
#     system('gcc -fPIC -shared -O1 gen.c -o gen.so')
# else:
#     from casadi import nlpsol
#     solver_comp = nlpsol('solver', 'ipopt', './gen.so')
#     S = solver_comp(x0=x0warm, p=p0, lbg=mpc._lbg_list, ubg=mpc._ubg_list)
#     traj = S['x']

#     traj_x = nlpx_to_mpcx(traj, params["N"], nc, nnc)
#     print(traj_x)
# This calls the solver

# S = mpc.solver(x0=x0warm, p=p0, lbg=mpc._lbg_list, ubg=mpc._ubg_list)
# solution = S['x']