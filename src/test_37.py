import pickle
from src.best_response import solve_warm_starts

warm_starts, response_veh_info, world, solver_params, params, ipopt_params, nonresponse_veh_info, cntrl_veh_info = pickle.load(
    open("/home/nbuckman/mpc-multiple-vehicles/src/temp02.p", 'rb'))

print(response_veh_info.x0)
print(nonresponse_veh_info[0].x0)
# print(cntrl_veh_info[0].x0)
ipopt_params['print_level'] = 5
# params["safety_constraint"] = False
min_cost_solution = solve_warm_starts(warm_starts, response_veh_info, world, solver_params, params, ipopt_params,
                                      nonresponse_veh_info, cntrl_veh_info)

print(min_cost_solution[2])