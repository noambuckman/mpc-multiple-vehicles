from src.multiagent_mpc import MultiMPC
from src.utils.ibr_argument_parser import IBRParser
import numpy as np
from src.traffic_world import TrafficWorld
from src.best_response import generate_solver_params


parser = IBRParser()
parser.add_argument("--nc", type=int, default=1, help="Number of Controlled Vehicles")
parser.add_argument("--nnc", type=int, default=4, help="Number of Non-Response Vehicles")

args = parser.parse_args()
params = vars(args)

# Determine number of control points in the optimization
params["N"] = max(1, int(params["T"] / params["dt"]))
params["number_ctrl_pts_executed"] = max(1, int(np.floor(params["N"] * params["p_exec"])))
world = TrafficWorld(params["n_lanes"], 0, 999999)

# Create the
ipopt_params = {"print_level": 5, "max_cpu_time":10.0}
params['slack'] = True
s_params = generate_solver_params(params, 0, 0)

n_coeff_d = 4

params["safety_constraint"] = True

for safety_constraint in [False]:
    for nc in range(0,3):
        for nnc in range(0, 11):
            mpc = MultiMPC(params["N"], nc, nnc, n_coeff_d, params, ipopt_params, safety_constraint)
            nlp_solver = mpc.solver
            precompiled_code_dir = "/home/nbuckman/mpc-multiple-vehicles/compiled_code/"
            
            solver_path = mpc.save_solver_pickle(precompiled_code_dir)
            bounds_path = mpc.save_bounds_pickle(precompiled_code_dir)
            
            print("Saving bounds to...nc %d  nnc %d"%(nc, nnc))
            print("Solver Path:  %s"%solver_path)
            print("Bounds Path: %s"%bounds_path) 
