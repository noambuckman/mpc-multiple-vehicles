from src.multiagent_mpc import MultiMPC
from src.utils.ibr_argument_parser import IBRParser
import numpy as np
from src.traffic_world import TrafficWorld
from src.best_response import generate_solver_params

from os import system
import pickle, os

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



params["safety_constraint"] = True


for nc in range(0,3):
    for nnc in range(0, 9):
        mpc = MultiMPC(params["N"], params["dt"], world, nc, nnc, params, ipopt_params, params["safety_constraint"])
        nlp_solver = mpc.solver
        precompiled_code_dir = "/home/nbuckman/mpc-multiple-vehicles/src/compiled_code/"
        
        mpc.save_solver_pickle(precompiled_code_dir)
        mpc.save_bounds_pickle(precompiled_code_dir)
        
        print("Saving bounds to...nc %d  nnc %d"%(nc, nnc))
