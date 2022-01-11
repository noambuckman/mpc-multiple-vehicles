from src.multiagent_mpc import MultiMPC
from src.utils.ibr_argument_parser import IBRParser
import numpy as np
from src.traffic_world import TrafficWorld
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

solver_params = {}
solver_params["slack"] = True
solver_params["n_warm_starts"] = params["default_n_warm_starts"]
solver_params["k_CA"] = params["k_CA_d"]
solver_params["k_CA_power"] = params["k_CA_power"]
solver_params["k_slack"] = params["k_slack_d"]


params["safety_constraint"] = True




TEST_FLAG = True
if TEST_FLAG:
    for nc in range(0,3):
        for nnc in range(0, 9):
            mpc = MultiMPC(params["N"], world, nc, nnc, solver_params, params, ipopt_params)
            nlp_solver = mpc.solver
            precompiled_code_dir = "/home/nbuckman/mpc-multiple-vehicles/src/compiled_code/"
            nlp_solver_name = "mpc_%02d_%02d.p" % (nc, nnc)
            nlp_solver_path = os.path.join(precompiled_code_dir, nlp_solver_name)      

            pickle.dump(nlp_solver, open(nlp_solver_path, 'wb'))
            
            nlp_lbg, nlp_ubg = mpc._lbg_list, mpc._ubg_list

            bounds = (nlp_lbg, nlp_ubg)

            bounds_path_name = "bounds_%02d%02d.p"%(nc, nnc)
            bounds_full_path = os.path.join(precompiled_code_dir, bounds_path_name)
            pickle.dump(bounds, open(bounds_full_path, 'wb'))

            print("Saving bounds to...nc %d  nnc %d"%(nc, nnc))
