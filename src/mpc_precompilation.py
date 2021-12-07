from src.multiagent_mpc import MultiMPC
from src.utils.ibr_argument_parser import IBRParser
import numpy as np
from src.traffic_world import TrafficWorld
from os import system
import datetime

parser = IBRParser()
parser.add_argument("--nc", type=int, default=2, help="Number of Controlled Vehicles")
parser.add_argument("--nnc", type=int, default=4, help="Number of Non-Response Vehicles")

args = parser.parse_args()
params = vars(args)

# Determine number of control points in the optimization
params["N"] = max(1, int(params["T"] / params["dt"]))
params["number_ctrl_pts_executed"] = max(1, int(np.floor(params["N"] * params["p_exec"])))
world = TrafficWorld(params["n_lanes"], 0, 999999)

# Create the
ipopt_params = {'ipopt': {"print_level": 5}}
params['slack'] = True

solver_params = {}
solver_params["solve_amb"] = True
solver_params["slack"] = True
solver_params["n_warm_starts"] = params["default_n_warm_starts"]
solver_params["k_CA"] = params["k_CA_d"]
solver_params["k_CA_power"] = params["k_CA_power"]
solver_params["k_slack"] = params["k_slack_d"]

solver_name_prefix = "mpc_%02d_%02d" % (args.nc, args.nnc)

params["safety_constraint"] = True

mpc = MultiMPC(params["N"], world, args.nc, args.nnc, solver_params, params, ipopt_params)
start_time = datetime.datetime.now()

print("Compiling %s" % (solver_name_prefix))
mpc.solver.generate_dependencies('%s.c' % solver_name_prefix)
# -O3 is the most optimized
system('gcc -fPIC -shared -O %s.c -o %s.so' % (solver_name_prefix, solver_name_prefix))
print("Done Compiling %s.  Duration: %s" % (solver_name_prefix, (datetime.datetime.now() - start_time)))
