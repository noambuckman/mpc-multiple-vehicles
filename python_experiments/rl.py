import time, datetime, os, sys, pickle, psutil, json, string, random
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt
import casadi as cas
import copy as cp
PROJECT_PATHS = ['/home/nbuckman/Dropbox (MIT)/DRL/2020_01_cooperative_mpc/mpc-multiple-vehicles/', 
                '/Users/noambuckman/mpc-multiple-vehicles/',
                os.path.expanduser("~") + "/mpc-multiple-vehicles/",
                ]

for p in PROJECT_PATHS:
    sys.path.append(p)

from src.traffic_world import TrafficWorld
import src.multiagent_mpc as mpc
import src.car_plotting_multiple as cmplot
import src.solver_helper as helper


from src.iterative_best_response import run_iterative_best_response
from src.ibr_argument_parser import IBRParser
from tqdm import trange


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

test_Vi_list = [np.array([[-1.49e-30,  2.23e+00,  4.49e+00,  6.78e+00,  9.10e+00,  1.15e+01,
         1.38e+01,  1.63e+01,  1.87e+01],
       [ 5.12e-30, -6.45e-03, -4.52e-02, -1.18e-01, -2.04e-01, -2.93e-01,
        -3.93e-01, -5.00e-01, -6.02e-01],
       [-1.43e-32, -8.67e-03, -2.54e-02, -3.59e-02, -3.73e-02, -3.95e-02,
        -4.38e-02, -4.35e-02, -3.97e-02],
       [-1.22e-32, -3.49e-02, -3.16e-02, -9.70e-03,  4.17e-03, -1.24e-02,
        -3.89e-03,  4.95e-03,  8.97e-03],
       [ 1.11e+01,  1.12e+01,  1.14e+01,  1.15e+01,  1.17e+01,  1.19e+01,
         1.20e+01,  1.22e+01,  1.23e+01],
       [ 0.00e+00,  2.23e+00,  4.49e+00,  6.78e+00,  0.00e+00,  2.36e+00,
         4.75e+00,  7.17e+00,  9.62e+00]]), np.array([[ 9.16e+00,  1.14e+01,  1.36e+01,  1.58e+01,  1.81e+01,  2.03e+01,
         2.25e+01,  2.48e+01,  2.70e+01],
       [ 3.70e+00,  3.69e+00,  3.65e+00,  3.54e+00,  3.37e+00,  3.18e+00,
         3.00e+00,  2.85e+00,  2.73e+00],
       [ 9.60e-33, -8.64e-03, -3.47e-02, -6.49e-02, -8.21e-02, -8.36e-02,
        -7.42e-02, -6.10e-02, -4.90e-02],
       [ 6.57e-32, -3.49e-02, -6.98e-02, -5.19e-02, -1.70e-02,  1.10e-02,
         2.66e-02,  2.68e-02,  2.12e-02],
       [ 1.11e+01,  1.12e+01,  1.12e+01,  1.12e+01,  1.12e+01,  1.12e+01,
         1.12e+01,  1.12e+01,  1.12e+01],
       [ 0.00e+00,  2.22e+00,  4.46e+00,  6.69e+00,  0.00e+00,  2.24e+00,
         4.47e+00,  6.71e+00,  8.94e+00]]), np.array([[ 1.60e+01,  1.82e+01,  2.04e+01,  2.27e+01,  2.49e+01,  2.71e+01,
         2.93e+01,  3.16e+01,  3.38e+01],
       [ 3.70e+00,  3.71e+00,  3.75e+00,  3.87e+00,  4.04e+00,  4.23e+00,
         4.42e+00,  4.55e+00,  4.62e+00],
       [ 6.55e-34,  8.64e-03,  3.47e-02,  6.61e-02,  8.54e-02,  8.75e-02,
         7.21e-02,  4.56e-02,  2.05e-02],
       [ 3.69e-33,  3.49e-02,  6.98e-02,  5.64e-02,  2.15e-02, -1.34e-02,
        -4.83e-02, -5.85e-02, -4.24e-02],
       [ 1.11e+01,  1.12e+01,  1.12e+01,  1.12e+01,  1.12e+01,  1.12e+01,
         1.12e+01,  1.12e+01,  1.12e+01],
       [ 0.00e+00,  2.22e+00,  4.46e+00,  6.69e+00,  0.00e+00,  2.24e+00,
         4.47e+00,  6.71e+00,  8.94e+00]]), np.array([[ 2.08e+01,  2.30e+01,  2.52e+01,  2.75e+01,  2.97e+01,  3.19e+01,
         3.42e+01,  3.64e+01,  3.86e+01],
       [-4.77e-32, -6.41e-03, -4.80e-02, -1.36e-01, -2.42e-01, -3.38e-01,
        -4.35e-01, -5.28e-01, -6.07e-01],
       [ 1.24e-33, -8.64e-03, -2.99e-02, -4.64e-02, -4.56e-02, -4.25e-02,
        -4.32e-02, -3.89e-02, -3.22e-02],
       [-2.54e-32, -3.49e-02, -5.07e-02, -1.57e-02,  1.92e-02, -6.90e-03,
         4.06e-03,  1.33e-02,  1.38e-02],
       [ 1.11e+01,  1.12e+01,  1.12e+01,  1.12e+01,  1.12e+01,  1.12e+01,
         1.12e+01,  1.12e+01,  1.12e+01],
       [ 0.00e+00,  2.22e+00,  4.46e+00,  6.69e+00,  0.00e+00,  2.24e+00,
         4.47e+00,  6.71e+00,  8.94e+00]]), np.array([[ 3.62e+01,  3.84e+01,  4.06e+01,  4.28e+01,  4.51e+01,  4.73e+01,
         4.96e+01,  5.18e+01,  5.40e+01],
       [ 3.70e+00,  3.70e+00,  3.71e+00,  3.71e+00,  3.72e+00,  3.72e+00,
         3.73e+00,  3.73e+00,  3.74e+00],
       [ 2.15e-34,  1.14e-03,  2.59e-03,  2.59e-03,  1.95e-03,  2.06e-03,
         2.58e-03,  2.37e-03,  1.79e-03],
       [-1.68e-33,  4.61e-03,  1.21e-03, -1.21e-03, -1.37e-03,  1.82e-03,
         2.54e-04, -1.09e-03, -1.25e-03],
       [ 1.11e+01,  1.12e+01,  1.12e+01,  1.12e+01,  1.12e+01,  1.12e+01,
         1.12e+01,  1.12e+01,  1.12e+01],
       [ 0.00e+00,  2.22e+00,  4.46e+00,  6.69e+00,  0.00e+00,  2.24e+00,
         4.47e+00,  6.71e+00,  8.94e+00]])]




class QNet(nn.Module):
    def __init__(self, n_agents_total, hidden_dimension = 25):
        super(QNet, self).__init__()
        self.fc_input = nn.Linear(n_agents_total, hidden_dimension)
        self.dropout = nn.Dropout(0.25)
        self.fc_output = nn.Linear(hidden_dimension, 1)

    def forward(self, x):

        x = self.fc_input(x)  ### input here is a flattened SVO matrix theta_ij
        x = F.relu(x)
        x = self.dropout(x)
        output = self.fc_output(x)  ### output here is a list of V_i

        return output



def run_simulation(load_log_dir, params, theta_ij):
    '''Runs a simulation with IBR using the determined svo matrix theta_ij
    Returns:  List of utilities and individual rewards
    '''

    if load_log_dir is None:
        ### Generate directory structure and log name
        params["start_time_string"] = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        alpha_num = string.ascii_lowercase[:8] + string.digits
        if params["log_subdir"] is None:
            subdir_name = ''.join(random.choice(alpha_num) for j in range(4)) + '-' + ''.join(random.choice(alpha_num) for j in range(4)) + "-" + params["start_time_string"]
        else:
            subdir_name = params["log_subdir"]
        log_dir = os.path.expanduser("~") + "/mpc_results/"  + subdir_name + "/"
        for f in [log_dir+"imgs/", log_dir+"data/", log_dir+"vids/", log_dir+"plots/"]:
            os.makedirs(f, exist_ok = True)

        ### Determine number of control points in the optimization
        i_mpc_start = 0
        params['N'] = max(1, int(params["T"]/params["dt"]))
        params['number_ctrl_pts_executed'] = max(1, int(np.floor(params['N']*params['p_exec'])))

        ### Create the world and vehicle objects
        world = TrafficWorld(params["n_lanes"], 0, 999999)
        
        ### Create the vehicle placement based on a Poisson distribution
        MAX_VELOCITY = 25 * 0.447 # m/s
        VEHICLE_LENGTH = 4.5 #m
        time_duration_s = (params["n_other"] * 3600.0 / params["car_density"] ) * 10 # amount of time to generate traffic
        initial_vehicle_positions = helper.poission_positions(params["car_density"], int(time_duration_s), params["n_lanes"] , MAX_VELOCITY, VEHICLE_LENGTH, position_random_seed = params["seed"])
        position_list = initial_vehicle_positions[:params["n_other"]]
        

        list_of_svo = theta_ij
       
        ambulance, amb_x0, all_other_vehicles, all_other_x0 = helper.initialize_cars_from_positions(params["N"], params["dt"], world,  
                                                                    True, 
                                                                    position_list, list_of_svo)    


        ### Save the vehicles and world for this simulation        
        for i in range(len(all_other_vehicles)):
            pickle.dump(all_other_vehicles[i], open(log_dir + "data/mpcother%03d.p"%i,'wb'))
        pickle.dump(ambulance, open(log_dir + "data/mpcamb.p",'wb'))
        pickle.dump(world, open(log_dir + "data/world.p",'wb'))
        print("Results saved in log %s:"%log_dir)
    else:
        print("Preloading settings from log %s"%args.load_log_dir)
        log_dir = args.load_log_dir
        with open(args.load_log_dir + "params.json",'rb') as fp:
            params = json.load(fp)
        i_mpc_start = args.mpc_start_iteration
        ambulance = pickle.load(open(log_dir + "data/mpcamb.p",'rb'))
        all_other_vehicles = [pickle.load(open(log_dir + "data/mpcother%03d.p"%i,'rb')) for i in range(params["n_other"])]
        world = pickle.load(open(log_dir + "data/world.p",'rb'))
        params["pid"] = os.getpid()

        ### We need to get initial conditions for the iterative best response
    
    
    #### Initialize the state and control arrays
    params["pid"] = os.getpid()
    if params['n_other'] != len(all_other_vehicles):
        raise Exception("n_other larger than  position list")
    with open(log_dir + 'params.json', 'w') as fp:
        json.dump(params, fp, indent=2)

    xamb_actual, xothers_actual = run_iterative_best_response(params, log_dir, load_log_dir, i_mpc_start, amb_x0, all_other_x0, ambulance, all_other_vehicles, world)

    V_i_all = [xamb_actual] + xothers_actual
    return V_i_all


def update_optimal_svo(theta_ij, q_network, params, random=False):
  ''' Use gradient descent to update SVO values 
    theta_ij:  previous thetas
    q_network:  Network without input |theta_ij| and output V_amb
    params: default
    random: True-returns random svos, False(default)

    returns:  updates theta_ij matrix
  '''

  
  if random:
    new_theta_ij = np.random.rand(*theta_ij.shape) * params["max_svo"]  ### Range of SVO is 0 < pi
  else:

    theta_ij = torch.from_numpy(theta_ij.flatten()).double().to(device)
    theta_ij.requires_grad = True
    
    svo_optimizer = optim.SGD([theta_ij], lr=0.1)
    svo_optimizer.zero_grad()
    V_hat = q_network.forward(theta_ij)    
    V_hat.backward()
    svo_optimizer.step()
    print("Updating SVO", theta_ij)
    new_theta_ij = theta_ij.detach().cpu().numpy()
    
  return new_theta_ij


if __name__ == '__main__':
    parser = IBRParser()
    parser.add_argument('--max-svo', type=float, default=np.pi/2.0, help="Max SVO we allow for random")
    parser.add_argument('--epochs', type=int, default = 10, help="Number of simulation epochs")
    parser.add_argument('--learning-rate', type=float, default=0.00001, help="learning rate alpha")
    parser.add_argument('--q-train-freq', type=float, default=2, help="Number of iterations before retraining")
    parser.add_argument('--random-svo-rl', type=bool, default=False, help="Just randomly choose the svo")
    parser.add_argument('--offline-history', type=str, default=None, help="preload an offine history of random svos")
    parser.set_defaults(
      n_other = 4,
      n_mpc = 2,
      T = 4,
    )
    args = parser.parse_args()
    params = vars(args)   

    # ## Over ride but I'm not sure of a better way to do this
    # params["n_other"] = 4
    # params["n_mpc"] = 2
    # params["T"] = 2

    writer = SummaryWriter()
    ### Initialize Q Network
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")        
    q_network = QNet(params["n_other"])  #Let's start by only changing theta's of other vehicles
    q_network.double()
    q_network.to(device)
    loss_function = nn.MSELoss()



    history = []
    theta_ij = np.random.rand(params["n_other"], 1) * params["max_svo"] ### Range of SVO is 0 < pi
    print("Random, initial theta_ij")
    for ep_ix in trange(params["epochs"]):

        TEST = False
        if not TEST:
            all_vehicle_trajectories = run_simulation(None, params, theta_ij)
            V_i_list = [-(x[0, -1] - x[0, 0]) for x in all_vehicle_trajectories]
        else:
            V_i_list = test_Vi_list





        

        ### Train a network to learn a function V(\theta_ij)
        history.append( (theta_ij, V_i_list) )  
        writer.add_scalar("theta_amb", theta_ij[0])
        writer.add_scalar("v_amb", V_i_list[0])

        if ep_ix % params["q_train_freq"]:
            print("Training!")
            optimizer = optim.SGD(q_network.parameters(), lr=params["learning_rate"])
            optimizer.zero_grad()
            loss = 0
            for idx, (theta_ijx, V_i_listx) in enumerate(history):
                V_ambx = np.array([V_i_listx[0]])
                theta_ijx = torch.from_numpy(theta_ijx.flatten()).double().to(device)
                V_ambx = torch.from_numpy(V_ambx).to(device)   
                V_hatx = q_network.forward(theta_ijx)
                loss += loss_function(V_hatx, V_ambx)
            loss.backward()
            writer.add_scalar("q_loss", loss, ep_ix)
            optimizer.step()

        writer.add_graph(q_network, torch.from_numpy(theta_ij.flatten()).double().to(device))

        theta_ij = update_optimal_svo(theta_ij, q_network, params, random=params["random_svo_rl"])
        
        # theta_ij = torch.from_numpy(theta_ij.flatten()).double().to(device)
        # # theta_ij = torch.rand((params["n_other"],1), dtype=torch.double).flatten().to(device)
        # print(theta_ij)
        # print(q_network)
        # V_hat = q_network.forward(theta_ij)
        # print("Actual V", V_hat)
        # print("Predicted V", V_ambulance)

        # ### Use differentiation to find V_hat/dx   <---where dx is the input now not the weights
        # theta_optimizer = optim.SGD(theta_ij, lr=params["learning_rate"])
        # theta_optimizer.zero_grad()
        # V_hat.backward()
        # theta_optimizer.step()    
        # print("Updated SVO", theta_ij)

    writer.flush()
    history_file = os.path.expanduser("~") + "/mpc_results/"  + params["log_subdir"] + "/history.p"

    with open(history_file, 'wb') as f:
        pickle.dump(history, f)
