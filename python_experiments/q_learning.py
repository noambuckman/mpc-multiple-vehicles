import os, sys, pickle
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt
PROJECT_PATHS = ['/home/nbuckman/Dropbox (MIT)/DRL/2020_01_cooperative_mpc/mpc-multiple-vehicles/', 
                '/Users/noambuckman/mpc-multiple-vehicles/',
                os.path.expanduser("~") + "/mpc-multiple-vehicles/",
                ]

for p in PROJECT_PATHS:
    sys.path.append(p)

from src.ibr_argument_parser import IBRParser
from tqdm import trange


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split

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

def update_optimal_svo(theta_ij, q_network, params, random=True):
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


class SVODataset(Dataset):

    def __init__(self, history_file, ambulance_only=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.history = pickle.load(open(history_file, 'rb'))
        self.ambulance_only = ambulance_only

    def __len__(self):
        return len(self.history)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        svo, values = self.history[idx]
        if self.ambulance_only:
          values = values[0].reshape(1)
        svo = svo.squeeze()
        sample = {'svos': svo, 'values': values}
        # print(svo.shape, values.shape)
        return sample


if __name__ == '__main__':
    parser = IBRParser()
    parser.add_argument('--max-svo', type=float, default=np.pi/2.0, help="Max SVO we allow for random")
    parser.add_argument('--epochs', type=int, default = 10, help="Number of simulation epochs")
    parser.add_argument('--learning-rate', type=float, default=0.001, help="learning rate alpha")
    parser.add_argument('--q-train-freq', type=float, default=2, help="Number of iterations before retraining")
    parser.add_argument('--random-svo-rl', type=bool, default=True, help="Just randomly choose the svo")
    parser.add_argument('--history-file', type=str, default=None, help="load a history file")
    parser.add_argument('--train-epochs', type=int, default=10000, help="load a history file")

    parser.set_defaults(
      n_other = 4,
      n_mpc = 2,
      T = 4,
    )
    args = parser.parse_args()
    params = vars(args)   

    assert params["history_file"] is not None

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



    # history = []
    dataset = SVODataset(params["history_file"], ambulance_only=True)
    n_train = int(.8 * len(dataset))
    n_test = len(dataset) - n_train
    training_set, validation_set = random_split(dataset, [n_train, n_test])
    
    training_loader = DataLoader(training_set, batch_size=4, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=4, shuffle=True)


    theta_ij = np.random.rand(params["n_other"], 1) * params["max_svo"] ### Range of SVO is 0 < pi
        ### Train a network to learn a function V(\theta_ij)
        # writer.add_scalar("theta_amb", theta_ij[0])
        # writer.add_scalar("v_amb", V_i_list[0])


    
    optimizer = optim.SGD(q_network.parameters(), lr=params["learning_rate"])
    for ep_tix in trange(params["train_epochs"]):
        # print("Training!")
        optimizer.zero_grad()
        total_loss = 0
        for idx, batch in enumerate(training_loader):
            # V_i_listx = 
            theta_ijx = batch["svos"]
            V_ambx = batch["values"]
            # print(theta_ijx.shape, V_ambx.shape)
            batch_size = V_ambx.shape[0]

            theta_ijx = theta_ijx.to(device)
            V_ambx = V_ambx.to(device)
            V_hatx = q_network.forward(theta_ijx)
            loss = loss_function(V_hatx, V_ambx)
            # print(V_hatx.shape, V_ambx.shape, loss.shape)
            total_loss += loss
            loss.backward()
            optimizer.step()

        total_loss = total_loss / len(training_loader)
        writer.add_scalar("q_loss_train", total_loss, ep_tix)

        total_loss = 0
        for idx, sample in enumerate(validation_loader):
            theta_ijx = sample["svos"]          
            V_ambx = sample["values"]

            V_ambx = V_ambx.to(device)
            theta_ijx = theta_ijx.to(device)

            # theta_ijx = torch.squeeze(theta_ijx)
            V_hatx = q_network.forward(theta_ijx)
            loss = loss_function(V_hatx, V_ambx)
            total_loss += loss
        total_loss = total_loss / len(validation_loader)     
        writer.add_scalar("q_loss_test", total_loss, ep_tix)

        if ep_tix % 1000 == 0:
          n_test_theta = 6
          sample_ambulance_theta = np.linspace(0, np.pi/2.0, n_test_theta)
          V_ambulance = []
          for itx in range(n_test_theta):
            amb_theta = sample_ambulance_theta[itx]
            V_total = 0
            for idx, sample in enumerate(training_loader):
              V_i_listx = sample["values"]
              theta_ijx = sample["svos"]      
              # theta_ijx = torch.squeeze(theta_ijx)

              theta_ijx[:, 0] = amb_theta
              theta_ijx = theta_ijx.to(device)
              # print(theta_ijx.shape, V_i_listx.shape)    

              V_hatx = q_network.forward(theta_ijx)
              V_total += V_hatx.sum()
            writer.add_scalar('V_amb'+str(ep_tix), V_total/len(training_set), itx)       


    writer.add_graph(q_network, torch.from_numpy(theta_ij.flatten()).double().to(device))

    theta_ij = update_optimal_svo(theta_ij, q_network, params, random=params["random_svo_rl"])

    
    writer.flush()
    # history_file = os.path.expanduser("~") + "/mpc_results/"  + params["log_subdir"] + "/history.p"

    # with open(history_file, 'wb') as f:
    #     pickle.dump(history, f)
