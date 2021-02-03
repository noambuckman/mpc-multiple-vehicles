import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset


class QNet(nn.Module):
    ''' QNetwork that predicts the performance of vehicles '''
    def __init__(self, n_agents_total, output_dimension=1, hidden_dimension=25, p_dropout=0.25, n_hidden_layers=2):
        super(QNet, self).__init__()
        self.n_hidden_layers = n_hidden_layers
        self.p_dropout = p_dropout
        self.hidden_dimension = hidden_dimension

        self.fc_input = nn.Linear(n_agents_total, self.hidden_dimension)
        self.dropout_input = nn.Dropout(0.1)
        self.input_layer = nn.Sequential(self.fc_input, nn.ReLU(), self.dropout_input)

        self.dropout_hidden = nn.Dropout(p_dropout)
        self.fc_hidden = nn.Linear(self.hidden_dimension, self.hidden_dimension)
        self.relu_hidden = nn.ReLU()
        self.hidden_layer = nn.Sequential(self.fc_hidden, self.relu_hidden, self.dropout_hidden)

        self.model = self.input_layer
        for li in range(self.n_hidden_layers):
            self.model = nn.Sequential(self.model, self.hidden_layer)
        self.fc_output = nn.Linear(hidden_dimension, output_dimension)
        self.model = nn.Sequential(self.model, self.fc_output)

    def forward(self, x):

        return self.model(x)


def update_optimal_svo(theta_ij, q_network, params, device, random=True):
    """ Use gradient descent to update SVO values
        theta_ij:  previous thetas
        q_network:  Network without input |theta_ij| and output V_amb
        params: default
        random: True-returns random svos, False(default)
        returns:  updates theta_ij matrix
    """

    if random:
        new_theta_ij = (np.random.rand(*theta_ij.shape) * params["max_svo"])  # Range of SVO is 0 < pi
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
    def __init__(self, history_file: str):
        """
        Args:
            history_file (string): Path to the pickle file.
            ambulance_only (): Only output the ambulance value
        """
        self.history = pickle.load(open(history_file, "rb"))

    def __len__(self):
        return len(self.history)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        svo, values = self.history[idx]
        values = np.array(values)
        svo = svo.squeeze()
        sample = {"svos": svo, "values": values}
        return sample


class SVODatasetFromLog(Dataset):
    def __init__(self, log_directory_paths: str, n_timesteps: int = None):
        """ Args """
        self.log_directory_paths = log_directory_paths
        self.n_timesteps = n_timesteps

        # LOG_DIRECTORY_PATH = self.log_directory_paths[idx]
        self.all_svos_experiments = []
        self.all_values_experiments = []
        self.all_logs_experiments = []

        for LOG_DIRECTORY_PATH in self.log_directory_paths:
            VEHICLE_PATH = LOG_DIRECTORY_PATH + 'ambulance.p'
            OTHER_VEHICLE_PATH = LOG_DIRECTORY_PATH + 'other_vehicles.p'
            TRAJ_PATH = LOG_DIRECTORY_PATH + 'trajectories.npy'
            PARAMS_PATH = LOG_DIRECTORY_PATH + 'params.json'

            with open(TRAJ_PATH, 'rb') as fp:
                traj = np.load(fp)

            if self.n_timesteps is not None:
                traj = traj[:, :, :self.n_timesteps]  #concat trajectory
                if traj.shape[2] != self.n_timesteps:
                    continue
            with open(VEHICLE_PATH, 'rb') as fp:
                ambulance = pickle.load(fp)
            ambulance_svo = ambulance.theta_ij[-1]

            with open(OTHER_VEHICLE_PATH, 'rb') as fp:
                other_vehicles = pickle.load(fp)
            other_svos = [veh.theta_ij[-1] for veh in other_vehicles]

            all_svos = np.array([ambulance_svo] + other_svos).squeeze()
            all_values = np.array(-(traj[:, 0, -1] - traj[:, 0, 0]))
            self.all_svos_experiments.append(all_svos)
            self.all_values_experiments.append(all_values)
            self.all_logs_experiments.append(LOG_DIRECTORY_PATH)

    def __len__(self):
        return len(self.all_svos_experiments)

    def __getitem__(self, idx):
        values = self.all_values_experiments[idx]
        svo = self.all_svos_experiments[idx]
        log = self.all_logs_experiments[idx]
        sample = {"svos": svo, "values": values, "log": log}
        return sample
