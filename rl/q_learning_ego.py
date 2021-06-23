import numpy as np
from collections import OrderedDict
import json

from argparse import ArgumentParser
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split, ConcatDataset

from rl.rl_auxilliary import QNet, SVODataset, SVODatasetFromLog

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--n-timesteps", type=int, default=None)
    parser.add_argument("--history_files", type=str, default=None, help="load a history file", nargs='+')
    parser.add_argument("--log_directories", type=str, default=None, help="load log dir", nargs="+")
    parser.add_argument("--dataset", type=str, default=None, help="load saved dataset")

    parser.add_argument("--max-svo", type=float, default=np.pi / 2.0, help="Max SVO we allow for random")
    parser.add_argument("--epochs", type=int, default=10, help="Number of simulation epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="learning rate alpha")
    parser.add_argument(
        "--q-train-freq",
        type=float,
        default=2,
        help="Number of iterations before retraining",
    )
    parser.add_argument("--n-hidden-layers", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--random-svo-rl", type=bool, default=True, help="Just randomly choose the svo")
    parser.add_argument("--train-epochs", type=int, default=100000, help="load a history file")
    parser.add_argument("--learning-agent", type=int, default=0, help="agent to select for optimizing")

    parser.add_argument('--n-other', type=int, default=4, help="Number non-ambulance vehicles")
    args = parser.parse_args()
    params = vars(args)

    writer = SummaryWriter()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Create/load the dataset of experiments
    datasets = []
    if params["history_files"]:
        for history_file in params["history_files"]:
            dataset = SVODataset(history_file)
            datasets.append(dataset)
        dataset = ConcatDataset(datasets)
    elif params["log_directories"]:
        dataset = SVODatasetFromLog(params["log_directories"], params["n_timesteps"])
    elif params["dataset"]:
        dataset = torch.load(params["dataset"])
    else:
        raise Exception("No simulations were provided")

    # Split training and test datset for test
    n_train = int(0.8 * len(dataset))
    n_test = len(dataset) - n_train
    training_set, validation_set = random_split(dataset, [n_train, n_test])

    training_loader = DataLoader(training_set, batch_size=5, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=5, shuffle=True)

    dataloaders = OrderedDict({'train': training_loader, 'validation': validation_loader})

    writer.add_text('Params', json.dumps(params, indent=1))
    writer.add_text('Dataset Sizes', 'Train: %d   Val: %d' % (len(training_set), len(validation_set)))

    print("Learning Agent:", args.learning_agent)

    q_network = QNet(1, 1, n_hidden_layers=args.n_hidden_layers)
    q_network.double()
    q_network.to(device)

    optimizer = optim.SGD(q_network.parameters(), lr=params["learning_rate"])
    loss_function = nn.MSELoss()

    for ep_tix in trange(params["train_epochs"]):
        # Train network
        for dataloader_type in dataloaders:
            if dataloader_type == 'train':
                q_network.train()
            else:
                q_network.eval()
            total_loss = 0
            dataloader = dataloaders[dataloader_type]

            for idx, batch in enumerate(dataloader):

                theta_ego = batch["svos"][:, 0:1]
                V_ego = batch["values"][:, 0:1]
                # theta_all = theta_all.reshape(theta_all.shape[0] * theta_all.shape[1], 1, 1)
                # V_all = V_all.reshape(V_all.shape[0] * V_all.shape[1], 1, 1)
                # cheat and reshape so only 1 agent in QNet

                theta_ego = theta_ego.to(device)
                V_ego = V_ego.to(device)

                V_hat_ego = q_network.forward(theta_ego)
                # V_hat_ego = torch.squeeze(V_hat_ego)
                loss = loss_function(V_hat_ego, V_ego)
                total_loss += loss
                if dataloader_type == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            total_loss = total_loss / len(dataloader)
            writer.add_scalar("q_loss_" + dataloader_type, total_loss, ep_tix)

        # Find the best SVO for the ego vehicle using V_hat
        if ep_tix % 1000 == 0:
            # Try multiple SVOs and predict the value
            n_test_svos = 20**params["n_other"] + 1
            q_network.eval()

            theta_ego = torch.rand(n_test_svos, 1, dtype=torch.double) * params["max_svo"]
            theta_ego = theta_ego.to(device)
            V_hat = q_network.forward(theta_ego)
            # V = V_hat.sum(axis=1)
            min_idx = torch.argmin(V_hat)
            theta_min = theta_ego[min_idx, :]
            for agent_i in range(theta_min.shape[0]):
                writer.add_scalar("theta_min_" + str(agent_i), theta_min[agent_i] * 180 / np.pi, ep_tix)
            writer.add_scalar("V_min", V_hat[min_idx], ep_tix)
            min_theta_dist = np.infty
            min_theta_log = ""
            min_theta_v = np.infty

            INPUT_OPTIMIZER_EPOCHS = int(4000)
            svo_inputs = torch.rand(1, 1, dtype=torch.double, device=device) * params["max_svo"]
            svo_inputs.requires_grad_(True)
            input_optimizer = optim.Adam([svo_inputs], params["learning_rate"])
            for iop_epoch in range(INPUT_OPTIMIZER_EPOCHS):
                input_optimizer.zero_grad()
                output = q_network(svo_inputs.clamp(min=0, max=np.pi / 2))
                loss_to_min = output.squeeze()
                writer.add_scalar("team_v_loss_min_%d" % ep_tix, loss_to_min, iop_epoch)
                loss_to_min.backward()
                input_optimizer.step()
                # svo_inputs = torch.clamp(svo_inputs, 0, np.pi / 2.0)
            print("Optimal SVO", svo_inputs)

            # Save a checkpoint
            PATH = writer.log_dir + "/model.pth"
            torch.save(
                {
                    'epoch': ep_tix,
                    'model_state_dict': q_network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss,
                }, PATH)
            continue

    writer.add_graph(q_network, theta_ego)

    # theta_ij = update_optimal_svo(theta_ij,
    #                               q_network,
    #                               params,
    #                               random=params["random_svo_rl"])

    writer.flush()
