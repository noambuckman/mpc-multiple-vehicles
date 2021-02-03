import numpy as np
from collections import OrderedDict
import json

from argparse import ArgumentParser
from numpy.core.arrayprint import DatetimeFormat
from tqdm import trange
import copy
import socket
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split, ConcatDataset
import datetime
from rl.rl_auxilliary import QNet, SVODataset, SVODatasetFromLog

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--n-timesteps", type=int, default=None)
    parser.add_argument("--history_files", type=str, default=None, help="load a history file", nargs='+')
    parser.add_argument("--log_directories", type=str, default=None, help="load log dir", nargs="+")

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
    parser.add_argument("--torch-log-dir", type=str, default='runs/')
    parser.add_argument('--n-other', type=int, default=4, help="Number non-ambulance vehicles")

    args = parser.parse_args()
    params = vars(args)

    torch_log_dir = args.torch_log_dir + datetime.datetime.now().strftime('%m%d_%H%M') + socket.gethostname() + "/"
    print("Logs: %s" % torch_log_dir)
    writer = SummaryWriter(log_dir=torch_log_dir)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)

    q_network = QNet(params["n_other"] + 1, params["n_other"] + 1, n_hidden_layers=args.n_hidden_layers)
    q_network.double()
    q_network.to(device)

    datasets = []
    if params["history_files"]:
        for history_file in params["history_files"]:
            dataset = SVODataset(history_file)
            datasets.append(dataset)
        dataset = ConcatDataset(datasets)
    elif params["log_directories"]:
        dataset = SVODatasetFromLog(params["log_directories"], params["n_timesteps"])
    else:
        raise Exception("No simulations were provided")

    n_train = int(0.8 * len(dataset))
    n_test = len(dataset) - n_train
    training_set, validation_set = random_split(dataset, [n_train, n_test])

    training_loader = DataLoader(training_set, batch_size=5, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=5, shuffle=True)

    dataloaders = OrderedDict({'train': training_loader, 'validation': validation_loader})

    theta_ij = (np.random.rand(params["n_other"], 1) * params["max_svo"])
    writer.add_text('Params', json.dumps(params, indent=1))
    writer.add_text('Dataset Sizes', 'Train: %d   Val: %d' % (len(training_set), len(validation_set)))
    optimizer = optim.SGD(q_network.parameters(), lr=params["learning_rate"])
    loss_function = nn.MSELoss()
    print("Learning Agent:", args.learning_agent)

    for ep_tix in trange(params["train_epochs"]):

        for dataloader_type in dataloaders:
            if dataloader_type == 'train':
                q_network.train()
            else:
                q_network.eval()
            total_loss = 0
            dataloader = dataloaders[dataloader_type]

            for idx, batch in enumerate(dataloader):

                theta_ijx = batch["svos"]
                # TODO: rename V_ambx to V_learningagent
                V_ambx = batch["values"]
                # print(V_ambx.shape, theta_ijx.shape)
                batch_size = V_ambx.shape[0]

                theta_ijx = theta_ijx.to(device)
                V_ambx = V_ambx.to(device)

                V_hatx = q_network.forward(theta_ijx)
                V_hatx = torch.squeeze(V_hatx)
                loss = loss_function(V_hatx, V_ambx)
                total_loss += loss
                if dataloader_type == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # if ep_tix == 0:
                #     V = V_ambx.sum(axis=1)
                #     avgV_batch = V.mean()
                #     writer.add_scalar("V_exp", avgV_batch, idx)

            total_loss = total_loss / len(dataloader)
            writer.add_scalar("q_loss_" + dataloader_type, total_loss, ep_tix)

            # q_network.eval()
            # total_loss = 0
            # for idx, sample in enumerate(validation_loader):
            #     theta_ijx = sample["svos"]
            #     V_ambx = sample["values"]

            #     V_ambx = V_ambx.to(device)
            #     theta_ijx = theta_ijx.to(device)

            #     # theta_ijx = torch.squeeze(theta_ijx)
            #     V_hatx = q_network.forward(theta_ijx)
            #     loss = loss_function(V_hatx, V_ambx)
            #     total_loss += loss
            # total_loss = total_loss / len(validation_loader)
            # writer.add_scalar("q_loss_test", total_loss, ep_tix)

        MODEL_PATH = torch_log_dir + 'model.pt'
        if ep_tix % 1000 == 0:
            q_network.eval()
            # torch.save(
            #     {
            #         'epoch': ep_tix,
            #         'model_state_dict': q_network.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'loss': total_loss,
            #     }, MODEL_PATH)
            torch.save(q_network, MODEL_PATH)

        if ep_tix % 1000 == 0:
            n_test_svos = 20**params["n_other"] + 1
            q_network.eval()

            theta_ij = torch.rand(n_test_svos, params["n_other"] + 1, dtype=torch.double) * params["max_svo"]
            theta_ij = theta_ij.to(device)
            V_hat = q_network.forward(theta_ij)
            V = V_hat.sum(axis=1)
            min_idx = torch.argmin(V)
            theta_min = theta_ij[min_idx, :]
            for agent_i in range(theta_min.shape[0]):
                writer.add_scalar("theta_min_" + str(agent_i), theta_min[agent_i] * 180 / np.pi, ep_tix)
            writer.add_scalar("V_min", V[min_idx], ep_tix)
            min_theta_dist = np.infty
            min_theta_log = ""
            min_theta_v = np.infty
            # theta_min = theta_min.to(device)
            for idx, batch in enumerate(dataloader):
                theta_dist = torch.sum((batch['svos'].to(device) - theta_min)**2, axis=1)
                min_idx = torch.argmin(theta_dist)
                if theta_dist[min_idx] < min_theta_dist:
                    min_theta_dist = theta_dist[min_idx]
                    min_theta_log = batch['log'][min_idx]
                    min_theta_v = batch['values'][min_idx].sum(axis=0)
            if ep_tix % 1000 == 0:
                writer.add_text("closet_exp", min_theta_log, ep_tix)
                writer.add_scalar("closest_log_theta_error", min_theta_dist, ep_tix)
                writer.add_scalar("closest_v", min_theta_v, ep_tix)

        if ep_tix % 1000 == 0:
            q_net_copy = copy.deepcopy(q_network)
            q_net_copy.eval()
            INPUT_OPTIMIZER_EPOCHS = int(4000)
            svo_inputs = torch.rand(1, params["n_other"] + 1, dtype=torch.double, device=device) * params["max_svo"]
            svo_inputs.requires_grad_(True)
            input_optimizer = optim.Adam([svo_inputs], params["learning_rate"])
            for iop_epoch in range(INPUT_OPTIMIZER_EPOCHS):
                input_optimizer.zero_grad()
                output = q_net_copy(svo_inputs.clamp(min=0, max=np.pi / 2))
                loss_to_min = output.sum()
                writer.add_scalar("team_v_loss_min_%d" % ep_tix, loss_to_min, iop_epoch)
                loss_to_min.backward()
                input_optimizer.step()
                print(svo_inputs)
                # svo_inputs = torch.clamp(svo_inputs, 0, np.pi / 2.0)
            print("Optimal SVO", svo_inputs)
            for agent_i in range(theta_min.size()[0]):
                writer.add_scalar("theta_min_learned" + str(agent_i), theta_min[agent_i] * 180 / np.pi, ep_tix)

    writer.add_graph(q_network, theta_ij)

    # theta_ij = update_optimal_svo(theta_ij,
    #                               q_network,
    #                               params,
    #                               random=params["random_svo_rl"])

    writer.flush()
