import argparse
import os, pickle, json
import numpy as np
from src.utils.plotting import car_plotting
import matplotlib.pyplot as plt
def post_process(args):
    ''' Load the directory and print the'''

    LOGDIR = args.logdir
    X0_PATH = os.path.join(LOGDIR, "x0.p")
    x0 = pickle.load(open(X0_PATH, "rb"))

    U_PATH = os.path.join(LOGDIR, "controls.npy")
    X_PATH = os.path.join(LOGDIR, "trajectories.npy")
    u = np.load(U_PATH)
    x = np.load(X_PATH)

    params = json.load(open(os.path.join(LOGDIR, "params.json")))

    np.set_printoptions(3, suppress=True)
    for ag_i in range(x.shape[0]):
        print(ag_i)
        print(x[ag_i, 2, :])
        print(ag_i, "theta$")
        print(np.rad2deg(x[ag_i, 3, :]))


    print(u[4, :, :])

    os.makedirs(os.path.join(LOGDIR, "plots/"), exist_ok=True)
    plt.figure()
    plt.plot(u[4,0,:])
    plt.ylabel("u0_steering_change")
    plt.xlabel("Timestep")
    plt.savefig(os.path.join(LOGDIR, "plots/", "u0_4.png"))

    plt.figure()
    plt.plot(u[4,1,:])
    plt.ylabel("u1_vel_change")
    plt.xlabel("Timestep")
    plt.savefig(os.path.join(LOGDIR, "plots/", "u1_4.png"))
    return None


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", type=str, help="log directory to analyze")
    parser.add_argument("--analyze_controls", action="store_true", help="load controls")
    args = parser.parse_args()



    post_process(args)
