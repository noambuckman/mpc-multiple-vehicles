# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from src.multiagent_mpc import MultiMPC
from rl.rl_auxilliary import SVODatasetFromLog
import numpy as np
import glob, json, pickle, os, shutil

# %%
results_dir = "/home/nbuckman/mpc_results/rl_txt_0430"
log_dirs = glob.glob(results_directory + "/*/")

# %%
# Remove directories without trajectories
for log in log_dirs:
    with open(log + "/params.json", 'rb') as f:
        params = json.load(f)
    try:
        traj = np.load(log + "/trajectories.npy")
    except FileNotFoundError:
        shutil.rmtree(log)

# %%
dataset = SVODatasetFromLog(log_dirs)

# %%
dataset[6]

# %%

# %%
