# mpc-multiple-vehicles
Code for a multiple cooperative-mpc for multiple vehicles (esp. ambulance)

Video: https://www.youtube.com/watch?v=iwibkEfj8CI

Screenshot: ![Screenshot](https://github.com/noambuckman/mpc-multiple-vehicles/blob/master/images/sample_sim.gif)


The main code being developed is:

1. src/
2. python_experiments/
3.  jupyter_notebooks/ -- these are the scripts for obtaining the results


The main script for running iterative best response is:
python_experiments/iterative_best_response.py

The main classes are located in src/ with brief description bellow:
1.  src/vehicle.py :  Vehicle() contains dynamics, dimensions, vehicle-specific costs
2.  src/traffic_world.py:  TrafficWorld() contains road and lane dimensions
3.  src/multiagent_mpc.py : MultiMPC: Optimization class which uses CASADI to create an optimization for a single vehicle planning with other vehicles on road
4. src/car_plotting.py:  scripts for plotting and animating vehicle trajectories


Dependents:
-  Scipy
-  Numpy
-  Matplotlib
-  Casadi 3.51

Installation:
conda install -c conda-forge/label/cf202003 casadi
conda install matplotlib
conda install scipy

If you use the code, please cite the paper:
N. Buckman, W. Schwarting, S. Karaman and D. Rus, "Semi-Cooperative Control for Autonomous Emergency Vehicles," 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2021, pp. 7052-7059, doi: 10.1109/IROS51168.2021.9636849.


The main settings that can be changed:
- MPC Time Horizon (T), % of ctrl pts executed, time discretization (dt)
- Iterative Best Response:  # rounds of IBR, allowed amount of slack
-  Vehicle Preferences:  SVO wrt ambulance, collision costs, 



Notes on installing on Supercloud:
1. curl -sL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > "Miniconda3.sh"
2. bash Miniconda3.sh
3. source ~/.bashrc
4. git clone https://github.com/noambuckman/mpc-multiple-vehicles.git 
5. conda env create -f env.yml
6. conda activate mpc
7. python iterative response
8. mv results to /afs/csail.mit.edu/u/n/nbuckman/mpc_results_afs/
