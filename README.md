# mpc-multiple-vehicles
Code for a multiple cooperative-mpc for multiple vehicles (esp. ambulance)

Screenshot: ![Screenshot](https://github.com/noambuckman/mpc-multiple-vehicles/blob/master/images/sample.png)


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
<Insert Title>

Working paper can be found on Overleaf:


The main settings that can be changed:
- MPC Time Horizon (T), % of ctrl pts executed, time discretization (dt)
- Iterative Best Response:  # rounds of IBR, allowed amount of slack
-  Vehicle Preferences:  SVO wrt ambulance, collision costs, 
