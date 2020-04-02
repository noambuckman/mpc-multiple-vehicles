# mpc-multiple-vehicles
Code for a multiple cooperative-mpc for multiple vehicles (esp. ambulance)

Screenshot: ![Screenshot](https://github.com/noambuckman/mpc-multiple-vehicles/blob/master/images/sample.png)


The main code being developed is:


1.  MPC_Casadi -- homemade clasess for creating CASADI optimizations for MPC
2.  car_plotitng -- homemade plotting to create the animations/plots
3.  jupyter_notebooks/ -- these are the scripts for obtaining the results


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
