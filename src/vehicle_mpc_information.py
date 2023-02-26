import numpy as np
import copy as cp

class Trajectory:
    def __init__(self, u:np.array, x:np.array, xd:np.array):
        self.x = x # state
        self.u = u # control
        self.xd = xd # desired control


    def transform_to_local(self, x0):
        new_x = cp.deepcopy(self.x)
        new_x[0:2, :] = self.x[0:2, :] - x0[0:2].reshape(2, 1)

        return Trajectory(self.u, new_x, self.xd)

    def transform_to_global(self, x0):
        new_x = cp.deepcopy(self.x)
        new_x[0:2, :] = self.x[0:2, :] + x0[0:2].reshape(2, 1)

        return Trajectory(self.u, new_x, self.xd)



class VehicleMPCInformation:
    ''' Helper class that holds the state of each vehicle and vehicle information'''
    def __init__(self, vehicle, x0: np.array, u: np.array = None, x: np.array = None, xd: np.array = None):
        self.vehicle = vehicle
        self.x0 = x0

        self.u = u
        self.x = x
        self.xd = xd

    def update_state(self, u:np.array, x:np.array, xd:np.array):
        self.u = u
        self.x = x
        self.xd = xd

    def update_state_from_traj(self, traj: Trajectory):
        self.u = traj.u
        self.x = traj.x
        self.xd = traj.xd

    
    def transform_to_local(self, x0):

        new_x = cp.deepcopy(self.x)
        new_x[0:2, :] = self.x[0:2, :] - x0[0:2].reshape(2, 1)
        

        new_x0 = cp.deepcopy(self.x0)
        new_x0[0:2] = self.x0[0:2] - x0[0:2]
        return VehicleMPCInformation(self.vehicle, new_x0, self.u, new_x, self.xd)        

    def transform_to_global(self, x0):

        new_x = cp.deepcopy(self.x)
        new_x[0:2, :] = self.x[0:2, :] + x0[0:2].reshape(2, 1)
        

        new_x0 = cp.deepcopy(self.x0)
        new_x0[0:2] = self.x0[0:2] + x0[0:2]

        return VehicleMPCInformation(self.vehicle, new_x0, self.u, new_x, self.xd)     

