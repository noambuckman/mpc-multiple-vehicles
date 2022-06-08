import numpy as np

class Trajectory:
    def __init__(self, u:np.array, x:np.array, xd:np.array):
        self.x = x
        self.u = u
        self.xd = xd


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



