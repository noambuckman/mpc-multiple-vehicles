import numpy as np
from src.traffic_world import TrafficWorld
from typing import List, Tuple
import casadi as cas


class DesiredTrajectoryCoefficients:
    def __init__(self, x_coeff: List[float], y_coeff: List[float], phi_coeff: List[float]):
        # coeff [a, b, c, d] corresponds to a + b*s + c*s^2 + d*b*s^3
        
        self.x_coeff = x_coeff # 4 x 1
        self.y_coeff = y_coeff # 4 x 1
        self.phi_coeff = phi_coeff # 4 x 1
        self.n_coeff = 4


    def gen_f_desired_lane_poly(self) -> cas.Function:
        ''' polynomials: 
            x(s) = cx0 + cx1*s + cx2*s^2 ... cx3*s^3
            y(s) = cy0 + cy1*s + cy2*s^2 ... cy3*s^3
            phi(s) = cphi0 + ... cphi3*s^3
        '''
        s = cas.MX.sym('s')

        assert self.x_coeff.shape == self.y_coeff.shape == self.phi_coeff.shape #For now we require them to be equal
        n_coeff = self.x_coeff.shape[0]

        xd = self.x_coeff[0]
        yd = self.y_coeff[0]
        phid = self.phi_coeff[0]
        for ci in range(1, n_coeff):
            xd += self.x_coeff[ci] * s**ci
            yd += self.y_coeff[ci] * s**ci
            phid += self.phi_coeff[ci] * s**ci
        
        des_traj = cas.vertcat(xd, yd, phid)
        fd = cas.Function('fd', [s], [des_traj], ['s'], ['des_traj'])
        return fd


class LaneFollowingPolynomial(DesiredTrajectoryCoefficients):
    def __init__(self, vehicle, world: TrafficWorld):

        desired_lane_number = vehicle.desired_lane
        yd = world.get_lane_centerline_y(desired_lane_number, right_direction=True)

        x_coeff = [0, 1, 0, 0] # equiv to x(s) = s
        y_coeff = [yd, 0, 0, 0] # equiv to y(s) = yd
        phi_coeff = [0, 0, 0, 0] # equiv to phi(s) = 0

        DesiredTrajectoryCoefficients.__init__(self, x_coeff, y_coeff, phi_coeff)


class LaneChangeTrajectoryCoefficients(DesiredTrajectoryCoefficients):
    ''' Generate a smooth cubic spline for changing lanes over a traj_length
        x(s) = s
        y(s=L)=delta_y,  y(s=0)=y'(0)=y'(L)=0
        phi(0) = phi(L) = 0,   phi(L/2) = max_angle = 2*atan2(delta_y, traj_length)
    '''
    def __init__(self, traj_length, delta_y):

        x_coeff = np.array([0, 1, 0, 0]) # Linearly vary over length of traj
        y_coeff = np.array([0, 0, 3*delta_y/traj_length**2, -2*delta_y/traj_length**3])  ## Cubic varying over length of trajectory


        avg_angle = np.arctan2(delta_y, traj_length)
        max_angle = 2*avg_angle
        constant_multiplier = max_angle * 4.0 / traj_length**2  
        phi_coeff = np.array([0.0, traj_length*constant_multiplier, -constant_multiplier, 0.0])  # phi = constant * s * (traj_lengh - s)

        DesiredTrajectoryCoefficients.__init__(self, x_coeff, y_coeff, phi_coeff)        

#######################################################################################################################                

class PiecewiseDesiredTrajectory:
    def __init__(self, polynomial1: DesiredTrajectoryCoefficients = None, 
                        polynomial2: DesiredTrajectoryCoefficients = None, 
                        polynomial3: DesiredTrajectoryCoefficients = None, 
                        length1: float = None, 
                        length2: float = None, 
                        length3: float = None):
        ''' Object that holds a piecewise desired trajectory 
            Currently, we only allow three parts
        
        
        '''
        self.polynomial1 = polynomial1
        self.polynomial2 = polynomial2
        self.polynomial3 = polynomial3
        self.length1 = length1
        self.length2 = length2
        self.length3 = length3

        self.n_splines = 3


    def to_array(self)-> Tuple[np.array, np.array, np.array, np.array]: 
        x_coeff_array = []
        y_coeff_array = []
        phi_coeff_array = []
        lengths_array = []

        for (polynomial, length) in [(self.polynomial1, self.length1), (self.polynomial2, self.length2), (self.polynomial3, self.length3)]:
            x_coeff_array.append(polynomial.x_coeff)
            y_coeff_array.append(polynomial.y_coeff)
            phi_coeff_array.append(polynomial.phi_coeff)
            lengths_array.append(length)

        x_coeff_array = np.array(x_coeff_array)
        y_coeff_array = np.array(y_coeff_array)
        phi_coeff_array = np.array(phi_coeff_array)
        lengths_array = np.array(lengths_array)

        return x_coeff_array, y_coeff_array, phi_coeff_array, lengths_array

    def from_array(self, x_coeff_array: np.array, y_coeff_array: np.array, phi_coeff_array: np.array, lengths_array: np.array):

        polynomial1 = DesiredTrajectoryCoefficients(x_coeff_array[0, :], y_coeff_array[0, :], phi_coeff_array[0, :])
        polynomial2 = DesiredTrajectoryCoefficients(x_coeff_array[1, :], y_coeff_array[1, :], phi_coeff_array[1, :])
        polynomial3 = DesiredTrajectoryCoefficients(x_coeff_array[2, :], y_coeff_array[2, :], phi_coeff_array[2, :])

        length1 = lengths_array[0]
        length2 = lengths_array[1]
        length3 = lengths_array[2]

        return PiecewiseDesiredTrajectory(polynomial1, polynomial2, polynomial3, length1, length2, length3)





    def gen_f_desired_3piecewise_poly(self) -> cas.Function:
        ''' Generate a piecewise function consisting of polynomials
            
        '''
        s = cas.MX.sym('s')

        fd1 = self.polynomial1.gen_f_desired_lane_poly()
        fd2 = self.polynomial2.gen_f_desired_lane_poly()
        fd3 = self.polynomial3.gen_f_desired_lane_poly()

        s1 = cas.fmax(cas.fmin(s - 0, self.length1), 0)
        s2 = cas.fmax(cas.fmin(s - self.length1, self.length2), 0)
        s3 = cas.fmax(cas.fmin(s - self.length1 - self.length2, self.length3), 0)
        fd_piecewise = fd1(s1)   + fd2(s2) + fd3(s3)

        return cas.Function('fd', [s], [fd_piecewise], ['s'], ['des_traj'])





class LaneFollowingPiecewiseTrajectory(PiecewiseDesiredTrajectory):
    def __init__(self, vehicle, world: TrafficWorld, spline_lengths: float = 99999):

        polynomial = LaneFollowingPolynomial(vehicle, world)
        
        PiecewiseDesiredTrajectory.__init__(self, polynomial, polynomial, polynomial, spline_lengths, spline_lengths, spline_lengths)



class LaneChangeManueverPiecewise(PiecewiseDesiredTrajectory):
    def __init__(self, traj_length_1, traj_length_2, traj_length_3, delta_y):
        ''' Lane changing equation based on distances for each segment
                _____(s3)____
                /   
                /     <---(s2)
        __(s1)__/      
        '''

        traj1 = LaneChangeTrajectoryCoefficients(traj_length_1, 0.0) #straight away
        traj2 = LaneChangeTrajectoryCoefficients(traj_length_2, delta_y) #lane change part
        traj3 = LaneChangeTrajectoryCoefficients(traj_length_3, 0.0) #straight away

        PiecewiseDesiredTrajectory.__init__(self, traj1, traj2, traj3, traj_length_1, traj_length_2, traj_length_3)




# def gen_f_desired_3piecewise_coeff(x_coeff, y_coeff, phi_coeff, spline_lengths):
#     ''' 
#         x_coeff:  n_coeff x n_splines  where the coefficients correspond 
    
#             polynomials: 
#                 x(s) = cx0 + cx1*s + cx2*s^2 ... cx3*s^3
#                 y(s) = cy0 + cy1*s + cy2*s^2 ... cy3*s^3
#                 phi(s) = cphi0 + ... cphi3*s^3
#     '''

#     assert x_coeff.shape[0] == y_coeff.shape[0] == phi_coeff.shape[0] #For now we require them to be equal
#     assert x_coeff.shape[1] == y_coeff.shape[1] == phi_coeff.shape[1] == spline_lengths.shape[0] == 3

#     n_splines = spline_lengths.shape[0]

#     # Generate each polynomial spline
#     polynomial_functions = []
#     for idx in range(n_splines):
#         fd = gen_f_desired_lane_poly(x_coeff[:, idx], y_coeff[:, idx], phi_coeff[:, idx])
#         polynomial_functions.append(fd)

#     # Combine into one piecewise function. Right now we only allow 3 polynomial functions

#     poly1 = polynomial_functions[0]
#     poly2 = polynomial_functions[1]
#     poly3 = polynomial_functions[2]
#     L1 = spline_lengths[0]
#     L2 = spline_lengths[1]
#     L3 = spline_lengths[2]

#     piecewise_fd = gen_f_desired_3piecewise_poly(poly1, poly2, poly3, L1, L2, L3)
    
#     return piecewise_fd





# def gen_f_desired_3piecewise_coeff(x_coeff, y_coeff, phi_coeff, spline_lengths):
#     ''' 
#         x_coeff:  n_coeff x n_splines  where the coefficients correspond 
    
#             polynomials: 
#                 x(s) = cx0 + cx1*s + cx2*s^2 ... cx3*s^3
#                 y(s) = cy0 + cy1*s + cy2*s^2 ... cy3*s^3
#                 phi(s) = cphi0 + ... cphi3*s^3
#     '''

#     assert x_coeff.shape[0] == y_coeff.shape[0] == phi_coeff.shape[0] #For now we require them to be equal
#     assert x_coeff.shape[1] == y_coeff.shape[1] == phi_coeff.shape[1] == spline_lengths.shape[0] == 3

#     n_splines = spline_lengths.shape[0]

#     # Generate each polynomial spline
#     polynomial_functions = []
#     for idx in range(n_splines):
#         fd = gen_f_desired_lane_poly(x_coeff[:, idx], y_coeff[:, idx], phi_coeff[:, idx])
#         polynomial_functions.append(fd)

#     # Combine into one piecewise function. Right now we only allow 3 polynomial functions

#     poly1 = polynomial_functions[0]
#     poly2 = polynomial_functions[1]
#     poly3 = polynomial_functions[2]
#     L1 = spline_lengths[0]
#     L2 = spline_lengths[1]
#     L3 = spline_lengths[2]

#     piecewise_fd = gen_f_desired_3piecewise_poly(poly1, poly2, poly3, L1, L2, L3)
    
#     return piecewise_fd

           

# def generate_desired_polynomial_coeff(vehicle: Vehicle, world: TrafficWorld) -> DesiredTrajectoryCoefficients: 
#     ''' For now we assume the desired coefficients are a straight line along the center of the lane''' 
#     desired_lane_number = vehicle.desired_lane
#     yd = world.get_lane_centerline_y(desired_lane_number, right_direction=True)

#     x_coeff = [0, 1, 0, 0] # equiv to x(s) = s
#     y_coeff = [yd, 0, 0, 0] # equiv to y(s) = yd
#     phi_coeff = [0, 0, 0, 0] # equiv to phi(s) = 0

#     return DesiredTrajectoryCoefficients(x_coeff, y_coeff, phi_coeff)


# def generate_desired_piecewise_coeff(vehicle: Vehicle, world: TrafficWorld) -> PiecewiseDesiredTrajectory:
#     ''' Generate a lane keeping trajectory. It is piecewise but repeates'''

#     poly1 = generate_desired_polynomial_coeff(vehicle, world)
#     l1 = 9999

#     piecewise_traj = PiecewiseDesiredTrajectory(poly1, poly1, poly1, l1, l1, l1)

#     return piecewise_traj





# def generate_desired_polynomial_coeff_constant_lane(vehicle: Vehicle, world: TrafficWorld) -> DesiredTrajectoryCoefficients: 
#     ''' For now we assume the desired coefficients are a straight line along the center of the lane''' 
#     desired_lane_number = vehicle.desired_lane
#     yd = world.get_lane_centerline_y(desired_lane_number, right_direction=True)

#     x_coeff = [0, 1, 0, 0] # equiv to x(s) = s
#     y_coeff = [yd, 0, 0, 0] # equiv to y(s) = yd
#     phi_coeff = [0, 0, 0, 0] # equiv to phi(s) = 0

#     return DesiredTrajectoryCoefficients(x_coeff, y_coeff, phi_coeff)

# def generate_lane_change_coeff(traj_length, delta_y) -> DesiredTrajectoryCoefficients:
#     ''' Generate a smooth cubic spline for changing lanes over a traj_length
#         x(s) = s
#         y(s=L)=delta_y,  y(s=0)=y'(0)=y'(L)=0
#         phi(0) = phi(L) = 0,   phi(L/2) = max_angle = 2*atan2(delta_y, traj_length)
#     '''
#     x_coeff = [0, 1, 0, 0] # Linearly vary over length of traj
#     y_coeff = [0, 0, 3*delta_y/traj_length**2, -2*delta_y/traj_length**3]  ## Cubic varying over length of trajectory


#     avg_angle = np.arctan2(delta_y, traj_length)
#     max_angle = 2*avg_angle
#     constant_multiplier = max_angle * 4.0 / traj_length**2  
#     phi_coeff = [0.0, traj_length*constant_multiplier, -constant_multiplier, 0.0]  # phi = constant * s * (traj_lengh - s)

#     return DesiredTrajectoryCoefficients(x_coeff, y_coeff, phi_coeff)








# def generate_piecewise_lane_change_maneuver_coeff(s1, s2, s3, delta_y) -> DesiredTrajectoryCoefficients:
#     ''' Lane changing equation based on distances for each segment
#                _____(s3)____
#               /   
#              /     <---(s2)
#     __(s1)__/      
#     '''

#     traj1 = generate_lane_change_coeff(s1, 0.0) # straight away
#     traj2 = generate_lane_change_coeff(s2, delta_y) # lane change part
#     traj3 = generate_lane_change_coeff(s3, 0.0) # straight_away
#     fd1 = gen_f_desired_lane_poly(np.array(traj1.x_coeff), np.array(traj1.y_coeff), np.array(traj1.phi_coeff))
#     fd2 = gen_f_desired_lane_poly(np.array(traj2.x_coeff), np.array(traj2.y_coeff), np.array(traj2.phi_coeff))
#     fd3 = gen_f_desired_lane_poly(np.array(traj3.x_coeff), np.array(traj3.y_coeff), np.array(traj3.phi_coeff))

#     fd = gen_f_desired_3piecewise_poly(fd1, fd2, fd3, s1, s2, s3)
    
#     return fd
    


def generate_lane_changing_desired_trajectories(vehicle, world: TrafficWorld, x0: np.array, T: float, v_mph_range: float = 10.0, n_speed_increments: int = 5) -> List[LaneChangeManueverPiecewise]:
    ''' Generate a few possible desired trajectories based on a lane change or lane following maneuver'''

    x_start = x0[0]
    y_start = x0[1]
    phi_start = x0[2]

    v_start = x0[4]

    MPH_TO_MS = 0.447
    lower_speed = v_start - v_mph_range / 2.0 * MPH_TO_MS
    upper_speed = v_start + v_mph_range / 2.0 * MPH_TO_MS
    lower_traj_length = lower_speed * T
    upper_traj_length = upper_speed * T
    trajectory_lengths = np.linspace(start=lower_traj_length, stop=upper_traj_length, num=n_speed_increments, endpoint=True)
    
    delta_ys = [world.get_lane_centerline_y(lane_i) - y_start for lane_i in range(world.n_lanes)]
    
    desired_trajectories = []
    for trajectory_length in trajectory_lengths:

        s1 = 0.25*trajectory_length
        s2 = 0.50*trajectory_length
        s3 = 0.25*trajectory_length

        for delta_y in delta_ys:
            desired_trajectories += [LaneChangeManueverPiecewise(s1, s2, s3, delta_y)]

    
    # desired_trajectories += [generate_desired_polynomial_coeff_constant_lane(vehicle, world)]


    return desired_trajectories






# def generate_fancy_lane_changing_desired_trajectories(vehicle: Vehicle, world: TrafficWorld, x0: np.array, T: float, v_mph_range: float = 10.0, n_speed_increments: int = 5) -> List[DesiredTrajectoryCoefficients]:


#     initial_straight_length = 10.0
#     change_length = 5.0
#     final_straight_length = 10.0
    
#     traj1 = generate_lane_change_coeff(initial_straight_length, 0, 0.0)
#     traj2 = generate_lane_change_coeff(change_length, 0, delta_y)
#     traj3 = generate_lane_change_coeff(final_straight_length, 0, 0.0)
#     fd1 = gen_f_desired_lane_poly(np.array(traj1.x_coeff), np.array(traj1.y_coeff), np.array(traj1.phi_coeff))
#     fd2 = gen_f_desired_lane_poly(np.array(traj2.x_coeff), np.array(traj2.y_coeff), np.array(traj2.phi_coeff))
#     fd3 = gen_f_desired_lane_poly(np.array(traj3.x_coeff), np.array(traj3.y_coeff), np.array(traj3.phi_coeff))

#     fd = gen_f_desired_3piecewise_poly(fd1, fd2, fd3, L1, L2, L3)





# def gen_f_desired_lane_poly(x_coeff: List[float], y_coeff: List[float], phi_coeff: List[float]):
#     ''' polynomials: 
#         x(s) = cx0 + cx1*s + cx2*s^2 ... cx3*s^3
#         y(s) = cy0 + cy1*s + cy2*s^2 ... cy3*s^3
#         phi(s) = cphi0 + ... cphi3*s^3
#     '''
#     s = cas.MX.sym('s')

#     assert x_coeff.shape == y_coeff.shape == phi_coeff.shape #For now we require them to be equal
#     n_coeff = x_coeff.shape[0]

#     xd = x_coeff[0]
#     yd = y_coeff[0]
#     phid = phi_coeff[0]
#     for ci in range(1, n_coeff):
#         xd += x_coeff[ci] * s**ci
#         yd += y_coeff[ci] * s**ci
#         phid += phi_coeff[ci] * s**ci
    
#     des_traj = cas.vertcat(xd, yd, phid)
#     fd = cas.Function('fd', [s], [des_traj], ['s'], ['des_traj'])
#     return fd




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from src.vehicle import Vehicle
    world = TrafficWorld(2, 0)
    vehicle = Vehicle(0.2)
    vehicle.desired_lane = 0

    x0 = np.array([0, world.get_lane_centerline_y(1), 0, 0, 4.0, 0])
    desired_trajectories = generate_lane_changing_desired_trajectories(vehicle, world, x0, 10.0)
    ns = 50
    arrow_length = 10.0 * x0[4] / ns 


    L = x0[4] * 10.0


    fig, ax = plt.subplots(1,1)
    s = np.linspace(0, 9.0, ns)
    fd = LaneChangeManueverPiecewise(10, 5, 10, 2.0).gen_f_desired_3piecewise_poly()
    Xd = fd.map(ns)(s)
    x = Xd[0, :]
    y = Xd[1,:]
    phi = Xd[2,:]
    ax.plot(x, y, 'o')
    ax.hlines(y=world.get_lane_centerline_y(0), xmin=0, xmax = s[-1], linestyle='--')
    ax.hlines(y=world.get_lane_centerline_y(1), xmin=0, xmax = s[-1], linestyle='--')
    for i in range(len(s)):
        ax.arrow(float(x[i]), float(y[i]), float(np.cos(phi[i])*arrow_length), float(np.sin(phi[i])*arrow_length), label=None)
    ax.set_xlim([0, max(s)])
    ax.set_ylim([0, 5])    
    plt.show()
    # y_final = 3.6
    # plt.plot(s, y_final/L**2*s**2 - 2*y_final/L**3**s**3)
    fig, axs = plt.subplots(4,3)
    for idx, traj in enumerate(desired_trajectories):
        s = np.linspace(0, x0[4] * (10.0+6.0), ns)

        fd = traj.gen_f_desired_3piecewise_poly()
        Xd = fd.map(ns)(s)
        x = Xd[0, :]
        y = Xd[1,:]
        phi = Xd[2,:]
        print(x)
        print(y)
        print(phi)
        # x = traj.x_coeff[0] + traj.x_coeff[1]*s + traj.x_coeff[2]*s**2 + traj.x_coeff[3]*s**3 
        # y = traj.y_coeff[0] + traj.y_coeff[1]*s + traj.y_coeff[2]*s**2 + traj.y_coeff[3]*s**3 
        # phi = traj.phi_coeff[0] + traj.phi_coeff[1]*s + traj.phi_coeff[2]*s**2 + traj.phi_coeff[3]*s**3 
        # print(idx%4, idx//4)
        ax = axs[idx%4, idx//4]
        ax.plot(x, y, 'o', label="%d"%idx)
        ax.hlines(y=world.get_lane_centerline_y(0), xmin=0, xmax = s[-1], linestyle='--')
        ax.hlines(y=world.get_lane_centerline_y(1), xmin=0, xmax = s[-1], linestyle='--')
        print(idx)
        for i in range(len(s)):
            print("x", x[i])
            print("y", y[i])
            print("phi", phi[i])
            print(np.cos(phi[i])*arrow_length)
            print(np.sin(phi[i])*arrow_length)

            ax.arrow(float(x[i]), float(y[i]), float(np.cos(phi[i])*arrow_length), float(np.sin(phi[i])*arrow_length), label=None)
        ax.set_xlim([0, max(s)])
        ax.set_ylim([-5, 5])
        # plt.plot()
    plt.legend()
    plt.show()


    plt.plot()