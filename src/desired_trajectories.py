import numpy as np
from src.traffic_world import TrafficWorld
from typing import List, Tuple
import casadi as cas
from numpy.polynomial import Polynomial

class CubicSpline:
    def __init__(self, a = None, b = None, c = None, d = None):
        ''' f(s) = a + b*s + c*s^3 + d*c^3'''
        self.a = a
        self.b = b
        self.c = c
        self.d = d


    def from_2pts(self, s0, value0, valuedot0, s1, value1, valuedot1):
        ''' Derive the coefficients from two points along the spline (s0, s1).  
            Needed are the values and 1st deriv at s0 and s1
        '''
        A_matrix = np.array([[1, s0, s0**2, s0**3], [0, 1, 2*s0, 3*s0**2],[1, s1, s1**2, s1**3], [0, 1, 2*s1, 3*s1**2]])
        b_array = np.array([[value0, valuedot0, value1, valuedot1]]).T
        [a, b, c, d] = np.matmul(np.linalg.inv(A_matrix), b_array)

        return CubicSpline(float(a), float(b), float(c), float(d))

    def as_array(self):
        return np.array([self.a, self.b, self.c, self.d])        

class QuadraticSpline(CubicSpline):
    def __init__(self, a = None, b = None, c = None):
        CubicSpline.__init__(self, a, b, c, 0)    
    
    def from_3pts(self, s0, value0, s1, value1, s2, value2):
        ''' Derive the coefficients from three points along the spline'''

        A_array = np.array([[1, s0, s0**2], [1, s1, s1**2], [1, s2, s2**2]])
        b_array = np.array([[value0, value1, value2]]).T
        [a, b, c] = np.matmul(np.linalg.inv(A_array), b_array)
        return QuadraticSpline(float(a), float(b), float(c))

class LinearSpline(CubicSpline):
    def __init__(self, a = 0.0, b = 1.0):
        ''' y = a + b*s'''
        CubicSpline.__init__(self, a, b, 0.0, 0.0)


####################################################################################################
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
        fd = cas.Function('fd', [s, self.x_coeff, self.y_coeff, self.phi_coeff], [des_traj], ['s', 'x_coeff', 'y_coeff', 'phi_coeff'], ['des_traj'])
        return fd


def cubic_function(n_coeff=4):
    s = cas.MX.sym('s')
    coeff = cas.MX.sym('coeff', n_coeff, 1)

    val = coeff[0]
    for ci in range(1, n_coeff):
        val += coeff[ci] * s**ci

    f = cas.Function("cubic", [s, coeff], [val], ['s','coeff'], ['value'])

    return f


def spline_function(n_coeff=4):
    s = cas.MX.sym('s')
    x_coeff = cas.MX.sym('x_coeff', n_coeff, 1)
    y_coeff = cas.MX.sym('y_coeff', n_coeff, 1)
    phi_coeff = cas.MX.sym('phi_coeff', n_coeff, 1)

    cube = cubic_function(n_coeff)

    f_x_val = cube(s, x_coeff)
    f_y_val = cube(s, y_coeff)
    f_phi_val = cube(s, phi_coeff)

    vec_value = cas.vertcat(f_x_val, f_y_val, f_phi_val)

    fd = cas.Function('fd_vec', [s, x_coeff, y_coeff, phi_coeff], [vec_value], ['s', 'x_coeff', 'y_coeff', 'phi_coeff'], ['vec_value'])
    
    return fd

def piecewise_function(n_splines=3, n_coeff=4):
    s = cas.MX.sym('s')
    x_coeffs = cas.MX.sym('x_coeffs', n_splines, n_coeff)
    y_coeffs = cas.MX.sym('y_coeffs', n_splines, n_coeff)
    phi_coeffs = cas.MX.sym('phi_coeffs', n_splines, n_coeff)
    lengths = cas.MX.sym('lengths', n_splines, 1)

    s0 = cas.fmax(cas.fmin(s - 0, lengths[0]), 0)
    s1 = cas.fmax(cas.fmin(s - lengths[0], lengths[1]), 0)
    s2 = cas.fmax(cas.fmin(s - lengths[0] - lengths[1], lengths[2]), 0)

    fspline = spline_function()
    fd_piecewise_value = fspline(s0, x_coeffs[0,:], y_coeffs[0,:], phi_coeffs[0,:]) + \
                        fspline(s1, x_coeffs[1,:], y_coeffs[1,:], phi_coeffs[1,:]) + \
                        fspline(s2, x_coeffs[2,:], y_coeffs[2,:], phi_coeffs[2,:])

    fd = cas.Function('fd_piecewise', [s, x_coeffs, y_coeffs, phi_coeffs, lengths], [fd_piecewise_value], 
                                    ['s', 'x_coeffs', 'y_coeffs', 'phi_coeffs', 'lengths'], ['fd_piecewise_value']   )
    
    return fd


class LaneFollowingPolynomial(DesiredTrajectoryCoefficients):
    def __init__(self, vehicle, world: TrafficWorld):

        desired_lane_number = vehicle.desired_lane
        yd = world.get_lane_centerline_y(desired_lane_number, right_direction=True)

        x_coeff = [0, 1, 0, 0] # equiv to x(s) = s
        y_coeff = [yd, 0, 0, 0] # equiv to y(s) = yd
        phi_coeff = [0, 0, 0, 0] # equiv to phi(s) = 0

        DesiredTrajectoryCoefficients.__init__(self, x_coeff, y_coeff, phi_coeff)

def phi_from_xy_splines(x_coeff, y_coeff, traj_length):
    ''' Generate a spline where phi is approx parallel the x, y curve'''
    x_poly = Polynomial(x_coeff)
    y_poly = Polynomial(y_coeff)
    s = np.linspace(0, traj_length, 25)
    ys = y_poly(s)
    xs = x_poly(s)

    dy = np.gradient(ys)
    dx = np.gradient(xs)
    phix = np.arctan2(dy, dx)

    cubic_spline = Polynomial.fit(s, phix, 3)

    return cubic_spline.coef


class LaneChangeTrajectoryCoefficients(DesiredTrajectoryCoefficients):
    ''' Generate a smooth cubic spline for changing lanes over a traj_length
        x(s) = s
        y(s=L)=delta_y,  y(s=0)=y'(0)=y'(L)=0
        phi(0) = phi(L) = 0,   phi(L/2) = max_angle = 2*atan2(delta_y, traj_length)
    '''
    def __init__(self, traj_length, delta_y, delta_phi):

        x_coeff = np.array([0, 1, 0, 0]) # Linearly vary over length of traj
        y_coeff = np.array([0, 0, 3*delta_y/traj_length**2, -2*delta_y/traj_length**3])  ## Cubic varying over length of trajectory


        avg_angle = np.arctan2(delta_y, traj_length)
        max_angle = 2*avg_angle
        constant_multiplier = max_angle * 4.0 / traj_length**2  
        # phi_coeff = np.array([0.0, traj_length*constant_multiplier, -constant_multiplier, 0.0])  # phi = constant * s * (traj_lengh - s)

        phi_coeff = QuadraticSpline().from_3pts(0, 0, traj_length/2.0, max_angle, traj_length, delta_phi).as_array()

        # phi_coeff = phi_from_xy_splines(x_coeff, y_coeff, traj_length, phi0)
        DesiredTrajectoryCoefficients.__init__(self, x_coeff, y_coeff, phi_coeff) 


class TrajectorySplineFromPoints(DesiredTrajectoryCoefficients):
    def __init__(self, s0, s1, y0, y1, y0dot, y1dot, s2, phi0, phi1, phi2):

        y_spline = CubicSpline().from_2pts(s0, y0, y0dot, s1, y1, y1dot)
        phi_spline = QuadraticSpline().from_3pts(s0, phi0, s1, phi1, s2, phi2)
        x_spline = LinearSpline()

        DesiredTrajectoryCoefficients.__init__(x_spline.as_array(), y_spline.as_array(), phi_spline.as_array())



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
        self.x_coeff_array, self.y_coeff_array, self.phi_coeff_array, self.lengths_array = self.to_array()


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
        fd_piecewise = fd1(s1, self.polynomial1.x_coeff, self.polynomial1.y_coeff, self.polynomial1.phi_coeff) + \
                fd2(s2, self.polynomial2.x_coeff, self.polynomial2.y_coeff, self.polynomial2.phi_coeff) + \
                fd3(s3, self.polynomial3.x_coeff, self.polynomial3.y_coeff, self.polynomial3.phi_coeff)

        
        vars = [s, self.polynomial1.x_coeff, self.polynomial1.y_coeff, self.polynomial1.phi_coeff, 
                self.polynomial2.x_coeff, self.polynomial2.y_coeff, self.polynomial2.phi_coeff,
                self.polynomial3.x_coeff, self.polynomial3.y_coeff, self.polynomial3.phi_coeff,
                self.length1, self.length2, self.length3
                ]

        return cas.Function('fd', vars, [fd_piecewise], ['s'], ['des_traj'])





class LaneFollowingPiecewiseTrajectory(PiecewiseDesiredTrajectory):
    def __init__(self, vehicle, world: TrafficWorld, spline_lengths: float = 99999):

        polynomial = LaneFollowingPolynomial(vehicle, world)
        
        PiecewiseDesiredTrajectory.__init__(self, polynomial, polynomial, polynomial, spline_lengths, spline_lengths, spline_lengths)



class LaneChangeManueverPiecewise(PiecewiseDesiredTrajectory):
    def __init__(self, traj_length_1, traj_length_2, traj_length_3, delta_y, delta_phi):
        ''' Lane changing equation based on distances for each segment
                _____(s3)____
                /   
                /     <---(s2)
        __(s1)__/      
        '''

        traj1 = LaneChangeTrajectoryCoefficients(traj_length_1, 0.0, 0.0) #straight away
        traj2 = LaneChangeTrajectoryCoefficients(traj_length_2, delta_y, delta_phi) #lane change part
        traj3 = LaneChangeTrajectoryCoefficients(traj_length_3, 0.0, 0.0) #straight away

        PiecewiseDesiredTrajectory.__init__(self, traj1, traj2, traj3, traj_length_1, traj_length_2, traj_length_3)

class InitialFinalManeuver(PiecewiseDesiredTrajectory):
    def __init__(self, traj_length_1, traj_length_2, traj_length_3, delta_y, delta_phi):

        traj1 = LaneChangeTrajectoryCoefficients(traj_length_1, 0.0, 0.0) #straight away
        traj2 = LaneChangeTrajectoryCoefficients(traj_length_2, delta_y, delta_phi) #lane change part
        traj3 = LaneChangeTrajectoryCoefficients(traj_length_3, 0.0, 0.0) #straight away

        PiecewiseDesiredTrajectory.__init__(self, traj1, traj2, traj3, traj_length_1, traj_length_2, traj_length_3)



def complete_original_lane_shift(y_start, y_end, traj_length, y_current, phi_current):
    # Original Lane Change
    delta_phi = 0 - phi_current
    full_lane_change_up = LaneChangeTrajectoryCoefficients(traj_length, y_end - y_start, delta_phi)

    cy = full_lane_change_up.y_coeff
    p = Polynomial(cy)

    # Find where we are on the original trajectory by solving roots

    roots = (p - y_current).roots()
    roots = roots[(roots>0.0) & (roots<traj_length)]
    s_current = roots[0]

    # Return a new polynomial f(s) = f(s+s_current) - f(s_current)
    new_shifted_spline = p + p.deriv(1)*s_current + p.deriv(2)*s_current**2/2 + p.deriv(3)*s_current**3/6 - p(s_current)
    new_coef = new_shifted_spline.coef
    if new_coef.shape[0] < 4:
        new_coef = np.pad(new_coef, (0, 4-new_coef.shape[0]))

    return new_coef, s_current, new_shifted_spline
    


def find_s_on_spline(coeff, value, s_max):
    ''' Find how far have you traversed on spline'''
    p = Polynomial(coeff)
    
    roots = (p - value).roots()
    roots = roots[(roots>0.0) & (roots<s_max)]
    try:
        s_current = roots[0]
    except IndexError as e:
        roots = (p - value).roots()
        print(roots)
        print(roots[(roots>0.0) & (roots<s_max)])
        raise e

    return s_current

def shift_polynomial(coeff, s_current):
    ''' Assume that s_min = 0'''
    p = Polynomial(coeff)
    assert p.degree() <= 3

    # Return a new polynomial f(s) = f(s+s_current) - f(s_current)
    new_shifted_spline = p + p.deriv(1)*s_current + p.deriv(2)*s_current**2/2 + p.deriv(3)*s_current**3/6 - p(s_current)
    new_coef = new_shifted_spline.coef
    if new_coef.shape[0] < 4:
        new_coef = np.pad(new_coef, (0, 4-new_coef.shape[0]))
    return new_coef


def shift_desired_trajectory(desired_trajectory: DesiredTrajectoryCoefficients, y_current, trajectory_length):
    ''' Shift the desired trajectory so s=0 corresponds to the point x,y,phi'''

    s_current = find_s_on_spline(desired_trajectory.y_coeff, y_current, trajectory_length)

    x_coeff = shift_polynomial(desired_trajectory.x_coeff, s_current)
    y_coeff = shift_polynomial(desired_trajectory.y_coeff, s_current)
    phi_coeff = shift_polynomial(desired_trajectory.phi_coeff, s_current)
    
    new_traj = DesiredTrajectoryCoefficients(x_coeff, y_coeff, phi_coeff)
    new_traj_length = trajectory_length - s_current
    return new_traj, new_traj_length

def generate_lane_changing_desired_trajectories(world: TrafficWorld, x0: np.array, T: float, v_mph_range: float = 10.0, n_speed_increments: int = 5) -> List[LaneChangeManueverPiecewise]:
    ''' Generate a few possible desired trajectories based on a lane change or lane following maneuver'''

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
    delta_phi = 0 - phi_start # We want to get to phi=0 by end of trajectory
    
    desired_trajectories = []
    for trajectory_length in trajectory_lengths:

        s1 = 0.25*trajectory_length
        s2 = 0.50*trajectory_length
        s3 = 0.25*trajectory_length + 999999 #This ensures that it doesn't end abruptly

        for delta_y in delta_ys:

            desired_trajectories += [LaneChangeManueverPiecewise(s1, s2, s3, delta_y, delta_phi)]

        if world.n_lanes != 2:
            raise Exception("Error,desired trajectories assumes only two lanes")
            
        top_lane_centerline_y = world.get_lane_centerline_y(1)
        bottom_lane_centerline_y = world.get_lane_centerline_y(0)
        bottom_grass_centerline_y = bottom_lane_centerline_y - world.lane_width
        top_grass_centerline_y = top_lane_centerline_y + world.lane_width
        
        INFTY = 9999999999
        y_splits = sorted([top_lane_centerline_y, bottom_lane_centerline_y, bottom_grass_centerline_y, top_grass_centerline_y, INFTY])
        idx_y = np.searchsorted(y_splits, y_start)
        bottom_line = y_splits[idx_y-1]
        top_line = y_splits[idx_y]
        straight_ahead = LaneChangeTrajectoryCoefficients(9999999, 0, delta_phi)

        EPSILON = 0.001 # needed since we're extrapoliting and thuse current_delta_y needs to be greater than zero
        if y_start > top_lane_centerline_y + EPSILON:
            top_line = max(top_grass_centerline_y, y_start + EPSILON)
            bottom_line = top_lane_centerline_y
            print(top_line, bottom_line)
            full_lane_change_down = LaneChangeTrajectoryCoefficients(trajectory_length, bottom_line - top_line, 0.0)
            current_delta_y = y_start - top_line
            shifted_lane_change_down, new_traj_length = shift_desired_trajectory(full_lane_change_down, current_delta_y, trajectory_length)
            shifted_lane_change_down_piecewise = PiecewiseDesiredTrajectory(shifted_lane_change_down, straight_ahead, straight_ahead, new_traj_length, 99999, 99999)
            desired_trajectories += [shifted_lane_change_down_piecewise]
        elif y_start < bottom_lane_centerline_y - EPSILON:
            top_line = bottom_lane_centerline_y
            bottom_line = min(bottom_grass_centerline_y, y_start - EPSILON)
            full_lane_change_up = LaneChangeTrajectoryCoefficients(trajectory_length, top_line - bottom_line, 0.0)
            current_delta_y = y_start - bottom_line
            shifted_lane_change_up, new_traj_length = shift_desired_trajectory(full_lane_change_up, current_delta_y, trajectory_length)
            shifted_lane_change_up_piecewise = PiecewiseDesiredTrajectory(shifted_lane_change_up, straight_ahead, straight_ahead, new_traj_length, 99999, 99999)
            desired_trajectories += [shifted_lane_change_up_piecewise]            
        elif (bottom_lane_centerline_y + EPSILON < y_start < top_lane_centerline_y - EPSILON):
        # Trajectories for continuing a lane change
            bottom_line = bottom_lane_centerline_y
            top_line = top_lane_centerline_y
            
            full_lane_change_up = LaneChangeTrajectoryCoefficients(trajectory_length, top_line - bottom_line, 0.0)
            current_delta_y = y_start - bottom_line
            shifted_lane_change_up, new_traj_length = shift_desired_trajectory(full_lane_change_up, current_delta_y, trajectory_length)
            shifted_lane_change_up_piecewise = PiecewiseDesiredTrajectory(shifted_lane_change_up, straight_ahead, straight_ahead, new_traj_length, 99999, 99999)
            desired_trajectories += [shifted_lane_change_up_piecewise]

            
            full_lane_change_down = LaneChangeTrajectoryCoefficients(trajectory_length, bottom_line - top_line, 0.0)
            current_delta_y = y_start - top_line
            shifted_lane_change_down, new_traj_length = shift_desired_trajectory(full_lane_change_down, current_delta_y, trajectory_length)
            shifted_lane_change_down_piecewise = PiecewiseDesiredTrajectory(shifted_lane_change_down, straight_ahead, straight_ahead, new_traj_length, 99999, 99999)
            desired_trajectories += [shifted_lane_change_down_piecewise]

        # desired_trajectories += [PiecewiseDesiredTrajectory(straight_ahead, straight_ahead, straight_ahead, 99999, 99999, 99999)]

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