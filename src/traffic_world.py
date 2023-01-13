import numpy as np


class TrafficWorld():
    def __init__(self, n_lanes_right, n_lanes_left, x_max=np.infty, lane_width=3.7):

        self.lane_width = lane_width  #m avg lane width
        self.n_lanes_right = n_lanes_right
        self.n_lanes_left = n_lanes_left
        self.n_lanes = self.n_lanes_left + self.n_lanes_right

        self.grass_width = self.lane_width 

        self.total_height = self.get_window_height()

        self.y_min = -0.5 * self.lane_width - self.grass_width
        self.y_max = (-0.5 + self.n_lanes) * self.lane_width + self.grass_width
        self.x_max = x_max
        self.x_min = -self.lane_width  # NO REAL GOOD REASON

    def get_window_height(self):
        return 2 * self.grass_width + self.n_lanes * self.lane_width

    def get_lane_centerline_y(self, lane_number, right_direction=True):
        # count starts at 0
        if right_direction:
            centerline_y = lane_number * self.lane_width
        else:
            centerline_y = (self.n_lanes_right + 1) * self.lane_width + lane_number * self.lane_width

        return centerline_y

    def get_bottom_grass_y(self):
        y_bottom = self.y_min
        y_top = self.y_min + self.grass_width
        y_center = self.y_min + self.grass_width / 2.0
        return y_bottom, y_center, y_top

    def get_top_grass_y(self):
        y_top = self.y_max
        y_bottom = self.y_max - self.grass_width
        y_center = self.y_max - self.grass_width / 2.0
        return y_bottom, y_center, y_top

    def get_lane_from_x0(self, x0, right_direction=True):
        ''' return the lane number from vehicle's position '''

        y = x0[1]
        min_lane = -1
        min_dist = np.infty

        if not right_direction:
            raise Exception("We haven't implemented left lanes yet")
        else:
            for li in range(self.n_lanes):
                center_y = self.get_lane_centerline_y(li, right_direction)
                y_dist = np.abs(center_y - y)
                if y_dist <= min_dist:
                    min_dist = y_dist
                    min_lane = li

        return min_lane


######################################################3
###Grass  ##########
###            Lane 2 Left
###
###            Lane 1 Left
###
###            Lane 2
###
###(xmin, 0)   Lane 1                       (xmax, 0)
###
###Grass #########
###
###
###