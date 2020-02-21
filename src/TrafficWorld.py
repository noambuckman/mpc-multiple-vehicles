
import numpy as np

class TrafficWorld():
    def __init__(self, lane_width, n_lanes_right, n_lanes_left, x_max):
        
        self.lane_width = lane_width
        self.n_lanes_right = n_lanes_right
        self.n_lanes_left = n_lanes_left
        self.n_lanes = self.n_lanes_left + self.n_lanes_right


        self.total_height = self.get_window_height()
        
        self.y_min = 0 - 1.5*self.lane_width
        self.y_max = 0 + (1 - 0.5 + self.n_lanes) * self.lane_width
        self.x_min = 0 - self.lane_width # NO REAL GOOD REASON
        self.x_max = x_max


    def get_window_height(self):
        n_lanes_wgrass = self.n_lanes + 2
        total_height = n_lanes_wgrass * self.lane_width
        return total_height    

    def get_lane_centerline_y(self, lane_number, right_direction=True):
        if right_direction:
            centerline_y = lane_number*self.lane_width
        else:
            centerline_y = (self.n_lanes_right + 1)*self.lane_width + lane_number*self.lane_width

        return centerline_y

    def get_bottom_grass_y(self):
        y_bottom = self.y_min
        y_top = self.y_min + self.lane_width
        y_center = self.y_min + self.lane_width/2.0
        return y_bottom, y_center, y_top

    def get_top_grass_y(self):
        y_top = self.y_max
        y_bottom = self.y_max - self.lane_width
        y_center = (y_top + y_bottom)/2.0
        return y_bottom, y_center, y_top    



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