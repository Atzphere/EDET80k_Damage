# This generates the text file for the annealing pattern (and gives you a visual representation of the pattern if you want).
# .run() will generate the text file and .print_graph() will generate the text file and plot the graph. As well as animation()

import numpy as np
import matplotlib.pyplot as plt
import math
import os
import random
import matplotlib.animation as animation

from coordinate_to_voltage_test import PositionVoltageConverter


# Place your file name here. Put complete file path
# I believe this is called "VoltagePositionPairs.txt" in the actual software
FILENAME = "C:\\Users\\Qiren\\Desktop\\EDET80k_Anneal\\TAPV-2\\Application\\pythonFiles\\DataTextFiles\\michaeltest1.txt"

# x1, y1 are the coordinates of the bottom left corner of the rectangle. x2, y2 are the coordinates of the top right corner of the rectangle
# Note that these are cartesian coordinates, NOT voltage coordinates
X1, Y1 = 5, 20
X2, Y2 = 16, 30

# Type of annealing pattern. Options are "snake", "zigzag", "spiral", "random", "halton_sequence"
OPTION = "halton_sequence" 

# Number of annealing points. Highly recommended to change it based on the pattern. (Tip: spiral would need less points than zigzag)
NUM_POINTS = 5000

# Current and pulse time
CURRENT = 2.03
PULSE_TIME = 1.03

DELAY_TIME = 0.5

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Not recommended to change anything below this line (EXCEPT FOR THE FUNCTION write_laser_path_to_file 
# to change what the text file looks like)
# And, only if you are add new options or change the existing options, 
# you can change the code below.

class AnnealingPattern(object):
    def __init__(self, filename, x1, y1, x2, y2, option, num_points, pulse_time, current, delay_time):
        self.filename = filename
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.option = option
        self.num_points = num_points
        self.pulse_time = pulse_time
        self.current = current
        self.delay_time = delay_time

    

    def options(self):
        try:
            if self.option == "snake":
                laser_path = self.create_snake()
            elif self.option == "zigzag":
                laser_path = self.create_zigzag()
            elif self.option == "spiral":
                laser_path = self.create_spiral()
            elif self.option == "random":
                laser_path = self.create_random()
            elif self.option == "halton_sequence":
                laser_path = self.generate_halton_points()
            else:
                print("Error: Option not recognized.")
                return None
        except AttributeError:
            print("Error: Option attribute not found.")
            return None

        return laser_path

    
    # EDIT THIS FUNCTION TO CHANGE THE FORMAT OF THE TEXT FILE
    # The format is X_Galvo_Voltage, Y_Galco_Voltage, Time_to_hold_instructions, Laser_Current_A

    # Time_to_hold_instructions is the time to hold the laser at that point.
    # Laser_Current_A is the current of the laser at that point.
    def write_laser_path_to_file_cartesian(self, laser_path):

        out_file = open(self.filename, 'w')
        for i in range(len(laser_path)):
            
            out_file.write(f"{laser_path[i][0]}, {laser_path[i][1]}, {self.pulse_time}, {self.current}\n")
        out_file.close()
        return None
    
    def write_laser_path_to_file_voltage(self, laser_path):

        out_file = open(self.filename, 'w')
        for i in range(len(laser_path)):
            x, y = laser_path[i]
            converter = PositionVoltageConverter()
            x_voltage, y_voltage = converter.voltage_given_position((x,y))
            out_file.write(f"{x_voltage}, {y_voltage}, {self.pulse_time}, {self.current}\n")
        out_file.close()
        return None
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    
    def plot_from_laser_path_in_cartesian(self, laser_path):
        # The points are where the laser will be and the red is the path of the laser

        x = [i[0] for i in laser_path]
        y = [i[1] for i in laser_path]
        plt.scatter(x, y)
        plt.plot(x, y, color='red', alpha=0.3)
        plt.show()
        return None
    
    def plot_from_file(self):
        # read the file and plot the points
        # plots the laser path as directed from the text file

        in_file = open(self.filename, 'r')
        x = []
        y = []
        for line in in_file:
            line = line.strip()
            line = line.split(',')
            x.append(float(line[0]))
            y.append(float(line[1]))
        plt.scatter(x, y)
        plt.plot(x, y, color='red', alpha=0.3)
        plt.show()
        return None
    
    def create_snake(self):
        # Like a S shape

        laser_path = []
        width = self.x2 - self.x1
        height = self.y2 - self.y1
        num_rows = int(np.ceil(np.sqrt(self.num_points * (height / width))))
        num_cols = int(np.ceil(self.num_points / num_rows))

        x_coords = np.linspace(self.x1, self.x2, num_cols)
        y_coords = np.linspace(self.y1, self.y2, num_rows)
        
        for i, y in enumerate(y_coords):
            if i % 2 == 0:
                for x in x_coords:
                    laser_path.append((x, y))
            else:
                for x in reversed(x_coords):
                    laser_path.append((x, y))
        return laser_path
    
    def create_zigzag(self):
        
        laser_path = []
        width = self.x2 - self.x1
        height = self.y2 - self.y1
        num_rows = int(np.ceil(np.sqrt(self.num_points * (height / width))))
        num_cols = int(np.ceil(self.num_points / num_rows))

        x_coords = np.linspace(self.x1, self.x2, num_cols)
        y_coords = np.linspace(self.y1, self.y2, num_rows)
        
        for y in y_coords:
            for x in x_coords:
                laser_path.append((x, y))
        
        return laser_path
    

    def create_spiral(self):

        def nearly_equal(a, b, epsilon):
            return abs(a - b) < epsilon

        # Initialize the boundaries of the rectangle
        left, right, bottom, top = self.x1, self.x2, self.y1, self.y2
        laser_path = []

        total_width = right - left
        total_height = top - bottom
        step_size_x = total_width / (self.num_points // 2)
        step_size_y = total_height / (self.num_points // 2)
        
        # Starting point
        x, y = self.x1, self.y1
        dx = step_size_x 
        dy = 0

        center_x, center_y = (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

        while True:
            # Add current point to the list
            laser_path.append((x, y))

            # Check if the center is reached
            if (nearly_equal(x, center_x, step_size_x)) and (nearly_equal(y, center_y, step_size_y)):
                break

            # Move to the next point
            x, y = x + dx, y + dy

            # Check if we need to turn
            if dx > 0 and x >= right:  # Move right
                x, y = right, y
                dx, dy = 0, step_size_y
                right -= step_size_x
            elif dy > 0 and y >= top:  # Move up
                x, y = x, top
                dx, dy = -step_size_x, 0
                top -= step_size_y
            elif dx < 0 and x <= left:  # Move left
                x, y = left, y
                dx, dy = 0, -step_size_y
                left += step_size_x
            elif dy < 0 and y <= bottom:  # Move down
                x, y = x, bottom
                dx, dy = step_size_x, 0
                bottom += step_size_y

        laser_path.append((center_x, center_y))

        return laser_path

    def create_random(self):

        laser_path = []
        for _ in range(self.num_points):
            x = random.uniform(self.x1, self.x2)
            y = random.uniform(self.y1, self.y2)
            laser_path.append((x, y))
        return laser_path
    

    def generate_halton_points(self):
        
        def halton_sequence(index, base):
            result = 0
            f = 1.0
            i = index
            while i > 0:
                f = f / base
                result = result + f * (i % base)
                i = i // base
            return result
    
        laser_path = []
        for i in range(1, self.num_points + 1):
            x = halton_sequence(i, 2) * (self.x2 - self.x1) + self.x1
            y = halton_sequence(i, 3) * (self.y2 - self.y1) + self.y1
            laser_path.append((x, y))
        return laser_path
    
    def run_with_cartesian_coordinates(self):
        # This will generate the text file for the annealing pattern (Recommended to use this)
        laser_path = self.options()

        self.write_laser_path_to_file_cartesian(laser_path)
        
        print("Done")
        return None
    
    def run_with_voltage_coordinates(self):
        # This will generate the text file for the annealing pattern (Recommended to use this)
        laser_path = self.options()

        self.write_laser_path_to_file_voltage(laser_path)
        
        print("Done")
        return None
    
    def print_graph(self):
        # This will generate the text file and plot the graph (mostly used to test)
        laser_path = self.options()
        # self.write_laser_path_to_file_cartesian(laser_path)
        self.write_laser_path_to_file_voltage(laser_path)
        self.plot_from_laser_path_in_cartesian(laser_path)
        print( "Done")
        return None
    
    def animation(self):
        extra_margin = 2

        # This will generate the text file and plot the graph (mostly used to test)
        laser_path = self.options()
        # self.write_laser_path_to_file_cartesian(laser_path)
        self.write_laser_path_to_file_voltage(laser_path)
        
        x_nums = [i[0] for i in laser_path]
        y_nums = [i[1] for i in laser_path]

        fig, axes = plt.subplots()

        scat = axes.scatter(x_nums[0], y_nums[0], c="b", linewidths= 0.5, label="Annealing Points", linestyle='-')
        axes.set(xlim=[self.x1 - extra_margin, self.x2 + extra_margin], ylim=[self.y1 - extra_margin, self.y2 + extra_margin], xlabel="X Coordinate", ylabel="Y Coordinate", title="Annealing Pattern on Chip")
        axes.legend()

        def update(frame):
            # for each frame, update the data stored on each artist.
            x = x_nums[:frame]
            y = y_nums[:frame]
            # update the scatter plot:
            data = np.stack([x, y]).T
            scat.set_offsets(data)
            # update the line plot:
            return (scat)
        
        # change the speed by changing the interval
        ani = animation.FuncAnimation(fig=fig, func=update, interval=1)
        plt.show()

        print( "Done")
        return None
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Running the program
ap = AnnealingPattern(FILENAME, X1, Y1, X2, Y2, OPTION, NUM_POINTS, PULSE_TIME, CURRENT, DELAY_TIME)
# ap.run_with_voltage_coordinates()
ap.animation()

# ----------------------------------------------------------------------------------------------------------------------------------------------------









