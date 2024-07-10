import numpy as np
import matplotlib.pyplot as plt
import math
import os
import random

FILENAME = "C:\\Users\\Qiren\\Desktop\\EDET80k_Anneal\\TAPV-2\\Application\\pythonFiles\\DataTextFiles\\michaeltest1.txt"

def write_laser_path_to_file(filename):
    # voltage x, y

    x = -0.704
    y = 3.983

    # x = -1.129
    # y = 4.000

    # x = 0.271
    # y = 3.147

    # x = 1.28
    # y = 3.903

    # x = 1.847
    # y = 3.881
    # 27, 30 is the chip coordinate (cartesian coordinate)

    # time laser is turned on
    hit_time = 1
    current = 2.5

    # time between each hit, all time in seconds
    time_stop = 120
    total_test_time = 7200

    with open(filename, 'w') as f:
        i = 0
        while i < total_test_time:
            # write the hit
            f.write(f"{x:.3f}, {y:.3f}, {hit_time:.3f}, {current:.3f}\n")
            i += 1

            # write the time stop between hits
            # you can comment the bottom out if you want it to be
            # a continuous line or no stop
            f.write(f"{x:.3f}, {y:.3f}, {time_stop:.3f}, 0.000\n")
            i += time_stop

    return None

write_laser_path_to_file(FILENAME)
