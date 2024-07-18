'''

File for calling and testing a simulation. Nothing here is critical, but this might hold some useful examples of
what you can do with the simulation.

'''

import simulationlib as sl
import lasinglib as ll
import measurelib as ml
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import logging
import modulators as mo
import random
import shapes

logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)

# initialize simulation grid and material

SILICON = sl.Material(88, 0.09, 0.7, 0.002329002)
CHIP = sl.SimGrid(30, 91, 0.03, use_spar=False,
                  spar_thickness=0.5, spar_width=1)

sim = sl.Simulation(CHIP, SILICON, duration=8, pulses=None, ambient_temp=300,
                    starting_temp=300, neumann_bc=True,
                    edge_derivative=0, sample_framerate=24, intended_pbs=1,
                    dense_logging=False, timestep_multi=1, radiation=True, progress_bar=True, silent=False)

# useful builtin constant
CENTER = (CHIP.CENTERPOINT)

# create a MeasureArea to mask the left border of the chip
LEFT_EDGE = ml.MeasureArea(CHIP, (0, CHIP.center), lambda x, y: x == 0)
# initalize a Measurement object with this area
BORDER_MAXTEMP = ml.Measurement(LEFT_EDGE, modes=["MAX"])
# then bind it to a Measurer to start recording in a certain time interval and label
RECORD_BMAXTEMP = ml.Measurer(0, 13, BORDER_MAXTEMP, "BORDER")

# ditto but to measure the exact centerpoint of the chip
CENTERPOINT = ml.MeasureArea(CHIP, CHIP.CENTERPOINT, lambda x, y: np.logical_and(x == 0, y == 0))
CENTERMEASURE = ml.Measurement(CENTERPOINT, modes=["MEAN"])
RECORD_CENTER_TEMPERATURE = ml.Measurer(0, 10, CENTERMEASURE, "CENTER")

# compile measurers
measurers = [RECORD_CENTER_TEMPERATURE, RECORD_BMAXTEMP]


# assemble laser pulses
pulses = []

CURRENT = 1.1

centerpulse = ll.LaserPulse(CHIP, start=0.5, duration=5, position=CENTER, power=CURRENT, modulators=[mo.doubleSinePulse(1, 3, 1)], measure_target=True)
pulses.append(centerpulse)
sim.pulses = pulses

sim.simulate()
print("Simulate done.")
data = sim.recorded_data
print(data.keys())

data_mean_temp = data[f"PULSE_0.5s+5s_1.1A MEAN"]

print(f"Max temp at center of pulse: {max(data_mean_temp):.2f} C")

sim.animate(repeat_delay=0, cmap="magma", vmin=0, vmax=450)
