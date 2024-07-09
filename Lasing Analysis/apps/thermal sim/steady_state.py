import simulationlib as sl
import lasinglib as ll
import measurelib as ml
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)

SILICON = sl.Material(88, 0.09, 0.7, 0.002329002)
CHIP = sl.SimGrid(dimension=30, resolution=51, thickness=0.03, use_spar=False,
                  spar_thickness=0.5, spar_width=1)

CENTER = (CHIP.CENTERPOINT)

pulses = []

DURATION = 3
START_TIME = 0
power = np.arange(START_TIME, DURATION, 0.1)

color = iter(cm.rainbow(np.linspace(0, 1, len(power))))

for p in power:
    c = next(color)
    p_string = f"PULSE_{START_TIME}s+{DURATION}s_{p}A"
    data_ind = p_string + " MEAN"
    time_ind = p_string + " time"
    a = ll.LaserPulse(CHIP, START_TIME, DURATION, CHIP.CENTERPOINT, p, sigma=0.3, measure_target=True, measure_timestep=0.1)
    sim = sl.Simulation(CHIP, SILICON, duration=DURATION, pulses=[a], ambient_temp=300,
                        starting_temp=300, neumann_bc=True,
                        edge_derivative=0, sample_framerate=0, intended_pbs=1,
                        dense_logging=False, timestep_multi=1, radiation=True, progress_bar=False, silent=True)
    sim.simulate()
    data = sim.recorded_data
    plt.plot(data[time_ind], data[data_ind], label=f"{p:.2f}A", c=c)


plt.legend()
plt.show()