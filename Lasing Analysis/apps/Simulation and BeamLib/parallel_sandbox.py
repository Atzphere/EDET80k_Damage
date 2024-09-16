'''
This file contains a rough framework for parallelized execution of simulations.
This may be useful for searching through parameter space.

As of July 2024, this is being used to test steady-state lasing behavior (maximum temps)
'''


import simulationlib as sl
import lasinglib as ll
import measurelib as ml
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import logging
import multiprocess as mp

logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)

SILICON = sl.Material(88, 0.09, 0.7, 0.002329002)
CHIP = sl.SimGrid(dimension=30, resolution=71, thickness=0.03, use_spar=False,
                  spar_thickness=0.5, spar_width=1)

CENTER = (CHIP.CENTERPOINT)

pulses = []

DURATION = 12
START_TIME = 0


def trial(p):
    p_string = f"PULSE_{START_TIME}s+{DURATION}s_{p}A"
    data_ind = p_string + " MEAN"
    time_ind = p_string + " time"
    a = ll.LaserPulse(CHIP, START_TIME, DURATION, CHIP.CENTERPOINT,
                      p, sigma=0.3, measure_target=True, measure_timestep=0.025)
    sim = sl.Simulation(CHIP, SILICON, duration=DURATION, pulses=[a], ambient_temp=300,
                        starting_temp=300, neumann_bc=True,
                        edge_derivative=0, sample_framerate=0, intended_pbs=1,
                        dense_logging=False, timestep_multi=1, radiation=True, progress_bar=False, silent=True)
    sim.simulate()

    data = sim.recorded_data
    return data[time_ind], data[data_ind]


if __name__ == "__main__":
    power = np.arange(0, 2, 0.1)
    with mp.Pool(mp.cpu_count()) as pol:
        dat = pol.map(trial, power)

    color = iter(cm.rainbow(np.linspace(0, 1, len(power))))
    for d, p in zip(dat, power):
        c = next(color)
        T, A = d
        plt.plot(T, A - 273.15, label=f"{p:.2f}A", c=c)


    plt.legend()
    plt.show()
