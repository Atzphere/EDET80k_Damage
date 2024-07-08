import simulationlib as sl
import lasinglib as ll
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SILICON = sl.Material(88, 0.09, 0.7, 0.002329002)
CHIP = sl.SimGrid(30, 101, 0.03, use_spar=False,
                  spar_thickness=0.5, spar_width=1)

sim = sl.Simulation(CHIP, SILICON, duration=10, pulses=None, ambient_temp=300,
                    starting_temp=300, neumann_bc=True,
                    edge_derivative=0, sample_framerate=24, intended_pbs=1,
                    dense_logging=False, timestep_multi=1, radiation=True, progress_bar=True)

CENTER = (CHIP.CENTERPOINT)

pulses = []


def make_exp_pulse(up_time, hold_time, power_rampup):
    duration = up_time + hold_time

    def function(t):
        if t <= up_time:
            val = np.exp(((3 * t) / up_time) - 3) ** power_rampup
        elif t < up_time + hold_time:
            val = 1
        elif t <= duration:
            val = 0
        return val

    return function


def makeRampedPulse(up_time, hold_time, down_time, power_rampup=1, power_rampdown=1):
    duration = up_time + down_time + hold_time

    def function(t):
        if t <= up_time:
            val = np.sin((np.pi / (2 * up_time)) * t)**power_rampup
        elif t < up_time + hold_time:
            val = 1
        elif t <= duration:
            val = np.sin((np.pi / (2 * up_time)) *
                         (t - (up_time + hold_time)) + np.pi / 2)**power_rampdown
        return val

    return function


CENTERPOINT = sl.MeasureArea(CHIP, CHIP.CENTERPOINT, lambda x, y: np.logical_and(x == 0, y == 0))
CENTERMEASURE = sl.Measurement(CENTERPOINT, modes=["MEAN"])

RECORD_CENTER_TEMPERATURE = sl.Measurer(0, 5.5, CENTERMEASURE, "CENTER")

measurements = [RECORD_CENTER_TEMPERATURE]

a = ll.LaserPulse(CHIP, 0.5, 5, CHIP.CENTERPOINT, 2, sigma=0.3,
                  modulators=[makeRampedPulse(3, 1, 3, 4, 4)])
b = ll.LaserPulse(CHIP, 5.5, 4, CHIP.CENTERPOINT, 2, sigma=0.3,
                  modulators=[make_exp_pulse(3, 1, 1)])
# a = ll.LaserStrobe(CHIP, 0.5, 3, CHIP.CENTERPOINT, 6, sigma=0.18, modulators=(lambda t: np.exp(
#     (t - 0.5)) / 1.5,), parameterization=ll.genericpolar((4 * np.pi) / 3, lambda t: np.exp(t), phase=0), params=())

# make class for list of pulses to auto sequence

pulses = [a, b]

sim.pulses = pulses


sim.simulate(measurements)
# sim.animate(repeat_delay=0, cmap="magma", vmin=0, vmax=450)

plt.plot(sim.recorded_data["CENTER time"], sim.recorded_data["CENTER MEAN"])
plt.show()
