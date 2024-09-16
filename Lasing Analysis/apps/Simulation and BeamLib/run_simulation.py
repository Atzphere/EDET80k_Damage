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

logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)

SILICON = sl.Material(88, 0.09, 0.7, 0.002329002)
CHIP = sl.SimGrid(30, 101, 0.03, use_spar=False,
                  spar_thickness=0.5, spar_width=1)

sim = sl.Simulation(CHIP, SILICON, duration=6, pulses=None, ambient_temp=300,
                    starting_temp=300, neumann_bc=True,
                    edge_derivative=0, sample_framerate=24, intended_pbs=1,
                    dense_logging=False, timestep_multi=1, radiation=True, progress_bar=True, silent=False)

CENTER = (CHIP.CENTERPOINT)

pulses = []


LEFT_EDGE = ml.MeasureArea(CHIP, (0, CHIP.center), lambda x, y: x == 0)
BORDER_MAXTEMP = ml.Measurement(LEFT_EDGE, modes=["MAX"])
RECORD_BMAXTEMP = ml.Measurer(0, 13, BORDER_MAXTEMP, "BORDER")

CENTERPOINT = ml.MeasureArea(CHIP, CHIP.CENTERPOINT, lambda x, y: np.logical_and(x == 0, y == 0))
CENTERMEASURE = ml.Measurement(CENTERPOINT, modes=["MEAN"])
RECORD_CENTER_TEMPERATURE = ml.Measurer(0, 10, CENTERMEASURE, "CENTER")

measurements = [RECORD_CENTER_TEMPERATURE, RECORD_BMAXTEMP]

def scf(time, timestep):
    n = int(time // timestep)
    xpos = np.random.randint(-20, 20, n + 1)
    ypos = np.random.randint(-20, 20, n + 1)
    def x(t):
        return xpos[int(t // timestep)]
    def y(t):
        return ypos[int(t // timestep)]
    
    return x, y

def flick(x1, y1, x2, y2, time, timestep):
    n = int(time // timestep)
    xpos = np.zeros(n + 1)
    ypos = np.zeros(n + 1)

    xpos[::2] = x1
    xpos[1::2] = x2

    ypos[::2] = y1
    ypos[1::2] = y2
    def x(t):
        return xpos[int(t // timestep)]
    def y(t):
        return ypos[int(t // timestep)]
    
    return x, y

doublegauss = ll.LaserPulse(CHIP, 0.5, 1, CHIP.CENTERPOINT, 1, sigma=0.3,
                  modulators=[mo.doubleGaussianRamp(2.5, 4, 2, cutoff=2, boost=0)], measure_target=True, target_r=2, measure_timestep=0.01)

singlepulse = ll.LaserPulse(CHIP, 0.5, 0.2, CHIP.CENTERPOINT, 6, sigma=0.3)

puls = []

for i in range(0, 100):
    puls.append(ll.LaserPulse(CHIP, 0.5, 0.0001, (random.randint(0 ,30), random.randint(0 ,30)), 1.4, sigma=0.3))

randomspam = ll.LaserSequence(puls, 0, 1)

# exp_pulse = ll.LaserPulse(CHIP, 5.5, 4, CHIP.CENTERPOINT, 2, sigma=0.3,
#                   modulators=[make_exp_pulse(3, 1, 1)])


# gauss_pulse = ll.LaserPulse(CHIP, 0.5, 5, CHIP.CENTERPOINT, 2, sigma=0.3,
#                   modulators=[make_bell_curve(2, 1)])

exponentialspiral = ll.LaserStrobe(CHIP, 0.5, 4, CHIP.CENTERPOINT, 1, sigma=0.18, modulators=[lambda t: 1 + (t / 4) * 0.8], parameterization=ll.genericpolar((4 * np.pi) / 3, lambda t: np.exp(t), phase=0), params=())

def x(t):
    return t * (20 / 0.06) - 10

wiggle = lambda t: 10 * np.sin((3 * 2 * (np.pi / 0.06)) * t)
wiggle2 = lambda t: 10 * np.cos((3 * 2 * (np.pi / 0.06)) * t)
circle = ll.LaserStrobe(CHIP, 0.5, 0.06, CHIP.CENTERPOINT, 1.4, sigma=0.18, parameterization=(wiggle2, wiggle), params=())
sinewave = ll.LaserStrobe(CHIP, 0.5, 0.06, CHIP.CENTERPOINT, 1.4, sigma=0.18, parameterization=(x, wiggle), params=())

strobetest = ll.LaserPulse(CHIP, 0.5, 6, (18, 18), 1.1, sigma=0.18, modulators=(lambda t: 1 if round(t / 0.05) % 2 == 0 else 0,))

a_l = ll.LaserSequence([circle, sinewave] + puls, 0.25, 1)
a_l = ll.LaserSequence([strobetest], 0.5, 6)

# b = ll.LaserStrobe(CHIP, 0.5, 4, CHIP.CENTERPOINT, 1.1, sigma=0.18, parameterization=flick(-15, -15, 15, 15, 4, 0.01), params=())
# a_l = ll.LaserSequence([b], 0.25, 1)
pulses = [a_l]
sim.pulses = pulses
a_l.write_to_cycle_code(r"C:\Users\ssuub\Desktop\MPSD-TAP\TAPV-2\Application\pythonFiles\DataTextFiles\michaeltest1.txt", 0.05)

# sim.simulate(measurements)
# data = sim.recorded_data
# print(data.keys())
# fig, ax = plt.subplots(2)
# fig.tight_layout()
# ax[0].plot(data["PULSE_0.5s+1s_1A time"], data["PULSE_0.5s+1s_1A MEAN"] - 273.15, label="Center temperature")
# ax[0].plot(data["BORDER time"], data["BORDER MAX 0"] - 273.15, label="Max temperature along edge")
# ax[0].legend()
# ax[0].set_xlabel("Time (s)")
# ax[0].set_ylabel("Temperature (C)")

# j = mo.doubleGaussianRamp(2.5, 4, 2, cutoff=2, boost=0)
# t = np.linspace(0, 13)
# y = [j(x) for x in t]
# ax[1].plot(t, y)
# ax[1].set_ylabel("Laser power (W)")
# ax[1].set_xlabel("Time (s)")

# plt.show()
# sim.animate(repeat_delay=0, cmap="magma", vmin=0, vmax=450)

# plt.plot(sim.recorded_data["CENTER time"], sim.recorded_data["CENTER MEAN"])
# plt.show()
