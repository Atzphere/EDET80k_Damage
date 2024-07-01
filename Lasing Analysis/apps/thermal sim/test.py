import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib_animtools as ma
import math

logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)

CHIP_DIMENSION = 30  # mm
RESOLUTION = 100
CHIP_THICKNESS = 0.030 # mm
dx = CHIP_DIMENSION / RESOLUTION
cell_area = dx**2

AMBIENT_TEMPERATURE = 300
SBC = 5.670E-14 # stefan-boltzmann constant per square mm

DIFFUSIVITY = 80  # mm^2 / s
EMISSIVITY = 0.09
SPECIFIC_HEAT = 0.7 # j g^-1 C^-1
DENSITY = 0.002329002 # g/mm^3

cell_mass = cell_area * CHIP_THICKNESS * DENSITY # in g

DISPLAY_FRAMERATE = 24
STOP_TIME = 7

def get_minimum_stable_timestep(dx, a):
    return dx**2 / (4 * a)


TIMESTEP = get_minimum_stable_timestep(dx, DIFFUSIVITY)

print(f"Starting simulation: {round(STOP_TIME / TIMESTEP)} iterations.")

gamma = DIFFUSIVITY * (TIMESTEP / dx**2)
times = np.arange(0, STOP_TIME, TIMESTEP)

timesteps_per_second = round(1 / TIMESTEP)
timesteps_per_frame = round(timesteps_per_second / DISPLAY_FRAMERATE)


x = np.linspace(0, CHIP_DIMENSION, RESOLUTION)
y = np.linspace(0, CHIP_DIMENSION, RESOLUTION)


grid = np.ones((RESOLUTION + 2, RESOLUTION + 2))
grid[:, 0] = 0
grid[:, RESOLUTION + 1] = 0
grid[0, :] = 0
grid[RESOLUTION + 1, :] = 0

roi_mask = grid != 0
grid[:, :] = AMBIENT_TEMPERATURE

grid[roi_mask] = AMBIENT_TEMPERATURE + 56



left = np.roll(roi_mask, -1)
right = np.roll(roi_mask, 1)
below = np.roll(roi_mask, 1, axis=0)
above = np.roll(roi_mask, -1, axis=0)

states = []
deltas = []
deltas.append(np.zeros(np.shape(grid)))
states.append(grid)

BEAM_X = 0
BEAM_Y = 0

bx = np.linspace(0, CHIP_DIMENSION, RESOLUTION) - CHIP_DIMENSION / 2 - BEAM_X
by = np.linspace(0, CHIP_DIMENSION, RESOLUTION) - CHIP_DIMENSION / 2 + BEAM_Y

xm, ym = np.meshgrid(bx, by)

xm -= 0
ym -= 0

r = np.sqrt(xm**2 + ym**2)


def gaussian(r, sigma):
    '''
    Returns a normalized gaussian profile.
    '''
    return (1 / (2 * np.pi * sigma**2)) * np.exp((-1 / 2) * (r**2 / sigma**2))


def laser_beam(r, sigma, power):
    '''
    Returns the intensity profile of the laser, given its total power output in watts.
    '''
    return gaussian(r, sigma) * power


K1 = (EMISSIVITY * SBC * cell_area) / (cell_mass * SPECIFIC_HEAT) * TIMESTEP
beam1 = (laser_beam(r, 0.1, 0.5) * TIMESTEP / (cell_mass * SPECIFIC_HEAT)).flatten()

temps = []

for n, t in enumerate(times):
    roi = grid[roi_mask]
    conduction = gamma * (grid[below] + grid[above] + grid[left] + grid[right] - 4 * roi)
    radiation_power = (AMBIENT_TEMPERATURE - roi)**4 # power output from radiation
    radiation_temp = radiation_power * K1 # convert to temperature drop from radiation

    delta = radiation_temp
    delta += conduction

    if t > 1 and t <= 6:
        grid[roi_mask] += beam1 * np.sqrt(np.sin((-np.pi / 5) * t + ((6 * np.pi) / 5)))
    temps.append(grid[50, 50])


    grid[roi_mask] += delta

    if n % timesteps_per_frame == 0:
        # print("yes")
        deltas.append(delta.copy())
        states.append(grid.copy())


# from scipy.optimize import curve_fit

# def arctan(x, a, m, b):
#     return a * np.arctan(m * (x - b))

# def logistic(x, L, k, x0):
#     return L / (1 + np.exp(-k * (x - x0)))


# tmask = np.logical_and(times >= 1, times <= 3)
# temps = np.array(temps)

# popt, pcov = curve_fit(logistic, times[tmask], temps[tmask], p0=(920,6,1))

plt.plot(times, temps)
# plt.plot(times, logistic(times, *popt))

# popt, pcov = curve_fit(arctan, times[tmask], temps[tmask], p0=(920,6,1))

# plt.plot(times, arctan(times, *popt))
# print(popt)
plt.show()
# print(len(states) * 1 / DISPLAY_FRAMERATE)

ma.animate_2d_arrays(states, interval=(1 / DISPLAY_FRAMERATE) * 1000, repeat_delay=0, cmap="magma", vmin=250, vmax=600)