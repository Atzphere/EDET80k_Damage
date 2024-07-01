import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib_animtools as ma

logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)

CHIP_DIMENSION = 30  # mm
RESOLUTION = 100
CHIP_THICKNESS = 0.030  # mm
dx = CHIP_DIMENSION / RESOLUTION
cell_area = dx**2

AMBIENT_TEMPERATURE = 300
SBC = 5.670E-14  # stefan-boltzmann constant per square mm

DIFFUSIVITY = 80  # mm^2 / s
EMISSIVITY = 0.09
SPECIFIC_HEAT = 0.7  # j g^-1 C^-1
DENSITY = 0.002329002  # g/mm^3

cell_mass = cell_area * CHIP_THICKNESS * DENSITY  # in g

DISPLAY_FRAMERATE = 24
STOP_TIME = 7

LASER_SIGMA = 0.08


def get_minimum_stable_timestep(dx, a):
    return dx**2 / (4 * a)


TIMESTEP = get_minimum_stable_timestep(dx, DIFFUSIVITY)


gamma = DIFFUSIVITY * (TIMESTEP / dx**2)
times = np.arange(0, STOP_TIME, TIMESTEP)

timesteps_per_second = round(1 / TIMESTEP)
timesteps_per_frame = round(timesteps_per_second / DISPLAY_FRAMERATE)

timesteps_per_percent = round(len(times) / 100)


grid = np.ones((RESOLUTION + 2, RESOLUTION + 2))
grid[:, 0] = 0
grid[:, RESOLUTION + 1] = 0
grid[0, :] = 0
grid[RESOLUTION + 1, :] = 0

roi_mask = grid != 0
grid[:, :] = AMBIENT_TEMPERATURE

center = CHIP_DIMENSION / 2
CENTERPOINT = (center, center)

left = np.roll(roi_mask, -1)
right = np.roll(roi_mask, 1)
below = np.roll(roi_mask, 1, axis=0)
above = np.roll(roi_mask, -1, axis=0)


grid[roi_mask] = AMBIENT_TEMPERATURE + 56

states = []
deltas = []
deltas.append(np.zeros(np.shape(grid)))
states.append(grid)

# for p in pulses:
#     p.bake()

print(" ...done")

def start_simulation():
    print(f"Starting simulation: {round(STOP_TIME / TIMESTEP)} iterations.")
    print("[" + " " * 24 + "25" + " " * 23 + "50" + " " * 23 + "75" + " " * 24 + "]")
    print("[", end="")
    # precompute constants to optimize
    K1 = (EMISSIVITY * SBC * cell_area) / (cell_mass * SPECIFIC_HEAT) * TIMESTEP
    temps = []

    progress = 0

    for n, t in enumerate(times):
        roi = grid[roi_mask]
        conduction = gamma * (grid[below] + grid[above] + grid[left] + grid[right] - 4 * roi)
        # power output from radiation
        radiation_power = (AMBIENT_TEMPERATURE - roi)**4
        # convert to temperature drop from radiation
        radiation_temp = radiation_power * K1

        delta = radiation_temp
        delta += conduction

        # for p in pulses:  # fire any lasing activities that should occur
        #     if p.is_active(t):
        #         delta += p.run()

        temps.append(grid[50, 50])

        grid[roi_mask] += delta

        if n % timesteps_per_frame == 0:
            # print("yes")
            deltas.append(delta.copy())
            states.append(grid.copy())

        if n % timesteps_per_percent == 0:
            print("#", end="")
            progress += 1
    print("]")

    plt.plot(times, temps)


    for n, s in enumerate(states):
        states[n] = s - 273.15

    plt.show()

    ma.animate_2d_arrays(states, interval=(1 / DISPLAY_FRAMERATE)
                         * 1000, repeat_delay=0, cmap="magma", vmin=0, vmax=450)



