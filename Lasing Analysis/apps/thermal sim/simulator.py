import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib_animtools as ma

logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)

CHIP_DIMENSION = 30  # mm
RESOLUTION = 99
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
PLAYBACKSPEED = 0.05 # affects display framerate too
STOP_TIME = 7

LASER_SIGMA = 0.08


def get_minimum_stable_timestep(dx, a):
    return dx**2 / (4 * a)


TIMESTEP = get_minimum_stable_timestep(dx, DIFFUSIVITY)  # / 4


gamma = DIFFUSIVITY * (TIMESTEP / dx**2)
times = np.arange(0, STOP_TIME, TIMESTEP)

timesteps_per_second = round(1 / TIMESTEP)
timesteps_per_frame = round((timesteps_per_second * PLAYBACKSPEED)/ (DISPLAY_FRAMERATE))

timesteps_per_percent = round(len(times) / 100)


grid = np.ones((RESOLUTION + 2, RESOLUTION + 2))
grid[:, 0] = 0
grid[:, RESOLUTION + 1] = 0
grid[0, :] = 0
grid[RESOLUTION + 1, :] = 0

roi_mask = grid != 0
grid[:, :] = AMBIENT_TEMPERATURE

center = CHIP_DIMENSION / 2
half_grid = RESOLUTION // 2 + 1
CENTERPOINT = (center, center)

left = np.roll(roi_mask, -1)
right = np.roll(roi_mask, 1)
below = np.roll(roi_mask, 1, axis=0)
above = np.roll(roi_mask, -1, axis=0)


grid[roi_mask] = AMBIENT_TEMPERATURE

states = []
deltas = []
deltas.append(np.zeros(np.shape(grid)))
states.append(grid)

xspace = np.linspace(0 - dx, CHIP_DIMENSION + dx, RESOLUTION + 2)

def get_offset_meshgrid(x, y):
    '''
    Builds a meshgrid of values corresponding to coordinates on the sim
    with the origin at x, y
    '''
    # x and y are the cartesian coordinates of the origin
    bx = np.linspace(0, CHIP_DIMENSION, RESOLUTION) - x
    by = np.linspace(0, CHIP_DIMENSION, RESOLUTION) - CHIP_DIMENSION + y

    return np.meshgrid(bx, by)


def radial_meshgrid(x, y):
    '''
    Returns meshgrid of radius values which can be passed to a distribution
    function for lasing.
    '''
    xm, ym = get_offset_meshgrid(x, y)
    r = np.sqrt(xm**2 + ym**2)

    return r


def gaussian(r, sigma):
    '''
    Returns a normalized gaussian profile.
    '''
    return (1 / (2 * np.pi * sigma**2)) * np.exp((-1 / 2) * (r**2 / sigma**2))


def laser_beam(r, sigma, power):
    '''
    Returns the intensity profile of the laser, given total power output (W)
    This gives a radial distribution.
    '''
    return gaussian(r, sigma) * power


class LaserPulse(object):
    def __init__(self, start, duration, position, power, sigma=LASER_SIGMA, modulators=None, params=None):
        self.x, self.y = position
        self.sigma = sigma
        self.power = power
        self.start = start
        self.duration = duration
        self.end = start + duration
        self.modulators = modulators
        if params is not None:
            self.params = params

        self.rendered_beam_profile = []
        self.beam_modulation = []
        self.beam_instructions = None

        self.eval_time = np.arange(self.start, self.end, TIMESTEP) - self.start

        for time in self.eval_time:
            coeff = 1
            if modulators is not None:
                for m, p in zip(modulators, params):
                    coeff *= m(time, *p)
            self.beam_modulation.append(coeff)

    def bake(self):
        r = radial_meshgrid(self.x, self.y)
        for coeff in self.beam_modulation:
            self.rendered_beam_profile.append((coeff * laser_beam(r, self.sigma, self.power)
                                               * (TIMESTEP / (cell_mass * SPECIFIC_HEAT))).flatten().copy())
        self.beam_instructions = iter(self.rendered_beam_profile)

    def is_active(self, time):
        return self.start <= time and time <= self.end

    def run(self):
        return next(self.beam_instructions)


class LaserStrobe(LaserPulse):
    def __init__(self, start, duration, position, power, parameterization, pargs=None, offset=None, **kwargs):
        super().__init__(start, duration, position, power, **kwargs)
        fx, fy = parameterization
        if pargs is not None:
            px, py = pargs
        else:
            px = py = ()

        if offset is not None:
            ox, oy = offset
        else:
            ox, oy = 0, 0

        self.xc = fx(self.eval_time, *px) + self.x + ox
        self.yc = fy(self.eval_time, *py) + self.y + oy

    def bake(self):
        for x, y, coeff in zip(self.xc, self.yc, self.beam_modulation):
            r = radial_meshgrid(x, y)
            self.rendered_beam_profile.append((coeff * laser_beam(r, self.sigma, self.power)
                                               * (TIMESTEP / (cell_mass * SPECIFIC_HEAT))).flatten().copy())
        self.beam_instructions = iter(self.rendered_beam_profile)


pulses = []



def radialgeneric(radius, duration, n=1, phase=0, r0=None):
    '''
    Generates parameterizations for x(t), y(t)
    which complete n revolutions over a duration on radius r, and phase shift
    Start from radius r0, go to r linearly (circular if r0 not specified).

    '''
    # a revolution occurs over 2pi
    # duration needs to mapped to 2pi * n
    omega = (2 * np.pi * n) / duration

    def r(t):
        if r0 is None:
            return radius
        else:
            return (radius - r0) * (t / duration)

    def xfunc(t):
        return r(t) * np.cos(omega * t + phase)

    def yfunc(t):
        return r(t) * np.sin(omega * t + phase)

    return xfunc, yfunc

print("Generating pulses", end="")

# pulses.append(LaserStrobe(0.5, 5, CENTERPOINT, 0.5, radialgeneric(10, 5, 5, r0=0)))


# t = 1
# for x in range(4, 28, 4):
#     for y in range(4, 28, 4):
#         pulses.append(LaserPulse(t, 0.0578, (x, y), 0.5, sigma=0.3))
#         t += 2 * (0.05 + 0.008)
# print(" ...done")

pulses.append(LaserPulse(0, 2, CENTERPOINT, 0.5, sigma=0.3))

print("\nRendering pulses", end="")

for p in pulses:
    p.bake()
print(" ...done")


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
    delta = conduction

    # power output from radiation
    # convert to temperature drop from radiation
    radiation_power = (AMBIENT_TEMPERATURE - roi)**4
    radiation_temp = radiation_power * K1

    delta += radiation_temp

    for p in pulses:  # fire any lasing activities that should occur
        if p.is_active(t):
            delta += p.run()

    temps.append(grid[half_grid, half_grid])

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
plt.plot(times, 1600 * np.sin(times))


for n, s in enumerate(states):
    states[n] = s - 273.15

plt.show()

ma.animate_2d_arrays(states, interval=(1 / (DISPLAY_FRAMERATE))
                     * 1000, repeat_delay=0, cmap="magma", vmin=0, vmax=450)
