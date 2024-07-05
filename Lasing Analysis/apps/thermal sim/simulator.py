import numpy as np
import logging
import matplotlib.pyplot as plt
import pickle
import blosc
import os
import matplotlib_animtools as ma
import PositionVoltageConverter_Standalone as pvcs

os.chdir(os.path.dirname(__file__))

TAG = "foobtest_"  # prefix to save the results under

DENSE_LOGGING = False  # whether or not to dump the entire simulation to file

logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)

CHIP_DIMENSION = 30  # mm
RESOLUTION = 101
CHIP_THICKNESS = 0.030  # mm
USE_SPAR = False
SPAR_THICKNESS = 0.5  # mm
SPAR_WIDTH = 1  # mm

NEUMANN = True
EDGE_DERIVATIVE = 0  # du/dx on boundaries. 

AMBIENT_TEMPERATURE = 300
STARTING_TEMP = 300
SBC = 5.670E-14  # stefan-boltzmann constant per square mm

DIFFUSIVITY = 80  # mm^2 / s
EMISSIVITY = 0.09
SPECIFIC_HEAT = 0.7  # j g^-1 C^-1
DENSITY = 0.002329002  # g/mm^3

DISPLAY_FRAMERATE = 24
PLAYBACKSPEED = 1  # changes the sampling rate of the video array and playback speed accordingly to maintain DISPLAY_FRAMERATE.
STOP_TIME = 7

DEFAULT_LASER_SIGMA = 0.08


center = CHIP_DIMENSION / 2
half_grid = RESOLUTION // 2
CENTERPOINT = (center, center)

dx = CHIP_DIMENSION / RESOLUTION
cell_area = dx**2
cell_mass = cell_area * CHIP_THICKNESS * DENSITY  # in g

spar_multi = CHIP_THICKNESS / SPAR_THICKNESS
spar_width_cells = int(SPAR_WIDTH // dx)
spar_extension_cells = (spar_width_cells - 1) // 2

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

spar_coefficients = np.ones((RESOLUTION, RESOLUTION))
spar_coefficients[:, half_grid - spar_extension_cells:half_grid + 1 + spar_extension_cells] = spar_multi
spar_coefficients[half_grid - spar_extension_cells:half_grid + 1 + spar_extension_cells, :] = spar_multi

spar_coefficients = spar_coefficients.flatten()


# plt.imshow(spar_coefficients)
# plt.show()

roi_mask = grid != 0
grid[:, :] = STARTING_TEMP


left = np.roll(roi_mask, -1)
right = np.roll(roi_mask, 1)
below = np.roll(roi_mask, 1, axis=0)
above = np.roll(roi_mask, -1, axis=0)

left_boundary = np.zeros((RESOLUTION + 2, RESOLUTION + 2), dtype=bool)

left_boundary[1:-1,0] = True
bottom_boundary = np.rot90(left_boundary)
right_boundary = np.rot90(bottom_boundary)
top_boundary = np.rot90(right_boundary)

left_boundary_inner = np.zeros((RESOLUTION + 2, RESOLUTION + 2), dtype=bool)
left_boundary_inner[1:-1,1] = True
bottom_boundary_inner = np.rot90(left_boundary_inner)
right_boundary_inner = np.rot90(bottom_boundary_inner)
top_boundary_inner = np.rot90(right_boundary_inner)


grid[roi_mask] = STARTING_TEMP


if DENSE_LOGGING:
    dense_states = []
    dense_deltas = []
    dense_deltas.append(np.zeros(np.shape(grid)))
    dense_states.append(grid)

else:
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
    return gaussian(r, sigma) * power * cell_area


class LaserPulse(object):
    '''
    Object representing a single laser pulse.

    Attributes:

    x, y: float: the location of the pulse on the chip. Origin is the bottom left corner.

    sigma: float: parameter determining the Gaussian FWHM

    power: float: total power output of the laser. 100% of power incident upon the chip is absorbed in the sim;
           you need significantly less power than IRL.

    start: float: the start time of the pulse.

    duration: float: the duration of the beam pulse.

    modulators: functions: functions of time to modulate the beam power with. If multiple are given in
          (float -> float) iterable, then their combined product is used. Canonically, this should be
                           a function of range [0, 1] and domain encompassing [0, duration].

    params: [tuple(...)] : parameters to pass to the modulators.
    '''
    def __init__(self, start, duration, position, power, sigma=DEFAULT_LASER_SIGMA, modulators=None, params=None):
        self.x, self.y = position
        self.sigma = sigma
        self.power = power
        self.start = start
        self.duration = duration
        self.end = start + duration
        self.modulators = modulators
        if params is not None:
            self.params = params

        # self.rendered_beam_profile = []
        self.beam_modulation = []
        self.beam_instructions = None
        self.rmg = radial_meshgrid(self.x, self.y)
        self.eval_time = np.arange(self.start, self.end, TIMESTEP) - self.start

        for time in self.eval_time:
            coeff = 1
            if modulators is not None:
                for m, p in zip(modulators, params):
                    coeff *= m(time, *p)
            self.beam_modulation.append(coeff)
        self.beam_modulation = iter(self.beam_modulation)


    def is_active(self, time):
        return self.start <= time and time <= self.end

    def run(self):
        return (next(self.beam_modulation) * laser_beam(self.rmg, self.sigma, self.power)
                                               * (TIMESTEP / (cell_mass * SPECIFIC_HEAT))).flatten()

    def __str__(self):
        if self.modulators is not None:
            m = str(len(self.modulators)) + "MOD)"
        else:
            m = "NOMOD)"
        return f"Pulse({self.power}A{self.duration}S" + m


class LaserStrobe(LaserPulse):
    '''
    A subclass of LaserPulse, containing methods to simulate a strobe - physically moving the laser
    the chip during firing.

    Novel attributes:

    parameterizion: Tuple(x(t), y(t)): Time - parameterization of the motion you wish to anneal with.
                                       Be default, the point x(0), t(0) is placed at the specified position

    pargs: Tuple(ParamX, ParamY): parameters (if needed) to pass to the parameterizer. 

    offset: (offX, offY) : How to offset the parameterized model from its default position relative to the specified 
                           position.
    '''
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

        self.xym = zip(self.xc, self.yc, list(self.beam_modulation))

    def run(self):
        x, y, m = next(self.xym)
        r = radial_meshgrid(x, y)
        return (m * laser_beam(r, self.sigma, self.power) * (TIMESTEP / (cell_mass * SPECIFIC_HEAT))).flatten()


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


if __name__ == "__main__":
    print("Generating pulses", end="")

    # pulses.append(LaserStrobe(0.5, 5, CENTERPOINT, 6, radialgeneric(15, 5, 5, r0=5)))

    # pulses.append(LaserStrobe(0.5, 5, CENTERPOINT, 6, (lambda t: 14 * np.sin(18 * np.pi * t), lambda t: 30 * (t / 5))))

    # t = 1
    # for x in range(4, 28, 4):
    #     for y in range(4, 28, 4):
    #         pulses.append(LaserPulse(t, 0.1, (x, y), 10, sigma=0.15))
    #         t += 2 * (0.1)
    # print(" ...done")

    # pulses.append(LaserPulse(0, 6, (x, y), 0.2, sigma=0.15))

    # pulses.append(LaserPulse(0, 0.5, CENTERPOINT, 1, sigma=0.3))
    pulses.append(LaserPulse(0, 1, (15, 15), 15, sigma=0.3))

    # print("\nRendering pulses", end="")

    # for p in pulses:
    #     p.bake()
    # print(" ...done")


    print(f"Starting simulation: {round(STOP_TIME / TIMESTEP)} iterations.")
    print("[" + " " * 24 + "25" + " " * 23 + "50" + " " * 23 + "75" + " " * 24 + "]")
    print("[", end="")
    # precompute constants to optimize
    K1 = (EMISSIVITY * SBC * cell_area) / (cell_mass * SPECIFIC_HEAT) * TIMESTEP
    temps = []

    progress = 0
    for n, t in enumerate(times):
        roi = grid[roi_mask]
        # print(roi)
        if NEUMANN:
            grid[left_boundary] = grid[left_boundary_inner] - EDGE_DERIVATIVE * dx
            grid[bottom_boundary] = grid[bottom_boundary_inner] - EDGE_DERIVATIVE * dx
            grid[right_boundary] = grid[right_boundary_inner] - EDGE_DERIVATIVE * dx
            grid[top_boundary] = grid[top_boundary_inner] - EDGE_DERIVATIVE * dx
            # print(grid[left_boundary])

        conduction = gamma * (grid[below] + grid[above] + grid[left] + grid[right] - 4 * roi)
        delta = conduction

        # power output from radiation
        # convert to temperature drop from radiation
        radiation_power = (AMBIENT_TEMPERATURE**4 - roi**4)
        radiation_temp = radiation_power * K1 * 10
        if USE_SPAR:
            radiation_temp *= spar_coefficients
            multi = spar_coefficients
        else:
            multi = 1

        delta += radiation_temp

        for p in pulses:  # fire any lasing activities that should occur
            if p.is_active(t):
                delta += p.run() * multi

        temps.append(grid[half_grid, half_grid])

        grid[roi_mask] += delta

        if n % timesteps_per_frame == 0 and not DENSE_LOGGING:
            deltas.append(delta.copy())
            states.append(grid.copy())
        elif DENSE_LOGGING:
            dense_deltas.append(delta.copy())
            dense_states.append(grid.copy())

        if n % timesteps_per_percent == 0:
            print("#", end="")
            progress += 1
    print("]")

    plt.plot(times, temps)

    for n, s in enumerate(states):
        states[n] = s - 273.15

    plt.show()

    ma.animate_2d_arrays(states, interval=(1 / (DISPLAY_FRAMERATE))
                         * 1000, repeat_delay=0, cmap="magma", vmin=0, vmax=450)

    if DENSE_LOGGING:
        pickled_data = pickle.dumps((dense_states, dense_deltas))
        TAG += "_DENSE"
    else:
        pickled_data = pickle.dumps((states, deltas))  # returns data as a bytes object
    compressed_pickle = blosc.compress(pickled_data)

    fname = TAG + " ".join([str(p) for n, p in enumerate(pulses) if n > 3]) + ".pkl"
    print(fname)
    with open("../saves/" + fname, "wb") as f:
        f.write(compressed_pickle)
