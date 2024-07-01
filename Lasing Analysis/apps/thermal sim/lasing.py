from simulator import CHIP_DIMENSION, RESOLUTION, LASER_SIGMA, CENTERPOINT, TIMESTEP, cell_mass, SPECIFIC_HEAT
import numpy as np
import multiprocess as mp

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

pulses.append(LaserStrobe(0.5, 5, CENTERPOINT, 0.5, radialgeneric(10, 5, 5, r0=0)))

t = 5

for x in range(4, 28, 2):
    for y in range(4, 28, 2):
        pulses.append(LaserPulse(t, 0.0578, (x, y), 0.5, sigma=0.3))
        t += 0.05 + 0.008
print(" ...done")

print("\nRendering pulses", end="")

for p in pulses:
    p.bake()
print(" ...done")