import numpy as np

DEFAULT_LASER_SIGMA = 0.08


def gaussian(r, sigma):
    '''
    Returns a gaussian profile with its integral over R^2 normalized to identity.
    '''
    return (1 / (2 * np.pi * sigma**2)) * np.exp((-1 / 2) * (r**2 / sigma**2))


class LaserPulse(object):
    '''
    Object representing a single laser pulse.

    Attributes:

    grid: a simulationlib SimGrid object representing the surface the laser will be
          projected over.

    position -> x, y: float: the location of the pulse on the chip. Origin is the bottom left corner.

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

    def __init__(self, grid, start, duration, position, power, sigma=DEFAULT_LASER_SIGMA, modulators=None, params=None):
        self.simgrid = grid
        self.sigma = sigma
        self.power = power
        self.start = start
        self.duration = duration
        self.x, self.y = position
        self.end = start + duration
        self.modulators = modulators
        if params is not None:
            self.params = params

        # self.rendered_beam_profile = []
        self.beam_modulation = []
        self.beam_instructions = None
        self.rmg = self.radial_meshgrid(self.x, self.y)

    def get_offset_meshgrid(self, x, y):
        '''
        Builds a meshgrid of values corresponding to coordinates on the sim
        with the origin at x, y
        '''
        # x and y are the cartesian coordinates of the origin
        cdim = self.simgrid.CHIP_DIMENSION
        res = self.simgrid.RESOLUTION
        bx = np.linspace(0, cdim, res) - x
        by = np.linspace(0, cdim, res) - cdim + y

        return np.meshgrid(bx, by)

    def radial_meshgrid(self, x, y):
        '''
        Returns meshgrid of radius values which can be passed to a distribution
        function for lasing.
        '''
        xm, ym = self.get_offset_meshgrid(x, y)
        r = np.sqrt(xm**2 + ym**2)

        return r

    def laser_beam(self, r, sigma, power):
        '''
        Returns the intensity profile of the laser, given total power output (W)
        This gives a radial distribution.
        '''
        return gaussian(r, sigma) * power * self.simgrid.cell_area

    def modulate_beam(self, time):
        t = time - self.start
        coeff = 1
        if self.modulators is not None:
            for m, p in zip(self.modulators, self.params):
                coeff *= m(t, *p)
        return coeff

    def is_active(self, time):
        return self.start <= time and time <= self.end

    def run(self, time):
        return (self.modulate_beam(time) * self.laser_beam(self.rmg, self.sigma, self.power)).flatten().copy()

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

    def __init__(self, grid, start, duration, position, power, parameterization, pargs=None, offset=None, **kwargs):
        super().__init__(grid, start, duration, position, power, **kwargs)
        self.fx, self.fy = parameterization
        if pargs is not None:
            self.px, self.py = pargs
        else:
            self.px = self.py = ()

        if offset is not None:
            self.ox, self.oy = offset
        else:
            self.ox, self.oy = 0, 0

    def move_beam(self, time):
        t = time - self.start
        return (self.fx(t, *self.px) + self.x + self.ox,
                self.fy(t, *self.py) + self.y + self.oy)

    def run(self, time):
        m = self.modulate_beam(time)
        x, y, = self.move_beam(time)
        r = self.radial_meshgrid(x, y)
        return (m * self.laser_beam(r, self.sigma, self.power)).flatten()


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