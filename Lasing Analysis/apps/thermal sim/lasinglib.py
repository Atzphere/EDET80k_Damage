DEFAULT_LASER_SIGMA = 0.08

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
                * TIMESTEP).flatten()

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
        return (m * laser_beam(r, self.sigma, self.power) * TIMESTEP).flatten()


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