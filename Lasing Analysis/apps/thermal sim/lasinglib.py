import numpy as np
import measurelib as ml
import copy
from collections.abc import Iterable
import coordinate_to_voltage_test as pvcs

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

    def __init__(self, grid, start, duration, position, power, sigma=DEFAULT_LASER_SIGMA, modulators=None, params=None, measure_target=False, target_r=0, target_modes=["MEAN"], measure_padding=(0, 0), measure_tag=None, measure_timestep=None, aux_measurers=[]):
        self.simgrid = grid
        self.sigma = sigma
        self.power = power
        self.start = start
        self.duration = duration
        self.x, self.y = position
        self.end = start + duration
        self.modulators = modulators

        self.measure_tag = measure_tag
        if measure_tag is None:
            self.measure_tag = f"PULSE_{start}s+{duration}s_{power}A"

        self.measurers = copy.deepcopy(aux_measurers)

        if aux_measurers != []:
            for m in self.measurers:
                m.tag = f"{self.measure_tag}_{m.tag}"

        if measure_target:
            pad_left, pad_right = measure_padding
            measure_duration = pad_left + pad_right + duration
            measure_start = start - pad_left

            self.measure_area = ml.MeasurePoint(self.simgrid, position, radius=target_r)
            self.beam_measurement = ml.Measurement(self.measure_area, modes=target_modes)
            self.target_measurer = ml.Measurer(measure_start, measure_duration, self.beam_measurement, tag=self.measure_tag, timestep=measure_timestep)
            self.measurers.append(self.target_measurer)

        if params is not None:
            self.params = params
        else:
            self.params = None
        # self.rendered_beam_profile = []
        self.beam_modulation = []
        self.beam_instructions = None
        self.rmg = self.radial_meshgrid(self.x, self.y)

    def has_measurers(self):
        return self.measurers is not None

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
            if self.params is not None:
                for m, p in zip(self.modulators, self.params):
                    coeff *= m(t, *p)
            else:
                for m in self.modulators:
                    coeff *= m(t)
        return coeff

    def is_active(self, time):
        return self.start <= time and time <= self.end

    def run(self, time):
        if self.is_active(time):
            return (self.modulate_beam(time) * self.laser_beam(self.rmg, self.sigma, self.power)).flatten().copy()
        else:
            # print(str(self), "did not fire")
            return None

    def __str__(self):
        if self.modulators is not None:
            m = str(len(self.modulators)) + "MOD)"
        else:
            m = "NOMOD)"
        return f"Pulse({self.power}A, {self.start} + {self.duration}S -> {self.end}" + m

    def __repr__(self):
        return str(self)


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
        if self.is_active(time):
            m = self.modulate_beam(time)
            x, y, = self.move_beam(time)
            r = self.radial_meshgrid(x, y)
            return (m * self.laser_beam(r, self.sigma, self.power)).flatten()
        else:
            return None


class LaserSequence(LaserPulse):
    def __init__(self, pulses, delay, start_time):
        self.start_time = start_time
        if isinstance(delay, Iterable):
            if len(delay) != len(pulses):
                raise ValueError("Supplied delay array-like must same size as pulses.")
            else:
                self.delays = delay
        else:
            self.delays = delay * np.ones(len(pulses))

        self.build_sequence(pulses)

    def build_sequence(self, pulses):
        self.pulses = []
        t = self.start_time
        for pulse, delay in zip(pulses, self.delays):
            p = copy.deepcopy(pulse)
            p.start = t
            p.end = t + p.duration
            t += p.duration + delay
            self.pulses.append(p)

        self.measurers = []
        for p in self.pulses:
            if p.has_measurers:
                self.measurers += p.measurers

    def run(self, time):
        for p in self.pulses:
            result = p.run(time)
            if result is not None:
                return result

    def append(self, pulse, delay):
        last_pulse = self.pulses[-1]
        self.delays.append(delay)
        copied_pulse = copy.deepcopy(pulse)
        copied_pulse.start = last_pulse.end + delay
        self.pulses.append(copied_pulse)

    def write_to_cycle_code(self, file, time_interval):
        with open(file, "w") as f:
            for pulse, delay in zip(self.pulses, self.delays):
                times = np.arange(0, pulse.duration, time_interval)
                x, y = 0, 0
                for t in times:  # t is in the domain of the pulse
                    if isinstance(pulse, LaserStrobe):
                        x, y = pulse.move_beam(t + pulse.start)
                    else:
                        x, y = pulse.x, pulse.y
                    current = pulse.modulate_beam(t) * pulse.power
                    x, y = pvcs.voltage_from_position(x, y)
                    f.write(cycle_code_line(x, y, time_interval, current) + "\n")
                f.write(cycle_code_line(x, y, delay, 0) + "\n")  # beam off and wait


def cycle_code_line(xv, yv, hold, current):
    return f"{xv:.3f},{yv:.3f},{hold:.3f},{current:.3f}"


def genericpolar(omega, r, phase=0):
    '''
    Generates parameterizations for x(t), y(t)
    which complete n revolutions over a duration on radius r, and phase shift
    Start from radius r0, go to r linearly (circular if r0 not specified).

    '''
    # a revolution occurs over 2pi
    # duration needs to mapped to 2pi * n

    def xfunc(t):
        return r(t) * np.cos(omega * t + phase)

    def yfunc(t):
        return r(t) * np.sin(omega * t + phase)

    return xfunc, yfunc


def linearspiral(radius, duration, n=1, phase=0, r0=None):
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
