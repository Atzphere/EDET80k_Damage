'''
This module contains everything to do with simulation laser pulses.

Primitive lasing actions can be modelled with the following objects:

LaserPulse: A static pulse with a certain duration at a particular position.
The amplitude of the pulse can be modulated as an arbitrary function of time.

The file modulators.py has a set of primitive modulation generators, as well as an
example of the overall signature for convenient use and expansion.

LaserStrobe: A subclass of LaserPulse. Works effectively the same, but its position
is controllable as a function of time.

The file shapes.py has set of primitive shape generators as well as a function template
describing a generalized shapemaker function.

LaserSequence: A subclass of LaserPulse representing an arbitrary ordered set of LaserPulse actions, intended to be executed
sequentially. The timing of the individual pulses can be adjusted.
These sequences can also be compiled to cycle code to be executed as an actual annealing cycle,
but be sure that such sequences are physically safe to execute;

@@@@@laser POWER in the simulation is directly interpreted as CURRENT when converted to CYCLE CODE!!@@@@@
6 "watts" in the sim translates to 6 amps. This will destroy your chip. Once we understand the mapping between
laser current and laser power output (efficiency is non-linear as you pump more), one could implement a translation layer.


'''

import numpy as np
import measurelib as ml
import copy
from collections.abc import Iterable
import position_voltage_converter as pvcs

DEFAULT_LASER_SIGMA = 0.18


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

    measure_target: bool: whether or not to measure the position of the pulse using a built-in Measurer object.
            default false

    target_r0: float: the size of the circular region around the measure target to record. 0 corresponds to
                      just the exact center.
            default 0
    target_modes: list[Str, Callable]: The method(s) to evaluate the target measure region with. See measurelib for more details.
            defaults "MEAN".

    measure_padding: Tuple(float, float): The time before and after the duration of the laser pulse to begin recording.
            default (0, 0)

    measure_tag: str: The tag to identify the measurer with in the recorded_data dictionary.
            default None

    measure_timestep: float: the time interval to wait between subsequent measurements.
            default 0

    aux_measurers: list[Measurer]: other measurers to be bound to this laser. Not location dependent.
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
        if measure_tag is None:  # generate a procedural measure tag if nothing else is supplied
            self.measure_tag = f"PULSE_{start}s+{duration}s_{power}A"

        self.measurers = copy.deepcopy(aux_measurers)
        # tag the other measurers assigned to the laser pulse
        if aux_measurers != []:
            for m in self.measurers:
                m.tag = f"{self.measure_tag}_{m.tag}"

        if measure_target:  # set recording time padding on either side of the pulse
            pad_left, pad_right = measure_padding
            measure_duration = pad_left + pad_right + duration
            measure_start = start - pad_left

            # initialize pulse measurement area and add it to measurers

            self.measure_area = ml.MeasurePoint(self.simgrid, position, radius=target_r)
            self.beam_measurement = ml.Measurement(self.measure_area, modes=target_modes)
            self.target_measurer = ml.Measurer(measure_start, measure_duration, self.beam_measurement, tag=self.measure_tag, timestep=measure_timestep)
            self.measurers.append(self.target_measurer)

        if params is not None:  # process any parameters meant to be passed to modulation
            self.params = params

        else:
            self.params = None

        # framework for generating the applied beam
        self.beam_modulation = []
        self.beam_instructions = None
        self.rmg = self.radial_meshgrid(self.x, self.y)

    def has_measurers(self):
        '''
        Reports whether or not the pulse has measurers assigned it
        '''
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
        function to build a radial intensity distribution (namely gaussian)
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
        '''
        Produces the modulation coefficient to apply to the pulse's
        base intensity as a function of time.
        '''
        t = time - self.start
        coeff = 1

        # the coefficient is the multiplicative sum of all the supplied modulators
        if self.modulators is not None:
            if self.params is not None:
                for m, p in zip(self.modulators, self.params):
                    coeff *= m(t, *p)
            else:
                for m in self.modulators:
                    coeff *= m(t)
        return coeff

    def is_active(self, time):
        '''
        Returns whether or not a laser pulse should be firing at a given time
        '''
        return self.start <= time and time <= self.end

    def run(self, time):
        '''
        "Fires" the laser, if it is supposed to be active at the time supplied.

        Returns a array mapping to the physical portion of the simulation grid, but flattened.
        This array represents the intensity distribution of the beam.
        This can be directly applied to this region by adding it, which is done in the simulation.

        '''
        if self.is_active(time):
            return (self.modulate_beam(time) * self.laser_beam(self.rmg, self.sigma, self.power)).flatten().copy()

    def __str__(self):
        '''
        String representation of the laser's properties for easy visual identification.
        Template is:

        Pulse(<power>W, <start time> -> <duration> -> <end time> + <MODulated or NOMOD>)

        '''
        if self.modulators is not None:
            m = str(len(self.modulators)) + "MOD)"
        else:
            m = "NOMOD)"
        return f"Pulse({self.power:.3f}W, {self.start:.3f} + {self.duration:.3f}S -> {self.end:.3f}" + m

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
        '''
        With strobes, the beam is "moved" in a bit of a weird way...
        instead of moving a discrete point around, we move the origin of
        an independent NxN meshgrid. An intensity distribution function is then
        evaluated on this grid, and the output is applied to the ROI.

        This works fairly okay, but expect some level of aliasing - especially with
        beams that have intensity distributions (i.e. low sigma). This will occur because
        you are moving the beam between individual discretized spatial points such that the peak intensity
        isn't being recorded.

        Let O represent points in the discretized simulation space, and I the maximum intensity peak of the laser
        distribution. If you increasingly concentrate laser power near this peak:

        OI            O

        will result in a higher than

        O      I      O

        As I moves from left to right, you will see a drop-off of effective power and then a rise
        as it approaches the next discretization point.

        This could hypothetically be fixed by integrating the intensity distribution over each spatial cell,
        but this would be very slow, and another option is to simply use a higher spatial resolution. 

        '''

        if self.is_active(time):
            m = self.modulate_beam(time)
            x, y, = self.move_beam(time)
            r = self.radial_meshgrid(x, y)
            return (m * self.laser_beam(r, self.sigma, self.power)).flatten()
        else:
            return None


class LaserSequence(LaserPulse):
    '''
    A subclass of LaserPulse representing an arranged sequence of LaserPulses. This is useful for choreographing repetitive
    individually crafted LaserPulses and/or LaserStrobes to form a simulated (or real!) annealing cycle.

    Novel attributes:

    pulses: Iterable[LaserPulse]: An iterable of LaserPulse-like objects to arrange.
                                  This should hypothetically support other LaserSequence objects... but this has not be tested.
                                  Note that in a LaserSequence, the start times of the individual pulses will be overridden - you do
                                  not need to specify something realistic.

    delay: float or Iterable[float]: A single delay value to wait between laser pulses, or an array mapping unique delay times to pulses

    start_time: when in sim-time to begin the LaserSequence. Not used for cycle code.

    '''
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
        self.build_trace()

    def build_trace(self):
        '''
        Generates an array of x-y positions of the component pulses/strobes for previewing
        '''
        self.trace_x = []
        self.trace_y = []
        for pulse in self.pulses:
            if not isinstance(pulse, LaserStrobe):
                x, y = pulse.x, pulse.y
                self.trace_x.append(x)
                self.trace_y.append(y)
            else:
                times = np.arange(0, pulse.duration, 0.03)
                for t in times:  # t is in the domain of the pulse
                    x, y = pulse.move_beam(t + pulse.start)
                    self.trace_x.append(x)
                    self.trace_y.append(y)

    def build_sequence(self, pulses):
        self.pulses = []
        t = self.start_time
        for pulse, delay in zip(pulses, self.delays):
            p = copy.deepcopy(pulse)
            p.start = t
            p.end = t + p.duration
            t += p.duration + delay
            self.pulses.append(p)

        self.duration = t - self.start_time

        self.measurers = []
        for p in self.pulses:
            if p.has_measurers:
                self.measurers += p.measurers

    def run(self, time):
        '''
        Runs the first pulse that should be running.
        '''
        for p in self.pulses:
            result = p.run(time)
            if result is not None:
                return result

    def append(self, pulse, delay):
        '''
        Method for appending new pulses to the LaserSequence.
        Although this is provided, it is preferable (and better-tested)
        to build a list of pulses first
        and then convert the entire thing to a sequence.
        '''
        last_pulse = self.pulses[-1]
        self.delays.append(delay)
        copied_pulse = copy.deepcopy(pulse)
        copied_pulse.start = last_pulse.end + delay
        self.pulses.append(copied_pulse)
        self.measurers.append(copied_pulse.measurers)

    def __str__(self):
        return f"LaserSequence({list([str(p) for p in self.pulses])})"

    def write_to_cycle_code(self, file, time_interval):
        '''
        Converts the LaserSequence to a cycle code file.
        Does not require a simulation to be run.

        Parameters
        file str : filepath to write the cycle code to.

        time_interval : how frequently to sample the pulses to the cycle code file for
        time-continuous processes i.e. modulation and strobing.

        '''

        with open(file, "w") as f:
            for pulse, delay in zip(self.pulses, self.delays):
                if pulse.modulators is not None and not isinstance(pulse, LaserPulse):
                    # write non-modulated binary pulses with 100% precision
                    x, y = pulse.x, pulse.y
                    f.write(cycle_code_line(x, y, pulse.duration, pulse.power) + "\n")

                else:
                    # if a pulse has some sort of continuous profile, sample it with the resolution specified by time_interval
                    # and write individual cycle code lines per sample
                    times = np.arange(0, pulse.duration, time_interval)
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
