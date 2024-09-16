import numpy as np


class Measurer(object):
    '''
    An object passed to a simulation, composed of a time interval and a set of
    measurements to record during that interval.

    Attributes:

    start_time: float: the time at which the measurer activates

    duration: float: the length of time to remain active for after the start_time

    measurement: Measurement or Iterable[Measurement]
                 A set of measurements to execute when the measurer is active.
                 Each measurement should have a tag.

    tag: str ot Iterable[str]: the label for the measurement(s). Used to access the dictionary of results
              produced at the end of the simulation.

    timestep: float: the sampling interval between measurements in time.
                     Defaults to taking a measurement every iteration if None.
              Defaults to None.



    '''

    def __init__(self, start_time, duration, measurement, tag, timestep=None):
        self.start_time = start_time
        self.duration = duration
        self.measurement = measurement
        self.timestep = timestep
        self.tag = tag

        # for interval checking
        self.last_meaurement_time = -np.inf

    def is_active(self, time):
        '''
        Returns whether or not a measurement should be taken at a given time.

        '''
        if self.timestep is not None:
            return (time - self.last_meaurement_time >= self.timestep) and time >= self.start_time and time < self.start_time + self.duration
        else:
            return time >= self.start_time and time < self.start_time + self.duration

    def check_measure(self, time, grid):
        '''
        Records a measurement if appropriate at time.

        Returns a dictionary of measurement results and their tags.
        '''
        if self.is_active(time):
            results = {}
            self.last_meaurement_time = time
            measurement = self.measurement.measure(grid)
            for key in measurement.keys():
                mkey = self.tag + " " + key
                results.update({mkey: measurement[key]})

            results.update({self.tag + " " + "time": time})

            return results

        else:
            return None



class Measurement(object):
    '''
    A measurement, passed to a Measurer to be carried out when specified.

    Attributes:

    measurearea MeasureArea: An MeasureArea object specifying a region of the
                             simulation to be read from.

    modes: str or Callable(NDArray) or Iterable[str, Callable(NDArray)]:
    Functions to evaluate on the MeasureArea. Basic ones are predefined as strings:
    "ALL", "MEAN", "STD", "MAX... but you can also specify arbitrary ones.

    For functions with multiple outputs (i.e. temperatures and their locations),
    the simulation will create a dict entry for each array of outputs numbered as
    measure_tag ... 0, 1, 2, etc.
    '''
    default_samplers = {"ALL": lambda T, x, y: (T, x, y),
                        "MEAN": lambda T, x, y: np.mean(T),
                        "STD": lambda T, x, y: T.std(),
                        "MAX": lambda T, x, y: (np.max(T), x[np.where(T == np.max(T))], y[np.where(T == np.max(T))])}

    def __init__(self, measurearea, modes=["ALL"]):
        self.modes = modes
        self.methods = {}
        self.measurearea = measurearea
        for n, mode in enumerate(modes):
            if mode in Measurement.default_samplers.keys():
                self.methods.update({mode: Measurement.default_samplers[mode]})
            else:
                self.methods.update({f"METHOD{n}": mode})

    def measure(self, state):
        '''
        state is RxR
        '''
        measurements = {}
        temps = state[self.measurearea.mask]
        x = self.measurearea.x_pos
        y = self.measurearea.y_pos
        for k in self.methods.keys():
            measurements.update({k: self.methods[k](temps, x, y)})

        return measurements


class MeasureArea(object):
    '''
    mask of cells to read along with their x, y positions.

    Generate an offset meshgrid, with origin centered on the location of the measure area.

    shapemaker is a function that takes that meshgrid and returns true/false depending on whether
    or not to measure in a region.

    these are applied to ROI, not the entire grid (RxR)

    '''

    def __init__(self, grid, location, shapemaker):
        '''
        location: (x, y)
        '''
        x, y = grid.physical_meshgrid
        ox, oy = grid.get_offset_meshgrid(*location)
        self.mask = shapemaker(ox, oy)
        self.x_pos, self.y_pos = x[self.mask], y[self.mask]
        self.mask = self.mask.flatten()


class MeasurePoint(MeasureArea):
    '''
    A type of MeasureArea which yields a circular region around a point of interest.

    Novel attributes:

    radius float: the radius around the point of interest to sample. Uses the units
                  of the simgrid. Defaults to 0 (a point).
    '''
    def __init__(self, grid, location, radius=0):
        def _centered_circle(x, y):
            return x**2 + y**2 <= radius**2

        super().__init__(grid, location, shapemaker=_centered_circle)
