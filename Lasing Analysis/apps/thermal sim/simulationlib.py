import numpy as np
import logging
import matplotlib.pyplot as plt
import pickle
import blosc
import os
import matplotlib_animtools as ma
import PositionVoltageConverter_Standalone as pvcs

os.chdir(os.path.dirname(__file__))

SBC = 5.670E-14  # stefan-boltzmann constant per square mm


def get_minimum_stable_timestep(dx, a):
    return dx**2 / (4 * a)


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

    def __init__(self, start_time, duration, measurement, tags, timestep=None):
        self.start_time = start_time
        self.duration = duration
        self.measurements = measurements
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
            for m, t in zip(self.measurementsself.tags):
                if t not in results.keys():
                    results.update({t: [m.measure(grid)]})
                else:
                    results[t].append(m.measure(grid))

            return results


class Measurement(object):
    '''
    A measurement, passed to a Measurer to be carried out when specified.

    Attributes:

    measurearea MeasureArea: An MeasureArea object specifying a region of the
                             simulation to be read from.

    modes: str or Callable(NDArray) or Iterable[str, Callable(NDArray)]:
    Functions to evaluate on the MeasureArea. Basic ones are predefined as strings:
    "ALL", "MEAN", "STD", "MAX... but you can also specify arbitrary ones. 
    '''
    default_samplers = {"ALL": lambda T, x, y: T,
                        "MEAN": lambda T, x, y: np.mean(T),
                        "STD": lambda T, x, y: T.std(),
                        "MAX": lambda T, x, y: (np.max(T), x[np.where(T == np.max(T))], y[np.where(T == np.max(T))])}

    def __init__(self, measurearea, modes=["ALL"]):
        self.modes = modes
        self.methods = []
        self.measurearea = measurearea
        for mode in modes:
            if mode in Measurement.default_samplers.keys():
                self.methods.append(Measurement.default_samplers[mode])
            else:
                self.methods.append()

    def measure(self, state):
        '''
        state is RxR
        '''
        measurements = []
        temps = state[self.measurearea.mask]
        x = self.measurearea.x_pos
        y = self.measurearea.y_pos
        for m in self.methods:
            measurements.append(m(temps, x, y))

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


class Material(object):
    '''
    Class representing a materials to be used in the heat simulation.
    This should work fine as long as a units conform (i.e. all mm, meters, etc)
    but this has only been tested with SI-millimeter units.
    '''
    def __init__(self, diffusivity, emissivity, specific_heat, density):
        self.DIFFUSIVITY = diffusivity
        self.EMISSIVITY = emissivity
        self.SPECIFIC_HEAT = specific_heat
        self.DENSITY = density


class SimGrid(object):
    '''
    A class respresenting the physical environment over which the simulation will take place.
    Units must be consistent with the Material intended to be used.

    Attributes:

    CHIP_DIMENSION float: the dimension of the simulation. Currently as only squares
                          are supported, only one side-length is required.

    RESOLUTION int: the number of subdivisions along the X-Y axis, i.e. the spatial resolution
                    of the simulation. Computation time scales with O(N^4), so be reasonable. 

    CHIP_THICKNESS float: the thickness of the chip. used for thermal mass calculations.

    USE_SPAR bool: semi-functional parameter which introduces a cross-shaped chunk of
                   extra mass on the chip. This is included in radiation and irradiation
                   calculations but not conduction for now because I don't want to think
                   about how to ensure inhomogenous heat simulation stays stable.
    
    SPAR_THICKNESS float: the entire thickness of the chip in the spar regions, including
                          the nominal thickness of the chip.

    SPAR_WIDTH float: how wide the spars are.

    '''
    def __init__(self, dimension, resolution, thickness, use_spar=False, spar_thickness=0.5, spar_width=1):
        self.CHIP_DIMENSION = dimension
        self.RESOLUTION = resolution
        self.CHIP_THICKNESS = thickness
        self.USE_SPAR = use_spar
        self.SPAR_THICKNESS = spar_thickness
        self.SPAR_WIDTH = spar_width

        self.center = self.CHIP_DIMENSION / 2
        self.half_grid = self.RESOLUTION // 2
        self.CENTERPOINT = (self.center, self.center)
        self.dx = self.CHIP_DIMENSION / self.RESOLUTION
        self.cell_area = self.dx**2

        self.spar_multi = self.CHIP_THICKNESS / self.SPAR_THICKNESS
        self.spar_width_cells = int(self.SPAR_WIDTH // self.dx)
        self.spar_extension_cells = (self.spar_width_cells - 1) // 2
        self.grid_template = np.ones(
            (self.RESOLUTION + 2, self.RESOLUTION + 2))
        self.innergrid_template = np.ones(
            (self.RESOLUTION, self.RESOLUTION))

        self.physical_meshgrid = self.get_offset_meshgrid(0, 0)

    def get_offset_meshgrid(self, x, y):
        '''
        Builds a meshgrid of values corresponding to coordinates on the sim
        with the origin at x, y
        '''
        # x and y are the cartesian coordinates of the origin
        bx = np.linspace(0, self.CHIP_DIMENSION, self.RESOLUTION) - x
        by = np.linspace(0, self.CHIP_DIMENSION,
                         self.RESOLUTION) - self.CHIP_DIMENSION + y

        return np.meshgrid(bx, by)


class Simulation(object):
    '''
    The main simulation object. Simulations can be run by calling Simulation.simulate(measurers).

    Attributes:

    simgrid Simgrid: the simgrid to solve the heat equations on.
    
    material Material: the material to use.

    pulses Iterable[LaserPulse]: The sequence of laser pulses to apply to the chip.

    ambient_temp float: The ambient environment temperature.

    starting_temp float: pre-heat the simgrid to this temperature

    neumann_bc bool: Whether or not to use Neumann boundary conditions. If false, use Dirichet instead (edges set to ambient)

    edge_derivative float: The edge temperature flux if using Neumann boundary conditions. Currently only supports symmetrical BC.

    sample_framerate int: How many times to capture the state of the simulation per second. Useful for animations.

    INTENDED_PBS float: The intended playback speed of an animation of the simulation. Use in tandem with sample_framerate to record slow-mo.

    DENSE_LOGGING bool: Whether or not to bypass sample_framerate and record all states from the simlation. Useful for studying rapid dynamics,
                        but otherwise can result in large amounts of memory consumption.

    TIMESTEP_MULTI float: Multiplier applied to the minimum stable iteration timestep for the simulation. useful for more resolution.

    progress_bar bool: whether or not to print a progress bar.

    radiation bool: whether or not to simulate radiation.

    '''
    def __init__(self, simgrid, material, duration, pulses, ambient_temp, starting_temp=300, neumann_bc=True, edge_derivative=0, sample_framerate=24, intended_pbs=1, dense_logging=False, timestep_multi=1, radiation=True, progress_bar=True):
        self.simgrid = simgrid
        self.material = material
        self.pulses = pulses
        self.STOP_TIME = duration
        self.AMBIENT_TEMPERATURE = ambient_temp
        self.STARTING_TEMP = starting_temp
        self.NEUMANN_BC = neumann_bc
        self.EDGE_DERIVATIVE = edge_derivative
        self.SAMPLE_FRAMERATE = sample_framerate
        self.INTENDED_PBS = 1
        self.DENSE_LOGGING = dense_logging
        self.TIMESTEP_MULTI = timestep_multi
        self.evaluated = False
        self.PLAYBACKSPEED = intended_pbs
        self.use_radiation = radiation
        self.progress_bar = progress_bar

        self.cell_mass = self.simgrid.cell_area * \
            self.simgrid.CHIP_THICKNESS * self.material.DENSITY  # in g

        self.TIMESTEP = get_minimum_stable_timestep(
            self.simgrid.dx, self.material.DIFFUSIVITY)  * self.TIMESTEP_MULTI

        self.gamma = self.material.DIFFUSIVITY * \
            (self.TIMESTEP / self.simgrid.dx**2)
        self.times = np.arange(0, self.STOP_TIME, self.TIMESTEP)

        self.timesteps_per_second = round(1 / self.TIMESTEP)
        self.timesteps_per_frame = round(
            (self.timesteps_per_second * self.PLAYBACKSPEED) / (self.SAMPLE_FRAMERATE))

        self.timesteps_per_percent = round(len(self.times) / 100)

    def simulate(self, analyzers=None):
        # initalize simulation plane
        grid = self.simgrid.grid_template.copy()
        grid[:, 0] = 0
        grid[:, self.simgrid.RESOLUTION + 1] = 0
        grid[0, :] = 0
        grid[self.simgrid.RESOLUTION + 1, :] = 0

        # ROI is where the the physically consistent environment is (no ghost cells)
        roi_mask = grid != 0

        # initial conditions
        grid[:, :] = self.STARTING_TEMP

        # initialize spar
        spar_coefficients = self.simgrid.innergrid_template.copy()
        spar_coefficients[:, self.simgrid.half_grid - self.simgrid.spar_extension_cells:self.simgrid.half_grid +
                          1 + self.simgrid.spar_extension_cells] = self.simgrid.spar_multi
        spar_coefficients[self.simgrid.half_grid - self.simgrid.spar_extension_cells:self.simgrid.half_grid +
                          1 + self.simgrid.spar_extension_cells, :] = self.simgrid.spar_multi

        # applications to the ROI must be a flattened array.
        spar_coefficients = spar_coefficients.flatten()


        # neighboring cell masks for heat conduction
        left = np.roll(roi_mask, -1)
        right = np.roll(roi_mask, 1)
        below = np.roll(roi_mask, 1, axis=0)
        above = np.roll(roi_mask, -1, axis=0)

        # Ghost cell boundaries of the simulation to enforce boundary conditions
        left_boundary = np.zeros(
            (self.simgrid.RESOLUTION + 2, self.simgrid.RESOLUTION + 2), dtype=bool)

        left_boundary[1:-1, 0] = True
        bottom_boundary = np.rot90(left_boundary)
        right_boundary = np.rot90(bottom_boundary)
        top_boundary = np.rot90(right_boundary)

        # Boundaries of the physical area to enforce Neumann BC
        left_boundary_inner = np.zeros(
            (self.simgrid.RESOLUTION + 2, self.simgrid.RESOLUTION + 2), dtype=bool)
        left_boundary_inner[1:-1, 1] = True
        bottom_boundary_inner = np.rot90(left_boundary_inner)
        right_boundary_inner = np.rot90(bottom_boundary_inner)
        top_boundary_inner = np.rot90(right_boundary_inner)

        grid[roi_mask] = self.STARTING_TEMP

        if self.DENSE_LOGGING:
            dense_states = []
            dense_deltas = []
            dense_deltas.append(np.zeros(np.shape(grid)))
            dense_states.append(grid)

        else:
            states = []
            deltas = []
            deltas.append(np.zeros(np.shape(grid)))
            states.append(grid)

        print(
            f"Starting simulation: {round(self.STOP_TIME / self.TIMESTEP)} iterations.")

        if self.progress_bar:
            print("[" + " " * 24 + "25" + " " * 23 +
                  "50" + " " * 23 + "75" + " " * 24 + "]")
            print("[", end="")

        # precompute constants to optimize
        K1 = (self.material.EMISSIVITY * SBC * self.simgrid.cell_area) / \
            (self.cell_mass * self.material.SPECIFIC_HEAT) * self.TIMESTEP
        temps = []
        laser_delta = self.simgrid.innergrid_template.copy().flatten()
        progress = 0

        for n, t in enumerate(self.times):
            roi = grid[roi_mask]

            if self.NEUMANN_BC:
                grid[left_boundary] = grid[left_boundary_inner] - \
                    self.EDGE_DERIVATIVE * self.simgrid.dx
                grid[bottom_boundary] = grid[bottom_boundary_inner] - \
                    self.EDGE_DERIVATIVE * self.simgrid.dx
                grid[right_boundary] = grid[right_boundary_inner] - \
                    self.EDGE_DERIVATIVE * self.simgrid.dx
                grid[top_boundary] = grid[top_boundary_inner] - \
                    self.EDGE_DERIVATIVE * self.simgrid.dx

            conduction = self.gamma * \
                (grid[below] + grid[above] +
                 grid[left] + grid[right] - 4 * roi)
            delta = conduction

            # power output from radiation
            # convert to temperature drop from radiation
            radiation_power = (self.AMBIENT_TEMPERATURE**4 - roi**4)
            radiation_temp = radiation_power * K1 * 10
            if self.simgrid.USE_SPAR:
                radiation_temp *= spar_coefficients
                spar_multi = spar_coefficients
            else:
                spar_multi = 1

            if self.use_radiation:
                delta += radiation_temp

            if self.pulses is not None:
                laser_delta[:] = 0
                for p in self.pulses:  # fire any lasing activities that should occur
                    if p.is_active(t):
                        laser_delta += p.run(t)
                delta += laser_delta * (self.TIMESTEP * spar_multi) / (self.cell_mass * self.material.SPECIFIC_HEAT)

            temps.append(grid[self.simgrid.half_grid, self.simgrid.half_grid] - 273.15)
            grid[roi_mask] += delta

            if n % self.timesteps_per_frame == 0 and not self.DENSE_LOGGING:
                deltas.append(delta.copy())
                states.append(grid.copy())
            elif self.DENSE_LOGGING:
                dense_deltas.append(delta.copy())
                dense_states.append(grid.copy())

            if n % self.timesteps_per_percent == 0 and self.progress_bar:
                print("#", end="")
                progress += 1
        if self.progress_bar:
            print("]")
        print("Simulation done.")

        plt.plot(self.times, temps)
        plt.show()

        for n, s in enumerate(states):
            states[n] = s - 273.15  # convert to celsius

        ma.animate_2d_arrays(states, interval=(1 / (self.SAMPLE_FRAMERATE))
                             * 1000, repeat_delay=0, cmap="magma", vmin=0, vmax=450)

        self.evaluated = True
        if self.DENSE_LOGGING:
            self.sim_states = dense_states
            self.sim_deltas = dense_deltas
            return dense_states, dense_deltas
        else:
            self.sim_states = states
            self.sim_deltas = deltas
            return states, deltas

    def save(self, fname=None):
        if self.DENSE_LOGGING:
            pickled_data = pickle.dumps((dense_states, dense_deltas))
            TAG = "foobtest_"  # prefix to save the results under
            TAG += "_DENSE"
        else:
            # returns data as a bytes object
            pickled_data = pickle.dumps((states, deltas))
        compressed_pickle = blosc.compress(pickled_data)

        fname = TAG + " ".join([str(p)
                                for n, p in enumerate(pulses) if n < 3]) + ".pkl"
        print(fname)
        with open("../saves/" + fname, "wb") as f:
            f.write(compressed_pickle)
        print(f"Saved simulation to /saves/{fname}.")
