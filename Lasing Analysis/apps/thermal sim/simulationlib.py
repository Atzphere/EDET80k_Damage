'''

This module defines the Simulation object as well as its requisite classes.
The workflow of using the simulation goes like this:

Initialize a SimGrid (the discretized spatial domain over which the thermal simulation is to be run)
and a Material (the material to model conduction and radiation with).

WIth the SimGrid, initialize a list of LaserPulses or LaserSequences (see lasinglib)
to be carried out at some point.

Initialize the Simulation object using your simgrid, material, and laser pulses.
Key parameters of interest are the boundary/initial conditions, simulation length, and
the duration/temporal resolution of the simulation.

If desired, you can use measurelib (see file for more details) to create Measurer objects
which can also be passed to the simulation. These objects specify arbitrary regions of the
SimGrid to be measured, and what types of measurements to be performed (mean, max, etc.).

Once you are happy with your simulation, run it with s.simulate().
You can now access the data from the simulation with s.recorded_data, animate it with s.animate(),
and also save the entire Simulation object as a pickle file for easy future analysis with s.save()


'''

import numpy as np
import logging
import dill
import blosc
import os
import matplotlib_animtools as ma
from collections.abc import Iterable

os.chdir(os.path.dirname(__file__))

SBC = 5.670E-14  # stefan-boltzmann constant per square mm


def get_minimum_stable_timestep(dx, a):
    return dx**2 / (4 * a)


def ragged(lst):
    # returns true if there's any list with unequal length to the first one
    return not any(len(lst[0]) != len(i) for i in lst)


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

    Recorded data can then be accessed via Simulation.recorded_data (dict).

    Attributes:

    simgrid Simgrid: the simgrid to solve the heat equations on.

    material Material: the material to use.

    pulses Iterable[LaserPulse]: The sequence of laser pulses to apply to the chip.

    ambient_temp float: The ambient environment temperature.

    starting_temp float: pre-heat the simgrid to this temperature

    neumann_bc bool: Whether or not to use Neumann boundary conditions. If false, use Dirichet instead (edges set to ambient temperature).
                     Neumann BC currently only supports a heat flux of zero across the boundary (thermally isolated aside from radiation).

    edge_derivative float: The edge temperature flux if using Neumann boundary conditions. Currently only supports symmetrical BC.

    sample_framerate int: How many times to capture the state of the simulation per second. Useful for animations. If zero, do not record states.

    INTENDED_PBS float: The intended playback speed of an animation of the simulation. Use in tandem with sample_framerate to record slow-mo.

    DENSE_LOGGING bool: Whether or not to bypass sample_framerate and record all states from the simlation. Useful for studying rapid dynamics,
                        but otherwise can result in large amounts of memory consumption.

    TIMESTEP_MULTI float: Multiplier applied to the minimum stable iteration timestep for the simulation. useful for more resolution.

    progress_bar bool: whether or not to print a progress bar.

    radiation bool: whether or not to simulate radiation.

    silent bool: whether or not to print anything to stdout regarding sim status

    '''

    def __init__(self, simgrid, material, duration, pulses, ambient_temp, starting_temp=300, neumann_bc=True, edge_derivative=0, sample_framerate=24, intended_pbs=1, dense_logging=False, timestep_multi=1, radiation=True, progress_bar=True, silent=True):
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
        self.recorded_data = {}
        self.silent = silent

        self.record_states = (sample_framerate != 0)

        self.cell_mass = self.simgrid.cell_area * \
            self.simgrid.CHIP_THICKNESS * self.material.DENSITY  # in g

        self.TIMESTEP = get_minimum_stable_timestep(
            self.simgrid.dx, self.material.DIFFUSIVITY) * self.TIMESTEP_MULTI

        self.gamma = self.material.DIFFUSIVITY * \
            (self.TIMESTEP / self.simgrid.dx**2)
        self.times = np.arange(0, self.STOP_TIME, self.TIMESTEP)

        self.timesteps_per_second = round(1 / self.TIMESTEP)

        if self.record_states:
            self.timesteps_per_frame = round((self.timesteps_per_second * self.PLAYBACKSPEED) / (self.SAMPLE_FRAMERATE))

        self.timesteps_per_percent = round(len(self.times) / 100)

    def simulate(self, analyzers=[]):
        recorded_data = {}

        # load laser pulse associated measurers

        for pulse in self.pulses:
            if pulse.has_measurers():
                analyzers += pulse.measurers


        # initalize simulation plane
        # grid is a (N+2) x (N+2) array. The bordering cells are used to calculate boundary conditions.
        grid = self.simgrid.grid_template.copy()
        grid[:, 0] = 0
        grid[:, self.simgrid.RESOLUTION + 1] = 0
        grid[0, :] = 0
        grid[self.simgrid.RESOLUTION + 1, :] = 0

        # ROI is where the the physically consistent environment is (no ghost cells)
        roi_mask = grid != 0

        # initial conditions
        grid[:, :] = self.STARTING_TEMP

        def _try_measurements(t):
            if analyzers != []:
                for a in analyzers:
                    result = a.check_measure(t, grid[roi_mask])
                    if result is not None:
                        for k in result.keys():
                            if k not in recorded_data.keys():
                                recorded_data.update({k: [result[k]]})
                            else:
                                recorded_data[k].append(result[k])

        # initialize spar geometry
        spar_coefficients = self.simgrid.innergrid_template.copy()
        spar_coefficients[:, self.simgrid.half_grid - self.simgrid.spar_extension_cells:self.simgrid.half_grid +
                          1 + self.simgrid.spar_extension_cells] = self.simgrid.spar_multi
        spar_coefficients[self.simgrid.half_grid - self.simgrid.spar_extension_cells:self.simgrid.half_grid +
                          1 + self.simgrid.spar_extension_cells, :] = self.simgrid.spar_multi

        # all array operations on the ROI must be done with flattened arrays of size NxN.
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

        # initial condition and measurements
        grid[roi_mask] = self.STARTING_TEMP
        _try_measurements(0)

        # initialize data logging arrays
        if self.record_states:
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

        if not self.silent:
            print(f"Starting simulation: {round(self.STOP_TIME / self.TIMESTEP)} iterations.")

        # progress bar reporting
        if self.progress_bar and not self.silent:
            print("[" + " " * 24 + "25" + " " * 23 +
                  "50" + " " * 23 + "75" + " " * 24 + "]")
            print("[", end="")

        # precompute constants to optimize
        K1 = (self.material.EMISSIVITY * SBC * self.simgrid.cell_area) / \
            (self.cell_mass * self.material.SPECIFIC_HEAT) * self.TIMESTEP

        # empty array to which laser energy is accumulated before adding to the grid
        laser_delta = self.simgrid.innergrid_template.copy().flatten()
        progress = 1

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
                    ld = p.run(t)
                    if ld is not None:
                        laser_delta += ld
                delta += laser_delta * (self.TIMESTEP * spar_multi) / (self.cell_mass * self.material.SPECIFIC_HEAT)

            grid[roi_mask] += delta

            _try_measurements(t)

            if self.record_states and n % self.timesteps_per_frame == 0 and not self.DENSE_LOGGING:
                deltas.append(delta.copy())
                states.append(grid.copy())
            elif self.DENSE_LOGGING:
                dense_deltas.append(delta.copy())
                dense_states.append(grid.copy())

            if n % self.timesteps_per_percent == 0 and self.progress_bar and not self.silent:
                print("#", end="")
                progress += 1

        # simulation iteration now ends

        # process recorded data:
        # break down tuple data (measurers reporting multiple values) into individual series in recorded_data
        new_data = {}
        tuple_keys = []

        for k in recorded_data.keys():
            data = recorded_data[k]
            if isinstance(data[0], tuple):
                tuple_keys.append(k)
                for n in range(len(data[0])):
                    new_data.update({f"{k} {n}": []})

                for line in data:
                    for n, item in enumerate(line):
                        new_data[f"{k} {n}"].append(item)

        recorded_data.update(new_data)

        # clean up the old unexpanded tuple arrays
        for tup in tuple_keys:
            del recorded_data[tup]

        # convert data series to numpy arrays if supported

        for k in recorded_data.keys():
            read = recorded_data[k]
            if isinstance(read[0], Iterable) and not ragged(read):
                pass
            else:
                recorded_data[k] = np.array(read)

        # convert temperature of simulation recording for animation readability

        if self.record_states:
            if self.DENSE_LOGGING:
                for n, s in enumerate(dense_states):
                    dense_states[n] = s - 273.15  # convert to celsius
                self.sim_states = dense_states
                self.sim_deltas = dense_deltas

            else:
                for n, s in enumerate(states):
                    states[n] = s - 273.15  # convert to celsius
                self.sim_states = states
                self.sim_deltas = deltas

            recorded_data.update({"states": self.sim_states, "deltas": self.sim_deltas})

        # saved processed data to the class
        self.recorded_data = recorded_data
        self.evaluated = True

        if not self.silent:
            if self.progress_bar:
                print("]")
            print("Simulation done.")

        return self.recorded_data

    def animate(self, fname=None, **kwargs):
        '''
        Animates the result of the animation according to the parameters supplied earlier.
        '''
        if not self.evaluated or not self.record_states:
            logging.error("No information to animate.")
            return None
        return ma.animate_2d_arrays(self.recorded_data["states"], interval=(1 / (self.SAMPLE_FRAMERATE))
                             * 1000, **kwargs)

    def save(self, fname, auto=True):
        '''
        Pickles the simulation object and saves it to a file. Used for easy analysis in the future.

        Parameters:

        fname str: the file path to save the simulation to.

        auto bool: Whether or not to append information to the file name based on the pulses in the simulation.
        '''
        if not self.evaluated:
            return None
        pickled_data = dill.dumps(self)
        if auto:
            TAG = fname  # prefix to save the results under
            if self.DENSE_LOGGING:
                TAG += "_DENSE"
            fname = TAG + " ".join([str(p) for n, p in enumerate(self.pulses) if n < 3]) + ".pkl"
        compressed_pickle = blosc.compress(pickled_data)

        with open(fname, "wb") as f:
            f.write(compressed_pickle)
        print(f"Saved simulation to {fname}.")

def load_sim(fname):
    with open(fname, 'rb') as f:
        return dill.loads(blosc.decompress(f.read()))
