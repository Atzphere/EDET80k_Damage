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
    def __init__(self, start_time, duration, measurement, tag, timestep=None):
        self.start_time = start_time
        self.duration = duration
        self.measurement = measurement
        self.timestep = timestep
        self.tag = tag

        self.last_meaurement_time = -np.inf

    def is_active(self, time):
        if self.timestep is not None:
            return (time - self.last_meaurement_time >= self.timestep) and time >= self.start_time and time < self.start_time + self.duration
        else:
            return time >= self.start_time and time < self.start_time + self.duration

    def check_measure(self, time, grid):
        if self.is_active(time):
            return self.measurement.measure(grid)


class Measurement(object):
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
    def __init__(self, diffusivity, emissivity, specific_heat, density):
        self.DIFFUSIVITY = diffusivity
        self.EMISSIVITY = emissivity
        self.SPECIFIC_HEAT = specific_heat
        self.DENSITY = density


class SimGrid(object):
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
    def __init__(self, simgrid, material, duration, pulses, ambient_temp, starting_temp=300, neumann_bc=True, edge_derivative=0, sample_framerate=24, intended_pbs=1, dense_logging=False, timestep_multi=1):
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

        self.cell_mass = self.simgrid.cell_area * \
            self.simgrid.CHIP_THICKNESS * self.material.DENSITY  # in g

        self.TIMESTEP = get_minimum_stable_timestep(
            self.simgrid.dx, self.material.DIFFUSIVITY)  # / 4

        self.gamma = self.material.DIFFUSIVITY * \
            (self.TIMESTEP / self.simgrid.dx**2)
        self.times = np.arange(0, self.STOP_TIME, self.TIMESTEP)

        self.timesteps_per_second = round(1 / self.TIMESTEP)
        self.timesteps_per_frame = round(
            (self.timesteps_per_second * self.PLAYBACKSPEED) / (self.SAMPLE_FRAMERATE))

        self.timesteps_per_percent = round(len(self.times) / 100)

    def simulate(self, analyzers=None):
        grid = self.simgrid.grid_template.copy()
        grid[:, 0] = 0
        grid[:, self.simgrid.RESOLUTION + 1] = 0
        grid[0, :] = 0
        grid[self.simgrid.RESOLUTION + 1, :] = 0

        roi_mask = grid != 0
        grid[:, :] = self.STARTING_TEMP

        spar_coefficients = self.simgrid.innergrid_template.copy()
        spar_coefficients[:, self.simgrid.half_grid - self.simgrid.spar_extension_cells:self.simgrid.half_grid +
                          1 + self.simgrid.spar_extension_cells] = self.simgrid.spar_multi
        spar_coefficients[self.simgrid.half_grid - self.simgrid.spar_extension_cells:self.simgrid.half_grid +
                          1 + self.simgrid.spar_extension_cells, :] = self.simgrid.spar_multi

        spar_coefficients = spar_coefficients.flatten()

        left = np.roll(roi_mask, -1)
        right = np.roll(roi_mask, 1)
        below = np.roll(roi_mask, 1, axis=0)
        above = np.roll(roi_mask, -1, axis=0)

        left_boundary = np.zeros(
            (self.simgrid.RESOLUTION + 2, self.simgrid.RESOLUTION + 2), dtype=bool)

        left_boundary[1:-1, 0] = True
        bottom_boundary = np.rot90(left_boundary)
        right_boundary = np.rot90(bottom_boundary)
        top_boundary = np.rot90(right_boundary)

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

        # xspace = np.linspace(0 - self.dx, self.CHIP_DIMENSION + self.dx, self.RESOLUTION + 2)

        print("Generating pulses", end="")

        # pulses.append(LaserStrobe(0.5, 5, self.CENTERPOINT, 6, radialgeneric(15, 5, 5, r0=5)))

        # pulses.append(LaserStrobe(0.5, 5, self.CENTERPOINT, 6, (lambda t: 14 * np.sin(18 * np.pi * t), lambda t: 30 * (t / 5))))

        # t = 1
        # for x in range(4, 28, 4):
        #     for y in range(4, 28, 4):
        #         pulses.append(LaserPulse(t, 0.1, (x, y), 10, sigma=0.15))
        #         t += 2 * (0.1)
        # print(" ...done")

        # pulses.append(LaserPulse(0, 6, (x, y), 0.2, sigma=0.15))

        # pulses.append(LaserPulse(0, 0.5, CENTERPOINT, 1, sigma=0.3))
        # pulses.append(LaserPulse(3, 0.5, CENTERPOINT, 1, sigma=0.3))

        # print("\nRendering pulses", end="")

        # for p in pulses:
        #     p.bake()
        # print(" ...done")

        print(
            f"Starting simulation: {round(self.STOP_TIME / self.TIMESTEP)} iterations.")
        print("[" + " " * 24 + "25" + " " * 23 +
              "50" + " " * 23 + "75" + " " * 24 + "]")
        print("[", end="")
        # precompute constants to optimize
        K1 = (self.material.EMISSIVITY * SBC * self.simgrid.cell_area) / \
            (self.cell_mass * self.material.SPECIFIC_HEAT) * self.TIMESTEP
        temps = []

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
                # print(grid[left_boundary])

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
                multi = spar_coefficients
            else:
                multi = 1

            delta += radiation_temp

            if self.pulses is not None:
                 / (cell_mass * SPECIFIC_HEAT)
                laser_delta = self.simgrid.innergrid_template.copy()
                laser_delta[:,:] = 0
                for p in self.pulses:  # fire any lasing activities that should occur
                    if p.is_active(t):
                        delta += p.run() * multi

            temps.append(grid[self.simgrid.half_grid, self.simgrid.half_grid])

            grid[roi_mask] += delta

            if n % self.timesteps_per_frame == 0 and not self.DENSE_LOGGING:
                deltas.append(delta.copy())
                states.append(grid.copy())
            elif self.DENSE_LOGGING:
                dense_deltas.append(delta.copy())
                dense_states.append(grid.copy())

            if n % self.timesteps_per_percent == 0:
                print("#", end="")
                progress += 1
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

        if False:
            if self.DENSE_LOGGING:
                pickled_data = pickle.dumps((dense_states, dense_deltas))
                TAG = "foobtest_"  # prefix to save the results under
                TAG += "_DENSE"
            else:
                # returns data as a bytes object
                pickled_data = pickle.dumps((states, deltas))
            compressed_pickle = blosc.compress(pickled_data)

            fname = TAG + " ".join([str(p)
                                    for n, p in enumerate(pulses) if n > 3]) + ".pkl"
            print(fname)
            with open("../saves/" + fname, "wb") as f:
                f.write(compressed_pickle)


logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)
