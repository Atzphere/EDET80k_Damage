import simulationlib as sl
import lasinglib as ll
import numpy as np

SILICON = sl.Material(80, 0.09, 0.7, 0.002329002)
CHIP = sl.SimGrid(30, 101, 0.03, use_spar=True,
                  spar_thickness=0.5, spar_width=1)

sim = sl.Simulation(CHIP, SILICON, 7, pulses=None, ambient_temp=300,
                    starting_temp=3000, neumann_bc=True,
                    edge_derivative=0, sample_framerate=24, intended_pbs=1,
                    dense_logging=False, timestep_multi=1)

CENTER = (CHIP.CENTERPOINT)

# p = ll.LaserPulse()

sim.simulate()