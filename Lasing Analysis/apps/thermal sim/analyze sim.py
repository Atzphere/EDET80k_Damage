import matplotlib_animtools as ma
import os
import numpy as np
import blosc
import pickle
from simulator import TIMESTEP

DISPLAY_FRAMERATE = 24

SLOWMO = 0.05
SPEEDUP = round(1 / SLOWMO)

SHAPE = 101

os.chdir(os.path.dirname(__file__))
DPATH = "../saves/foobtest_Pulse(0.5A2SNOMOD).pkl"


with open(DPATH, "rb") as f:
    compressed_pickle = f.read()
    depressed_pickle = blosc.decompress(compressed_pickle)

data = pickle.loads(depressed_pickle)

states = data[0]
grad = np.array([np.gradient(g) for g in states])
xgrad = grad[:,0]
ygrad = grad[:,1]

mag = np.sqrt(xgrad**2 + ygrad**2)

mag = mag[::SPEEDUP]
states = states[::SPEEDUP]

deltas = data[1]
del deltas[0]
deltas = [a.reshape((SHAPE, SHAPE)) / TIMESTEP for a in deltas]

ma.animate_2d_arrays(states, interval=(1 / (DISPLAY_FRAMERATE))
                     * 1000, repeat_delay=0, cmap="magma", vmax=400)
