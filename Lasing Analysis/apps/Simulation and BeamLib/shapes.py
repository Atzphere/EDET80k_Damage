import numpy as np


def flick(x1, y1, x2, y2, time, timestep):
    n = int(time // timestep)
    xpos = np.zeros(n + 1)
    ypos = np.zeros(n + 1)

    xpos[::2] = x1
    xpos[1::2] = x2

    ypos[::2] = y1
    ypos[1::2] = y2

    def x(t):
        return xpos[int(t // timestep)]

    def y(t):
        return ypos[int(t // timestep)]

    return x, y


def genericpolar(phi, r, phase=0):
    '''
    Generates parameterizations for x(t), y(t) in polar coordinates r, phi

    phi: Callable float -> float: parameterized azimuthal angle as a function of t. Should return an angle in radians.

    r: Callable float -> float: paramterization of radius as a function of t.

    phase: float: phase shift to add on top of phi(t).

    '''
    # a revolution occurs over 2pi
    # duration needs to mapped to 2pi * n

    def xfunc(t):
        return r(t) * np.cos(phi(t) + phase)

    def yfunc(t):
        return r(t) * np.sin(phi(t) + phase)

    return xfunc, yfunc


def genericradial(omega, r, phase=0):
    '''
    Generates parameterizations for x(t), y(t)
    for a function with a constant angular velocity and r(t).

    Parameters:

    omega float: Angular velocity.

    r Callable float -> float: paramterization of radius as a function of t.

    phase float: phase shift to add on top of phi(t).

    '''
    # a revolution occurs over 2pi
    # duration needs to mapped to 2pi * n

    return genericpolar(lambda t: omega * t, r, phase=phase)


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

    return genericradial(omega, r, phase=phase)
