'''
Maps laser input current to effective raw power output.
Mapping is done via a curve-fitted quadratic model.
'''
import numpy as np


LASING_CUTOFF_CURRENT = 1.12  # Amps, essentially no lasing below this
QUAD_MODEL_PARAMS = (-0.07902737, 6.67298871, -7.34059295)


def quadratic(current, a, b, c):
    return a * current**2 + b * current + c


def quad_formula(a, b, c):
    r1 = (-b + np.sqrt(b**2 - (4 * a * c))) / (2 * a)
    r2 = (-b - np.sqrt(b**2 - (4 * a * c))) / (2 * a)
    return r1, r2


def inverse_quadratic(power, a, b, c):
    return quad_formula(a, b, c - power)


def current_to_power(current):
    if current <= LASING_CUTOFF_CURRENT:
        return 0
    else:
        return quadratic(current, *QUAD_MODEL_PARAMS)


def power_to_current(power):
    current = inverse_quadratic(power, *QUAD_MODEL_PARAMS)[0]
    if current <= LASING_CUTOFF_CURRENT:
        return 0
    else:
        return current
