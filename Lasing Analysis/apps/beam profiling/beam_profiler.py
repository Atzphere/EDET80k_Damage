'''
Helper functions for analyzing radial lasing temperature profiles.

'''

from scipy.optimize import curve_fit
import numpy as np
import logging


def gaussian_fitfun(x, mu, var, a, b):
    # gaussian fit function to profile beams with
    return a * np.exp(-((x - mu)**2 / (2 * var))) + b


def profile_beam(time, x, data, p0=(0, 0, 0, 0), sigma=None, error_tolerance=(np.inf, np.inf, np.inf, np.inf), **kwargs):
    '''
    Takes temperature vs x vs time data and attempts to fit it to a Gaussian
    beam. Returns the fitted parameters as a function of time, as well as whether
    a fit at time t either succeeds or fails/exceeds error tolerances.

    Can also pass parameters to curve_fit through kwargs.

    params

    time : np.array(float): time array data, shape (N).

    x : np.array(float): position array data, shape (M)

    data : temperature vs x vs time data: shape (N x M).

    p0 : tuple of initial guess parameters to be passed to curve_fit

    error_tolerance : tuple of maximum tolerable variances per parameter.
                      failure is flagged upon one or more tolerances exceeded.

    '''
    tolerance = np.array(error_tolerance)

    # accumulates parameter evolution over time
    params = np.zeros((len(time), len(p0)))
    # accumulates parameter variance estimates over time
    variance = np.zeros((len(time), len(p0)))
    # accumulates fit pass/fail results over time.
    validity = np.zeros(len(time))

    if sigma is None:
        sigma = np.ones(np.shape(data))

    for index, time in enumerate(time):
        try:
            popt, pcov = curve_fit(gaussian_fitfun, x, data[index], p0=p0, sigma=sigma[index], **kwargs)
            var = np.diag(pcov)
            params[index, :] = popt
            variance[index, :] = var
            if np.any(var > tolerance):
                logging.info(f"Param variance exceeded tolerance at time {time}")
                logging.info(f"{popt}, {var}")
            else:
                validity[index] = 1
        except RuntimeError:
            logging.info(f"Param variance exceeded tolerance at time {time}")

    return params, variance, validity
