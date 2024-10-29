import numpy as np
import math


def make_exp_pulse(up_time, hold_time, power_rampup):
    duration = up_time + hold_time

    def function(t):
        if t <= up_time:
            val = np.exp(((3 * t) / up_time) - 3) ** power_rampup
        elif t < up_time + hold_time:
            val = 1
        elif t <= duration:
            val = 0
        return val

    return function


def normal_curve(start_time, sigma):
    def function(t):
        return (1 / (sigma * math.sqrt(2 * np.pi))) * np.exp((-1 / 2) * ((t - start_time)**2 / sigma**2))

    return function


def doubleSinePulse(up_time, hold_time, down_time, power_rampup=1, power_rampdown=1):
    duration = up_time + down_time + hold_time

    def function(t):
        if t <= up_time:
            val = np.sin((np.pi / (2 * up_time)) * t)**power_rampup
        elif t < up_time + hold_time:
            val = 1
        elif t <= duration:
            val = np.sin((np.pi / (2 * up_time)) * (t - (up_time + hold_time)) + np.pi / 2)**power_rampdown
        return val

    return function


def doubleGaussianRamp(I0, hold_time, sigma, cutoff=3, boost=0,):
    I_reduced = I0 - boost
    half_time = hold_time / 2

    print(f"pulse duration: {2 * cutoff * sigma + hold_time:.2f} seconds")

    def gaussian(t):
        return np.exp((- 1 / 2) * (t**2 / sigma**2))

    def curve_vec(t):
        y = t.copy()
        up_filter = t < -(hold_time / 2)
        down_filter = t > (hold_time / 2)
        y[up_filter] = I_reduced * gaussian(t[up_filter] + (hold_time / 2)) + boost
        y[down_filter] = I_reduced * gaussian(t[down_filter] - (hold_time / 2)) + boost
        y[~np.logical_or(up_filter, down_filter)] = I0
        return y

    def curve(t):
        if t < -(hold_time / 2):
            return I_reduced * gaussian(t + half_time) + boost
        elif t > (hold_time / 2):
            return I_reduced * gaussian(t - half_time) + boost
        else:
            return I0

    def function(t):
        return curve(t - hold_time / 2 - cutoff * sigma)

    return function

def double_exp(a, b):
    def function(t):
        cutoff_time = np.log(2) / a
        if t < cutoff_time:
            output = np.exp(a * t) - 1
        else:
            output = np.exp(-b * (t - cutoff_time))
        return output
    return function
        

if __name__ == "__main__":
    j = doubleGaussianRamp(2.5, 4, 2, cutoff=2, boost=0)
    import matplotlib.pyplot as plt
    f = np.linspace(0, 12, 900)
    y = []
    for v in f:
        y.append(j(v))
    plt.plot(f, y)
    plt.xlim(0)
    plt.show()
