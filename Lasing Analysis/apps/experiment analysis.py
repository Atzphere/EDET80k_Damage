import pickle
import matplotlib.pyplot as plt
import logging
import beam_profiler as bp
import numpy as np
from scipy.signal import savgol_filter
import dirtools as dt
from matplotlib.pyplot import cm

logging.basicConfig()
logging.getLogger().setLevel(logging.WARN)


# files = ["3.0A 0.1S.pickle", "3.0A 0.3S.pickle", "3.0A 0.6S.pickle", "3.0A 1.0S.pickle",
#          "3.0A 1.1S.pickle", "3.0A 1.2S.pickle", ]
# DPATH = "../data/3A Time evolution tests/processed/"

DPATH = "../data/1s Current evolution tests/processed/"
files = sorted(list(filter(lambda name: "pickle" in name, dt.get_files(DPATH))))
files.insert(0, files[-1])
del files[-1]
print(files)

# just something that remotely resembles the beam profile
BASE_GUESS = (-0.07, 0.05, 50, 40)
NONGAUSSIAN_TOLERANCE = (100, 100, 100, 100)


def process_file(f):
    '''
    Loads a beam profile pickle file, centers it on its start time, and fits parameters.
    '''
    with open(f, 'rb') as handle:
        b = pickle.load(handle)
        params, var, valid = bp.profile_beam(
            b["time"], b["x"], b["temp"], p0=BASE_GUESS, error_tolerance=NONGAUSSIAN_TOLERANCE, sigma=b["temp_sigma"], absolute_sigma=True)
        b["time"] = b["time"] - b["start"]
        return b, params, var, valid


def find_deviation_points(data, threshold=0.10, window_length=4, polyorder=3):
    '''
    Attempts to locate points at which a dataset deviates from being linear.

    params

    data: 1D array to be analyzed, spacings between data points is assumed to be constant

    threshold: float: the difference threshold between subsequent slopes
                      over which the algorithm will detect deviations.

    '''
    smoothed = savgol_filter(data, window_length, polyorder)
    points = []
    for num, point in enumerate(smoothed):
        if num > 0 and num < len(smoothed) - 2:
            slope_diff = smoothed[num] / smoothed[num - 1] - smoothed[num + 1] / smoothed[num]
            if np.abs(slope_diff) > threshold:
                points.append(True)
            else:
                points.append(False)
        else:
            points.append(False)
    return points

def find_pulse_end(time, data, pulse_length, ignore_cutoff=0.5, **kwargs):
    '''
    Attempts to find the point at which the laser pulse ends via a 
    the temperature profile becoming non-gaussian. Does this by analyzing
    Gaussian beam width parameter fit uncertainty over time. Used to clean
    temperature profile vs time datasets to include only the lasing.

    params:

    time: 1D array corresponding to time. Zero should correspond to estimated
          beam start.

    data: 1D array, variance of fit for sigma over time.

    pulse_length: the expected length of the pulse.

    ignore_cutoff: the period of time to skip forwards by as a fraction of
                   pulse_length. Used to skip over initial noise which may
                   confuse the algorithm.

    returns: index of the point when the temperature profile is sufficiently
             non-gaussian.

    '''

    time_from_zero = time
    cutoff_time = ignore_cutoff * pulse_length
    cull_time_mask = (time_from_zero) > cutoff_time
    cutoff_index = np.where(time_from_zero > cutoff_time)[0][0]

    # flattens everything before the ignore window to avoid false positives
    data[~cull_time_mask] = data[cull_time_mask][0]
    nonlinearities = np.where(find_deviation_points(data))[0]
    consecutive_nonlinearities = nonlinearities[np.diff(nonlinearities, prepend=nonlinearities[0]) == 1]

    if len(consecutive_nonlinearities) != 0:
        laser_cutoff = consecutive_nonlinearities[0]
        if laser_cutoff > cutoff_index + 100:
            logging.warn("Laser cutoff significantly beyond expected point. Detection may have failed.")
        return laser_cutoff
    else:
        logging.error("No pulse end detected.")
        return None


def get_stable_interval(time, data, pulse_length, start_cutoff=0.3, **kwargs):
    '''
    Returns the indicies bordering the closed interval in which there is a 
    stable gaussian temperature profile on the chip. This can then be used to
    calculate properties of the beam. Ideally, this should find the interval
    in which the gaussian temperature profile has a constant variance.

    start_cutoff: float: initial fraction of the data from t=0 to ignore.
                         allows conditions to stabilize. default 0.3

    '''
    # ignore_cutoff=0.8, threshold=0.05
    beam_off = find_pulse_end(time, svar, pulse_length, **kwargs)
    cutoff_time = start_cutoff * pulse_length
    start_index = np.where(time >= cutoff_time)[0][0]

    return np.array([start_index, beam_off])

def get_sigma(time, sigma_arr, svar, pulse_length, **kwargs):
    start, stop = get_stable_interval(time, svar, pulse_length, **kwargs)
    clean_sigma = sigma_arr[start: stop]
    return clean_sigma.mean(), clean_sigma.std()

fig, ax = plt.subplots(len(files), 2)

stable_beam_indices = []

for num, f in enumerate(files):
    b, params, var, valid = process_file(DPATH + f)
    time = b["time"]
    pulse_length = b["duration"]
    sigma_array = params[:, 1]
    svar = var[:, 1]
    temp = b["temp"]
    # ax[num, 0].set_ylim(top=7)
    # ax[num, 0].set_xlim(0)
    # plt.show(block=False)
    ax[num, 1].semilogy(time, svar, label="Sigma Variance " + f)
    # ax[num, 1].set_xlim(0)
    beam_off = find_pulse_end(time, svar, pulse_length, ignore_cutoff=0.8, threshold=0.05)
    ax[num, 1].scatter(time[beam_off], 5, color="red", label="Detected beam end")
    ax[num, 1].scatter(pulse_length, 5, color="orange", label="Set beam end", alpha=0.5)
    start, stop = get_stable_interval(time, svar, pulse_length, ignore_cutoff=0.8, threshold=0.05)
    stable_beam_indices.append((start, stop))
    ax[num, 1].axvline(x=time[start], c="magenta")
    ax[num, 1].axvline(x=time[stop], c="magenta")

    ax[num, 0].plot(time[start:stop], sigma_array[start:stop], label="Sigma " + f)
    ax[num, 0].plot(time[start:stop], np.log(temp[:, 19])[start:stop], label="center temperature " + f)

    ax[num, 0].grid()
    ax[num, 1].grid()
    ax[num, 1].legend(loc="upper right")
    ax[num, 0].legend(loc="upper right")
# plt.show()

fig, ax = plt.subplots(len(files))
fig.tight_layout()
for num, f in enumerate(files):
    b, params, var, valid = process_file(DPATH + f)
    time = b["time"]
    pulse_length = b["duration"]
    sigma_array = params[:, 1]
    svar = var[:, 1]
    temp = b["temp"]
    x = b["x"]
    current = b["current"]
    temp_sigma = b["temp_sigma"]
    start, stop = stable_beam_indices[num]
    xstart, xstop = x[0], x[-1]
    smooth_x = np.linspace(xstart, xstop, 500)
    color = iter(cm.rainbow(np.linspace(0, 1, len(time))))
    ax[num].set_title(f"{current} A {pulse_length} S")
    for ind, time in enumerate(time):
        c=next(color)
        ax[num].errorbar(x, temp[ind], yerr=temp_sigma[ind], linestyle="None", ecolor=c)
        ax[num].scatter(x, temp[ind], c=c, s=4, alpha=0.5)
        ax[num].plot(smooth_x, bp.gaussian_fitfun(smooth_x, *params[ind]), c=c, label=f"{time:.3f}")
        ax[num].set_ylim(bottom=-1, top=500)
        ax[num].axhline(300, xstart, xstop, c="red", lw=0.5)

ax[-1].legend(loc="lower right")
plt.show(block=False)
color = iter(cm.rainbow(np.linspace(0, 1, len(files))))
fig, ax = plt.subplots()
for num, f in enumerate(files):
    c = next(color)
    b, params, var, valid = process_file(DPATH + f)
    time = b["time"]
    pulse_length = b["duration"]
    sigma_array = params[:, 1]
    svar = var[:, 1]
    temp = b["temp"]
    current = b["current"]
    temp_sigma = b["temp_sigma"] # standard deviation

    start, stop = get_stable_interval(time, svar, pulse_length, ignore_cutoff=0.8, threshold=0.05, c=c)
    y, dy = get_sigma(time, sigma_array, svar, pulse_length, ignore_cutoff=0.8, threshold=0.05)
    ax.errorbar(pulse_length, y, yerr=dy, linestyle="None", marker="o", c=c)
    ax.scatter(time, sigma_array, label=f"current {current} pl {pulse_length:.2f}", c=c, s=4)
    ax.set_ylim(0, 10)
    ax.legend()

plt.show()
# with open(f, 'rb') as handle:
#     b = pickle.load(handle)

# with open(f, 'wb') as handle:
#     pickle.dump(b, handle, protocol=pickle.HIGHEST_PROTOCOL)
