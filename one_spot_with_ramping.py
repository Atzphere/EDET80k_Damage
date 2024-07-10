import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def gaussian_envelope(amplitude, single_duration, num_points):
    t = np.linspace(0, single_duration, num_points)
    mean = single_duration / 2
    std_dev = single_duration / 6  # Controls the spread of the Gaussian envelope
    envelope = amplitude * norm.pdf(t, mean, std_dev)
    envelope = envelope / np.max(envelope) * amplitude  # Normalize and scale to the max amplitude
    return t, envelope

def repeated_gaussian_wave(amplitude, single_duration, num_shots, time_delay, num_points, total_duration = None):
    if num_shots is None and total_duration is None:
        raise ValueError("Either num_shots or total_duration must be specified.")
    
    if num_shots is not None:
        total_duration = num_shots * (single_duration + time_delay) - time_delay
    else:
        num_shots = int(total_duration // (single_duration + time_delay))

    t_single, envelope_single = gaussian_envelope(amplitude, single_duration, num_points)
    
    t_total = np.array([])
    envelope_total = np.array([])
    
    for i in range(num_shots):
        t_total = np.concatenate((t_total, t_single + i * (single_duration + time_delay)))
        envelope_total = np.concatenate((envelope_total, envelope_single))
        if time_delay > 0 and i < num_shots - 1:
            t_total = np.concatenate((t_total, np.linspace(t_total[-1], t_total[-1] + time_delay, int(time_delay * num_points / single_duration))[1:]))
            envelope_total = np.concatenate((envelope_total, np.zeros(int(time_delay * num_points / single_duration) - 1)))
    

    # if the amplitude is betwen 0 and 0.1 exclusive, then we set it to 0
    # this is the issue with labview. We will need to fix this in the future and it is really important.
    # envelope_total = np.where(envelope_total < 0.1, 0, envelope_total)

    return t_total, envelope_total

def plot_gaussian_wave(t, envelope):
    plt.figure(figsize=(10, 6))
    plt.plot(t, envelope)
    plt.xlabel('Time')
    plt.ylabel('Current')
    plt.grid(True)
    plt.show()

def save_wave_data(x, y, unit, envelope, filename, full_power_time, delay):
    envelope = np.round(envelope, 3)


    data = pd.DataFrame({'Voltage X': x, 'Voltage Y': y, 'Time': unit, 'Amplitude': envelope})

    # whenever the amplitude is max, we will set the time column to the full power time
    data.loc[data['Amplitude'] == 2.5, 'Time'] = full_power_time

    # insert a pulse of 90 seconds at the end of the wave
    data.loc[data['Amplitude'] == 0.028, 'Time'] = delay/2

    # delete all rows with amplitude 0
    data = data[data['Amplitude'] != 0.0]

    # 'Amplitude' needs to be CHANGED
    data.loc[data['Amplitude'] == 0.028, 'Amplitude'] = 0.0


    data.to_csv(filename, index=False, sep=str(","), header = False)

    print("Done!")

if __name__ == "__main__":

    FILENAME = "michaeltest1.txt"

    # Voltage X and Y
    X = -0.433
    Y = 3.361

    CURRENT = 2.5 # This is the maximum current in the wave
    SHOT_DURATION = 60 # This is the duration of each shot
    FULL_POWER_TIME = 2 # This is the time for the full power of the wave
    DELAY = 60 # Time delay between shots


    NUM_POINTS = 21 # This is the number of points in the wave per shot (odd number recommended)

    NUM_SHOTS =  60 # total number of shots

    
    # This is the time unit. The length of each instruction in the file and round to 2 decimal places
    unit = SHOT_DURATION/NUM_POINTS 
    unit = round(unit, 2)

    t, envelope = repeated_gaussian_wave(CURRENT, SHOT_DURATION, NUM_SHOTS, DELAY, NUM_POINTS)

    # print(envelope)

    # plot_gaussian_wave(t, envelope)

    save_wave_data(X, Y, unit, envelope, FILENAME, FULL_POWER_TIME, DELAY)
