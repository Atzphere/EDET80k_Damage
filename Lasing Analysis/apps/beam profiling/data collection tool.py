'''
Somewhat streamlined script to half-manually collect beam profiles from Optris RAVI recordings.
Set the working path you want to dump information to, run and follow the instructions given.

'''


import comport_asyncio as cs
import time
import glob
import os
import optris_csv as ocsv
import numpy as np
import pickle
import temperaturemap
import matplotlib.pyplot as plt

os.chdir(r"C:\Users\ssuub\Desktop\Damage analysis\EDET80k_Damage\Lasing Analysis\apps")
# gets hotspot location from optris and reports to user. gives instructions to user.
# takes file, unpacks profile, lets user specify time range region of interest and standardized start time (begin lasing event)
# cuts file down and saves dataset as a pickle

done = False
optris_connection = cs.COMInterface("COM4")

DPATH = "../data/2.5A 1S Current modulation/"
OUTPUT_PATH = "processed/"


class TempProfileDataset:
    '''A 2d dataset containing T vs x vs time values'''

    def __init__(self, dset, key, start, stop, num, precull=None, internal=False, metadata={}):
        self.metadata=metadata
        if not internal:
            dset_dict = dset.array_data
            if precull is not None:
                tstart, tstop = precull
            else:
                tstart = 0
                tstop = dset_dict["time"][-1]
            self.time = dset.slice_by_time("time", tstart, tstop)
            self.data = dset.slice_by_time(key, tstart, tstop)
            self.xvals = np.linspace(start, stop, num)
            self.dsigma = 0

    def __add__(self, val2):
        if np.any(self.time != val2.time):
            raise ValueError("Added arrays must have matching timestamps.")
        new_data = np.concatenate((self.data, val2.data), axis=1)
        new_x = np.concatenate((self.xvals, val2.xvals))
        sorted_indices = np.argsort(new_x)
        new_x = new_x[sorted_indices]
        new_data = new_data[:, sorted_indices]
        to_return = TempProfileDataset(
            None, None, None, None, None, internal=True)
        to_return.time = self.time
        to_return.data = new_data
        to_return.xvals = new_x
        return to_return

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def slice_by_time(self, start, stop, preview=True):
        sliced_time = self.time[np.logical_and(start <= self.time, self.time <= stop)]
        sliced_data = self.data[np.logical_and(start <= self.time, self.time <= stop)]
        if preview:
            return sliced_time, sliced_data
        else:
            self.time = sliced_time
            self.data = sliced_data

    def save(self, file):
        '''saves a dictionary of the time, x, temperature dsets as a pickle'''
        data = {'time': self.time, 'x': self.xvals, 'temp': self.data, 'start' : self.start_time, 'temp_sigma': self.dsigma}
        data.update(self.metadata)
        with open(file, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def add_metadata(self, mdata):
        self.metadata.update(mdata)

def block_until_ready(prompt):
    ready = False
    while not ready:
        resp = input(prompt)
        if resp == "":
            ready = True
        if resp == "s":
            return True
    return False


def is_profile(name):
    return "profile" in name


def get_profile_params(name):
    params = {}
    param_strings = name.split('|')[0].split(';')
    for p in param_strings:
        keyvalpair = p.split(':')
        try:
            val = int(keyvalpair[1])
        except ValueError:
            val = float(keyvalpair[1])
        # print(keyvalpair)
        params.update({keyvalpair[0]: val})
    return params


def process_file(f):
    dset = ocsv.OptrisDataset(f)
    data = dset.build_array_data()
    profiles = []
    for profile_key in filter(is_profile, data.keys()):
        print(data[profile_key])
        params = get_profile_params(profile_key)
        profiles.append(TempProfileDataset(dset, profile_key, **params))
    processed = sum(profiles)
    processed.data, processed.dsigma = temperaturemap.maps['Al'].true_temperature(processed.data, auto_uncertainty=True)
    
    return processed

def get_values(prompts):
    responses = []
    for prompt in prompts:
        responses.append(float(input(prompt)))
    return tuple(responses)



while not done:
    d = block_until_ready(
        "Press Return when ready to retrieve hotspot. To skip getting hotspot, reply with 's'.")
    if not d:
        optris_connection.query_port_blocking("!AreaName(2)=Thermal_Corona")
        time.sleep(1)
        optris_connection.query_port_blocking("!ImgTemp")
        print(optris_connection.query_port_blocking("?AreaLoc(0)"))
        print("Input this into the google sheet for coordinates, then apply those to your temperature profiles.")

    block_until_ready(
        f"Playback your video before the lasing event to collect data, then save the file to {DPATH}")
    # * means all if need specific format then *.csv
    list_of_files = glob.glob(DPATH + '*.dat')
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Using latest file: {latest_file}")
    overall_profile = process_file(latest_file)

    satisfied = False
    fig, ax = plt.subplots()
    ax.plot(overall_profile.time, overall_profile.data)
    ax.set_title("Temperature vs time preview")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Converted temperature (C)")
    plt.show(block=False)
    tstart = None
    while not satisfied:
        if input("Would you like to cull this dataset by time? (y/n)") == 'n':
            break
        else:
            tstart, tstop = get_values(["Choose a start time.", "Choose a stop time."])

        fig, ax = plt.subplots()
        ax.plot(*overall_profile.slice_by_time(tstart, tstop))
        ax.set_title("Temperature vs time preview")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Converted temperature (C)")
        plt.show(block=False)
        if input("Would you like to keep this cull? (y/n)") == 'y':
            overall_profile.slice_by_time(tstart, tstop, preview=False)
            satisfied = True
    overall_profile.start_time = float(input("To the best of your ability, specify the start time of the pulse."))
    overall_profile.add_metadata({"current" : float(input("What was the current of the pulse?"))})
    overall_profile.add_metadata({"duration" : float(input("What was the duration of the pulse?"))})
    if not os.path.exists(DPATH + OUTPUT_PATH):
        os.makedirs(DPATH + OUTPUT_PATH)
    overall_profile.save(DPATH + OUTPUT_PATH + f"{overall_profile.metadata['current']}A {overall_profile.metadata['duration']}S.pickle")
    print("saved data.\n")
