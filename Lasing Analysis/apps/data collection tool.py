import comport_asyncio as cs
import time
import glob
import os
import optris_csv as ocsv
import numpy as np

os.chdir("C:\\Users\\ssuub\\Desktop\\Damage analysis\\EDET80k_Damage\\Lasing Analysis\\apps")
# gets hotspot location from optris and reports to user. gives instructions to user.
# takes file, unpacks profile, lets user specify time range region of interest and standardized start time (begin lasing event)
# cuts file down and saves dataset as a pickle

done = False
optris_connection = cs.COMInterface("COM4")
DPATH = "../data/3A Time evolution tests/"

class TempProfileDataset:
    '''A 2d dataset containing T vs x vs time values'''
    def __init__(self, dset, key, start, stop, num, precull=None, internal=False):
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

    def __add__(self, val2):
        if np.any(self.time != val2.time):
            raise ValueError("Added arrays must have matching timestamps.")
        new_data = np.concatenate((self.data, val2.data), axis=1)
        new_x = np.concatenate((self.xvals, val2.xvals))
        sorted_indices = np.argsort(new_x)
        new_x = new_x[sorted_indices]
        new_data = new_data[:, sorted_indices]
        to_return = TempProfileDataset(None, None, None, None, None, internal=True)
        to_return.time = time
        to_return.data = new_data
        to_return.xvals = new_x
        return to_return
    
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)



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

while not done:
    d = block_until_ready("Press Return when ready to retrieve hotspot. To skip getting hotspot, reply with 's'.")
    if not d:
        optris_connection.query_port_blocking("!AreaName(2)=Thermal_Corona")
        time.sleep(1)
        optris_connection.query_port_blocking("!ImgTemp")
        print(optris_connection.query_port_blocking("?AreaLoc(0)"))
        print("Input this into the excel sheet for coordinates, then apply those to your temperature profiles.")

    block_until_ready(f"Playback your video before the lasing event to collect data, then save the file to {DPATH}")
    list_of_files = glob.glob(DPATH + '*.dat') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Using latest file: {latest_file}")
    dset = ocsv.OptrisDataset(latest_file)
    data = dset.build_array_data()
    profiles = []
    for profile_key in filter(is_profile, data.keys()):
        params = get_profile_params(profile_key)
        profiles.append(TempProfileDataset(dset, profile_key, **params))
    print(sum(profiles).xvals)
        





