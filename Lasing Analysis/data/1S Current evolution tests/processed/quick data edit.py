import pickle
import temperaturemap as tm
from os import listdir
from os.path import isfile, join, splitext
import numpy as np

DPATH = "./"

onlyfiles = [f for f in listdir(DPATH) if isfile(join(DPATH, f)) and splitext(f)[1] == '.pickle']
print(onlyfiles)

for f in onlyfiles:
    with open(f, 'rb') as handle:
        b = pickle.load(handle)
        print(b.keys())
    # ot = tm.maps["Al"].original_temperature(b["temp"])
    # actual_temps, uncertainties = tm.maps["Al"].true_temperature(ot, np.ones(np.shape(ot)) * 2)
    # b.update({"temp_sigma" : uncertainties})
    # with open(f, 'wb') as handle:
    #     pickle.dump(b, handle, protocol=pickle.HIGHEST_PROTOCOL)