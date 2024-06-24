import pickle
from os import listdir
from os.path import isfile, join, splitext

DPATH = "./"

onlyfiles = [f for f in listdir(DPATH) if isfile(join(DPATH, f)) and splitext(f)[1] == '.pickle']
del onlyfiles[-1]

for f in onlyfiles:
    with open(f, 'rb') as handle:
        b = pickle.load(handle)
        print(b['current'])
        b['current'] += -0.02
        print(f)
        print(b['current'])
    with open(f, 'wb') as handle:
        pickle.dump(b, handle, protocol=pickle.HIGHEST_PROTOCOL)