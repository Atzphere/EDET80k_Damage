# Optris Temperature vs. time reader for arbitrary number of ROIs. Useful for testing.
# Use OptrisDataset(filepath).

import csv
import numpy as np
import pandas as pd


def timestr_to_seconds(time_str) -> float:
    # HHours:MMinutes:SSeconds.Miliseconds
    units = [3600, 60, 1, 1E-3]
    working_string = time_str
    components = working_string.split(":")
    components += components[2].split(".")
    del components[2]
    time_seconds = sum(
        [unit * int(measure) for unit, measure in zip(units, components)])
    return time_seconds


class DictDataset:
    def __init__(self):
        self.writeabledata = {}

    def organize(self, name, value):
        if name in self.writeabledata.keys():
            self.writeabledata[name].append(value)
        else:
            self.writeabledata.update({name: [value]})

    def build(self):
        self.array_data = {}
        for key in self.writeabledata.keys():
            self.array_data.update({key: np.array(self.writeabledata[key])})
        return self.array_data

    def build_array_data(self):
        self.dataframe = pd.DataFrame.from_dict(self.build())
        return self.dataframe


class OptrisDataset(DictDataset):
    def __init__(self, filepath):  # auto builds dataaset
        super().__init__()
        self.filepath = filepath
        self.times = []
        with open(filepath) as f:
            reader = csv.reader(f)

            for i in range(6):  # blow past useless header data
                next(reader)
            series_headers = next(reader)[1:-1]  # extract ROI names

            for num, row in enumerate(reader):
                try:  # extract values for time + each ROI
                    time = timestr_to_seconds('.'.join(row[:2]))
                    self.organize("time", time)
                    for num, series in enumerate(series_headers):
                        working_column = 2 * num + 2
                        value_components = [
                            int(i) for i in row[working_column:working_column + 2]]
                        value = value_components[0] + value_components[1] / 100

                        self.organize(series, value)
                except IndexError:
                    pass  # horrible but deals with end of file strings


if __name__ == "__main__":
    fpath = "OvenCoolingExperiment_TIW_01_rough.dat"
    test = OptrisDataset(fpath)
    print(test.writeabledata)
