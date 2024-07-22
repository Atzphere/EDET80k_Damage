# Optris Temperature vs. time reader for arbitrary number of ROIs. Useful for testing.
# Use OptrisDataset(filepath).

import csv
import numpy as np
import re
import time


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


def get_profile_and_index(name):
    separated = re.split(r'\s+(?=\d)|(?<=\d)\s+', name)
    profile_name = separated[0] + " " + separated[1][0]
    index = int(separated[1][1:])
    return profile_name, index


def is_profile(name):
    return "profile" in name


class DictDataset:
    def __init__(self):
        self.writeabledata = {}

    def organize(self, name, value, rownum):
        if is_profile(name):  # handles 1D temperature data
            # get relative position of the profile point and the profile's name
            profile_name, index = get_profile_and_index(name)

            if profile_name in self.writeabledata.keys():
                # add a new empty row to array if needed
                if rownum + 1 > len(self.writeabledata[profile_name]):
                    self.writeabledata[profile_name].append([0])
                data_row = self.writeabledata[profile_name][rownum]
                length_diff = (index + 1) - len(data_row)
                if length_diff > 0:  # expand length of row appropriately
                    data_row += [0] * length_diff
                data_row[index] = value  # update value in row
                # update data with new row
                self.writeabledata[profile_name][rownum] = data_row
            else:
                data_row = [0] * (index + 1)
                data_row[index] = value
                self.writeabledata.update({profile_name: [data_row]})
        else:
            if name in self.writeabledata.keys():
                self.writeabledata[name].append(value)
            else:
                self.writeabledata.update({name: [value]})

    def build_array_data(self):
        self.array_data = {}
        for key in self.writeabledata.keys():
            self.array_data.update({key: np.array(self.writeabledata[key])})
        return self.array_data


class TemperatureProfile:
    # A
    def __init__(self):
        pass


class OptrisDataset(DictDataset):
    def __init__(self, filepath):  # auto builds dataaset
        super().__init__()
        self.filepath = filepath
        self.times = []
        self.profiles = []
        with open(filepath, encoding="ISO-8859-1") as f:
            reader = csv.reader(f)

            for i in range(6):  # blow past useless header data
                next(reader)
            series_headers = next(reader)[1:-1]  # extract ROI names

            for row_num, row in enumerate(reader):  # per row
                try:  # extract values for time + each ROI
                    time = timestr_to_seconds('.'.join(row[:2]))
                    self.organize("time", time, row_num)
                    for num, series in enumerate(series_headers):
                        working_column = 2 * num + 2
                        value_components = [
                            int(i) for i in row[working_column:working_column + 2]]
                        value = value_components[0] + value_components[1] / 10

                        self.organize(series, value, row_num)
                except IndexError:
                    pass  # horrible but deals with end of file strings

    def slice_by_time(self, label, start, stop, closed=True):
        '''
        Returns a time slice of a specified subdataset
        '''
        if self.array_data is None:
            self.build_array_data()

        if closed:
            return self.array_data[label][np.logical_and(start <= self.array_data["time"], self.array_data["time"] <= stop)]
        else:
            return self.array_data[label][np.logical_and(start < self.array_data["time"], self.array_data["time"] < stop)]


if __name__ == "__main__":
    pass
