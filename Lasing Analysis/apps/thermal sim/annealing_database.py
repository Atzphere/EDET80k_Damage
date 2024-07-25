import datetime
import hashlib
import pandas as pd
import blosc
import dill
import matplotlib.pyplot as plt


class ChipRecord(object):
    '''
    A database object to store information on what lasing activities have taken place on a chip.

    The database has two storage methods: a full-depth serialized pickle which stores the original
    LaserSequence objects used to generate cycle code, and a "front-end" csv file which is more human readable,
    and can be annotated. Modifying the csv by deleting entries will reflect on the serialized data, and so
    pulses that were never actually fired can be removed from the record.
    '''

    def __init__(self, name, filepath):
        self.name = name
        self.entry_data = {}
        self.creation_date = datetime.datetime.now()
        self.filepath = filepath
        self.last_file_hash = " "

        cols = ["PulseID", "Date", "Number of Pulses",
                "Max Current", "Pulse Max Duration", "Pulse Total Duration", "Notes"]
        with open(filepath, "w") as f:
            f.write(",".join(cols))

        self.last_hash = self.get_filehash()
        self.save()

    def refresh_data(self):
        '''
        Checks to see if the database's csv file has been modified, rebuilding the
        in-python database if so.
        '''
        if self.file_modified():
            data = pd.read_csv(self.filepath)
            ID_list = data["PulseID"]
            removed_IDs = []
            for ID in self.entry_data.keys():
                if ID not in list(ID_list):
                    print(
                        f"Pulse {ID} no longer found in csv record, removing...")
                    removed_IDs.append(ID)

            for ID in removed_IDs:
                del self.entry_data[ID]
        self.last_hash = self.get_filehash()
        self.save()

    def file_modified(self):
        '''
        Csv file modification check using a hash comparison
        '''
        return self.last_hash == self.get_filehash()

    def commit_data(self, entry):
        '''
        write fired laser sequences to python and csv storage.

        individual locations, max power, duration, notes about modulation

        '''

        self.refresh_data()
        self.entry_data.update({entry.ID: entry})
        self.write_to_csv(entry)
        self.save()

    def write_to_csv(self, entry):
        with open(self.filepath, "a") as f:
            f.write("\n" + ",".join(entry.csv_line()))

    def get_filehash(self):
        with open(self.filepath, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def save(self):
        '''
        Save this object to a dill file.
        '''
        pickled_data = dill.dumps(self)
        fname = f"{self.name}.dill"
        compressed_pickle = blosc.compress(pickled_data)

        with open(fname, "wb") as f:
            f.write(compressed_pickle)


class DataEntry(object):
    def __init__(self, sequence, notes=None):
        self.date = datetime.datetime.now()
        sequence_string = str(sequence) + str(self.date)
        sequence_ID = hashlib.md5(sequence_string.encode()).hexdigest()
        self.ID = sequence_ID
        self.sequence = sequence

        self.num_pulses = len(sequence.pulses)
        self.max_current = max([p.power for p in sequence.pulses])
        self.max_duration = max([p.duration for p in sequence.pulses])
        self.total_duration = sequence.duration

        if notes is not None:
            self.notes = notes
        else:
            self.notes = ""

    def __repr__(self):
        return str(self.sequence)

    def csv_line(self):
        return [str(thing) for thing in [self.ID, self.date, self.num_pulses, self.max_current, self.max_duration, self.total_duration, self.notes]]


class DatabaseWrapper:
    def __init__(self, dbpath):
        self.dbpath = dbpath

    def write_sequence(self, seq):
        database = load_db(self.dbpath)
        database.commit_data(DataEntry(seq))

    def visualize(self):
        database = load_db(self.dbpath)
        fig, ax = plt.subplots()
        ax.set_xlim(-3, 35)
        ax.set_ylim(-3, 35)
        for key in database.entry_data.keys():
            entry = database.entry_data[key]
            ax.plot(entry.sequence.trace_x, entry.sequence.trace_y)
        plt.show()


def load_db(fname):
    with open(fname, 'rb') as f:
        return dill.loads(blosc.decompress(f.read()))
