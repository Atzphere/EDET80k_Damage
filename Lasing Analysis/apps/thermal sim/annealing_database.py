import datetime
import hashlib
import pandas as pd
import blosc
import dill
import matplotlib.pyplot as plt
import copy


class ChipRecord(object):
    '''
    A database object to store information on what lasing activities have taken place on a chip.

    The database has two storage methods: a full-depth serialized pickle which stores the original
    LaserSequence objects used to generate cycle code, and a "front-end" csv file which is more human readable,
    and can be annotated. Modifying the csv by deleting entries will reflect on the serialized data, and so
    pulses that were never actually fired can be removed from the record.
    '''

    def __init__(self, name, dpath, csvpath):
        self.name = name
        self.entry_data = {}
        self.creation_date = datetime.datetime.now()
        self.csvpath = csvpath
        self.dpath = dpath
        self.backups = {}

        cols = ["PulseID", "Date", "Number of Pulses",
                "Max Current", "Pulse Max Duration", "Pulse Total Duration", "Notes"]
        with open(csvpath, "w") as f:
            f.write(",".join(cols))

        # use this to detect file updates to the csv file
        self.last_hash = "NONE"
        self.save()

    def save(self):
        '''
        Save this object to a dill file. Should only be called once on initalization.
        '''
        pickled_data = dill.dumps(self)
        fname = self.dpath
        compressed_pickle = blosc.compress(pickled_data)

        with open(fname, "wb") as f:
            f.write(compressed_pickle)


class DataEntry(object):
    '''
    A class representing an laser sequence which has been deployed as an annealing cycle.
    DataEntry are stored in a ChipRecord.
    Stores the serialized laser sequence as well as some human-useful information
    (date of creation, number of pulses, max current etc.) bound to a unique identifying ID hash.

    Attributes:

    date datetime.datetime: the date the DataEntry was created on

    ID str: the unique indentifying hash of a DataEntry instance.
            Created on initalization by hashing the string representation of a lasersequence salted with the current time and date.

    sequence LaserSequence: the LaserSequence object to be recorded.

    num_pulses int: auto-assigned human-readabe reprsenting the total number of individual subpulses
    containe in the sequence

    max_current, max_duration float: the same as num_pulses, but for the maximum current and duration of
    any pulse in the sequence. These two are more useful for identifying anealing patterns which consist of
    moving the same base pulse around.

    notes str: human-provided notes to attach to the entry. Although the option is presented here, this is better done
    in the csv file.
    '''
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
        '''
        Custom string alignment for easy indentification via print statements
        '''
        datestr = self.date.strftime("%m/%d/%Y, %H:%M:%S")
        return f"{datestr}\nAnnealing cycle: {str(self.sequence)}"

    def csv_line(self):
        '''
        Returns a csv-line representation of the stored lasersequence.
        '''
        return [str(thing) for thing in [self.ID, self.date, self.num_pulses, self.max_current, self.max_duration, self.total_duration, self.notes]]


    class DatabaseWrapper:
        '''
        Wrapper class to modularize interactions with a ChipRecord objects/files.
        '''

        def __init__(self, dbpath):
        self.dbpath = dbpath

    def write_sequence(self, seq):
        '''
        Writes a new row of data to the ChipRecord
        '''
        database = self.load_data()
        refresh_data(database)
        commit_data(database, DataEntry(seq))

    def load_data(self):
        '''
        Exposes the ChipRecord's data for read/write.
        '''
        record = load_db(self.dbpath)
        return refresh_data(record)

    def visualize(self):
        '''
        Plot all prior recorded annealing sequences
        '''
        database = self.load_data()
        refresh_data(database)
        fig , ax = plt.subplots()
        ax.set_aspect("equal")
        ax.set_xlim(-3, 35)
        ax.set_ylim(-3, 35)
        ax.axvline(0, 0, 32, c="gray")
        ax.axvline(32, 0, 32, c="gray")
        ax.axhline(0, 0, 32, c="gray")
        ax.axhline(32, 0, 32, c="gray")
        for key in database.entry_data.keys():
            entry = database.entry_data[key]
            label = entry.date.strftime("%m/%d/%Y, %H:%M:%S")
            ax.plot(entry.sequence.trace_x, entry.sequence.trace_y,
                    marker="o", alpha=0.2, lw=1, label=label)
        ax.legend()
        plt.show()

    def backup(self):
        '''
        Back up the current annealing history. When called, this method
        will return the name of the backup created in the event it needs to be
        modified.
        '''
        date = datetime.datetime.now()
        datestr = date.strftime("%m/%d/%Y, %H:%M:%S")
        backup_str = f"Backup: {datestr}"
        database = self.load_data()
        database.backups.update({backup_str: copy.deepcopy(database.entry_data)})
        save(database)
        return backup_str

    def load_backup(self, key):
        '''
        Attempts to load a backup of the annealing history by name.
        This does not save the current annealing data, so be sure to backup() that
        first if you would like to not lose it.
        '''
        database = self.load_data()
        try:
            loaded_backup = database.backups[key]
        except KeyError:
            print(f"Backup not found. Existing backups:\n{database.backups.keys()}")
            raise
        database.entry_data = copy.deepcopy(loaded_backup)
        refresh_csv(database)

    def delete_backup(self, key):
        database = self.load_data()
        try:
            del database.backups[key]
        except KeyError:
            print(f"Backup not found. Existing backups:\n{database.backups.keys()}")
            raise

        save(database)


def refresh_csv(db):
    '''
    Rebuilds a database's csv file, preserving notes and culling excess pulses not found in the ChipRecord.
    This should only be run in the case of restoring backups.
    '''
    data = pd.read_csv(db.csvpath)
    ID_list = data["PulseID"]
    unknown_IDs = []
    for ID in ID_list:
        if ID not in db.entry_data.keys():
            unknown_IDs.append(ID)

    data.set_index("PulseID", inplace=True, drop=False)

    for ID in unknown_IDs:
        print(f"Dropping Pulse {ID} from csv database (not in previous backup)")
        data.drop(index=ID, inplace=True)

    data.to_csv(db.csvpath, index=False)
    db.last_hash = get_csvhash(db)
    save(db)


def refresh_data(db):
    '''
    Checks to see if a database's csv file has been modified, rebuilding the
    object if so.
    '''

    if csv_modified(db):
        data = pd.read_csv(db.csvpath)
        ID_list = data["PulseID"]
        removed_IDs = []
        for ID in db.entry_data.keys():
            if ID not in list(ID_list):
                print(
                    f"Pulse {ID} no longer found in csv record, removing...")
                removed_IDs.append(ID)

        for ID in removed_IDs:
            del db.entry_data[ID]
    db.last_hash = get_csvhash(db)

    save(db)
    return db


def csv_modified(db):
    '''
    Csv file modification check using a hash comparison
    '''
    return db.last_hash == get_csvhash(db)


def commit_data(db, entry):
    '''
    write fired laser sequences to python and csv storage.

    individual locations, max power, duration, notes about modulation

    '''

    db.entry_data.update({entry.ID: entry})
    write_to_csv(db, entry)
    save(db)


def write_to_csv(db, entry):
    with open(db.csvpath, "a") as f:
        f.write("\n" + ",".join(entry.csv_line()))


def get_csvhash(db):
    with open(db.csvpath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def save(db):
    '''
    Overwrite an old database with an updated version
    '''
    pickled_data = dill.dumps(db)
    fname = db.dpath
    compressed_pickle = blosc.compress(pickled_data)

    with open(fname, "wb") as f:
        f.write(compressed_pickle)


def load_db(fname):
    with open(fname, 'rb') as f:
        return dill.loads(blosc.decompress(f.read()))


def record_exists(dpath):
    try:
        with open(fname, 'rb') as f:
            loaded = dill.loads(blosc.decompress(f.read()))
            return isinstance(loaded, ChipRecord)
    except FileNotFoundError:
        return False


def new_chiprecord(name, dpath, csvpath, overwrite=False):
    if overwrite or not record_exists(dpath):
        record = ChipRecord(name, dpath, csvpath)

