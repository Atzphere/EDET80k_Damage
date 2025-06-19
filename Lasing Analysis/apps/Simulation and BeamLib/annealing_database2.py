import datetime
import hashlib
import pandas as pd
import blosc
import dill
import matplotlib.pyplot as plt
import copy
import os
from pathlib import Path
import zipfile
import dirtools as dt
import csv

from typing import Generator, Iterator, Dict

import numpy as np

from matplotlib import cm


class ChipRecord(object):
    '''
    A database object to store information on what lasing activities have taken place on a chip.

    The database has two storage methods: a full-depth serialized pickle which stores the original
    LaserSequence objects used to generate cycle code, and a "front-end" csv file which is more human readable,
    and can be annotated. Modifying the csv by deleting entries will reflect on the serialized data, and so
    pulses that were never actually fired can be removed from the record.
    '''

    def __init__(self, name, path, dname, csvname, backup_frequency=3):
        '''
        Entry data points to paths to serialized Entry objects

        '''
        self.name = name
        self.entry_data = {}
        self.creation_date = datetime.datetime.now()
        self.path = path
        self.csvpath = f"{path}/{csvname}"
        self.dpath = f"{path}/{dname}"
        self.entry_folder = path + f"/{self.name} entries"
        self.backup_folder = path + f"/{self.name} backups"
        self.write_counter = 0

        self.backup_frequency = backup_frequency

        Path(self.entry_folder).mkdir(parents=True, exist_ok=True)
        Path(self.backup_folder).mkdir(parents=True, exist_ok=True)

        cols = ["PulseID", "Date", "Number of Pulses",
                "Max Current", "Pulse Max Duration", "Pulse Total Duration", "Notes"]
        with open(self.csvpath, "w") as f:
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
        self.version = "ADBV2"
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
        if self.notes != "":
            return f"{datestr}\nAnnealing cycle: {self.notes}"
        else:
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

    def write_sequence(self, seq, notes=None):
        '''
        Writes a new row of data to the ChipRecord
        '''
        database = self.load_data()

        refresh_data(database)
        commit_data(database, DataEntry(seq, notes=notes))
        if (database.write_counter + 1) % database.backup_frequency == 0:
            self.backup(note="automated backup")

    def get_entry_files(self, fullpath=True) -> Iterator[str]:
        '''
        Returns a list of entry filepaths
        '''
        record = refresh_data(load_db(self.dbpath))
        files = dt.get_files(record.entry_folder, fullpath=fullpath, extensions=(".dill",))
        files_f = filter(lambda name: not "gitignore" in name, files)
        return files_f

    def load_data(self):
        '''
        Exposes the ChipRecord for read/write.
        '''
        record = refresh_data(load_db(self.dbpath))
        return record

    def sorted_ids(self):
        '''
        Returns a list of datewise-sorted entry IDs.
        '''
        record = refresh_data(load_db(self.dbpath))
        csv_data = read_csv(record)
        files = self.get_entry_files()

        ids = csv_data["PulseID"]
        dates = csv_data["Date"]

        assert len(list(files)) == len(ids)

        ids_sorted = [id for _, id in sorted(zip(dates, ids))]

        return ids_sorted

    def get_diff(self):
        '''
        Used for debugging purposes: returns any entries in the csv file that aren't in the entry database.
        '''
        record = refresh_data(load_db(self.dbpath))
        csv_data = read_csv(record)
        csv_ids = csv_data["PulseID"]

        print("CSV IDs:")
        print(csv_ids)
        
        entry_ids = list([t.split(".")[0] for t in self.get_entry_files(fullpath=False)])

        print("Entry IDs:")
        print(entry_ids)
        
        diff = [item for item in csv_ids if item not in entry_ids]
        return list(diff)
        

    def get_file(self, ID) -> str:
        '''
        Gets an entry's file given its ID.

        Returns a filepath string.
        '''
        record = refresh_data(load_db(self.dbpath))
        return f"{record.entry_folder}/{ID}.dill"

    def get_entries(self) -> Generator[DataEntry, None, None]:
        '''
        Generator that returns a timewise (oldest-first) sequence of lasing entries.

        Try to keep the generator structure whenever possible instead of pulling the entire database
        otherwise, you will quickly encounter memory issues with large records.

        Returns a generator object of (ID, DataEntry).
        '''
        for ID in self.sorted_ids():
            path = self.get_file(ID)
            yield (ID, read_entry_file(path))

    def load_entries(self) -> Dict:
        '''
        Loads a dictionary of entries and their pulse IDs
        '''
        record = refresh_data(load_db(self.dbpath))

        files = self.get_entry_files()
        entrydata = {}

        for f in files:
            entry = read_entry_file(f)
            entrydata.update({entry.ID: entry})

        return entrydata

    def load_entry(self, ID):
        '''
        Returns a loaded DataEntry corresponding to a provided id.
        Use is not recommeded as it will easily eat up your memory with large databases.

        Use get_entries or other generator-based methods instead.
        '''

        return read_entry_file(self.get_file(ID))


    def visualize(self, alpha=0.2, fpath='annealing record visualization.png'):
        '''
        Plot all prior recorded annealing sequences.
        Also capable of exporting figures to a file.

        params:

        alpha float, default 0.2: transparency of annealing spot markers.
                        Helpful to visualize the intensity of successive lasing on the same location.

        fpath str, default 'annealing record visualization.png':
                                   Filepath to export the finished visualization under.
        '''
        record = self.get_entries()
        gen_length = sum(1 for _ in record)
        record = self.get_entries()
        color = iter(cm.rainbow(np.linspace(0, 1, len(list(self.get_entry_files())))))
        
        database = self.load_data()
        refresh_data(database)
        fig, ax = plt.subplots(tight_layout=True)
        ax.set_aspect("equal")
        ax.set_xlim(-3, 35)
        ax.set_ylim(-3, 35)
        ax.axvline(0, 0, 32, c="gray")
        ax.axvline(32, 0, 32, c="gray")
        ax.axhline(0, 0, 32, c="gray")
        ax.axhline(32, 0, 32, c="gray")

        for (ID, entry), c in zip(record, color):
            datestr = entry.date.strftime("%m/%d/%Y, %H:%M")
            label = f"{datestr}: {entry.notes}"
            ax.plot(entry.sequence.trace_x, entry.sequence.trace_y,
                    marker="o", alpha=alpha, lw=1, label=label, c=c)
        lgnd = ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        fig.savefig(fpath, bbox_extra_artists=(lgnd,), bbox_inches="tight")
        plt.show()

    def backup(self, note=""):
        '''
        Back up the current annealing history (entry pickles) to a zip file. When called, this method
        will return the name of the backup created in the event it needs to be
        modified.
        '''
        date = datetime.datetime.now()
        datestr = date.strftime("%m-%d-%Y, %H;%M")
        backup_str = f"Backup {datestr}; {note}"
        database = self.load_data()

        save_folder(database.entry_folder, f"{database.backup_folder}/{backup_str}.zip")
        return backup_str

    def load_backup(self, name):
        '''
        Attempts to load a backup of the annealing history by filename.
        This does not save the current annealing data, so be sure to backup() that
        first if you would like to not lose it.
        '''
        database = self.load_data()

        # get and remove current records
        current_files = dt.get_files(database.entry_folder, fullpath=True, extensions=(".dill",))
        try:
            # ignore git-related files
            for f in filter(lambda name: not "gitignore" in name, current_files):
                os.remove(f)
            load_zip(f"{database.backup_folder}/{name}", dump=database.path)
        except FileNotFoundError:
            print(
                f"Backup not found. Existing backups:\n{self.get_backups()}")
            raise
        refresh_csv(database)

    def delete_backup(self, name):
        '''
        Deletes a backup file by its local filename. You can also just manually delete them.=,
        but this is provided for convenience and some documentation examples.
        '''
        database = self.load_data()
        try:
            os.remove(f"{database.backup_folder}/{name}.zip")
        except FileNotFoundError:
            print(
                f"Backup not found. Existing backups:\n{self.get_backups()}")
            raise

        save(database)

    def get_backups(self):
        '''
        Returns a list of backups
        '''
        database = self.load_data()
        return dt.get_files(database.backup_folder, fullpath=False, extensions=(".zip",))


def new_chiprecord(name, path, dname, csvname, overwrite=False, **kwargs):
    dpath = f"{path}/{dname}"
    if overwrite or not record_exists(dpath):
        record = ChipRecord(name, path, dname=dname, csvname=csvname, **kwargs)

# The following functions are purely helper functions and are not meant to be used directly by the end user.

def read_csv(db):
    '''
    Parses the csv portion of an annealing record.
    ''' 
    return pd.read_csv(db.csvpath, sep=',', parse_dates=["Date"], dayfirst=False)    

def refresh_csv(db):
    '''
    Rebuilds a database's csv file, preserving notes and culling excess pulses not found as .dill files in the ChipRecord.
    This should only be run in the case of restoring backups.
    '''

    data = read_csv(db)
    ID_list = list(data["PulseID"])
    unknown_IDs = []
    entries_to_add = []

    new_lines = []

    entry_files = dt.get_files(db.entry_folder, extensions=(".dill",))
    entry_IDs = [name.split(".")[0] for name in entry_files]

    for ID in ID_list:
        if ID not in entry_IDs:
            unknown_IDs.append(ID)

    for ID in entry_IDs:
        if ID not in ID_list:
            entries_to_add.append(ID)

    data.set_index("PulseID", inplace=True, drop=False)

    for ID in unknown_IDs:
        print(f"Dropping Sequence {ID} from csv database (not in backup)")
        data.drop(index=ID, inplace=True)

    for ID in entries_to_add:
        print(f"Adding Sequence {ID} to csv database (recovered by backup)")
        entry_filename = f"{db.entry_folder}/{ID}.dill"
        new_lines.append(read_entry_file(entry_filename).csv_line())


    new_data = data.values.tolist() + new_lines

    new_data = pd.DataFrame(new_data, columns=list(data.columns.values))
    new_data.to_csv(db.csvpath, index=False, sep=",")
    db.last_hash = get_csvhash(db)
    save(db)


def remove_entry(db, ID):
    '''
    Deletes a DataEntry from a ChipRecord. Only for use when restoring backups; do not use this directly.
    '''
    try:
        os.remove(f"{db.entry_folder}/{ID}.dill")
    except FileNotFoundError as e:
        print(e)
        print(f"Sequence {ID} already deleted.")
        raise


def refresh_data(db):
    '''
    Checks to see if a database's csv file has been modified, rebuilding the
    object if so.
    '''
    if csv_modified(db):
        print("CSV modified")
        data = pd.read_csv(db.csvpath, sep=',')
        ID_list = data["PulseID"]
        removed_IDs = []

        entry_files = dt.get_files(db.entry_folder, extensions=(".dill",))
        entry_IDs = [name.split(".")[0] for name in entry_files]

        for ID in entry_IDs:
            if ID not in list(ID_list):
                print(f"Pulse {ID} no longer found in csv record, removing...")
                removed_IDs.append(ID)

        for ID in removed_IDs:
            remove_entry(db, ID)
    db.last_hash = get_csvhash(db)

    save(db)
    return db


def csv_modified(db):
    '''
    csv file modification check using a hash comparison
    '''
    return not (db.last_hash == get_csvhash(db))


def commit_data(db, entry):
    '''
    write fired laser sequences to python and csv storage.

    individual locations, max power, duration, notes about modulation

    '''

    write_entry(entry, db.entry_folder)
    db.write_counter += 1
    write_to_csv(db, entry)
    save(db)


def write_entry(entry, path):
    pickled_data = dill.dumps(entry)
    compressed_pickle = blosc.compress(pickled_data)

    fpath = path + f"/{entry.ID}.dill"

    with open(fpath, "wb") as f:
        f.write(compressed_pickle)


def read_entry_file(fname):
    with open(fname, 'rb') as f:
        return dill.loads(blosc.decompress(f.read()))


def write_to_csv(db, entry):
    '''
    writes an entry to the csv file
    '''

    with open(db.csvpath, 'a', newline='') as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(list(entry.csv_line()))
        file.flush()


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


def record_exists(fname):
    '''
    dpath str: the path of the dill file containing the ChipRecord
    '''
    try:
        with open(fname, 'rb') as f:
            loaded = dill.loads(blosc.decompress(f.read()))
            return isinstance(loaded, ChipRecord)
    except FileNotFoundError:
        return False



def zip_file(target, fname):
    with zipfile.ZipFile(target, "w") as zf:
        zf.write(target, compress_type=zipfile.ZIP_DEFLATED)


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), 
                       os.path.relpath(os.path.join(root, file), 
                                       os.path.join(path, '..')))
# https://stackoverflow.com/questions/1855095/how-to-create-a-zip-archive-of-a-directory


def save_folder(path, fname):
    with zipfile.ZipFile(fname, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipdir(path, zipf)


def load_zip(fname, dump):
    with zipfile.ZipFile(fname, "r") as zf:
        zf.extractall(dump)
