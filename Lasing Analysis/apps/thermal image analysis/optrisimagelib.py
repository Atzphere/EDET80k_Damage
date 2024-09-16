import numpy as np


def load_optris_imcsv(f):
    '''
    Takes a radiometric data snapshot CSV file saved from Pix Connect
    and converts it into a 2D numpy array for analysis.

    param f: filename of the snapshot.
    '''

    raw_data = np.loadtxt(f, delimiter=",")

    # I guess because Optris is european they use commas as decimals...
    # WHILE also using them to separate their CSV values?!??!?!?!
    # So we have to do some data rearranging...

    left_vals = raw_data[:, ::2]  # digits LEFT of the decimal place
    right_vals = raw_data[:, 1::2]  # the misread decimal places

    return left_vals + right_vals / 100
