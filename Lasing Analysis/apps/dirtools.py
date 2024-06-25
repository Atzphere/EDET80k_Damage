'''
Useful directory-manipulation functions used in GaribaldiPPP.

Functions:

    get_parent_dir(directory, depth)
    gets the (depth)th parent directory of (directory) -> os.path

    get_subdirs(directory, fullpath)
    gets a list of child folder names by default in (directory)
    Or their entire paths if (fullpath=True) -> list(str) or list(os.path)

'''
import os


def get_parent_dir(directory, depth=1):
    path = directory
    for i in range(0, depth):
        path = os.path.dirname(path)
    return path


def get_subdirs(directory, fullpath=False):
    '''
    Gets the folder in a folder. Not recursive.
    Only returns folder names by default.

    Parameters
        directory (path str) : The parent folder to get the children of

        fullpath (bool) : Whether or not to return full folder paths.
                          Default value: False
    '''
    if fullpath:
        return [os.path.join(directory, dI).replace("\\", "/")
                for dI in os.listdir(directory)
                if os.path.isdir(os.path.join(directory, dI))]
    else:
        return [dI.replace("\\", "/") for dI in os.listdir(directory)
                if os.path.isdir(os.path.join(directory, dI))]


def get_files(directory, fullpath=False):
    '''
    Gets the folder in a folder. Not recursive.
    Only returns folder names by default.

    Parameters
        directory (path str) : The parent folder to get the files of.

        fullpath (bool) : Whether or not to return full file paths.
                          Default value: False
    '''
    if fullpath:
        return [os.path.join(directory, f) for f in os.listdir(
            directory) if os.path.isfile(os.path.join(directory, f))]
    else:
        return [f for f in os.listdir(
            directory) if os.path.isfile(os.path.join(directory, f))]
