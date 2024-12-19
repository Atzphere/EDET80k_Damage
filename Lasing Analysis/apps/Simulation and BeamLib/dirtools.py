'''
Useful directory-access functions

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


def get_file_extension(path):
    filename, ext = os.path.splitext(path)
    return ext

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


def get_files(directory, fullpath=False, extensions=None):
    '''
    Gets the folder in a folder. Not recursive.
    Only returns folder names by default.

    Parameters
        directory (path str) : The parent folder to get the files of.

        fullpath (bool) : Whether or not to return full file paths.
                          Default value: False

        extensions List, default None: Whitelist of file extensions to return
    '''
    if fullpath:
        results = [os.path.join(directory, f) for f in os.listdir(
            directory) if os.path.isfile(os.path.join(directory, f))]
    else:
        results = [f for f in os.listdir(
            directory) if os.path.isfile(os.path.join(directory, f))]
    if extensions is not None:
        try:
            return filter(lambda file: get_file_extension(file) in extensions, results)
        except ValueError:
            raise ValueError("extensions must be an iterable of strings")
    else:
        return results
