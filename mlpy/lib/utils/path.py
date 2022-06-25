import os
import numpy as np
import shutil


def get_dir_list(dir_root):
    """

    :param dir_root:
    :return:
    """
    dir_list = [f for f in os.listdir(dir_root)
                if os.path.isdir(os.path.join(dir_root, f))]
    return np.sort(dir_list)


def makedirs(path, clean=False):
    if clean:
        if os.path.exists(path) is True:
            shutil.rmtree(path)

    if os.path.exists(path) is False:
        os.makedirs(path)

    return path


def makedirs_clean(path):
    if os.path.exists(path) is True:
        shutil.rmtree(path)
    os.makedirs(path)
    return path


def path_split(path):
    folders = []
    while 1:
        path, folder = os.path.split(path)

        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)
            break
    folders.reverse()
    return folders


def tag_path(path, nback=1):
    """
    example:
        tag_path(os.path.abspath(__file__), 1) # return file name
    :param path: 
    :param nback: 
    :return: 
    """
    folders = path_split(path)
    nf = len(folders)

    assert nback >= 1, "nback={} should be larger than 0.".format(nback)
    assert nback <= nf, "nback={} should be less than the number of folder {}!".format(nback, nf)

    # exclude the file suffix, potential problem: the file has no a suffix '.py'
    tag = '.'.join(folders[-1].split('.')[:-1])
    # concatenate parents
    if nback > 0:
        for i in range(2, nback + 1):
            tag = folders[-i] + '_' + tag
    return tag



