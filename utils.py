#!/usr/bin/env python

# Made by Oliver Tautz

# This skript is part of the Bachelor-Thesis of Oliver Tautz,
# created in the summer semester of 2020 for obtaining a bachelor's degree
# in 'Naturwissenschaftliche Informatik' at the Universität Bielefeld.


from functools import reduce

import numpy as np


def get_dir(filepath):
    return filepath.rpartition('/')[0]


def get_filename(filepath):
    return filepath.rpartition('/')[2]


def map_over_dict(f, d):
    for k in d.keys():
        d[k] = f(d[k])


def concat_list_of_lists(lol):
    # print(lol[0])
    return reduce(lambda x, y: x + y, lol)


def get_all_values(dict):
    return concat_list_of_lists(list(dict.values()))


def shuffle_unison(*arrays):
    arr_length = len(arrays[0])
    no_arrays = len(arrays)

    for arr in arrays:
        assert len(arr) == arr_length

    p = np.random.RandomState(seed=42).permutation(arr_length)

    return list(map(lambda x: x[p], arrays))


if __name__ == "__main__":
    arr1 = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    arr2 = np.array([['a', 'b'], ['c', 'd'], ['e', 'f'], ['g', 'h'], ['i', 'j']])
    arr3 = np.array([['eins', 'zwei'], ['drei', 'vier'], ['fünf', 'sechs'], ['sieben', 'acht'], ['neun', 'zehn']])
    arr4 = np.array([1, 0, 1, 1, 1])

    arr1, arr2, arr3, arr4 = shuffle_unison(arr1, arr2, arr3, arr4)

    print(arr1)
    print(arr2)
    print(arr3)
    print(arr4)
