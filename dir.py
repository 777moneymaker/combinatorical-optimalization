#!/usr/bin/python3
"""
Script for sub-folders creation
"""

import os

from itertools import product


def main():
    sizes = {40, 45, 50, 55, 60, 65, 70}
    sub_folders = {'INTENSITY', 'ALPHA', 'BETA', 'COLONY'}
    for folder in product(sizes, sub_folders):
        for i in range(5):
            name = 'V{}'.format(str(folder[0]))
            path = os.path.join('Tests', name)
            sub_name = os.path.join(path, folder[1])
            test = 'TEST_{}'.format(str(i+1))
            os.mkdir(os.path.join(sub_name, test))


if __name__ == '__main__':
    main()
