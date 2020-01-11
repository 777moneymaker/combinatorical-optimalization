#!/usr/bin/python3
"""
Script for sub-folders creation
"""

import os


def main():
    sizes = {40, 45, 50, 55, 60, 65, 70}
    sub_folders = {'INTENSITY', 'ALPHA', 'BETA', 'COLONY'}
    for size in sizes:
        for sub in sub_folders:
            for i in range(5):
                name = 'V{}'.format(str(size))
                path = os.path.join('Tests', name)
                sub_name = os.path.join(path, sub)
                test = 'TEST_{}'.format(str(i+1))
                os.mkdir(os.path.join(sub_name, test))


if __name__ == '__main__':
    main()
