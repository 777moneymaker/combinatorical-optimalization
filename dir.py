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
            name = 'V{}'.format(str(size))
            path = os.path.join('Tests', name)
            sub_name = os.path.join(path, sub)
            os.mkdir(sub_name)


if __name__ == '__main__':
    main()
