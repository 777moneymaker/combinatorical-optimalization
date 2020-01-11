#!/usr/bin/python3
"""ACO optimization main file.

Requires:
    packages: tkinter
    version: python3.7
    modules: aco.py
"""

__author__ = 'Milosz Chodkowski PUT'
__license__ = 'MIT'
__version__ = '1.0'
__status__ = 'Working'

import os

from datetime import datetime
from tkinter import messagebox
from aco import ACO

COLONY_SIZES = ['5', '10', '15', '25', '35']
PARAMETERS = {
    'ALPHA': ['0.35', '0.4', '0.55', '0.6', '0.7'],
    'BETA': ['0.35', '0.4', '0.45', '0.55', '0.6'],
    'INTENSITY': ['1.0', '2.0', '4.0', '10.0', '20.0']
}


def time_is():
    return datetime.now().time()


def main():
    print('Start optimization...')
    # test_file = input("Give a filename for tests \"file.txt\"")
    # instance_file = input("Give a file to load \"file.txt\"")

    for fh in ['v50.txt']:
        for parameter in PARAMETERS:
            size = fh.split('.')[0].split('v')[1]
            times, costs = list(), list()
            values = PARAMETERS[parameter]

            for val in values:
                full_name = 'test_{}_{}_{}.txt'.format(size, parameter, val)
                for i in range(5):  # For loop for tests.
                    print('Current file: {}, Current size: {}'.format(full_name, size))
                    print("Test no: {}".format(i+1))
                    print(time_is())
                    aco = ACO(
                        test_file=full_name,
                        instance_file=fh,
                        vertex=40,
                        colony_size=10,
                        iterations=50,
                        alpha=float(val) if parameter == 'ALPHA' else 0.45,
                        beta=float(val) if parameter == 'BETA' else 0.55,
                        pq=0.5,
                        pi=float(val) if parameter == 'INTENSITY' else 1.0)
                    best_c, best_s, time = aco.optimize()
                    times.append(time)
                    costs.append(best_c)
                    print('time: {:.2f}, cost: {:.2f}, solution (len): {}'.format(time, best_c, len(best_s)))

    messagebox.showinfo(
        'Data',
        'Avg Time {:.2f}\nAvg Cost {:.2f}\nBest cost {}'.format(
            sum(times) / len(times), sum(costs) / len(costs), min(costs)
        )
    )


if __name__ == "__main__":
    main()
