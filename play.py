#!/usr/bin/python3
"""
This is a simple implementation of Graphs in Python3.

Requirements:
:version = python3.8
:modules = random, math, numpy
"""

__author__ = 'Milosz Chodkowski PUT'
__license__ = 'MIT'
__version__ = '0.9'
__status__ = 'Testing'

from aco import ACO


def main():
    print('Start optimization...')
    aco = ACO(vertex=40, colony_size=9, iterations=70, a=2.3, b=1.7, pq=0.45, pi=190.0)
    best_c, best_s, time = aco.optimize()

    print('time:{}, cost:{}, solution:{}'.format(time, best_c, best_s))


if __name__ == "__main__":
    main()
