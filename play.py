#!/usr/bin/python3
"""ACO optimization main file.

Requires:
    version: python3.7
    modules: aco.py
"""

__author__ = 'Milosz Chodkowski PUT'
__license__ = 'MIT'
__version__ = '0.9'
__status__ = 'Testing'

from aco import ACO


def main():
    print('Start optimization...')
    aco = ACO(vertex=40, colony_size=20, iterations=100, alpha=0.40, beta=0.55, pq=0.5, pi=30.0)
    best_c, best_s, time = aco.optimize()

    print('time:{}, cost:{}, solution:{}'.format(time, best_c, best_s))


if __name__ == "__main__":
    main()
