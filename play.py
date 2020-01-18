#!/usr/bin/python3
"""ACO optimization main file.

Requires:
    version: python3.7
    modules: aco
"""

__author__ = 'Milosz Chodkowski PUT'
__license__ = 'MIT'
__version__ = '1.0'
__status__ = 'Working'

from aco import ACO


def main():
    file = input("Give a file to load \"/../../file.txt\"")
    print('Start optimization...')
    aco = ACO(
        instance_file=file,
        vertex=40,
        colony_size=10,
        iterations=50,
        alpha=0.45,
        beta=0.55,
        pq=0.5,
        pi=1.0,
        break_count=3,
        change_count=6,
    )
    best_c, best_s, time = aco.optimize()
    print('time: {:.2f}, cost: {:.2f}, solution (len): {}'.format(time, best_c, len(best_s)))


if __name__ == "__main__":
    main()
