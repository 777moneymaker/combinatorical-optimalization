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
    test_file = input("Give a filename for tests \"file.txt\"")
    instance_file = input("Give a file to load \"file.txt\"")
    # For loop just for tests
    for i in range(5):
        aco = ACO(
            test_file=test_file,
            instance_file=instance_file,
            vertex=40, colony_size=10,
            iterations=50,
            alpha=0.45,
            beta=0.55,
            pq=0.5,
            pi=1.0)
        best_c, best_s, time = aco.optimize()
        print('time:{}, cost:{}, solution:{}'.format(time, best_c, best_s))


if __name__ == "__main__":
    main()
