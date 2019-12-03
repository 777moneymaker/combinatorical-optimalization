#!/usr/bin/python3
"""Main module of ACO script.

Requirements:
:version = python3
:modules = random, math, numpy
"""

from aco import ACO
from graph import _Graph

if __name__ == "__main__":
    A = ACO(vertex=25, colony_size=6, iterations=25, a=1, b=5, pq=0.4, pi=14.7)
    best_c, best_s = A.solve()
    print('cost:{}, solution:{}'.format(best_c, best_s))
