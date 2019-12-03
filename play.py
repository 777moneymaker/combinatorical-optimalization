#!/usr/bin/python3
"""Simple implementation of Graphs in Python3.

Requirements:
:version = python3
:modules = random, math, numpy
"""

from aco import ACO
from graph import _Graph

if __name__ == "__main__":
	G = _Graph(300)
	G.show()
    # A = ACO(vertex=100, colony_size=6, iterations=25, a=1, b=5, pq=0.4, pi=14.7)
    # best_c, best_s = A.solve()
    # print('cost:{}, solution:{}'.format(best_c, best_s))
