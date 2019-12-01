#!/usr/bin/python3
"""
This is a simple implementation of Graphs in Python3.

Requirements:
:version = python3
:modules = random, math, numpy
"""

from aco import ACO

if __name__ == "__main__":
    A = ACO(vertex=100, colony_size=5, iterations=50, a=4.25, b=1.75, pq=0.35, pi=250.0)
    best_c, best_s = A.solve()
    print('cost:{}, solution:{}'.format(best_c, best_s))
