#!/usr/bin/python3

__author__ = "Milosz Chodkowski PUT"
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Milosz Chodkowski"
__email__ = "milosz.chodkowski@student.put.poznan.pl"
__status__ = "Production"


from graph import _Graph
from numpy import unique
from aco import ACO


if __name__ == "__main__":
    sols = []
    mins = []
    i = 0
    A = ACO(vertex=100, colony_size=7, iterations=100, a=2.1, b=1.1, pq=0.35, pi=25)
    best_c, best_s = A.solve()
    print('cost:{}, solution:{}'.format(best_c, best_s))
