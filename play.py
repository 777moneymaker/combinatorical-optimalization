#!/usr/bin/python3

__author__ = "Milosz Chodkowski PUT"
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Milosz Chodkowski"
__email__ = "milosz.chodkowski@student.put.poznan.pl"
__status__ = "Production"

from aco import ACO

if __name__ == "__main__":
    A = ACO(vertex=100, colony_size=5, iterations=50, a=4.25, b=1.75, pq=0.35, pi=250.0)
    best_c, best_s = A.solve()
    print('cost:{}, solution:{}'.format(best_c, best_s))
