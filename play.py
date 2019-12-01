#!/usr/bin/python3

__author__ = "Milosz Chodkowski PUT"
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Milosz Chodkowski"
__email__ = "milosz.chodkowski@student.put.poznan.pl"
__status__ = "Production"

from aco import ACO
from graph import Graph
from numpy import unique


if __name__ == "__main__":
    G = Graph(vertex=100)
    G.show()
    A = ACO(graph=G, colony_size=15, a=1.0, b=10.0, iterations=100, pq=0.5, pi=10)
    best_cost, best_solution = A.solve()
    print(best_cost, best_solution)
