#!/usr/bin/python3
"""
This is a simple implementation of Graphs in Python3.

Requirements:
:version = python3
:modules = random, math
"""
__author__ = "Milosz Chodkowski PUT"
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Milosz Chodkowski"
__email__ = "milosz.chodkowski@student.put.poznan.pl"
__status__ = "Production"

import random
from math import inf


class Graph:
    def __init__(self, vertex=10):
        self.vertex = vertex
        # if cell is equal to zero, that means the vertices i, j are not connected
        self.matrix = [[inf for i in range(vertex)] for j in range(vertex)]
        self.pheromone_matrix = [[1/vertex**2 for i in range(vertex)] for j in range(vertex)]
        self._reduce_edges()

    def _reduce_edges(self):
        for i in range(self.vertex):
            vertices = [random.randint(0, self.vertex - 1) for i in range(6)]
            for j in range(6):
                c, val = random.choice(vertices), random.randint(1, 100)
                self.matrix[i][c] = val
                self.matrix[c][i] = val
                vertices.remove(c)
        for i in range(self.vertex):
            for j in range(self.vertex):
                if self.matrix[i][j] == inf:
                    self.pheromone_matrix[i][j] = -inf

    def show(self):
        for line in self.matrix:
            print(line)
        for line in self.pheromone_matrix:
            print(line)