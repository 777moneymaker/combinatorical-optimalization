#!/usr/bin/python3
"""
This is a simple implementation of Graphs in Python3.

Requirements:
:version = python3.8
:modules = random, math, numpy
"""

__author__ = 'Milosz Chodkowski PUT'
__license__ = 'MIT'
__version__ = '0.5'
__status__ = 'Testing'

import random as rnd
from math import inf

import numpy as np


class _Graph:
    def __init__(self, rank=10):
        self.rank = rank
        self.matrix = np.zeros((rank, rank))
        # Load graph from file.
        self.pheromone_matrix = np.zeros((self.rank, self.rank))
        self._generate_edges()
        self._generate_pheromones()

    def _generate_edges(self):
        # Set random values to all edges.
        for i in range(self.rank):
            for j in range(self.rank):
                self.matrix[i][j] = rnd.randint(1, 100)
        # Make rank / 4 arcs an inf.
        for i in range(self.rank // 4):
            self.matrix[rnd.randint(0, self.rank - 1)][rnd.randint(0, self.rank - 1)] = inf
        # Each arc has to become an edge.
        for i in range(self.rank):
            for j in range(self.rank):
                if self.matrix[i][j] == inf:
                    self.matrix[j][i] = inf
        # Give new edges the same value.
        for i in range(self.rank):
            for j in range(self.rank):
                self.matrix[j][i] = self.matrix[i][j]
        # Set inf's on diagonal.
        np.fill_diagonal(self.matrix, inf)

    def _generate_pheromones(self):
        # Fill every cell with the same value.
        for i in range(self.rank):
            for j in range(self.rank):
                self.pheromone_matrix[i][j] = 1 / (self.rank / 2) ** 2
        # Apply the same pheromone to edge, not only arc.
        for i in range(self.rank):
            for j in range(self.rank):
                self.pheromone_matrix[j][i] = self.pheromone_matrix[i][j]
        # Set diagonal to -inf.
        for i in range(self.rank):
            self.pheromone_matrix[i][i] = -inf

    def generate_to_file(self):
        np.savetxt('demo.txt', self.matrix, fmt='%f')

    def load(self):
        self.matrix = np.loadtxt('demo.txt', dtype=float)
        self.rank = len(self.matrix)

    def show(self):
        for line in self.matrix:
            print(line)
        for line in self.pheromone_matrix:
            print(line)