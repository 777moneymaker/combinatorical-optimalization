#!/usr/bin/python3
"""
This is a simple implementation of Graphs in Python3.

Requirements:
:version = python3
:modules = random, math, numpy
"""

import random as rnd
from math import inf

import numpy as np


class _Graph:
    def __init__(self, rank=10):
        self.rank = rank
        self.matrix = np.zeros((rank, rank))
        self.pheromone_matrix = np.zeros((self.rank, self.rank))
        self._generate_edges()
        self._generate_pheromones()

    def _generate_edges(self):
        for i in range(self.rank):
            for j in range(self.rank):
                self.matrix[i][j] = rnd.randint(1, 100)
        for i in range(self.rank // 4):
            self.matrix[i][rnd.randint(1, self.rank - 1)] = inf
        for i in range(self.rank):
            for j in range(self.rank):
                if self.matrix[i][j] == inf:
                    self.matrix[j][i] = inf
        for i in range(self.rank):
            self.matrix[i][i] = inf

    def _generate_pheromones(self):
        for i in range(self.rank):
            for j in range(self.rank):
                self.pheromone_matrix[i][j] = 1 / (self.rank / 2) ** 2
        for i in range(self.rank):
            for j in range(self.rank):
                self.pheromone_matrix[j][i] = self.pheromone_matrix[i][j]
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
