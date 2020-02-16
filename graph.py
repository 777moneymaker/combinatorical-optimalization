#!/usr/bin/python3
"""Graph module for ACO project in python3.

Module contains methods for creation and handling Graph objects.

Requires:
    version: python3.7
    packages: numpy, more_itertools
"""

__author__ = 'Milosz Chodkowski PUT'
__license__ = 'MIT'
__version__ = '1.0'
__status__ = 'Working'

import os
import random as rnd
import numpy as np

from more_itertools import pairwise
from math import inf


class Graph:
    def __init__(self, instance_file: str = None, rank: int = 10):
        """Constructor for Graph class.
        
        Generates random matrix and pheromone_matrix based on number of vertices.
        
        Args:
            rank (int): Number of vertices.
        """
        self.rank = rank
        self.matrix = np.random.uniform(1.0, 100.0, (self.rank, self.rank))   # Init with random values.
        self.load(instance_file)
        # Set the same value in every cell.
        self.pheromone_matrix = np.full((self.rank, self.rank), 1 / (self.rank / 2) ** 2, dtype='float64')
        self.check_if_connected()

    def reset_pheromone(self):
        self.pheromone_matrix = np.full((self.rank, self.rank), 1 / (self.rank / 2) ** 2, dtype='float64')
        np.fill_diagonal(self.pheromone_matrix, -inf)

    def smooth(self):
        """Smooths the pheromone matrix."""
        minimal = np.median(self.pheromone_matrix[self.pheromone_matrix > -inf])
        maxim = np.max(self.pheromone_matrix)
        self.pheromone_matrix[self.pheromone_matrix == maxim] = minimal * np.log(maxim/minimal)

    def remove_random_edges(self):
        """Sets random corresponding cells to inf.
        
        Sets [i, j] and [j, i] to inf.
        
        Note:
            Diagonal is inf.
        """
        # Make rand edges an inf.
        for x in range(self.rank * 10):
            i, j = rnd.randint(0, self.rank - 1), rnd.randint(0, self.rank - 1)
            self.matrix[(i, j), (j, i)] = inf

    def save(self, file, name):
        """Saves matrix to specific txt file."""
        np.savetxt(os.path.join(file, name), self.matrix, fmt='%f')

    def load(self, path):
        """Loads matrix from specific txt file."""
        self.matrix = np.loadtxt(path, dtype=float)
        self.rank = len(self.matrix)

    def show_distances(self):
        """Prints distance matrix."""
        np.set_printoptions(threshold=np.inf)
        print(self.matrix)
        np.set_printoptions()

    def show_pheromones(self):
        """Prints pheromone matrix"""
        np.set_printoptions(threshold=np.inf)
        print(self.pheromone_matrix)
        np.set_printoptions()

    def check_if_connected(self):
        """Checks if graph is connected Graph.

        Makes a DFS traverse. If len DFS == rank then graph is connected.

        Raises:
                ValueError: If Graph is not connected.
        """
        def matrix_to_list():
            """Creates a adjacency list from matrix.

            Returns:
                AdjList: List of successors for each vertex.
            """
            graph = {
                node: [
                    neighbour for neighbour in range(self.rank) if self.matrix[node, neighbour] != inf
                ] for node in range(self.rank)
            }
            return graph

        def dfs(visited: list, graph: AdjList, node: int):
            """DFS algorithm for traversing."""
            if node not in visited:
                visited.append(node)
                for neighbour in graph[node]:
                    dfs(visited, graph, neighbour)

        # check_if_connected body.
        adj_list = matrix_to_list()
        visited = list()
        dfs(visited, adj_list, 0)
        if len(visited) != self.rank:
            raise ValueError('Graph not connected!')
