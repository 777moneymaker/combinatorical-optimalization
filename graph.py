#!/usr/bin/python3
"""Graph module for ACO project in python3.

Module contains methods for creation and handling Graph objects.

Requires:
    version: python3.7
    packages: numpy

TODO's:
    * Simplify code.
"""

__author__ = 'Milosz Chodkowski PUT'
__license__ = 'MIT'
__version__ = '0.9'
__status__ = 'Testing'

import random as rnd
import numpy as np
from math import inf
from typing import List, Dict


class Graph:
    def __init__(self, instance_file: str, rank: int = 10):
        """Constructor for Graph class.
        
        Generates random matrix and pheromone_matrix based on number of vertices.
        
        Args:
            instance_file (str): File to load instance.
            rank (int): Number of vertices.
        """
        self.instance_file = instance_file
        self.rank = rank
        # Init with random values.
        self.matrix = np.random.uniform(1.0, 100.0, (self.rank, self.rank))
        self.load()
        np.fill_diagonal(self.matrix, inf)
        # Set the same value in every cell.
        self.pheromone_matrix = np.full((self.rank, self.rank), 1 / (self.rank / 2) ** 2, dtype='float64')
        np.fill_diagonal(self.pheromone_matrix, -inf)

        self._remove_random_edges()
        self._check_if_connected()

    def _remove_random_edges(self):
        """Sets random correspoding cells to inf.
        
        Sets [i, j] and [j, i] to inf.
        
        Note:
            Diagonal is inf.
        """
        # Make rand edges an inf.
        for i in range(self.rank * 10):
            x, y = rnd.randint(0, self.rank - 1), rnd.randint(0, self.rank - 1)
            self.matrix[x, y] = inf
            self.matrix[y, x] = inf

    def generate_to_file(self):
        """Saves matrix to specific txt file."""
        file = input("Give a file to save instance \"file.txt\"")
        np.savetxt('Instances/' + file, self.matrix, fmt='%f')

    def load(self):
        """Loads matrix from specific txt file."""
        self.matrix = np.loadtxt('Instances/' + self.instance_file, dtype=float)
        self.rank = len(self.matrix)

    def show(self):
        """Prints matrix and pheromone matrix."""
        np.set_printoptions(threshold=np.inf)
        print(self.matrix)
        print(self.pheromone_matrix)
        np.set_printoptions()

    def _check_if_connected(self):
        """Checks if graph is connected Graph.

        Makes a DFS traverse. If len DFS == rank then graph is connected.

        Raises:
                ValueError: If Graph is not connected.
        """
        AdjList = Dict[int, List[int]]
        def matrix_to_list() -> AdjList:
            """Creates a adjacency list from matrix.

            Returns:
                AdjList: List of successors for each vertex.
            """
            graph = dict()
            for i in range(self.rank):
                nodes = list()
                for j in range(self.rank):
                    if self.matrix[i, j] != inf:
                        if j not in nodes:
                            nodes.append(j)
                graph[i] = nodes
            return graph

        def dfs(visited: list, graph: AdjList, node: int):
            """DFS algorithm for traversing."""
            if node not in visited:
                visited.append(node)
                for neighbour in graph[node]:
                    dfs(visited, graph, neighbour)

        adj_list = matrix_to_list()
        visited = list()
        dfs(visited, adj_list, 0)
        if len(visited) != self.rank:
            raise ValueError('Graph not connected!')
