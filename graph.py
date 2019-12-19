#!/usr/bin/python3
"""Graph module for ACO project in python3.

Module contains methods for creation and handling Graph objects.

Requires:
    version: python3.7
    packages: random, math, numpy

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
    def __init__(self, rank: int = 10):
        """Constructor for Graph class.
        
        Generates random matrix and pheromone_matrix based on number of vertices.
        
        Args:
            rank (int): Number of vertices.
        """
        self.rank = rank
        # Init with random values.
        self.matrix = np.random.uniform(1.0, 100.0, (self.rank, self.rank))
        np.fill_diagonal(self.matrix, inf)
        # Set the same value in every cell.
        self.pheromone_matrix = np.full((self.rank, self.rank), 1 / (self.rank / 2) ** 2, dtype='float64')
        np.fill_diagonal(self.pheromone_matrix, -inf)

        self._remove_random_edges()
        self._check_if_connected()

    def _remove_random_edges(self) -> None:
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

    def generate_to_file(self) -> None:
        """Saves matrix to specific txt file."""
        np.savetxt('TestsData/v40.txt', self.matrix, fmt='%f')

    def load(self) -> None:
        """Loads matrix from specific txt file."""
        self.matrix = np.loadtxt('TestsData/v40.txt', dtype=float)
        self.rank = len(self.matrix)

    def show(self) -> None:
        """Prints matrix and pheromone matrix."""
        np.set_printoptions(threshold=np.inf)
        print(self.matrix)
        print(self.pheromone_matrix)
        np.set_printoptions()

    def _check_if_connected(self):
        """Checks if graph is connected Graph.

        Makes a DFS traverse. If len DFS == rank then graph is connected.
        """
        AdjList = Dict[int, List[int]]
        def matrix_to_list() -> AdjList:
            """Creates a adjacency list from matrix.

            Returns:
                AdjList: List of successors for each vertex.
            """
            graph = {}
            for i in range(self.rank):
                nodes = []
                for j in range(self.rank):
                    if self.matrix[i, j] != inf:
                        if j not in nodes:
                            nodes.append(j)
                graph[i] = nodes
            return graph

        def dfs(visited: list, graph: AdjList, node: int) -> None:
            """DFS algorithm for traversing."""
            if node not in visited:
                visited.append(node)
                for neighbour in graph[node]:
                    dfs(visited, graph, neighbour)

        adj_list = matrix_to_list()
        visited = []
        dfs(visited, adj_list, 0)
        if len(visited) != self.rank:
            raise ValueError('Graph not connected!');

