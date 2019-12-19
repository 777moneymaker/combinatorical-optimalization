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

import sys
import random as rnd
import numpy as np
from math import inf
from typing import List, Tuple, Dict

class Graph:
    def __init__(self, rank: int = 10):
        """Constructor for Graph class.
        
        Generates random matrix and pheromone_matrix based on number of vertices.
        
        Args:
            rank(int): Number of vertices.
        """
        self.rank = rank
        self.matrix = np.zeros((rank, rank))
        self.load()
        self.pheromone_matrix = np.zeros((self.rank, self.rank))
        self._generate_edges()
        self._generate_pheromones()
        self._check_if_connected()

    def _generate_edges(self) -> None:
        """Fills every matrix cell.
        
        Matrix contains only numbers between [1, 100].
        Inf means there is no edge.
        
        Note:
            Diagonal is inf.
        """
        # Set random values to all edges.
        for i in range(self.rank):
            for j in range(self.rank):
                val = rnd.randint(1, 100)
                self.matrix[i, j] = val
                self.matrix[j, i] = val

        # Make rank edges an inf.
        for i in range(self.rank * 10):
            x, y = rnd.randint(0, self.rank - 1), rnd.randint(0, self.rank - 1)
            self.matrix[x, y] = inf
            self.matrix[y, x] = inf
        # Set inf's on diagonal.
        np.fill_diagonal(self.matrix, inf)
        # Print
        # np.set_printoptions(threshold=sys.maxsize)
        # print(self.matrix)

    def _generate_pheromones(self) -> None:
        """Fills pheromone matrix with specific value."""
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

    def generate_to_file(self) -> None:
        """Saves matrix to specific txt file."""
        np.savetxt('TestsData/v40.txt', self.matrix, fmt='%f')

    def load(self) -> None:
        """Loads matrix from specific txt file."""
        self.matrix = np.loadtxt('TestsData/v40.txt', dtype=float)
        self.rank = len(self.matrix)

    def show(self) -> None:
        """Prints matrix and pheromones matrix."""
        for line in self.matrix:
            print(line)
        for line in self.pheromone_matrix:
            print(line)

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
                    if self.matrix[i][j] != inf:
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

