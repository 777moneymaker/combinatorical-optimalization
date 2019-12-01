#!/usr/bin/python3
"""
This is a simple implementation of Graphs in Python3.

Requirements:
:version = python3
:modules = random, math, numpy
"""

import random as rnd
from math import inf

from numpy import isnan
from numpy.random import choice as np_choice

from graph import _Graph


class ACO:
    def __init__(self, vertex: int, colony_size: int, iterations: int, a: float, b: float, pq: float, pi: float):
        self.graph = _Graph(vertex)
        self.colony = colony_size
        self.iterations = iterations
        self.pheromone_impact = a
        self.distance_impact = b
        self.pheromone_vaporize_coefficient = pq
        self.pheromone_intensity = pi
        self.first_iter = True

    def _update_pheromones(self, ants: list):
        for i in range(len(self.graph.pheromone_matrix)):
            for j in range(len(self.graph.pheromone_matrix)):
                self.graph.pheromone_matrix[i][j] *= (1 - self.pheromone_vaporize_coefficient)
                for ant in ants:
                    self.graph.pheromone_matrix[i][j] += ant.left_pheromones[i][j]

    def solve(self):
        best_solution = []
        best_cost = inf

        for g in range(self.iterations):
            ants = [Ant(self) for a in range(self.colony)]
            for ant in ants:
                for i in range(self.graph.rank - 1):
                    ant.travel()
                if ant.total_cost < best_cost:
                    best_cost = ant.total_cost
                    best_solution = ant.visited_vertices
                ant._leave_pheromones()
            self._update_pheromones(ants)
            print('generation {}, cost: {}, path: {}'.format(g + 1, best_cost, len(best_solution)))
        return best_cost, best_solution


class Ant:
    def __init__(self, a: ACO):
        self.aco = a
        self.total_cost = 0.0
        self.start = rnd.randint(0, self.aco.graph.rank - 1)
        self.previous_vertex = None
        self.current_vertex = self.start
        self.allowed_moves = []
        self.left_pheromones = [[]]
        self.visited_vertices = [self.start]
        self.tabu_moves = dict()

    def travel(self):
        self._generate_allowed_moves()
        probabilities = list(map(self._get_probability, self.allowed_moves))
        self._validate_probabilities(probabilities)
        if not self.aco.first_iter:
            next_vertex = np_choice(self.allowed_moves, p=probabilities)
        else:
            next_vertex = rnd.choice(self.allowed_moves)
            self.aco.first_iter = False
        if self.previous_vertex is not None:
            if self.aco.graph.matrix[self.previous_vertex][self.current_vertex] > \
                    self.aco.graph.matrix[self.current_vertex][next_vertex]:
                self.total_cost -= self.aco.graph.matrix[self.previous_vertex][self.current_vertex]
                self.total_cost += self.aco.graph.matrix[self.previous_vertex][self.current_vertex] * 10

        self.visited_vertices.append(next_vertex)
        self.total_cost += self.aco.graph.matrix[self.current_vertex][next_vertex]
        self.tabu_moves[next_vertex] = self.current_vertex
        self.previous_vertex = self.current_vertex
        self.current_vertex = next_vertex

    def _validate_probabilities(self, probabilities: list):
        # lowest value that python3 can handle
        lowest = 2.2250738585072014e-308
        total = 0
        for i in range(len(probabilities)):
            if isnan(probabilities[i]):
                probabilities[i] = lowest
            probabilities[i] *= 1.1 ** self.aco.graph.rank
        total = sum(probabilities)
        for i in range(len(probabilities)):
            probabilities[i] /= total

    def _get_probability(self, j: int):
        current, denominator = self.current_vertex, 0.0
        numerator = (self.aco.graph.pheromone_matrix[current][j] ** self.aco.pheromone_impact) * \
                    ((1 / self.aco.graph.matrix[current][j]) ** self.aco.distance_impact)
        for x in self.allowed_moves:
            denominator += (self.aco.graph.pheromone_matrix[current][x] ** self.aco.pheromone_impact) * \
                           ((1 / self.aco.graph.matrix[current][x]) ** self.aco.distance_impact)
        return numerator / denominator

    def _generate_allowed_moves(self):
        current = self.current_vertex
        if len(self.visited_vertices) > (self.aco.graph.rank // 10) and \
                len(self.tabu_moves) > len(self.visited_vertices) // 2:
            for i in range(len(self.tabu_moves) // 2):
                self.tabu_moves.popitem()
        allowed = []
        for j in range(self.aco.graph.rank):
            if self.aco.graph.matrix[current][j] != inf and j not in self.tabu_moves.keys():
                allowed.append(j)
        self.allowed_moves = allowed

    def _leave_pheromones(self):
        left_pheromones = [[0] * self.aco.graph.rank] * self.aco.graph.rank
        for x in range(len(self.visited_vertices)):
            i, j = self.visited_vertices[x - 1], self.visited_vertices[x]
            left_pheromones[i][j] = self.aco.pheromone_intensity / self.total_cost ** 2
        self.left_pheromones = left_pheromones
