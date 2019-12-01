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
from numpy.random import choice as prob_choice
from numpy import unique
from numpy import isnan
from graph import Graph
from math import inf


class ACO:
    def __init__(self, graph: Graph, colony_size: int, iterations: int, a: float, b: float, pq: float, pi: float):
        self.graph = graph
        self.colony = colony_size
        self.generations = iterations
        self.pheromone_impact = a
        self.distance_impact = b
        self.pheromone_intensity = pi
        self.pheromone_vaporize_coefficient = pq
        self.ants = []

    def _update_pheromones(self):
        for i in range(self.graph.vertex):
            for j in range(self.graph.vertex):
                self.graph.pheromone_matrix[i][j] *= 1 - self.pheromone_vaporize_coefficient

    def solve(self):
        best_solution = []
        best_cost = inf
        for i in range(self.generations):
            self.ants = [Ant(self) for a in range(self.colony)]
            for ant in self.ants:
                while len(unique(ant.visited_vertices)) != self.graph.vertex:
                    ant.travel()
                    #print(ant.visited_vertices)
                if ant.total_cost < best_cost:
                    best_cost = ant.total_cost
                    best_solution = ant.visited_vertices
                ant._apply_pheromone()
            self._update_pheromones()
            #for line in self.graph.pheromone_matrix:
            #    print(line)
            print('generation {}, cost: {}, path: {}'.format(i+1, best_cost, len(best_solution)))
        return best_cost, best_solution


class Ant:
    def __init__(self, a: ACO):
        self.aco = a
        self.previous_vertex = None
        self.current_vertex = random.choice(range(0, self.aco.graph.vertex))
        self.allowed_moves = []
        self.visited_vertices = [self.current_vertex]
        self.total_cost = 0.0
        self._generate_neighborhood()

    def _generate_neighborhood(self):
        graph = self.aco.graph
        allowed_moves = []
        for j in range(graph.vertex):
            if graph.matrix[self.current_vertex][j] != inf:
                allowed_moves.append(j)
        if self.visited_vertices:
            for v in self.visited_vertices:
                if v in allowed_moves:
                    allowed_moves.remove(v)
        if self.previous_vertex is not None and self.previous_vertex in allowed_moves:
            allowed_moves.remove(self.previous_vertex)
        if not allowed_moves:
            p = self.previous_vertex
            self.current_vertex = self.previous_vertex
            for j in range(graph.vertex):
                if graph.matrix[self.current_vertex][j] != inf:
                    allowed_moves.append(j)
            if p in allowed_moves:
                allowed_moves.remove(p)
        self.allowed_moves = allowed_moves

    def _apply_pheromone(self):
        it = iter(self.visited_vertices)
        for i in self.visited_vertices:
            try:
                self.aco.graph.pheromone_matrix[i][next(it)] += self.aco.pheromone_intensity / self.total_cost
            except ValueError:
                pass

    def travel(self):
        graph = self.aco.graph
        self._generate_neighborhood()
        probabilities = [self._get_probability(self.current_vertex, j) for j in self.allowed_moves]
        for i in range(len(probabilities)):
            if isnan(probabilities[i]):
                probabilities[i] = 0
        total = 0
        for i in range(len(probabilities)):
            if probabilities[i] != -inf:
                total += probabilities[i]
        for i in range(len(probabilities)):
            probabilities[i] /= total
        next_vertex = prob_choice(self.allowed_moves, p=probabilities)
        if self.previous_vertex is not None:
            if graph.matrix[self.previous_vertex][self.current_vertex] > \
                    graph.matrix[self.current_vertex][next_vertex]:
                self.total_cost -= graph.matrix[self.previous_vertex][self.current_vertex]
                self.total_cost += graph.matrix[self.previous_vertex][self.current_vertex] * 10
        else:
            pass
        self.allowed_moves.remove(next_vertex)
        self.visited_vertices.append(next_vertex)
        self.total_cost += graph.matrix[self.current_vertex][next_vertex]
        self.previous_vertex = self.current_vertex
        self.current_vertex = next_vertex

    def _get_probability(self, i: int, j: int):
        denominator = 0.0
        numerator = (self.aco.graph.pheromone_matrix[i][j] ** self.aco.pheromone_impact) * \
                    ((1/self.aco.graph.matrix[i][j]) ** self.aco.distance_impact)
        for x in self.allowed_moves:
            denominator += (self.aco.graph.pheromone_matrix[i][j] ** self.aco.pheromone_impact) * \
                            ((1/self.aco.graph.matrix[i][j]) ** self.aco.distance_impact)
        return numerator/denominator
