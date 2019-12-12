#!/usr/bin/python3
"""
This is a simple implementation of Graphs in Python3.

Requirements:
:version = python3.8
:modules = random, math, numpy
"""

__author__ = 'Milosz Chodkowski PUT'
__license__ = 'MIT'
__version__ = '0.9'
__status__ = 'Testing'

import random as rnd
import timeit
from math import inf

from numpy import isnan, zeros, unique
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
        for i in range(self.graph.rank):
            for j in range(self.graph.rank):
                # Pheromones on every edge should evaporate.
                self.graph.pheromone_matrix[i][j] *= (1 - self.pheromone_vaporize_coefficient)
                # Every ant leaves a pheromone on the edges that she visited.
                for ant in ants:
                    self.graph.pheromone_matrix[i][j] += ant.left_pheromones[i][j]

    def optimize(self):
        best_solution, best_cost = list(), inf
        gen_count, was_changed = 0, False
        start = timeit.default_timer()
        elapsed_time = 0
        # Until we do as many generations as we need.
        while gen_count != self.iterations:
            # If we got past 10 minutes.
            if elapsed_time > 600:
                print('Time is over!')
                return best_cost, best_solution
            # Make a new list of ants.
            ants = [Ant(self) for a in range(self.colony)]
            # Make a new empty list for ants which found solution.
            best_ants = list()

            for ant in ants:
                # Every ant travels |V| - 1. First vertex is given at the start.
                for i in range(self.graph.rank - 1):
                    ant.travel()
                # If any of the ants found best solution, then set it.
                if ant.total_cost < best_cost and len(unique(ant.visited_vertices)) >= self.graph.rank:
                    best_cost, best_solution = ant.total_cost, ant.visited_vertices
                    was_changed = True
                    # Add ant which found a solution to list of best_ants.
                    best_ants.append(ant)

            gen_count += 1
            stop = timeit.default_timer()
            elapsed_time = stop - start
            print('ended gen no {}, curr elapsed time: {}'.format(gen_count, elapsed_time))

            # If any ant got solution, then update all pheromones and each best ant should leave pheromones.
            if was_changed:
                for b in best_ants:
                    b._leave_pheromones()
                self._update_pheromones(best_ants)
                print('Found Solution:\n cost: {}, path: {}\n'.format(best_cost, len(best_solution)))
                o_file = open('TestsData/data4.txt', 'a')
                o_file.write('generation: ' + str(gen_count) + ' cost: ' + str(best_cost) + ' solution ' + ' '.join(
                    str(v) for v in best_solution) + '\n')
                o_file.close()
                was_changed = False
        return best_cost, best_solution, elapsed_time


class Ant:
    def __init__(self, a: ACO):
        self.aco = a
        self.total_cost = 0.0

        self.start = rnd.randint(0, self.aco.graph.rank - 1)
        self.previous_vertex = None
        self.current_vertex = self.start

        self.allowed_moves = list()
        self.tabu_moves = list()

        self.left_pheromones = [[]]
        self.visited_vertices = [self.start]

    def travel(self):
        # Generate vertices which are valid to move to.
        self._generate_allowed_moves()
        probabilities = list(map(self._get_probability, self.allowed_moves))
        self._validate_probabilities(probabilities)
        # If it's not first iteration, then choice is based on different probabilities.
        if not self.aco.first_iter:
            next_vertex = np_choice(self.allowed_moves, p=probabilities)
        else:
            next_vertex = rnd.choice(self.allowed_moves)
            self.aco.first_iter = False

        # Add next edge value to total cost. If previous value was bigger,
        # then subtract the previous value, and add it but 10 times bigger.
        if self.previous_vertex is not None:
            if self.aco.graph.matrix[self.previous_vertex][self.current_vertex] > \
                    self.aco.graph.matrix[self.current_vertex][next_vertex]:
                self.total_cost -= self.aco.graph.matrix[self.previous_vertex][self.current_vertex]
                self.total_cost += self.aco.graph.matrix[self.previous_vertex][self.current_vertex] * 10

        self.total_cost += self.aco.graph.matrix[self.current_vertex][next_vertex]
        # On next move ant can't go to previous and current vertex.
        if self.previous_vertex is not None:
            self.tabu_moves.append(self.previous_vertex)
        self.tabu_moves.append(self.current_vertex)
        # Set a new current vertex, and update previous.
        self.visited_vertices.append(next_vertex)
        self.previous_vertex = self.current_vertex
        self.current_vertex = next_vertex

    def _validate_probabilities(self, probabilities: list):
        # lowest value that python3 can handle
        lowest = 2.2250738585072014e-308
        for i in range(len(probabilities)):
            # If value is == to NaN, then make it the lowest value, that python can handle.
            if isnan(probabilities[i]):
                probabilities[i] = lowest
            # Make every probability proportionally bigger.
            probabilities[i] *= 1.1 ** self.aco.graph.rank
        total = sum(probabilities)
        # To make probabilities sum to 1, divide every probability by their sum.
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
        def generate(ant: Ant):
            matrix, rank = ant.aco.graph.matrix, ant.aco.graph.rank
            curr_v, prev_v = ant.current_vertex, ant.previous_vertex

            al_moves, tabu_moves = [], ant.tabu_moves
            for j in range(rank):
                if matrix[curr_v][j] != inf and j not in tabu_moves:
                    al_moves.append(j)
            # Previous vertex can't be allowed to move.
            if prev_v in al_moves and len(al_moves) > 1:
                al_moves.remove(prev_v)
            return al_moves

        allowed = generate(self)
        # If not any legal move, then pop half of tabu moves and generate again.
        while not allowed:
            for x in range(len(self.tabu_moves) // 2):
                if len(self.tabu_moves) < 2:
                    break
                self.tabu_moves.pop()
            allowed = generate(self)

        self.allowed_moves = allowed

    def _leave_pheromones(self):
        v = self.aco.graph.rank
        # Make left pheromones on all edges equal to zero.
        left_pheromones = zeros((v, v))
        if len(self.visited_vertices) > 1:
            for x in range(1, len(self.visited_vertices)):
                i, j = self.visited_vertices[x - 1], self.visited_vertices[x]
                # Leave pheromones on edge i, j and j, i.
                left_pheromones[i][j] = self.aco.pheromone_intensity / self.total_cost ** 2
                left_pheromones[j][i] = self.aco.pheromone_intensity / self.total_cost ** 2

        self.left_pheromones = left_pheromones
