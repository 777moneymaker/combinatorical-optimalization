#!/usr/bin/python3
"""ACO and Ant Class used for optimization.

Contains methods for handling object of type(Graph) and type(Ant).

Requires:
    version: python3.8
    packages: random, math, numpy, typing, timeit

TODO's
    * Test if while loop will work better than for loop (in optimize method).
"""

__author__ = 'Milosz Chodkowski PUT'
__license__ = 'MIT'
__version__ = '0.9'
__status__ = 'Testing'

import timeit
import random as rnd
from math import inf
from numpy import isnan, zeros, unique
from numpy.random import choice as np_choice
from typing import Tuple, List

from graph import Graph


class ACO:
    def __init__(self, vertex: int, colony_size: int, iterations: int, alpha: float, beta: float, pq: float, pi: float):
        """Constructor for ACO class.

        Creates ACO object containing Graph, Ants and methods handling optimization process.

        Args:
            param1 (int): Number of vertices.
            param2 (int): Size of colony / number of ants.
            param3 (int): Number of iterations.
            param4 (float): Pheromone impact.
            param5 (float): Distance impact.
            param6 (float): Pheromone vaporize coefficient.
            param7 (float): Pheromone intensity.
        """
        self.graph = Graph(vertex)
        self.colony = colony_size
        self.iterations = iterations
        self.pheromone_impact = alpha
        self.distance_impact = beta
        self.pheromone_vaporize_coefficient = pq
        self.pheromone_intensity = pi
        self.first_iter = True

    def _update_pheromones(self, ants: list) -> None:
        """Method updating pheromones on visited edges.
        
        Travers through every visited edge and applies pheromone on it.

        Args:
            param (list): List of ants which found valid solution.

        Returns:
            None.
        """
        for i in range(self.graph.rank):
            for j in range(self.graph.rank):
                # Evaporate on every edge.
                self.graph.pheromone_matrix[i][j] *= (1 - self.pheromone_vaporize_coefficient)
                # Ant apply pheromones.
                for ant in ants:
                   self.graph.pheromone_matrix[i][j] += ant.left_pheromones[i][j]

    def optimize(self) -> Tuple[float, list, float]:
        """Main method optimizing solution.
        
        Every ant travel through graph and compute solutin.
        Ants which found solution are added to list of best ants.

        Returns:
            Tuple(float, list, float): best cost, best solution, elapsed time in specific generation.
        """
        best_solution, best_cost = list(), inf
        gen_count, was_changed = 0, False
        elapsed_time, start = 0, timeit.default_timer()

        # Until as many generations as need.
        while gen_count != self.iterations:
            # If past 30 minutes.
            if elapsed_time > 60 * 30 :
                print('Time is over!')
                return best_cost, best_solution

            # Make a new list of ants and best_ants which found solution.
            ants, best_ants = [Ant(self) for a in range(self.colony)], list()
            for ant in ants:
                # Every ant travels |V| - 1. First vertex is given at the start.
                for i in range(self.graph.rank - 1):
                    ant.travel()
                # If any of the ants found best solution - set it.
                if ant.total_cost < best_cost and len(unique(ant.visited_vertices)) >= self.graph.rank:
                    best_cost, best_solution = ant.total_cost, ant.visited_vertices
                    was_changed = True
                    # Add ant which found a solution to list of best_ants.
                    best_ants.append(ant)
            gen_count += 1
            stop = timeit.default_timer()
            elapsed_time = stop - start
            print('ended gen no {}, curr elapsed time: {}'.format(gen_count, elapsed_time))

            # If any ant got solution, then update all pheromones and each best applies pheromone.
            if was_changed:
                for b in best_ants:
                    b._leave_pheromones()
                self._update_pheromones(best_ants)
                print('Found Solution:\n cost: {}, path: {}\n'.format(best_cost, len(best_solution)))
                o_file = open('TestsData/data5.txt', 'a')
                o_file.write('generation: ' + str(gen_count) + ' cost: ' + str(best_cost) + ' solution ' + ' '.join(
                    str(v) for v in best_solution) + '\n')
                o_file.close()
                was_changed = False

        return best_cost, best_solution, elapsed_time


class Ant:
    def __init__(self, a: ACO):
        """Constructor for Ant class.

        Creates Ant object containing reference to ACO class and methods handling ant choice process.
        
        Args:
            param (ACO): Reference to ACO class in which Ants will be created.
        """
        self.aco = a
        self.total_cost = 0.0

        self.start = rnd.randint(0, self.aco.graph.rank - 1)
        self.previous_vertex = None
        self.current_vertex = self.start

        self.allowed_moves = list()
        self.tabu_moves = list()

        self.left_pheromones = [[]]
        self.visited_vertices = [self.start]

    def travel(self) -> None:
        """Makes ant travel to next vertex.

        Generates allowed moves and their probabilities. Makes choice.
        """
        # Generate valid vertices.
        self._generate_allowed_moves()
        probabilities = list(map(self._get_probability, self.allowed_moves))
        self._validate_probabilities(probabilities)

        # If not first iteration, then choice is based on different probabilities.
        if not self.aco.first_iter:
            next_vertex = np_choice(self.allowed_moves, p=probabilities)
        else:
            next_vertex = rnd.choice(self.allowed_moves)
            self.aco.first_iter = False

        '''Add next edge value to total cost. If previous value was bigger,
           then subtract the previous value, and add it 10 times bigger.'''
        if self.previous_vertex is not None:
            if self.aco.graph.matrix[self.previous_vertex][self.current_vertex] > \
                    self.aco.graph.matrix[self.current_vertex][next_vertex]:
                self.total_cost -= self.aco.graph.matrix[self.previous_vertex][self.current_vertex]
                self.total_cost += self.aco.graph.matrix[self.previous_vertex][self.current_vertex] * 10

        self.total_cost += self.aco.graph.matrix[self.current_vertex][next_vertex]
        
        # On next move ant can't go to previous and current.
        if self.previous_vertex is not None:
            self.tabu_moves.append(self.previous_vertex)
        self.tabu_moves.append(self.current_vertex)
        
        # Set a new current vertex, update previous.
        self.visited_vertices.append(next_vertex)
        self.previous_vertex = self.current_vertex
        self.current_vertex = next_vertex

    def _validate_probabilities(self, probabilities: list) -> None:
        """Checks if probabilities sum to one.

        If not, then make it sum to one.

        Args:
            param (list): list of probabilities to validate.
        """
        # Lowest value that python3 can handle
        lowest = 2.2250738585072014e-308
        for i in range(len(probabilities)):
            # If value is NaN, then make it the lowest value.
            if isnan(probabilities[i]):
                probabilities[i] = lowest
            # Make every probability proportionally bigger.
            probabilities[i] *= 1.1 ** self.aco.graph.rank
        total = sum(probabilities)

        # To make probabilities sum to 1, divide every probability by their sum.
        for i in range(len(probabilities)):
            probabilities[i] /= total

    def _get_probability(self, j: int) -> float:
        """Get probability for edge (current, param).
        Args:
            param (int): Vertex for which we compute probability.
        """
        current, denominator = self.current_vertex, 0.0
        
        # Set the numerator and denominator for current iteration.
        numerator = (self.aco.graph.pheromone_matrix[current][j] ** self.aco.pheromone_impact) * \
                    ((1 / self.aco.graph.matrix[current][j]) ** self.aco.distance_impact)
        for x in self.allowed_moves:
            denominator += (self.aco.graph.pheromone_matrix[current][x] ** self.aco.pheromone_impact) * \
                           ((1 / self.aco.graph.matrix[current][x]) ** self.aco.distance_impact)

        return numerator / denominator

    def _generate_allowed_moves(self) -> None:
        """Generate moves that are valid.

        Method checks which edges are not visited.
        When no allowed - mark several visited as unvisited and generate again.
        """
        def generate(ant: Ant) -> list:
            """Nested method for generating initial lost of allowed moves.
            
            Args:
                param (Ant): Reference to Ant on which generate() is called.
            """
            matrix, rank = self.aco.graph.matrix, self.aco.graph.rank
            curr_v, prev_v = ant.current_vertex, ant.previous_vertex
            tabu_moves, al_moves = ant.tabu_moves, list()
            # Initially append all visited vertices.
            for j in range(rank):
                if matrix[curr_v][j] != inf and j not in tabu_moves:
                    al_moves.append(j)
            # Previous vertex not valid.
            if prev_v in al_moves and len(al_moves) > 1:
                al_moves.remove(prev_v)
            return al_moves
        
        allowed = generate(self)
        # If not any legal move, then pop 1/4 of tabu moves and generate again.
        while not allowed:
            for x in range(len(self.tabu_moves) // 4):
                if len(self.tabu_moves) == 2:
                    self.tabu_moves.pop()
                    break
                self.tabu_moves.pop()
            allowed = generate(self)

        self.allowed_moves = allowed

    def _leave_pheromones(self) -> None:
        """Apply pheromones on visited edges.

        Traverse whole solution (path) and applies pheromone on every edge.
        """
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
