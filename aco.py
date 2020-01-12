#!/usr/bin/python3
"""ACO and Ant Class used for optimization.

Contains methods for handling object of type(Graph) and type(Ant).

Requires:
    version: python3.7
    packages: numpy
"""

__author__ = 'Milosz Chodkowski PUT'
__license__ = 'MIT'
__version__ = '1.0'
__status__ = 'Working'

import os
import sys
import timeit
import random as rnd
import numpy as np

from more_itertools import pairwise
from math import inf
from numpy.random import choice as np_choice
from typing import Tuple

from graph import Graph


class ACO:
    def __init__(self, instance_file: str, test_file: str, vertex: int, colony_size: int, iterations: int, alpha: float,
                 beta: float, pq: float, pi: float, break_count: int, change_count: int):
        """Constructor for ACO class.

        Creates ACO object containing Graph, Ants and methods handling optimization process.

        Args:
            instance_file (str): File to load.
            test_file (str) File to save results.
            vertex (int): Number of vertices.
            colony_size (int): Size of colony / number of ants.
            iterations (int): Number of iterations.
            alpha (float): Pheromone impact.
            beta (float): Distance impact.
            pq (float): Pheromone vaporize coefficient.
            pi (float): Pheromone intensity.
            break_count (int): Breaks allowed before pheromone smoothing.
            change_count (int): No-solution-change iterations allowed before pheromone smoothing
        """
        self.test_file = test_file
        self.graph = Graph(instance_file, vertex)
        self.colony = colony_size
        self.iterations = iterations
        self.pheromone_impact = alpha
        self.distance_impact = beta
        self.vaporize = pq
        self.intensity = pi
        self.break_count = break_count
        self.change_count = change_count

    def update_pheromones(self, ants: list):
        """Method updating pheromones on visited edges.
        
        Traverse through every visited edge and applies pheromone on it.

        Args:
            ants (list): List of ants which found valid solution.
        """
        for ant in ants:
            self.graph.pheromone_matrix += ant.left_pheromones
        self.graph.pheromone_matrix *= 1 - self.vaporize

    def smooth_pheromone(self, best_solution):
        median = np.median(self.graph.pheromone_matrix)
        for pair in pairwise(best_solution):
            i, j = pair[0], best_solution[1]
            self.graph.pheromone_matrix[(i, j), (j, i)] = median  # Reduction of pheromones
            # self.graph.pheromone_matrix[j, i] = median

    def optimize(self) -> Tuple[float, list, float]:
        """Main method optimizing solution.
        
        Every ant travel through graph and compute solution.
        Ants which found solution are added to list of best ants.

        Returns:
            Tuple(float, list, float): best cost, best solution, elapsed time.
        """
        with open(os.path.join('Tests', self.test_file), 'a') as o_file:
            o_file.write("\nInstance parameters : |V| {}, colony size {},"
                         " iterations {}, alpha {}, beta {}, "
                         " pq {}, pi {} \n".format(
                self.graph.rank, self.colony, self.iterations, self.pheromone_impact,
                self.distance_impact, self.vaporize, self.intensity
            ))

        best_solution, best_cost = None, inf
        solutions = list()
        no_change_count, gen_count = 0, 0
        was_changed = False
        elapsed_time, start = 0, timeit.default_timer()

        # Until as many generations as needed.
        while gen_count != self.iterations:
            if elapsed_time > 60 * 30:  # If past 30 minutes.
                print('Time is over!')
                return best_cost, best_solution, elapsed_time

            # If there is no change in solution and any new solutions exist
            if no_change_count >= self.change_count and solutions:
                for sol in solutions:
                    self.smooth_pheromone(sol)
                print('Matrix was smoothed')
                solutions.clear()
                no_change_count = 0

            # Make a list of best_ants which found solution.
            ants, best_ants = [Ant(self) for a in range(self.colony)], list()
            break_counter = 0
            for ant in ants:
                # Set pheromone matrix to median value, if ants breaking too much
                if break_counter > self.break_count and best_solution is not None:
                    self.smooth_pheromone(best_solution)
                    break_counter = 0

                local_start = timeit.default_timer()
                while len(np.unique(ant.visited_vertices)) < self.graph.rank:  # Until solution found.
                    ant.travel()
                    local_stop = timeit.default_timer()

                    if local_stop - local_start > 3:  # If ant is travelling more than 10 seconds
                        print('Ant was travelling too long. Breaking...')
                        break_counter += 1
                        break

                if ant.total_cost < best_cost:  # Solution better
                    best_cost, best_solution = ant.total_cost, ant.visited_vertices
                    was_changed = True
                    best_ants.append(ant)   # Add ant which found a solution to list of best_ants.
                    solutions.append(best_solution)

            stop = timeit.default_timer()
            elapsed_time = stop - start
            gen_count += 1

            print('End of gen no {}'.format(gen_count))

            # If any ant got solution, then update pheromones
            for ant in best_ants:
                ant.leave_pheromones()
            self.update_pheromones(best_ants)

            if was_changed:     # Print results to file.
                print('Solution!', 'cost: {:.2f}, path: {}'.format(best_cost, len(best_solution)), sep='\n')
                with open(os.path.join('Tests', self.test_file), 'a') as o_file:
                    o_file.write('generation: {} cost: {}, solution {}\n'.format(
                        str(gen_count), str(best_cost), ' '.join(str(v) for v in best_solution))
                    )
                was_changed = False
                no_change_count = 0
            else:
                no_change_count += 1

        with open(os.path.join('Tests', self.test_file), 'a') as o_file:
            o_file.write('Time {:.2f}, Best cost: {:.2f}'.format(elapsed_time, best_cost))

        return best_cost, best_solution, elapsed_time


class Ant:
    def __init__(self, a: ACO):
        """Constructor for Ant class.

        Creates Ant object containing reference to ACO class and methods handling ant choice process.
        
        Args:
            a (ACO): Reference to ACO class in which Ants will be created.
        """
        self.aco = a
        self.total_cost = 0.0

        self.start = rnd.randint(0, self.aco.graph.rank - 1)
        self.previous_vertex = None
        self.current_vertex = self.start

        self.allowed_moves = list()
        self.tabu_moves = list()

        self.left_pheromones = np.zeros((self.aco.graph.rank, self.aco.graph.rank))
        self.visited_vertices = [self.start]

    def travel(self):
        """Makes ant travel to next vertex.

        Generates allowed moves and their probabilities. Makes choice.
        """
        # Generate valid vertices.
        self.generate_allowed_moves()
        probabilities = np.array(list(map(self.get_probability, self.allowed_moves)), dtype='float64')
        self.validate_probabilities(probabilities)

        next_vertex = np_choice(self.allowed_moves, p=probabilities)

        '''Add next edge value to total cost. If previous value was bigger,
           then subtract the previous value, and add it 10 times bigger.'''
        if self.previous_vertex is not None:
            if self.aco.graph.matrix[self.previous_vertex, self.current_vertex] > \
                    self.aco.graph.matrix[self.current_vertex, next_vertex]:
                self.total_cost -= self.aco.graph.matrix[self.previous_vertex, self.current_vertex]
                self.total_cost += self.aco.graph.matrix[self.previous_vertex, self.current_vertex] * 10

        self.total_cost += self.aco.graph.matrix[self.current_vertex, next_vertex]

        # On next move ant can't go to previous and current.
        if self.previous_vertex is not None:
            self.tabu_moves.append(self.previous_vertex)
        self.tabu_moves.append(self.current_vertex)

        # Set a new current vertex, update previous.
        self.visited_vertices.append(next_vertex)
        self.previous_vertex = self.current_vertex
        self.current_vertex = next_vertex

    def validate_probabilities(self, probabilities):
        """Checks if probabilities sum to one.

        If not, then make it sum to one.

        Args:
            probabilities (ndarray): list of probabilities to validate.
        """
        # Lowest value that python3 can handle
        lowest = sys.float_info.min
        if np.nan in probabilities:
            probabilities[probabilities == np.nan] = lowest     # Change Nan values to lowest
            probabilities *= 1.1 ** self.aco.graph.rank
        if sum(probabilities) != 1:
            probabilities /= sum(probabilities)     # Make it sum to 1.

    def get_probability(self, j: int) -> float:
        """Get probability for edge (current, param).

        Args:
            j (int): Vertex for which we compute probability.

        Returns:
            (float): Probability of picking the j vertex as next.
        """
        current, denominator = self.current_vertex, 0.0

        # Set the numerator and denominator for current iteration.
        numerator = (self.aco.graph.pheromone_matrix[current, j] ** self.aco.pheromone_impact) * \
                    ((1 / self.aco.graph.matrix[current, j]) ** self.aco.distance_impact)
        for x in self.allowed_moves:
            denominator += (self.aco.graph.pheromone_matrix[current, x] ** self.aco.pheromone_impact) * \
                           ((1 / self.aco.graph.matrix[current, x]) ** self.aco.distance_impact)

        return numerator / denominator

    def generate_allowed_moves(self):
        """Generate moves that are valid.

        Method checks which edges are not visited.
        When no allowed - mark several visited as unvisited and generate again.
        """

        def generate(ant: Ant) -> list:
            """Nested method for generating initial lost of allowed moves.
            
            Args:
                ant (Ant): Reference to Ant on which generate() is called.

            Returns:
                (list): List of allowed moves.
            """
            matrix, rank = self.aco.graph.matrix, self.aco.graph.rank
            curr_v, prev_v = ant.current_vertex, ant.previous_vertex
            tabu_moves, al_moves = ant.tabu_moves, list()
            # Initially append all visited vertices.
            for j in range(rank):
                if matrix[curr_v, j] != inf and j not in tabu_moves:
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

    def leave_pheromones(self):
        """Apply pheromones on visited edges.

        Traverse through whole solution (path) and applies pheromone on every edge.
        """
        # Make left pheromones on all edges equal to zero.
        if len(self.visited_vertices) <= 1:
            return None

        left_pheromones = np.zeros_like(self.left_pheromones)
        for pair in pairwise(self.visited_vertices):
            i, j = pair[0], pair[1]
            # Leave pheromones on edge i, j and j, i.
            left_pheromones[(i, j), (j, i)] = (self.aco.intensity / self.total_cost) ** 2
            # left_pheromones[j, i] = (self.aco.intensity / self.total_cost) ** 2
        self.left_pheromones = left_pheromones
