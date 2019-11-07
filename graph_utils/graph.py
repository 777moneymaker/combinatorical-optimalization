#!/usr/bin/env python3
"""
This is a simple implementation of Graphs in Python3.
"""
import random

__author__ = "Milosz Chodkowski"
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Milosz Chodkowski"
__email__ = "milosz.chodkowski@student.put.poznan.pl"
__status__ = "Production"


class Graph:
    def __init__(self, vertex=5):
        self.vertex = vertex
        self.matrix = [[random.randint(0, 1) for i in range(vertex)] for i in range(vertex)]

    def show(self):
        for arr in self.matrix:
            for val in arr:
                print(val, end=" ")
            print("")
