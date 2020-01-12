#!/usr/bin/python3

from graph import Graph

if __name__ == '__main__':
    G = Graph('v40_2.txt', 40)
    G.check_if_connected()