#!/usr/bin/python3
"""ACO optimization main file.

Requires:
    version: python3.7
    modules: aco
"""

__author__ = 'Milosz Chodkowski PUT'
__license__ = 'MIT'
__version__ = '1.0'
__status__ = 'Working'


import argparse

from aco import ACO


def get_args():
    parser = argparse.ArgumentParser(description='Ant Colony Optimization for Graph Travel')
    parser.add_argument(
        '-f', '-F', '--file',
        required=True,
        dest='fh_input',
        help='instance filename')
    parser.add_argument(
        '-c', '-C', '--colony',
        dest='colony',
        default=10,
        help='colony size')
    parser.add_argument(
        '-it', '-IT', '--iter',
        dest='iterations',
        default=50,
        help='no of iterations')
    parser.add_argument(
        '-a', '-A', '--alpha',
        dest='alpha',
        default=0.45,
        help='alpha val')
    parser.add_argument(
        '-b', '-B', '--beta',
        dest='beta',
        default=0.55,
        help='beta val')
    parser.add_argument(
        '-v', '-V', '--vaporize',
        default=0.5,
        dest='vaporize',
        help='pheromone vaporize coefficient')
    parser.add_argument(
        '-i', '-I', '--intensity',
        default=1.0,
        dest='intensity',
        help='pheromone intensity')
    parser.add_argument(
        '-bc', '-BC', '-break_count',
        default=3,
        dest='br_count',
        help='break count')
    parser.add_argument(
        '-cc', '-CC', '--change_count',
        default=6,
        dest='ch_count',
        help='change_count')
    return parser.parse_args()


def main(fh_input: str, colony: int, iters: int, alpha: float, beta: float, vaporize: float, intensity: float, br_count: int, ch_count: int):
    print('Start optimization...')
    aco = ACO(
        instance_file=fh_input,
        vertex=40,
        colony_size=colony,
        iterations=iters,
        alpha=alpha,
        beta=beta,
        pq=vaporize,
        pi=intensity,
        break_count=br_count,
        change_count=ch_count,
    )
    best_c, best_s, time = aco.optimize()
    print('time: {:.2f}, cost: {:.2f}, solution (len): {}'.format(time, best_c, len(best_s)))


if __name__ == "__main__":

    args = get_args()
    main(args.fh_input, int(args.colony), int(args.iterations), float(args.alpha), float(args.beta), float(args.vaporize), float(args.intensity), int(args.br_count), int(args.ch_count))
