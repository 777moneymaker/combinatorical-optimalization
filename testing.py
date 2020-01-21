#!/usr/bin/python3

import os


if __name__ == '__main__':
    results = {}
    costs = []
    best_costs = {'False': [], 'True': []}
    for fh in os.listdir('Tests/MATRIX_SMOOTH'):
        size = fh.split('SMOOTH_')[1].split('.txt')[0]
        rank = fh.split('_MATRIX')[0].split('test_')[1]
        with open('Tests/MATRIX_SMOOTH/{}'.format(fh), 'r') as test_file:
            for line in test_file.readlines():
                if 'generation' in line:
                    costs.append(1)
            best_costs[str(size)].append('V'+ rank + ' ' + str(len(costs)/5))
        costs.clear()
    for size, cost in best_costs.items():
        print(size, cost)

