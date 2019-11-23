import argparse
import os
import random
import numpy

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='input.txt')
parser.add_argument('--output', type=str, default='output.txt')
parser.add_argument('--num_samples', type=int, default=1000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--onehot', action='store_true')
parser.add_argument('--num_classes', type=int, default=10)
args = parser.parse_args()

random.seed(args.seed)
numpy.random.seed(args.seed)

with open(args.input, 'r') as f:
    lines = [l.strip('\n') for l in f.readlines()]

if args.num_samples > 0:
    lines_sampled = numpy.random.choice(lines, args.num_samples)
else:
    lines_sampled = lines

with open(args.output, 'w') as f:
    for line in lines_sampled:
        if args.onehot:
            L = numpy.zeros(args.num_classes)
            L[int(line.split()[1])] = 1
            f.write(line.split()[0] + ' ' + ' '.join(map(str, L)) + '\n')
        else:
            f.write(line + '\n')
