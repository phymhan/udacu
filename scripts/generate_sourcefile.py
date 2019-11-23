import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='svhn')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--root', type=str, default='datasets')
args = parser.parse_args()

import os
labels = os.listdir(os.path.join(args.root, args.name, args.mode))

with open(os.path.join('sourcefiles', f'{args.name}_{args.mode}.txt'), 'w') as f:
    for l in labels:
        for img in os.listdir(os.path.join(args.root, args.name, args.mode, l)):
            f.write(f'{l}/{img} {l}\n')