import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
args = parser.parse_args()

src = f'/media/ligong/Picasso/Datasets/cifar/cifar10_{args.mode}.txt'
src_label = '/media/ligong/Picasso/Datasets/cifar/labels.txt'

with open(src_label, 'r') as f:
    label_map = [c.strip('\n') for c in f.readlines()]

label_map = {c: i for i, c in enumerate(label_map, 0)}
print(label_map)

with open(src, 'r') as f:
    sourcefile = [c.strip('\n') for c in f.readlines()]

with open(f'cifar10_{args.mode}.txt', 'w') as f:
    for name in sourcefile:
        c = name.replace('.png', '').split('_')[1]
        print(f'{name}, {c}')
        i = label_map[c]
        f.write(f'{name} {i}\n')
