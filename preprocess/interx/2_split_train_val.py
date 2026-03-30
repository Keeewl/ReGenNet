import os
import argparse
import itertools
import h5py
import numpy as np
from tqdm import tqdm

train_file = '../datasets/interx/splits/train.txt'
val_file = '../datasets/interx/splits/val.txt'
test_file = '../datasets/interx/splits/test.txt'

hhi_file = './regen/inter-x_regen.h5'

train_list = [line.rstrip('\n') for line in open(train_file, "r").readlines()]
val_list = [line.rstrip('\n') for line in open(val_file, "r").readlines()]
test_list = [line.rstrip('\n') for line in open(test_file, "r").readlines()]

os.makedirs('./regen', exist_ok=True)
fw_train = h5py.File('./regen/train.h5', 'w')
fw_val = h5py.File('./regen/val.h5', 'w')
fw_test = h5py.File('./regen/test.h5', 'w')

with h5py.File(hhi_file, 'r') as f:
    keys = list(f.keys())
    pbar = tqdm(keys)
    for k in pbar:
        data = f[k][:]
        if k in train_list:
            fw_train.create_dataset(k, data=data, dtype='f4')
        elif k in val_list:
            fw_val.create_dataset(k, data=data, dtype='f4')
        elif k in test_list:
            fw_test.create_dataset(k, data=data, dtype='f4')
