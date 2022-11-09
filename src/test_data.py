# Copyright (c) 2022-Current TANG Tianhao <tth502025390@gmail.com>
# License: TBD

import argparse
import pickle
import time
from dataset import *
from torch_geometric.loader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
opt = parser.parse_args()
print(opt)


def main():
    data = pickle.load(open('../datasets/retailrocket/' + 'train.txt', 'rb'))
    train_data = SessionDataset('sample', data)

    length = len(train_data)
    # print(length)
    np.random.seed()
    idx = np.random.randint(0, length-1)
    print(train_data[idx])
    print(train_data[idx].x)
    print(train_data[idx].edge_index)
    print(train_data[idx].n_node)

    train_data_norm = SessionDataset('sample', data, need_norm=True)
    print(train_data_norm[idx])
    print(train_data_norm[idx].x)
    print(train_data_norm[idx].edge_index)
    print(train_data_norm[idx].edge_weight)
    print(train_data_norm[idx].n_node)

    loader = DataLoader(train_data, 10)
    print(loader)
    for batch in loader:
        print(batch)
        print(batch.x)
        print(batch.edge_index)
        print(batch.y)
        print(batch.batch)
        print(batch.ptr)
        break

if __name__ == '__main__':
    main()
