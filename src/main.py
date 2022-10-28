# Copyright (c) 2022-Current TANG Tianhao <tth502025390@gmail.com>
# License: TBD

import argparse
import pickle
import time
import torch
from torch_geometric.loader import DataLoader
from dataset import SessionDataset, split_validation
from model import SR_GNN, train, test

parser = argparse.ArgumentParser()
# Base config
parser.add_argument('--dataset', default='sample', 
                    choices=['diginetica', 'yoochoose1_4', 'yoochoose1_64', 'sample'],
                    help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--n_hid', type=int, default=100, help='hidden state size')
parser.add_argument('--top_k', type=int, default=10, help='eval on top k items')
parser.add_argument('--cuda', type=bool, default=False, action='store_true', help='if true and if cuda exist, use cuda')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
# Optim config
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
# we haven't use them yet
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')

args = parser.parse_args()

def main():
    train_data = pickle.load(open('../datasets/' + args.dataset + '/train.txt', 'rb'))
    if args.validation:
        train_data, valid_data = split_validation(train_data, args.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('../datasets/' + args.dataset + '/test.txt', 'rb'))


    train_data = SessionDataset(args.dataset, train_data, shuffle=True)
    test_data = SessionDataset(args.dataset, test_data, shuffle=False)

    train_data_loader = DataLoader(train_data, args.batch_size)
    test_data_loader = DataLoader(test_data, args.batch_size)

    # Will be changed later after preprocess.py is finished
    if args.dataset == 'diginetica':
        n_node = 43098
    elif args.dataset == 'yoochoose1_64' or args.dataset == 'yoochoose1_4':
        n_node = 37484
    else:
        n_node = 310
    
    if args.cuda:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device('cpu')

    model = SR_GNN(args.n_hid, n_node, args.epoch, args.lr, args.l2, args.lr_dc_step, args.lr_dc).to(device)
    
    start = time.time()
    print('Start Training--------------------------------------------------------')
    train(model, train_data_loader, device, args.top_k, test_data_loader)
    print('Finish Training-------------------------------------------------------')

    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
