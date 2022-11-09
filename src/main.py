# Copyright (c) 2022-Current TANG Tianhao <tth502025390@gmail.com>
# Copyright (c) 2022-Current Jialiang Wang <jilwang804@gmail.com>
# License: TBD
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import logging
import argparse
import pickle
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from dataset import SessionDataset, split_validation
from model import SR_GNN, train, test


# ---- Session-based Recommendation with Graph Neural Networks Script ----
if 1:
    # ---- Command Line Arguments Section ------------------
    parser = argparse.ArgumentParser()

    # General config
    parser.add_argument('--dataset', type=str, default='retailrocket',
                        choices=['diginetica', 'yoochoose1_4', 'yoochoose1_64', 'retailrocket', '30music', 'aotm', 'clef', 'rsc15', 'nowplaying', 'tmall', 'xing'],
                        help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/retailrocket/30music.')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Input batch size.')
    parser.add_argument('--n_hid', type=int, default=100,
                        help='Hidden state size.')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Evaluation on top k items.')
    parser.add_argument('--use_cpu', action='store_true', default=False,
                        help='Disables CUDA training and uses CPU only.')
    parser.add_argument('--fast_mode', action='store_true', default=False,
                        help='Disables validation during training pass.')
    parser.add_argument('--valid_portion', type=float, default=0.1,
                        help='The portion of training set split as validation set.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for torch.')

    # Optim config
    parser.add_argument('--epoch', type=int, default=30,
                        help='The number of epochs for training.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')           # Suggested: [0.001, 0.0005, 0.0001]
    parser.add_argument('--lr_dc', type=float, default=0.1,
                        help='Learning rate decay rate.')
    parser.add_argument('--lr_dc_step', type=int, default=3,
                        help='The number of steps after which the learning rate decay.')
    parser.add_argument('--l2', type=float, default=1e-5,
                        help='l2 penalty.')              # Suggested: [0.001, 0.0005, 0.0001, 0.00005, 0.00001]

    # TODO We haven't used them yet
    #parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
    #parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
    #parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')

    # Command line arguments parser
    print('********** Parsing Parameter ***********')
    args = parser.parse_args()
    args.use_cuda = not args.use_cpu and torch.cuda.is_available()
    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.use_cuda:
            torch.cuda.manual_seed(args.seed)
        print('CUDA:', args.use_cuda, ' Seed:', args.seed)
    else:
        print('CUDA:', args.use_cuda)
    # ------------------------------------------------------


def main():
    # ---- Objection Initialization Section ----------------
    print('************ Initialization ************')
    # Loading dataset
    train_data = pickle.load(open('../datasets/' + args.dataset + '/train.txt', 'rb'))
    valid_data_loader = None
    if not args.fast_mode:
        train_data, valid_data = split_validation(train_data, args.valid_portion)
        valid_data = SessionDataset(args.dataset, valid_data)
        valid_data_loader = DataLoader(valid_data, args.batch_size, shuffle=False)


    test_data = pickle.load(open('../datasets/' + args.dataset + '/test.txt', 'rb'))
    train_data = SessionDataset(args.dataset, train_data)
    test_data = SessionDataset(args.dataset, test_data)
    train_data_loader = DataLoader(train_data, args.batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, args.batch_size, shuffle=False)

    # TODO Will be changed later after preprocess.py is finished
    if args.dataset == 'diginetica':
        n_node = 43098
    elif args.dataset == 'yoochoose1_64' or args.dataset == 'yoochoose1_4':
        n_node = 37484
    elif args.dataset == 'retailrocket':
        n_node = 466868
    elif args.dataset == '30music':
        n_node = 137942
    elif args.dataset == 'aotm':
        n_node = 53949
    elif args.dataset == 'clef':
        n_node = 1498
    elif args.dataset == 'nowplaying':
        n_node = 60861
    elif args.dataset == 'tmall':
        n_node = 622678
    elif args.dataset == 'xing':
        n_node = 351111
    else:
        n_node = 0

    # ---- Parameter Section -------------------------------
    device = torch.device('cuda') if args.use_cuda else torch.device('cpu')
    # Usage: SR_GNN(n_hid, n_node, epoch, lr, l2, lr_dc_step, lr_dc)
    model = SR_GNN(args.n_hid, n_node, args.epoch, args.lr, args.l2, args.lr_dc_step, args.lr_dc).to(device)
    # ------------------------------------------------------

    # ---- Training Section --------------------------------
    start = time.time()
    print('************ Training Start ************')
    train(model, train_data_loader, device, args.top_k, valid_data_loader)
    print('************ Training End **************')

    end = time.time()
    print("Run time: %f s" % (end - start))

    # Training loss plot
    plt.figure(figsize=(10, 5))
    plt.title("Training Loss")
    plt.plot(model.training_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    print('************ Training End **************')
    # ------------------------------------------------------

    # ---- Testing Section ---------------------------------
    print('************ Testing Start *************')
    hit, mrr = test(model, test_data_loader, device, args.top_k)
    print('************ Overall Performance *******')
    print('SR-GNN Precision:', str(hit), 'Mean Reciprocal Rank:', str(mrr))
    print('************ Finish ********************')
    # ------------------------------------------------------


if __name__ == '__main__':
    main()

