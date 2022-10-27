'''
MethodModule class for Gated Graph Sequence Neural Networks on Session-based Recommendation Task

@inproceedings{Wu:2019vb,
author = {Wu, Shu and Tang, Yuyuan and Zhu, Yanqiao and Wang, Liang and Xie, Xing and Tan, Tieniu},
title = {Session-based Recommendation with Graph Neural Networks},
booktitle = {Proceedings of The Twenty-Third AAAI Conference on Artificial Intelligence},
series = {AAAI '19},
year = {2019},
url = {http://arxiv.org/abs/1811.00855}
}
'''

# Copyright (c) 2022-Current Jialiang Wang <jilwang804@gmail.com>
# License: TBD

import math
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GatedGraphConv


class SR_GNN(nn.Module):

    # Initialization function
    # loss_function: nn.CrossEntropyLoss()
    # optimizer: torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.wd)
    # scheduler: torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)
    def __init__(self, n_hid, n_node, epoch, loss_function, optimizer, scheduler):
        super(SR_GNN, self).__init__()
        nn.Module.__init__(self)

        self.n_hidden = n_hid
        self.n_node = n_node

        # GatedGraphConv(out_channels: int, num_layers: int, aggr: str = 'add', bias: bool = True, **kwargs)
        self.embed = nn.Embedding(self.n_node, self.n_hidden)
        self.ggc = GatedGraphConv(self.n_hidden, num_layers=1)
        self.W_1 = nn.Linear(self.n_hidden, self.n_hidden)
        self.W_2 = nn.Linear(self.n_hidden, self.n_hidden)
        self.q = nn.Linear(self.n_hidden, 1)
        self.W_3 = nn.Linear(2 * self.n_hidden, self.n_hidden)

        self.max_epoch = epoch
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.training_loss = []

        self.reset_parameters()

    # From https://github.com/CRIPAC-DIG/SR-GNN/blob/master/pytorch_code/model.py
    def reset_parameters(self):

        stdv = 1.0 / math.sqrt(self.n_hidden)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    # Forward propagation function
    def forward(self, data):

        # (0)
        # Embed every item into a unified embedding space
        embedding = self.embed(data.x).squeeze()

        # (1) - (5)
        # Learning item embeddings on session graphs
        # The gated graph convolution operator
        v = self.ggc(embedding, data.edge_index)

        # Generating session embeddings
        n_items_in_sessions = torch.bincount(data.batch)
        v_by_sessions = torch.split(v, list(n_items_in_sessions))

        # s_l: the list of last-clicked item in each session; v_n
        s_l, v_n = [], []
        for s in v_by_sessions:
            s_l.append(s[-1])
            v_n.append(s[-1].view(-1).repeat(s.shape[0], 1))

        s_l = torch.stack(s_l, dim=0)
        v_n = torch.cat(v_n, dim=0)

        # (6)
        alpha = self.q(torch.sigmoid(self.W_1(v_n) + self.W_2(v)))
        s_g_split = torch.split(alpha * v, list(n_items_in_sessions))

        s_g = []
        for s in s_g_split:
            s_g_sum = torch.sum(s, dim=0)
            s_g.append(s_g_sum)

        s_g = torch.stack(s_g, dim=0)

        # (7)
        s_h = self.W_3(torch.cat([s_l, s_g], dim=-1))

        # Making recommendation
        # (8)
        # Multiply each candidate item's embedding by session representation s_h
        z = torch.mm(s_h, self.embed.weight.transpose(1, 0))

        return z


# Train function
def train(model, train_data_loader, device, top_k, validation_data_loader=None):

    # Switch to training mode
    model.train()

    # Initialize data components
    y_pred = []

    for epoch in range(model.max_epoch):
        for i, batch in enumerate(train_data_loader):
            # Forward step
            scores = model(batch.to(device))

            # Calculate the training loss
            train_loss = model.loss_function(scores, batch.y - 1)
            model.optimizer.zero_grad()

            # Backward step: error backpropagation
            train_loss.backward()

            # Update the variables according to the optimizer and the gradients calculated by the above loss function
            model.optimizer.step()

            if i % 1000 == 0:
                if validation_data_loader:
                    with torch.no_grad():
                        hit, mrr = test(model, validation_data_loader, device, top_k)
                    print('Epoch:', epoch + 1, 'Batch:', i + 1, 'Train Loss:', train_loss.item(), 'Top', top_k, 'Precision:', hit,
                          'Mean Reciprocal Rank:', mrr)
                else:
                    print('Epoch:', epoch + 1, 'Batch:', i + 1, 'Train Loss:', train_loss.item())

        model.training_loss.append(train_loss.item())

        model.scheduler.step()


# Test function
def test(model, validation_data_loader, device, top_k):

    # Switch to testing mode
    model.eval()

    hit, mrr = [], []

    for i, batch in enumerate(validation_data_loader):
        # Forward step
        scores = model(batch.to(device))
        labels = batch.y - 1

        # Find top-k item scores
        top_scores = scores.topk(top_k)[1]

        for score, label in zip(top_scores.detach().cpu().numpy(), labels.detach().cpu().numpy()):
            hit.append(np.isin(label, score))

            if len(np.where(score == label)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == label)[0][0] + 1))

    return np.mean(hit) * 100, np.mean(mrr) * 100

