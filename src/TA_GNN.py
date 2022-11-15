'''
MethodModule class for Target Attentive Graph Neural Network on Session-based Recommendation Task

@inproceedings{yu2020tagnn,
  title={TAGNN: target attentive graph neural networks for session-based recommendation},
  author={Yu, Feng and Zhu, Yanqiao and Liu, Qiang and Wu, Shu and Wang, Liang and Tan, Tieniu},
  booktitle={Proceedings of the 43rd international ACM SIGIR conference on research and development in information retrieval},
  pages={1921--1924},
  year={2020}
}
'''

# Copyright (c) 2022-Current Jialiang Wang <jilwang804@gmail.com>
# Copyright (c) 2022-Current TANG Tianhao <tth502025390@gmail.com>
# License: TBD

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv
from tqdm import tqdm


class TA_GNN(nn.Module):

    # Initialization function
    def __init__(self, n_hid, n_node, epoch=30, lr=0.001, l2=1e-5, lr_dc_step=3, lr_dc=0.1, n_layers=1):
        super(TA_GNN, self).__init__()
        nn.Module.__init__(self)

        self.n_hidden = n_hid
        self.n_node = n_node

        # GatedGraphConv(out_channels: int, num_layers: int, aggr: str = 'add', bias: bool = True, **kwargs)
        self.embed = nn.Embedding(self.n_node, self.n_hidden)
        self.ggc = GatedGraphConv(self.n_hidden, num_layers=n_layers)
        self.W_1 = nn.Linear(self.n_hidden, self.n_hidden)
        self.W_2 = nn.Linear(self.n_hidden, self.n_hidden)
        self.W_t = nn.Linear(self.n_hidden, self.n_hidden)
        self.q = nn.Linear(self.n_hidden, 1)
        self.W_3 = nn.Linear(2 * self.n_hidden, self.n_hidden)

        self.max_epoch = epoch
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=l2)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_dc_step, gamma=lr_dc)
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
        v = self.ggc(embedding, data.edge_index, data.edge_weight)

        # Generating session embeddings
        n_items_in_sessions = torch.bincount(data.batch)
        v_by_sessions = torch.split(v, list(n_items_in_sessions))

        # s_l: the list of last-clicked item in each session; v_n
        s_l, v_n, s_t = [], [], []
        for s in v_by_sessions:
            s_l.append(s[-1])
            v_n.append(s[-1].view(-1).repeat(s.shape[0], 1))

            # (6)
            s_t.append(torch.mm(F.softmax(torch.mm(self.embed.weight, self.W_t(s).transpose(1, 0)), -1), s))

        s_l = torch.stack(s_l, dim=0)
        v_n = torch.cat(v_n, dim=0)
        s_t = torch.stack(s_t, dim=0)

        # (8)
        alpha = self.q(torch.sigmoid(self.W_1(v_n) + self.W_2(v)))
        s_g_split = torch.split(alpha * v, list(n_items_in_sessions))

        # (9)
        s_g = []
        for s in s_g_split:
            s_g_sum = torch.sum(s, dim=0)
            s_g.append(s_g_sum)

        s_g = torch.stack(s_g, dim=0)

        # (10)
        s_h = self.W_3(torch.cat([s_l, s_g], dim=-1))
        s_h = s_h.view(s_g.shape[0], 1, s_g.shape[1])

        # Making recommendation
        # (7) (10) (11)
        # Multiply each candidate item's embedding by session representation s_h
        s_h = s_h + s_t
        z = torch.sum(s_h * self.embed.weight, -1)

        return z


# Train function
def train(model, train_data_loader, device, top_k, validation_data_loader=None, result_saver=None, logger=None):

    # Switch to training mode
    model.train()
    best_hit = 0

    # Initialize data components
    for epoch in range(model.max_epoch):
        train_loss = -1
        tbar = tqdm(enumerate(train_data_loader), total=len(train_data_loader), dynamic_ncols=True)
        for i, batch in tbar:
            # Set prefix
            tbar.set_description('Epoch %d: Current loss: %4f' % (epoch + 1, train_loss))

            # Forward step
            y_pred = model(batch.to(device))

            # Calculate the training loss
            train_loss = model.loss_function(y_pred, batch.y.to(device))
            model.optimizer.zero_grad()

            # Backward step: error backpropagation
            train_loss.backward()

            # Update the variables according to the optimizer and the gradients calculated by the above loss function
            model.optimizer.step()

            if (i + 1) % 500 == 0:
                if validation_data_loader:
                    with torch.no_grad():
                        hit, mrr = test(model, validation_data_loader, device, top_k, logger=logger)
                    print('Epoch:', epoch + 1, 'Batch:', i + 1, 'Train Loss:', train_loss.item(), 'Top', top_k,
                        'Precision:', hit, 'Mean Reciprocal Rank:', mrr)
                    logger.info('Epoch: ' + str(epoch + 1) + ' Batch: ' + str(i + 1) +
                                ' Train Loss: ' + str(train_loss.item()) + ' Top ' + str(top_k) +
                                ' Precision: ' + str(hit) + ' Mean Reciprocal Rank: ' + str(mrr))

                    # Save the model parameter with best hit
                    if epoch > 0 and hit > best_hit:
                        result_saver.data = model.state_dict()
                        result_saver.save_learned_model()
                        best_hit = hit
                else:
                    print('Epoch:', epoch + 1, 'Batch:', i + 1, 'Train Loss:', train_loss.item())
                    logger.info('Epoch: ' + str(epoch + 1) + ' Batch: ' + str(i + 1) +
                                ' Train Loss: ' + str(train_loss.item()))

        model.training_loss.append(train_loss.item())

        model.scheduler.step()


# Test function
def test(model, test_data_loader, device, top_k, best_model_path=None, logger=None):

    # Load the best model if provided
    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))

    # Switch to testing mode
    model.eval()

    hit, mrr = [], []

    for i, batch in tqdm(enumerate(test_data_loader), total=len(test_data_loader), 
            desc="Valid/Test phase", dynamic_ncols=True):
        # Forward step
        scores = model(batch.to(device))
        labels = batch.y

        # Find top-k item scores
        top_scores = scores.topk(top_k)[1]

        for score, label in zip(top_scores.detach().cpu().numpy(), labels.detach().cpu().numpy()):
            hit.append(np.isin(label, score))

            if len(np.where(score == label)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == label)[0][0] + 1))

    return np.mean(hit) * 100, np.mean(mrr) * 100

