# Copyright (c) 2022-Current TANG Tianhao <tth502025390@gmail.com>
# License: TBD

from collections import defaultdict
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
import numpy as np


def max_n_node(all_usr_pois):
    us_lens = [len(np.unique(upois).tolist()) for upois in all_usr_pois]
    len_max = max(us_lens)
    return len_max


def split_validation(train_set, valid_portion):
    """
    This function is used before sending data into Session dataset

    """
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class SessionDataset(Dataset):
    def __init__(self, name, data, total_n_node: int = None, need_shuffle: bool = False, need_norm: bool = False):
        super(SessionDataset, self).__init__()
        self.name = name
        self.data = data[0]
        self.max_n_node = max_n_node(data[0])
        self.targets = data[1]
        self.total_n_node = total_n_node
        self.total_n_graph = len(self.data)
        self.need_shuffle = need_shuffle
        self.need_norm = need_norm

        if need_shuffle:
            shuffle_range = np.range(self.total_n_graph)
            np.random.shuffle(shuffle_range)
            self.data = self.data[shuffle_range]
            self.targets = self.targets[shuffle_range]

    def __len__(self):
        return self.total_n_graph

    def __getitem__(self, idx):
        """
        We do not have padding here. Thus, in the original model, the padding 0 will also be seen as a 
        unique node in the graph, and at the same time the 0 node will also be shown in the adj matrix. 
        Here we do not have that in this dataset. 
            
        In the original dataset, there are cases that only one node in the session. In this case, the 
        `edge_index` will be an empty vector.

        In original dataset the adj matrix has been normalized. Here, if need_norm is enabled, an `edge_weight`
        attr will also be added to the graph.

        @return: 
            graph: torch_geometric.data.Data
            n_node: int
        """
        rough_nodes, target = self.data[idx], self.targets[idx]
        nodes = np.unique(rough_nodes)
        n_node = len(nodes)
        _in, _out = list(), list()
        if self.need_norm:
            _in_c = defaultdict(int)
        # Construct edge index
        for i in range(len(rough_nodes)-1):
            u = np.where(nodes == rough_nodes[i])[0][0] # output is Tuple(array(idx)), thus [0][0]
            v = np.where(nodes == rough_nodes[i + 1])[0][0]
            _in.append(u)
            _out.append(v)
            if self.need_norm:
                _in_c[u] += 1
        nodes = torch.tensor(nodes, dtype=torch.float64)
        edge_index = torch.tensor(np.vstack((np.array(_in), np.array(_out))), dtype=torch.long)
        target = torch.tensor(target, dtype=torch.float64)
        
        if self.need_norm:
            # If zero length return zero length 
            edge_weight = torch.zeros(edge_index.size(1), dtype=torch.float64)
            if edge_index.size(1) >= 1:
                for node in np.arange(len(nodes)):
                    edge_weight[edge_index[0] == node] = 1.0 / _in_c[node] if node in _in_c.keys() else 1.0
            graph = Data(x=nodes, edge_index=edge_index, y=target, edge_weight=edge_weight, n_node=n_node)
            return graph

        graph = Data(x=nodes, edge_index=edge_index, y=target, n_node=n_node)            
        return graph


