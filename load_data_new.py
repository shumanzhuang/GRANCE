import torch
import os.path as osp
import numpy as np
import scipy.sparse as sp
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.preprocessing import normalize
from torch_geometric.utils import to_undirected
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_sparse import SparseTensor
from utils import to_scipy
from types import SimpleNamespace
from scipy.sparse import csr_matrix

def load_new_dataset(load_path):
    data = np.load(load_path, allow_pickle=True)

    rows = data['edge_index'][0]
    cols = data['edge_index'][1]
    data_ = np.ones_like(rows)
    sparse_matrix = csr_matrix((data_, (rows, cols)))

    data_loaded = SimpleNamespace(
        num_nodes=data["num_nodes"],
        num_edges=data["num_edges"],
        num_node_features=data["num_node_features"],
        num_classes=data["num_classes"],
        adj=sparse_matrix,
        features=to_scipy(torch.tensor(data['x'])),
        idx_test=data["idx_test"],
        idx_train=data["idx_train"],
        idx_val=data["idx_val"],
        labels=data["labels"],
        train_mask=data["train_mask"],
        test_mask=data["test_mask"],
        val_mask=data["val_mask"],
        edge_index=data["edge_index"],
    )
    return data_loaded

def load_ogb_data():
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./dataset/')
    graph = dataset[0]
    # create train mask
    split_idx = dataset.get_idx_split()
    idx_train = split_idx['train'].numpy()
    idx_val = split_idx['valid'].numpy()
    idx_test = split_idx['test'].numpy()
    graph.x = normalize(graph.x)
    graph.x = torch.from_numpy(graph.x).float()
    graph.y = graph.y.squeeze()
    adj = to_sparsetensor(graph)
    graph.adj = normalize_adj_OBG(adj, 'DAD')
    graph.y = graph.y.squeeze()
    return graph, idx_train, idx_test, idx_val

def to_sparsetensor(data):
    N = data.num_nodes
    data.edge_index = to_undirected(data.edge_index)
    data.edge_index, _ = remove_self_loops(data.edge_index)
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=N)

    row, col = data.edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.set_diag()
    return adj

def normalize_adj_OBG(adj, mode='DA'):
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    if mode == 'DA':
        return deg_inv_sqrt.view(-1, 1) * deg_inv_sqrt.view(-1, 1) * adj
    if mode == 'DAD':
        return deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    return adj
