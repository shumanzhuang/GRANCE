import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree



class GNNLayer(MessagePassing):
    def __init__(self, edge_index, y, num_hidden, dropout):
        super(GNNLayer, self).__init__(aggr='add')
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * num_hidden, 1)
        self.edge_index = edge_index
        self.row, self.col = edge_index
        self.norm_degree = degree(self.row, num_nodes=y.shape[0]).clamp(min=1)
        self.norm_degree = torch.pow(self.norm_degree, -0.5)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)

    def forward(self, h):
        h2 = torch.cat([h[self.row], h[self.col]], dim=1)
        g = torch.tanh(self.gate(h2)).squeeze()
        norm =  g * self.norm_degree[self.row] * self.norm_degree[self.col]
        norm = self.dropout(norm)
        return self.propagate(self.edge_index, size=(h.size(0), h.size(0)), x=h, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1,1) * x_j

    def update(self, aggr_out):
        return aggr_out




class GNN_Classifier(nn.Module):
    def __init__(self, edge_index, edge_index_knn, y, num_features, num_hidden, num_classes, dropout, knn_weights, eps, layer_num=2):
        super(GNN_Classifier, self).__init__()
        self.knn_weights = knn_weights
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.layers_knn = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(GNNLayer(edge_index, y,num_hidden, dropout))
        for i in range(self.layer_num):
            self.layers_knn.append(GNNLayer(edge_index_knn, y,num_hidden, dropout))
        self.t1 = nn.Linear(num_features, num_hidden)
        self.t2 = nn.Linear(num_hidden, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, h):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h_org = self.layers[i](h)
            h_knn = self.layers_knn[i](h)
            h = self.eps * raw + h_org + self.knn_weights * h_knn
        h = self.t2(h)
        return F.log_softmax(h, 1)

