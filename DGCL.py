import torch.nn as nn
import copy
from deeprobust.graph.utils import *
from utils import sparse_dense_mul, neighborhood_reliability, sim

def get_contrastive_emb(args, logger, adj_pre, features, aug_adj1, lr, weight_decay, nb_epochs):
    ft_size = features.shape[2]
    adj = normalize_adj(adj_pre + (sp.eye(adj_pre.shape[0]) * 2))
    aug_adj1 = normalize_adj2(aug_adj1 + (sp.eye(adj.shape[0]) * 2))
    sp_adj = sparse_mx_to_torch_sparse_tensor((adj))
    sp_aug_adj1 = sparse_mx_to_torch_sparse_tensor(aug_adj1)
    model = GRACE(ft_size, 512, 'prelu')
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if torch.cuda.is_available():
        logger.info('Using CUDA')
        model.cuda()
        features = features.cuda()
        sp_adj = sp_adj.cuda()
        sp_aug_adj1 = sp_aug_adj1.cuda()

    cnt_wait = 0
    best = 1e9
    best_t = 0

    for epoch in range(nb_epochs):
        model.train()
        optimiser.zero_grad()

        z_0, z_aug1 = model(features, sp_adj, sp_aug_adj1, True)
        sp_adj_dense = sp_adj.to_dense()
        edge_index = sp_adj.coalesce().indices()
        z_0 = z_0.squeeze(0)
        z_aug1 = z_aug1.squeeze(0)
        aug_adj1_dense = sp_aug_adj1.to_dense()
        probs = neighborhood_reliability(z_0, edge_index, args.nclusters, args.beta)
        confmatrix = sim(probs, probs)
        loss = model.loss(z_0, sp_adj_dense, z_aug1, aug_adj1_dense, confmatrix, args.mean)

        logger.info('Loss:[{:.4f}]'.format(loss.item()))

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            weights = copy.deepcopy(model.state_dict())
        else:
            cnt_wait += 1

        if cnt_wait == 30:
            logger.info('Early stopping!')
            break

        loss.backward()
        optimiser.step()

    logger.info('Loading {}th epoch'.format(best_t))
    model.load_state_dict(weights)

    return model.embed(features, sp_adj, True, None)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize_adj2(adj, alpha=-0.5):
    """Symmetrically normalize adjacency matrix."""
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj = torch.add(torch.eye(adj.shape[0]), adj)
    degree = adj.sum(dim=1)
    in_degree_norm = torch.pow(degree.view(1, -1), alpha).expand(adj.shape[0], adj.shape[0])
    out_degree_norm = torch.pow(degree.view(-1, 1), alpha).expand(adj.shape[0], adj.shape[0])
    adj = sparse_dense_mul(adj, in_degree_norm)
    adj = sparse_dense_mul(adj, out_degree_norm)
    if alpha > 0:
        return to_scipy((adj / (adj.sum(dim=1).reshape(adj.shape[0], -1)))).tocoo()
    else:
        return to_scipy(adj).tocoo()


class GRACE(nn.Module):
    def __init__(self, n_in, n_h, activation, tau: float=0.5):
        super(GRACE, self).__init__()
        self.gcn = GRACE_GCN(n_in, n_h, activation)
        self.tau: float = tau
        self.sigm = nn.Sigmoid()

    def forward(self, features, adj, aug_adj1, sparse):
        h_0 = self.gcn(features, adj, sparse)
        h_1 = self.gcn(features, aug_adj1, sparse)
        return h_0, h_1

    def semi_loss(self, z1, adj1, z2, adj2, confmatrix, mean):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(sim(z1, z1))
        between_sim = f(sim(z1, z2))
        if mean:
            pos = between_sim.diag() + (refl_sim * adj1 * confmatrix).sum(1) / (adj1.sum(1) + 0.01)
        else:
            pos = between_sim.diag() + (refl_sim * adj1 * confmatrix).sum(1)
        neg = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag() - (refl_sim * adj1).sum(1) - (between_sim * adj2).sum(1)
        loss = -torch.log(pos / (pos + neg))

        return loss

    def loss(self, z1, graph1, z2, graph2, confmatrix, mean):
        h1 = z1
        h2 = z2

        l1 = self.semi_loss(h1, graph1, h2, graph2, confmatrix, mean)
        l2 = self.semi_loss(h2, graph2, h1, graph1, confmatrix, mean)

        ret = (l1 + l2) * 0.5
        ret = ret.mean()

        return ret

    # Detach the return variables
    def embed(self, features, adj, sparse, msk):
        h_1 = self.gcn(features, adj, sparse)
        return h_1.detach()


class GRACE_GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GRACE_GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, features, adj, sparse=False):
        seq_fts = self.fc(features)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)