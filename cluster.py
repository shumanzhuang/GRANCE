import torch
import torch.nn as nn
import torch.nn.functional as F

def cluster(data, k, temp, num_iter, init, cluster_temp):
    cuda0 = torch.cuda.is_available()  # False

    if cuda0:
        mu = init.cuda()
        data = data.cuda()
        cluster_temp = cluster_temp.cuda()
    else:
        mu = init

    data = data / (data.norm(dim=1)[:, None] + 1e-6)
    for t in range(num_iter):
        mu = mu / (mu.norm(dim=1)[:, None] + 1e-6)
        dist = torch.mm(data, mu.transpose(0, 1))

        r = F.softmax(cluster_temp * dist, dim=1)
        cluster_r = r.sum(dim=0)
        cluster_mean = r.t() @ data
        new_mu = torch.diag(1 / cluster_r) @ cluster_mean
        mu = new_mu

    r = F.softmax(cluster_temp * dist, dim=1)

    return mu, r


class Clusterator(nn.Module):
    def __init__(self, nout, K):
        super(Clusterator, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.K = K
        self.nout = nout

        self.init = torch.rand(self.K, nout)

    def forward(self, embeds, cluster_temp, num_iter=10):
        mu_init, _ = cluster(embeds, self.K, 1, num_iter, cluster_temp=torch.tensor(cluster_temp), init=self.init)
        mu, r = cluster(embeds, self.K, 1, 1, cluster_temp=torch.tensor(cluster_temp), init=mu_init.clone().detach())

        return mu, r
