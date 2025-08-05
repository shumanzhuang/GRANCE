import torch
import yaml
from sklearn.metrics.pairwise import cosine_similarity
import logging
import argparse
import random
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from deeprobust.graph.utils import *
from cluster import Clusterator
import scipy.sparse as ss
import configparser


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def adj_norm(adj, neighbor_only=False):
    if not neighbor_only:
        adj = torch.add(torch.eye(adj.shape[0]).cuda(), adj)
    if adj.is_sparse:
        degree = adj.to_dense().sum(dim=1)
    else:
        degree = adj.sum(dim=1)
    in_degree_norm = torch.pow(degree.view(1, -1), -0.5).expand(adj.shape[0], adj.shape[0])
    in_degree_norm = torch.where(torch.isinf(in_degree_norm), torch.full_like(in_degree_norm, 0), in_degree_norm)
    out_degree_norm = torch.pow(degree.view(-1, 1), -0.5).expand(adj.shape[0], adj.shape[0])
    out_degree_norm = torch.where(torch.isinf(out_degree_norm), torch.full_like(out_degree_norm, 0), out_degree_norm)
    adj = sparse_dense_mul(adj, in_degree_norm)
    adj = sparse_dense_mul(adj, out_degree_norm)
    return adj


def sparse_dense_mul(s, d):
    if not s.is_sparse:
        return s * d
    i = s._indices()
    v = s._values()
    dv = d[i[0, :], i[1, :]]
    return torch.sparse.FloatTensor(i, v * dv, s.size())


def evaluate(model, adj, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        test_labels = labels[mask]
        _, indices = logits.max(dim=1)
        correct = torch.sum(indices == test_labels)
        return correct.item() * 1.0 / test_labels.shape[0]


def get_reliable_neighbors(adj, features, k, degree_threshold):
    degree = adj.sum(dim=1)
    degree_mask = degree > degree_threshold
    assert degree_mask.sum().item() >= k
    sim = cosine_similarity(features.to('cpu'))
    sim = torch.FloatTensor(sim).to('cuda')
    sim[:, degree_mask == False] = 0
    _, top_k_indices = sim.topk(k=k, dim=1)
    for i in range(adj.shape[0]):
        adj[i][top_k_indices[i]] = 1
        adj[i][i] = 0
    return

def aug_random_edge(input_adj, adj_delete, recover_percent=0.2):
    percent = recover_percent
    adj_delete = sp.tril(adj_delete)
    row_idx, col_idx = adj_delete.nonzero()
    edge_num = int(len(row_idx))
    add_edge_num = int(edge_num * percent)
    print("the number of recovering edges: {:04d}" .format(add_edge_num))
    aug_adj = input_adj.tolil()
    edge_list = [(i, j) for i, j in zip(row_idx, col_idx)]  # List of removable edges
    add_idx = random.sample(range(edge_num), add_edge_num)  # Randomly select edges to recover

    # Recover the selected edges
    for i in add_idx:
        aug_adj[edge_list[i][0], edge_list[i][1]] = 1
        aug_adj[edge_list[i][1], edge_list[i][0]] = 1

    aug_adj = aug_adj.tocsr()
    return aug_adj

def preprocess_adj(features, adj, logger, metric='similarity', threshold=0.03, recover_percent=0.2 ,jaccard=True):
    """Drop dissimilar edges.(Faster version using numba)
    """
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)

    adj_triu = sp.triu(adj, format='csr')

    if sp.issparse(features):
        features = features.todense().A  # make it easier for njit processing

    if metric == 'distance':
        removed_cnt = dropedge_dis(adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=threshold)
    else:
        if jaccard:
            removed_cnt = dropedge_jaccard(adj_triu.data, adj_triu.indptr, adj_triu.indices, features,
                                           threshold=threshold)
        else:
            removed_cnt = dropedge_cosine(adj_triu.data, adj_triu.indptr, adj_triu.indices, features,
                                          threshold=threshold)
    logger.info('removed %s edges in the original graph' % removed_cnt)
    modified_adj = adj_triu + adj_triu.transpose()
    adj_delete = adj - modified_adj
    aug_adj1 = aug_random_edge(adj, adj_delete=adj_delete, recover_percent=recover_percent)
    return modified_adj, aug_adj1

def dropedge_dis(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            C = np.linalg.norm(features[n1] - features[n2])
            if C > threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1

    return removed_cnt


def dropedge_both(A, iA, jA, features, threshold1=2.5, threshold2=0.01):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            C1 = np.linalg.norm(features[n1] - features[n2])

            a, b = features[n1], features[n2]
            inner_product = (a * b).sum()
            C2 = inner_product / (np.sqrt(np.square(a).sum() + np.square(b).sum())+ 1e-6)
            if C1 > threshold1 or threshold2 < 0:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1

    return removed_cnt


def dropedge_jaccard(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            n1 = row
            n2 = jA[i]
            a, b = features[n1], features[n2]
            intersection = np.count_nonzero(a*b)
            J = intersection * 1.0 / (np.count_nonzero(a) + np.count_nonzero(b) - intersection)

            if J < threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1
    return removed_cnt


def dropedge_cosine(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            n1 = row
            n2 = jA[i]
            a, b = features[n1], features[n2]
            inner_product = (a * b).sum()
            C = inner_product / (np.sqrt(np.square(a).sum()) * np.sqrt(np.square(b).sum()) + 1e-8)
            if C <= threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1
    return removed_cnt


def sparse_mx_to_sparse_tensor(sparse_mx):
    """sparse matrix to sparse tensor matrix(torch)
    Args:
        sparse_mx : scipy.sparse.csr_matrix
            sparse matrix
    """
    sparse_mx_coo = sparse_mx.tocoo().astype(np.float32)
    sparse_row = torch.LongTensor(sparse_mx_coo.row).unsqueeze(1)
    sparse_col = torch.LongTensor(sparse_mx_coo.col).unsqueeze(1)
    sparse_indices = torch.cat((sparse_row, sparse_col), 1)
    sparse_data = torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparse_indices.t(), sparse_data, torch.Size(sparse_mx.shape))


def to_tensor(adj, features, labels=None, device='cpu'):
    """Convert adj, features, labels from array or sparse matrix to
    torch Tensor on target device.
    Args:
        adj : scipy.sparse.csr_matrix
            the adjacency matrix.
        features : scipy.sparse.csr_matrix
            node features
        labels : numpy.array
            node labels
        device : str
            'cpu' or 'cuda'
    """
    if sp.issparse(adj):
        adj = sparse_mx_to_sparse_tensor(adj)
    else:
        adj = torch.FloatTensor(adj)
    if sp.issparse(features):
        features = sparse_mx_to_sparse_tensor(features)
    else:
        features = torch.FloatTensor(np.array(features))

    if labels is None:
        return adj.to(device), features.to(device)
    else:
        labels = torch.LongTensor(labels)
        return adj.to(device), features.to(device), labels.to(device)

def fea_to_tensor(features, device='cpu'):
    """Convert adj, features, labels from array or sparse matrix to
    torch Tensor on target device.
    Args:
        adj : scipy.sparse.csr_matrix
            the adjacency matrix.
        features : scipy.sparse.csr_matrix
            node features
        labels : numpy.array
            node labels
        device : str
            'cpu' or 'cuda'
    """
    if sp.issparse(features):
        features = sparse_mx_to_sparse_tensor(features)
    else:
        features = torch.FloatTensor(np.array(features))

    return features.to(device)

def idx_to_mask(idx, nodes_num):
    """Convert a indices array to a tensor mask matrix
    Args:
        idx : numpy.array
            indices of nodes set
        nodes_num: int
            number of nodes
    """
    mask = torch.zeros(nodes_num)
    mask[idx] = 1
    return mask.bool()


def is_sparse_tensor(tensor):
    """Check if a tensor is sparse tensor.
    Args:
        tensor : torch.Tensor
                 given tensor
    Returns:
        bool
        whether a tensor is sparse tensor
    """
    # if hasattr(tensor, 'nnz'):
    if tensor.layout == torch.sparse_coo:
        return True
    else:
        return False


def to_scipy(tensor):
    """Convert a dense/sparse tensor to scipy matrix"""
    if is_sparse_tensor(tensor):
        values = tensor._values()
        indices = tensor._indices()
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)
    else:
        indices = tensor.nonzero().t()
        values = tensor[indices[0], indices[1]]
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)

def data_split(nnodes, Y):
    np.random.seed(15)
    idx = np.arange(nnodes)
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=None,
                                                   train_size=0.1 + 0.1,
                                                   test_size=0.8,
                                                   stratify=Y)

    if Y is not None:
        stratify = Y[idx_train_and_val]

    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=None,
                                          train_size=0.5,
                                          test_size=0.5,
                                          stratify=stratify)
    return idx_train, idx_val, idx_test

def get_transform(normalize_features, transform):
    if transform is not None and normalize_features:
        transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        transform = T.NormalizeFeatures()
    elif transform is not None:
        transform = transform
    return transform

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool).cuda()
    mask[index] = 1
    return mask

def mask_to_index(mask):
    index = torch.where(mask == True)[0].cuda()
    return index

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def set_seed(seed):

    np.random.seed(seed)
    random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
    return seed

def normalize_adj_tensor(adj, sparse=False):
    """Normalize adjacency tensor matrix.
    """
    device = adj.device
    if sparse:
        # warnings.warn('If you find the training process is too slow, you can uncomment line 207 in deeprobust/graph/utils.py. Note that you need to install torch_sparse')
        # TODO if this is too slow, uncomment the following code,
        # but you need to install torch_scatter
        # return normalize_sparse_tensor(adj)
        adj = to_scipy(adj)
        mx = normalize_adj(adj)
        return sparse_mx_to_sparse_tensor(mx).to(device)
    else:
        mx = adj + torch.eye(adj.shape[0]).to(device)
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
    return mx

def normalize_adj(mx):
    """Normalize sparse adjacency matrix,
    A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    Row-normalize sparse matrix

    Parameters
    ----------
    mx : scipy.sparse.csr_matrix
        matrix to be normalized

    Returns
    -------
    scipy.sprase.lil_matrix
        normalized matrix
    """

    # TODO: maybe using coo format would be better?
    if type(mx) is not sp.lil.lil_matrix:
        mx = mx.tolil()
    if mx[0, 0] == 0 :
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx

def accuracy(output, labels):
    """Return accuracy of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels

    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels), preds, labels

def neighborhood_reliability(x, edge_index, nclusters, beta):
    softkmeans = Clusterator(x.shape[1], nclusters)
    centroids1, logits = softkmeans(x, beta)
    probs = F.softmax(logits, dim=1)
    return probs

def sim(z1, z2):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def generate_knn(args, x):
    knn_data = Data()
    sim = F.normalize(x).mm(F.normalize(x).T).fill_diagonal_(0.0)
    dst = sim.topk(args.knn, 1)[1].cuda()
    src = torch.arange(x.size(0)).unsqueeze(1).expand_as(sim.topk(args.knn, 1)[1]).cuda()
    edge_index = torch.stack([src.reshape(-1), dst.reshape(-1)])
    edge_index = to_undirected(edge_index)
    knn_data.x = x.clone()
    knn_data.edge_index = edge_index
    knn_data = knn_data.cuda()
    return knn_data

def FAISS_KNN(args):
    save_direction = './adj_matrix/' + args.dataset + '_' + str(args.ptb_rate) + '/'
    adj = ss.load_npz(save_direction + 'knn' + str(args.knn) + '_adj.npz')
    print("adj: ", adj)
    knn_edge = xx1_to_set(adj)
    return knn_edge

def edge_index_to_sparse(edge_index):
    values = torch.ones(edge_index.shape[1], dtype=torch.float32).cuda()
    size = (edge_index.max().item() + 1, edge_index.max().item() + 1)
    edge_index_sparse = torch.sparse_coo_tensor(edge_index, values, size, dtype=torch.float32).to('cuda:0')
    return edge_index_sparse

def edge_index_sp_to_xx1(adj):
    adj = adj.cpu().coalesce()
    indices = adj.indices().t()
    values = adj.values().t()
    indices_np = indices.numpy()
    values_np = values.numpy()
    adj_sparse = sp.coo_matrix((values_np, (indices_np[:, 0], indices_np[:, 1])))
    return adj_sparse

def xx1_to_set(adj):
    edge_set = {(i, j) for i, j in zip(*adj.nonzero())}
    return edge_set

def add_knn_edges(args,perturbed_adj, features, n_nodes):
    _, features1 = to_tensor(perturbed_adj, features)
    if args.dataset == 'obg':
        knn_edges = FAISS_KNN(args)
    else:
        knn_data = generate_knn(args, features1.to_dense())
        knn_edges = set(zip(knn_data.edge_index[0].tolist(), knn_data.edge_index[1].tolist()))
    adj_pre_edge_index = sparse_mx_to_torch_sparse_tensor(perturbed_adj).coalesce().indices()
    pre_edges = set(zip(adj_pre_edge_index[0].tolist(), adj_pre_edge_index[1].tolist()))
    new_edges = knn_edges - pre_edges
    new_edge_index = [[],[]]
    for edge in new_edges:
        new_edge_index[0].append(edge[0])
        new_edge_index[1].append(edge[1])
    print("add {} edges in the original graph".format(len(new_edges)))
    new_edges_sp = edge_index_to_sparse(torch.tensor(new_edge_index).cuda())
    new_edges_values =  new_edges_sp.coalesce().values().fill_(args.knn_weight)
    values = new_edges_values
    size = (torch.tensor(new_edge_index).max().item() + 1, torch.tensor(new_edge_index).max().item() + 1)
    new_edges_sp = torch.sparse_coo_tensor(torch.tensor(new_edge_index).cuda(), values, size, dtype=torch.float32).to('cuda:0')
    adj_pre_sp = sparse_mx_to_torch_sparse_tensor(perturbed_adj)

    new_edges_indices = new_edges_sp.coalesce().indices().cuda()
    new_edges_values = new_edges_sp.coalesce().values().cuda()
    adj_pre_indices = adj_pre_sp.coalesce().indices().cuda()
    adj_pre_values = adj_pre_sp.coalesce().values().cuda()
    merged_indices = torch.cat((new_edges_indices, adj_pre_indices), dim=1)
    merged_values = torch.cat((new_edges_values, adj_pre_values))
    merged_tensor = torch.sparse_coo_tensor(merged_indices, merged_values, size=(n_nodes, n_nodes))
    merged_tensor = edge_index_sp_to_xx1(merged_tensor)
    return merged_tensor

def split_origin_knn(args, adj_temp, device):
    adj_temp_SP = sparse_mx_to_torch_sparse_tensor(to_scipy(adj_temp))
    edge_index_temp = adj_temp_SP.coalesce().indices().to(device)
    values = adj_temp_SP.coalesce().values().to(device)
    mask_1 = values == 1.0
    edge_index = edge_index_temp[:, mask_1]
    mask_knn = values == args.knn_weight
    edge_index_knn= edge_index_temp[:, mask_knn]
    return edge_index, edge_index_knn


def SparseTensor_to_xx1(adj):
    row, col, val = adj.coo()
    formatted_result = ["adj\n"]
    for r, c, v in zip(row, col, val):
        formatted_result.append(f"  ({r.item()}, {c.item()})\t{v.item():.4f}")

    return "\n".join(formatted_result)

def load_config(args):
    conf = configparser.ConfigParser()
    config_path = './config/config_{}.ini'.format(args.dataset)
    conf.read(config_path)
    ptb = str(args.ptb_rate)
    args.eps = conf.getfloat(ptb, 'eps')
    args.threshold = conf.getfloat(ptb, 'threshold')
    args.knn_weight = conf.getfloat(ptb, 'knn_weight')
    args.nclusters = conf.getint(ptb, 'nclusters')
    return args