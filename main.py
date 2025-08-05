import torch.nn as nn
from utils import *
from copy import deepcopy
from deeprobust.graph.utils import *
from DGCL import get_contrastive_emb
from dataset_adv import get_dataset
from GNN_Classifier import GNN_Classifier
from args import parameter_parser

# Training parameters for classifier
epochs =  200
n_hidden = 32
dropout = 0.5
weight_decay = 5e-4
lr = 1e-2
loss = nn.CrossEntropyLoss()


def train(model, args, optim, adj, run, logger, labels, train_mask, val_mask, test_mask, embeds, verbose=True):
    best_loss_val = 9999
    best_acc_val = 0
    for epoch in range(epochs):
        model.train()
        logits = model(embeds)
        l = loss(logits[train_mask], labels[train_mask])
        optim.zero_grad()
        l.backward()
        optim.step()
        acc = evaluate(model, adj, embeds, labels, val_mask)
        val_loss = loss(logits[val_mask], labels[val_mask])
        if val_loss < best_loss_val:
            best_loss_val = val_loss
            weights = deepcopy(model.state_dict())
        if acc > best_acc_val:
            best_acc_val = acc
            weights = deepcopy(model.state_dict())
        if verbose:
            if epoch % 10 == 0:
                logger.info("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f}"
                      .format(epoch, l.item(), acc))
    model.load_state_dict(weights)
    acc = evaluate(model, adj, embeds, labels, test_mask)
    logger.info("Run {:02d} Test Accuracy {:.4f}".format(run, acc))
    return acc

def main(args):
    if args.log:
        logger = get_logger('./log/' + args.attack + '/' + 'ours_' + args.dataset + '_' + str(args.ptb_rate) + '.log')
    else:
        logger = get_logger('./log/try.log')

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading data
    dataset_, data, split_idx, perturbed_data = get_dataset(args)
    features = to_scipy(dataset_.x)
    labels = np.array(dataset_.y)
    idx_train = dataset_.idx_train
    idx_val = dataset_.idx_val
    idx_test = dataset_.idx_test
    perturbed_adj = perturbed_data.adj

    n_nodes = features.shape[0]
    n_class = labels.max() + 1

    train_mask, val_mask, test_mask = idx_to_mask(idx_train, n_nodes), idx_to_mask(idx_val, n_nodes), \
                                      idx_to_mask(idx_test, n_nodes)
    train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)
    logger.info('train nodes:%d' % train_mask.sum())
    logger.info('val nodes:%d' % val_mask.sum())
    logger.info('test nodes:%d' % test_mask.sum())

    logger.info(args)
    logger.info('===start preprocessing the graph===')
    perturbed_adj_sparse = add_knn_edges(args, perturbed_adj, features, n_nodes)
    adj_pre, aug_adj1 = preprocess_adj(features, perturbed_adj_sparse, logger, threshold=args.threshold)
    features = fea_to_tensor(features)

    logger.info('===start getting contrastive embeddings===')
    embeds = get_contrastive_emb(args, logger, adj_pre, features.unsqueeze(dim=0).to_dense(), aug_adj1,
                                    lr=1e-3, weight_decay=0.0, nb_epochs=500)
    embeds = embeds.squeeze(dim=0)

    acc_total = []

    adj_clean = adj_pre
    adj_clean = sparse_mx_to_sparse_tensor(adj_clean)
    adj_clean = adj_clean.to_dense()
    adj_clean = adj_clean.to(device)
    labels = torch.LongTensor(labels)
    labels = labels.to(device)

    logger.info('===train ours on perturbed graph===')
    num_features = embeds.shape[1]
    for run in range(10):
        adj_temp = adj_clean.clone()
        edge_index, edge_index_knn = split_origin_knn(args, adj_temp, device)
        model = GNN_Classifier(edge_index, edge_index_knn, labels, num_features, n_hidden, n_class, dropout, args.knn_weight, args.eps, layer_num = 2)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        acc = train(model, args, optimizer, adj_temp, run, logger, labels, train_mask, val_mask, test_mask, embeds=embeds, verbose=False)
        acc_total.append(acc)
    logger.info('Mean Accuracy:%f' % np.mean(acc_total))
    logger.info('Standard Deviation:%f' % np.std(acc_total, ddof=1))

    return np.mean(acc_total)

if __name__ == '__main__':
    args = parameter_parser()
    args = load_config(args)
    main(args)

