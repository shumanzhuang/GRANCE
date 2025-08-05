import torch
import torch_geometric.transforms as T
import numpy as np
from deeprobust.graph.data import Dataset as DeepRobust_Dataset
from deeprobust.graph.data import PrePtbDataset as DeepRobust_PrePtbDataset
from torch_geometric.data import Data
import argparse
from utils import mask_to_index
from utils import str2bool
from load_data_new import load_new_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'pubmed', "obg", 'BlogCatalog'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--normalize_features', type=str2bool, default=True)
args = parser.parse_args()

def get_dataset(args, sparse=True):
    if sparse:
        transform = T.ToSparseTensor()
    else:
        transform = None

    assert args.dataset in ["cora", "cora_ml", "citeseer", "BlogCatalog"], 'dataset not supported'
    dataset, perturbed_data= get_adv_dataset(args.dataset, args.normalize_features, transform=transform, ptb_rate=args.ptb_rate,
                              args=args)

    data = dataset.data

    split_idx = {}
    split_idx['train'] = mask_to_index(data.train_mask)
    split_idx['valid'] = mask_to_index(data.val_mask)
    split_idx['test'] = mask_to_index(data.test_mask)

    return dataset, data, split_idx, perturbed_data


def get_adv_dataset(name, normalize_features=False, transform=None, ptb_rate=0.05, args=None):
    if name in ['BlogCatalog']:
        dataset = load_new_dataset(f"./datasets/{name}.npz")
    else:
        dataset = DeepRobust_Dataset(root='./datasets', name=name, setting='nettack', require_mask=True, seed=15)
    dataset.x = torch.FloatTensor(dataset.features.todense())
    dataset.y = torch.LongTensor(dataset.labels)
    dataset.num_classes = dataset.y.max().item() + 1

    if ptb_rate > 0:
        if args.attack == 'mettack':
            perturbed_data = DeepRobust_PrePtbDataset(
                root='./datasets',
                name=name,
                attack_method='meta',
                ptb_rate=ptb_rate)
        edge_index = torch.LongTensor(perturbed_data.adj.nonzero())
    else:
        perturbed_data = dataset
        edge_index = torch.LongTensor(dataset.adj.nonzero())
    data = Data(x=dataset.x, edge_index=edge_index, y=dataset.y)

    clean_edge_index = torch.LongTensor(dataset.adj.nonzero())
    clean_data = Data(x=dataset.x, edge_index=clean_edge_index, y=dataset.y)

    data.train_mask = torch.tensor(dataset.train_mask)
    data.val_mask = torch.tensor(dataset.val_mask)
    data.test_mask = torch.tensor(dataset.test_mask)

    dataset.data = data
    dataset.clean_data = clean_data
    dataset.data.clean_adj = dataset.clean_data.edge_index
    return dataset, perturbed_data

def main():
    dataset, data, split_idx, perturbed_data = get_dataset(args)
    print("perturbed_data: ", perturbed_data.adj)
    idx_train, idx_val, idx_test = dataset.idx_train, dataset.idx_val, dataset.idx_test
    modified_adj = data.edge_index
    #modified_adj = modified_adj.cpu().to_sparse()
    print("modified_adj: ", modified_adj)
    torch.save(modified_adj.cpu().to_sparse(), "./ptb_graphs/%s/%s_%s_%s.pt" % ('mettack', 'mettack', args.dataset, args.ptb_rate))
    np.save("./ptb_graphs/%s/%s_%s_%s_idx_train" % ('mettack', 'mettack', args.dataset, args.ptb_rate), idx_train)
    np.save("./ptb_graphs/%s/%s_%s_%s_idx_val" % ('mettack', 'mettack', args.dataset, args.ptb_rate), idx_val)
    np.save("./ptb_graphs/%s/%s_%s_%s_idx_test" % ('mettack', 'mettack', args.dataset, args.ptb_rate), idx_test)


if __name__ == '__main__':
    main()