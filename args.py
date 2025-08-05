import argparse
from utils import  str2bool

def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument('--attack', type=str, default='mettack', help='attack method')
    parser.add_argument('--ptb_rate', type=float, default=0.05, help='pertubation rate')
    parser.add_argument('--beta', type=float, default=100, help='inverse-temperature hyperparameter of soft-cluster')
    parser.add_argument("--log", action='store_true', help='run prepare_data or not')
    parser.add_argument('--normalize_features', type=str2bool, default=True)
    parser.add_argument('--num_hidden', type=int, default=16)
    parser.add_argument("--niter", type=int, default=20, help='Number of iteration for kmeans.')
    parser.add_argument('--mean', action="store_true", help='Calculate mean for neighbor pos')
    parser.add_argument('--knn', type=int, default=20)
    parser.add_argument('--seed', type=int, default=15)

    args = parser.parse_args()
    return args