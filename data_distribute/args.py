import argparse
import torch


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--E', type=int, default=5, help='number of rounds of training')
    parser.add_argument('--r', type=int, default=300, help='number of communication rounds')
    parser.add_argument('--K', type=int, default=100, help='number of total clients')
    parser.add_argument('--input_dim', type=int, default=28, help='input dimension')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--C', type=float, default=0.1, help='sampling rate')
    parser.add_argument('--B', type=int, default=50, help='local batch size')
    parser.add_argument('--mu', type=float, default=0.01, help='proximal term constant')
    parser.add_argument('--optimizer', type=str, default='sgd', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay per global round')
    clients = ['Task1_W_Zone' + str(i) for i in range(1, 11)]
    parser.add_argument('--clients', default=clients)
    parser.add_argument('--model', type=str, default='VGG16', help='model name')

    args = parser.parse_args()

    return args