# -*- coding:utf-8 -*-
from args import args_parser
from server import FedAvg


def main():
    args = args_parser()
    FedAvg = FedAvg(args)
    FedAvg.server()
    FedAvg.global_test()


if __name__ == '__main__':
    main()
