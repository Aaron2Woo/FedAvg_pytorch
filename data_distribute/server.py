# -*- coding:utf-8 -*-
import copy
import random

import numpy as np
import torch
from tqdm import tqdm

from model import VGG
from client import train, test
from get_data import train_loader, test_loader

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('result')

class FedAvg:
    def __init__(self, args):
        self.args = args
        self.nn = VGG(args.model).to(args.device)
        self.nns = []
        for i in range(self.args.K):
            temp = copy.deepcopy(self.nn)
            # temp.name = self.args.clients[i]    # e: Task1_W_Zone5 (i==5)
            self.nns.append(temp)

    def server(self):
        for t in tqdm(range(self.args.r)):
            print('round', t + 1, ':')
            # sampling
            m = np.max([int(self.args.C * self.args.K), 1])
            index = random.sample(range(0, self.args.K), m)  # st

            # dispatch
            self.dispatch(index)
            # local updating
            self.client_update(index)

            self.client_model_test(index)

            # aggregation
            # self.aggregation(index)
            self.server_aggregate(index)
            # test global model
            loss_g, acc_g = self.global_model_test()
            writer.add_scalar('global test acc', acc_g, t)
            writer.add_scalar('global test loss', loss_g, t)

        return self.nn


    def server_aggregate(self, index):
        """
        This function has aggregation method 'mean'
        """
        s = 0
        for j in index:
            # normal
            s += self.nns[j].len

        global_dict = self.nn.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.stack([self.nns[i].state_dict()[k].float() * (self.nns[i].len / s) for i in index],0).sum(0)
        self.nn.load_state_dict(global_dict)
        for model in self.nns:
            model.load_state_dict(self.nn.state_dict())


    def dispatch(self, index):
        for j in index:
            for old_params, new_params in zip(self.nns[j].parameters(), self.nn.parameters()):
                old_params.data = new_params.data.clone()

    def client_update(self, index):  # update nn
        for k in index:
            # self.nns[k] = train(self.args, self.nns[k], train_loader[k])
            train(self.args, self.nns[k], train_loader[k])

    def client_model_test(self, index):
        for i in index:
            loss, acc = test(self.nns[i], test_loader)
            # print('the {i}-th acc is: %0.3g | test loss is: %0.3g' % (acc, loss))
            print(f'the {i}-th acc is {acc} | test loss is {loss}')

    def global_model_test(self):
        loss, acc = test(self.nn, test_loader)
        print('the acc is: %0.3g | test loss is: %0.3g' % (acc, loss))
        # writer.add_scalar('global test loss', acc, epoch * len(trainloader) + i)
        return loss, acc