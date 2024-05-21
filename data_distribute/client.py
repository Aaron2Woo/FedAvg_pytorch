# -*- coding:utf-8 -*-
import copy
from itertools import chain

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F


# def train(model, optimizer, train_loader, epoch=5):   # 传入的train_loader会指定具体的client
def train(args, model, train_loader):
    """
    This function updates/trains client model on client data
    """
    model.train()
    model.len = len(train_loader.dataset)
    # print(model.len)
    lr = args.lr
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
                                    # momentum=0.9, weight_decay=args.weight_decay)
    # stepLR = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    loss_func = nn.CrossEntropyLoss().cuda()
    for e in range(args.E):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)    # client_model 已定义，见第141行
            # loss = F.nll_loss(output, target)  # import torch.nn.functional as F  # nll_loss与nllloss相同
            loss = loss_func(output, target.long())
            loss.backward()
            optimizer.step()
        # stepLR.step()
        model.train()
    return loss.item()


# def test(global_model, test_loader):
def test(model, test_loader):
    """This function test the global model on test data and returns test loss and test accuracy """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss  # '.item()'取出单元素张量的元素值并返回该值(数值型)，保持原元素类型不变
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability # 找到概率最大的'下标'
            correct += pred.eq(target.view_as(pred)).sum().item()  # a.eq(b) 张量a与张量b之间，在相同位置值相同则返回对应的True,返回的是一个列表
                                                                   # pred.eq(target.view_as(pred))
                                                                   # target.view_as(pred) 将张量target按pred的size输出
                                                                   # pred.eq(target.view_as(pred)).sum()  返回一个True的‘个数’的tensor值 （‘.item’将单个tensor值数值化）
    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)   # len(test_loader.dataset) = 10000
    return test_loss, acc

