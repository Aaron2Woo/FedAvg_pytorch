import os
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset

import math
from datasets import CIFAR10_truncated
from PIL import Image

import logging


torch.backends.cudnn.benchmark = True



"""
---------------------------Creating desired data distribution among clients ------------------------------------
"""
# Image augmentation
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),  # 现在图像上下左右填充4行，cifar-10中图像由32*32变为40*40，再随机裁剪为32*32
    transforms.RandomHorizontalFlip(),     # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# Loading CIFAR10 using torchvision.datasets
traindata = datasets.CIFAR10('./data', train=True, download=True,
                             transform=transform_train)

# Normalizing the test images
transform_test = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# =============================================================================
# 1.label skew Non-IID
# =============================================================================
n_parties = 100
batch_size = 128
beta = 0.5          # beta
min_require_size = 20
K = 10              # 数据集分类类别数
N = 50000           # All number of samples in training dataset

y_train = np.array(traindata.targets)
min_size = 0


np.random.seed(2020)
net_dataidx_map = {}

while min_size < min_require_size:
    idx_batch = [[] for _ in range(n_parties)]
    for k in range(K):
        idx_k = np.where(y_train == k)[0]                                       # 标签为k的数量
        np.random.shuffle(idx_k)                                                # 随机打乱标签为k的数据的排列
        proportions = np.random.dirichlet(np.repeat(beta, n_parties))           # dirichlet 分布
        ## Balance
        proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))] # 写法经典
        min_size = min([len(idx_j) for idx_j in idx_batch])
        
for j in range(n_parties):
    np.random.shuffle(idx_batch[j])
    net_dataidx_map[j] = idx_batch[j]


dl_obj = CIFAR10_truncated

# traindata = dl_obj('./data', dataidxs=dataidxs, train=True, transform=transform_train, download=True)
traindata = [dl_obj('./Non_IID/label_skew_NonIID data', dataidxs=dataidxs, train=True, download=True, transform=transform_train) for _, dataidxs in net_dataidx_map.items()]
train_loader = [torch.utils.data.DataLoader(traindata_, batch_size=batch_size, shuffle=True, drop_last=False) for traindata_ in traindata]

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./Non_IID/label_skew_NonIID data', train=False, transform=transform_test), batch_size=1000, shuffle=True)