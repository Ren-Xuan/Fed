#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random

import torch.nn.functional as F

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs,randLabel = False):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.randLabel = randLabel
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None,randLabel = False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss().to(self.args.device)
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True,drop_last=False,num_workers= 10)
        if len(self.ldr_train) == 0:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=1, shuffle=True,drop_last=True,num_workers= 10)
        self.randLabel = randLabel
    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
        
                if self.randLabel:
                    labelUpperBound = labels.max().tolist()
                    labels = torch.LongTensor([random.randint(0,labelUpperBound) for _ in labels.tolist()])
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

