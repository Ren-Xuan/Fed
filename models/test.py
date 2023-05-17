#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import math
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

def test_img(net_g, datatest, args,frac = 1,batch_size = 64):
    with torch.no_grad():
        net_g.eval()
        # testing
        test_loss = 0
        correct = 0
        data_loader = DataLoader(datatest, batch_size=batch_size)
        l = len(data_loader)
        cnt = 0
        for idx, (data, target) in enumerate(data_loader):
            if random.random()>frac:
                continue
            data, target = data.to(args.device), target.to(args.device)
            log_probs = net_g(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            cnt+=list(target.size())[0]
        if cnt == 0:
            return -1,-1
        test_loss /= cnt
        accuracy = 100.00 * correct / cnt
        if args.verbose:
            print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, correct, cnt, accuracy))
    return accuracy, test_loss

def test_agnews(net_g, datatest,frac = 1,collate_batch = None):
    total_acc, total_count = 0, 0
    data_loader = DataLoader(datatest, batch_size=16, shuffle=False, collate_fn=collate_batch)
    with torch.no_grad():
        net_g.eval()
        for idx, (textAndOff, label) in enumerate(data_loader):
            text,offsets = textAndOff[0],textAndOff[1]
            predited_label = net_g(text, offsets)
            # loss = criterion(predited_label, label)
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return 100*total_acc / total_count,100*total_acc / total_count
