#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from collections import OrderedDict
import copy
import math
from typing import Dict, List
import numpy as np
import torch
from torch import nn


def FedAvg(w:Dict,indexWeight:Dict = None):
    userIdx = list(w.keys())
    totWeight = sum(indexWeight.values())
    w_avg = copy.deepcopy(w[userIdx[0]])
    for k in w_avg:
        w_avg[k] = torch.zeros_like(w[userIdx[0]][k])
    for k in w_avg.keys():
        for i in userIdx:
            w_avg[k] += w[i][k]*(indexWeight[i]/totWeight)
    return w_avg
    w_avg = copy.deepcopy(w[userIdx[0]])
    for k in w_avg.keys():
        for i in userIdx[1:]:
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg



def FedSlice(userParamDict:Dict,indexWeight:Dict = None,randFlag = False):
    if indexWeight == None:
        indexWeight = {idx:1 for idx in userParamDict.keys()}
    if not randFlag:
        userIdx = list(userParamDict.keys())
        allKeyAndIdxList = list()
        w_avg = OrderedDict()
        for k in userParamDict[userIdx[0]].keys():
            w_avg[k] = torch.zeros_like(userParamDict[userIdx[0]][k])
        for k in w_avg.keys():
            assert len(w_avg[k].size()) != 3
            if len(w_avg[k].size()) == 1:
                for i in range(w_avg[k].size()[0]):
                    allKeyAndIdxList.append((k,i))
            elif len(w_avg[k].size()) == 2:
                for i in range(w_avg[k].size()[0]):
                    for j in range(w_avg[k].size()[1]):
                        allKeyAndIdxList.append((k,i,j))
            elif len(w_avg[k].size()) == 4:
                for firstDegree in range(w_avg[k].size()[0]):
                    for secondDegree in range(w_avg[k].size()[1]):
                        for i in range(w_avg[k].size()[2]):
                            for j in range(w_avg[k].size()[3]):
                                allKeyAndIdxList.append((k,firstDegree,secondDegree,i,j))#two dimention to one dimention 
        totIndexLenth = len(allKeyAndIdxList)
        totWeight = sum(indexWeight.values())
        for userId in userParamDict:
            choiceOnce = (int(totIndexLenth*indexWeight[userId]/totWeight))+1
            if choiceOnce>=len(allKeyAndIdxList):
                choiceOnce = len(allKeyAndIdxList)
            keyAndIndexToFill = np.random.choice(allKeyAndIdxList,choiceOnce,replace = False)
            allKeyAndIdxList =  list(set(allKeyAndIdxList) - set(keyAndIndexToFill))
            for arr in keyAndIndexToFill:
                if len(arr) == 2:
                    k,i = arr[0],arr[1]
                    w_avg[k][i] = userParamDict[userId][k][i]
                elif len(arr)  == 3:
                    k,i,j = arr[0],arr[1],arr[2]
                    w_avg[k][i][j] = userParamDict[userId][k][i][j]
                else:
                    k,r,c,i,j = arr[0],arr[1],arr[2],arr[3],arr[4]
                    w_avg[k][r][c][i][j] = userParamDict[userId][k][r][c][i][j]
        if len(allKeyAndIdxList) != 0:
            raise KeyError("The remaining parameters for these indexes are not loaded: "+str(allKeyAndIdxList))
        return w_avg

def get_corr(X:torch.Tensor, Y:torch.tensor)->torch.tensor:
        X, Y = X.reshape(-1), Y.reshape(-1)
        X_mean, Y_mean = torch.mean(X), torch.mean(Y)
        corr = (torch.sum((X - X_mean) * (Y - Y_mean))) / (
                    torch.sqrt(torch.sum((X - X_mean) ** 2)) * torch.sqrt(torch.sum((Y - Y_mean) ** 2)).add(0.0000000001))
        return corr
