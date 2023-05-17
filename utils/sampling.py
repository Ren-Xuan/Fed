#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from collections import defaultdict
import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users,size = None):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items =  size if size!=None else int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        if len(all_idxs)<num_items: 
            dict_users[i] = set(np.random.choice([i for i in range(len(dataset))], num_items, replace=False))
        else:
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users,{i:len(dict_users[i]) for i in dict_users}
def agnews_iid(dataset_lenth, num_users,size = None):
    """
    Sample I.I.D. client data from Agnews dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = size if size!=None else int(dataset_lenth/num_users)
    dict_users, all_idxs = {}, [i for i in range(dataset_lenth)]
    for i in range(num_users):
        if len(all_idxs)<num_items: 
            dict_users[i] = set(np.random.choice([i for i in range(dataset_lenth)], num_items, replace=False))
        else:
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users,{i:len(dict_users[i]) for i in dict_users}
def cifar_iid(dataset, num_users,size = None):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = size if size!=None else int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        if len(all_idxs)<num_items: 
            dict_users[i] = set(np.random.choice([i for i in range(len(dataset))], num_items, replace=False))
        else:
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users,{i:len(dict_users[i]) for i in dict_users}
def mnist_noniid(dataset, num_users,loc = 50,scale = 25):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dataset_lenth = len(dataset)
    dict_users, all_idxs = {}, [i for i in range(dataset_lenth)]
    idxs_size = defaultdict(int)
    for i in range(num_users):
        randSize = np.random.normal(loc=loc, scale=scale)
        if randSize<=1:randSize = 1
        idxs_size[i] = int(randSize)
    for i in range(num_users):
        if len(all_idxs)<idxs_size[i]: 
            dict_users[i] = set(np.random.choice([i for i in range(len(dataset))], idxs_size[i], replace=False))
        else:
            dict_users[i] = set(np.random.choice(all_idxs, idxs_size[i], replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users,{i:len(dict_users[i]) for i in dict_users}



def cifar_noniid(dataset, num_users,loc = 50,scale = 25):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dataset_lenth = len(dataset)
    dict_users, all_idxs = {}, [i for i in range(dataset_lenth)]
    idxs_size = defaultdict(int)
    for i in range(num_users):
        randSize = np.random.normal(loc=loc, scale=scale)
        if randSize<=1:randSize = 1
        idxs_size[i] = int(randSize)
    for i in range(num_users):
        if len(all_idxs)<idxs_size[i]: 
            dict_users[i] = set(np.random.choice([i for i in range(len(dataset))], idxs_size[i], replace=False))
        else:
            dict_users[i] = set(np.random.choice(all_idxs, idxs_size[i], replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users,{i:len(dict_users[i]) for i in dict_users}
def agnews_noniid(dataset_lenth, num_users,loc = 50,scale = 25):
    """
    Sample I.I.D. client data from Agnews dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users, all_idxs = {}, [i for i in range(dataset_lenth)]
    idxs_size = defaultdict(int)
    for i in range(num_users):
        randSize = np.random.normal(loc=loc, scale=scale)
        if randSize<=1:randSize = 1
        idxs_size[i] = int(randSize)
    for i in range(num_users):
        if len(all_idxs)<idxs_size[i]: 
            dict_users[i] = set(np.random.choice([i for i in range(dataset_lenth)], idxs_size[i], replace=False))
        else:
            dict_users[i] = set(np.random.choice(all_idxs, idxs_size[i], replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users,{i:len(dict_users[i]) for i in dict_users}
if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
