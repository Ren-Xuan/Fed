#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fedslice', action='store_true', help='use fedslice or not')
    # federated arguments
    parser.add_argument('--randFlag', action='store_true', help='use randFlag algorithm')
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")

    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")

    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0, help="SGD momentum (default: 0.0)")
    
    parser.add_argument('--save', action='store_true', help="save the local state dict for each client")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--dataset_size', type=int, default=None, help='training size of each client')

    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    
    parser.add_argument('--scale', type=int, default=25, help='nd distribution')
    parser.add_argument('--loc', type=int, default=50, help='nd distribution')
    
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")

    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    args = parser.parse_args()
    return args
