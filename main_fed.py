#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from collections import defaultdict
import datetime
import math
from random import randint
import matplotlib

from models.test import test_agnews, test_img

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from utils.sampling import agnews_iid, agnews_noniid, cifar_noniid, mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import BasicBlock, CNNCifar, ResidualNet, SimpleCNN
from models.Fed import FedAvg, FedSlice


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args.device)
    print(args)
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users,dict_usersWeight = mnist_iid(dataset_train, args.num_users,size=args.dataset_size)
        else:
            dict_users,dict_usersWeight = mnist_noniid(dataset_train, args.num_users,loc = args.loc,scale=args.scale)
    elif 'fashion' in args.dataset:
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('../data/fashion-mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST('../data/fashion-mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users,dict_usersWeight = mnist_iid(dataset_train, args.num_users,size=args.dataset_size)
        else:
            dict_users,dict_usersWeight = mnist_noniid(dataset_train, args.num_users,loc = args.loc,scale=args.scale)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users,dict_usersWeight = cifar_iid(dataset_train, args.num_users,size=args.dataset_size)
        else:
            dict_users,dict_usersWeight = cifar_noniid(dataset_train, args.num_users,loc = args.loc,scale=args.scale)
    elif args.dataset == 'agnews':
        import torch
        from torchtext.datasets import AG_NEWS
    
        train_iter = AG_NEWS(root='../data', split='train')      
        test_iter = AG_NEWS(root='../data', split='test')
        dataset_train = [(line,label) for label,line in train_iter]
        dataset_test = [(line,label) for label,line in test_iter]
        from torchtext.data.utils import get_tokenizer
        from collections import Counter
        from torchtext.vocab import vocab

        tokenizer = get_tokenizer('basic_english')      
        counter = Counter()
        for line,label in dataset_train:
            counter.update(tokenizer(line))
        for line,label in dataset_test:
            counter.update(tokenizer(line))
        vocab = vocab(counter, min_freq=1)
        text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]    
        label_pipeline = lambda x: int(x) - 1       


        def collate_batch(batch):
            label_list, text_list, offsets = [], [], [0]
            for (_text,_label) in batch:
                label_list.append(label_pipeline(_label))
                processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)      # torch.Size([41]), torch.Size([58])...
                text_list.append(processed_text)
                offsets.append(processed_text.size(0))

            label_list = torch.tensor(label_list, dtype=torch.int64)        # torch.Size([64])
            offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)      # torch.Size([64])
            text_list = torch.cat(text_list)        
            return (text_list.to(args.device), offsets.to(args.device)),label_list.to(args.device)
        dataset_length = len(dataset_train)
        dataset_test = list(dataset_test)
        if args.iid:
            dict_users,dict_usersWeight = agnews_iid(dataset_length, args.num_users,size=args.dataset_size)
        else:
            dict_users,dict_usersWeight = agnews_noniid(dataset_length, args.num_users,loc = args.loc,scale=args.scale)
        print('each client data lenth',len(dict_users[0]))
        print(len(dataset_train))
        print(len(dataset_test))
    else:
        exit('Error: unrecognized dataset')
    # build model
    if args.model == 'cnn' :
        net_glob = SimpleCNN().to(args.device)
    elif args.model == 'resnet' and args.dataset == 'cifar':
        net_glob = ResidualNet(BasicBlock, [2, 2,2, 2]).to(args.device)
    elif args.model == 'lstm' and 'agnews' in args.dataset:
        from models.LSTM import LSTMAttnGRU
        num_class = 4
        net_glob = LSTMAttnGRU(4).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net_glob.parameters())))
    net_glob.train()
    print('-----------Train/Test{:.3f}------------------'.format(len(dataset_train)/len(dataset_test)))
    print(len(dataset_train),'/',len(dataset_test))
    # copy weights
    w_glob = net_glob.state_dict()
    net_glob.load_state_dict(w_glob)
    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    acc_tests = []
    print(dict_usersWeight)
    print("each client has data len:",[i for i in dict_usersWeight.values()][:16],"(display up to 16 elements)")
    tmpAccFileName = './save/'+str(randint(0,10000))
    print("TempAccTestFileName:",tmpAccFileName)
    for iter in range(args.epochs):
        loss_locals = dict()
        w_locals = dict()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals[idx] = copy.deepcopy(w)
            loss_locals[idx] = copy.deepcopy(loss)
        # update global weights
        if args.fedslice:
            w_glob =FedSlice(w_locals,indexWeight= {idx:dict_usersWeight[idx] for idx in idxs_users})
        else:
            w_glob = FedAvg(w_locals,indexWeight= {idx:dict_usersWeight[idx] for idx in idxs_users})

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        loss_avg = sum(loss_locals.values()) / len(loss_locals)
        if args.dataset == 'agnews':
            acc_test, loss_test = test_agnews(net_glob, dataset_test,collate_batch = collate_batch)
            #acc_test = acc_test.tolist()
            print('Round {:3d}, Average loss {:.3f}, Acc {:.3f}'.format(iter, loss_test,acc_test)+"\t:"+datetime.datetime.now().strftime('%H:%M %m-%d'))
        else:
            acc_test, loss_test = test_img(net_glob, dataset_test, args,frac=0.2)
            acc_test = acc_test.tolist()
            print('Round {:3d}, Average loss {:.3f}, Acc {:.3f}'.format(iter, loss_test,acc_test)+"\t:"+datetime.datetime.now().strftime('%H:%M %m-%d'))
        loss_train.append(loss_test)
        acc_tests.append(acc_test)
        with open(tmpAccFileName+".txt",'w',encoding='UTF-8') as fw:
            for e in acc_tests:
                fw.writelines(str(e)+",")
            fw.writelines("\n")
            for e in loss_train:
                fw.writelines(str(e)+",")
            fw.writelines("\n"+str(args))
