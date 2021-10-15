#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from sampling import *
import pandas as pd
import numpy as np
import torch
import copy
from torch.utils import data as Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.nn import init
from torch import nn


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
            
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
    
            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)
    
            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)
            
        else:
            data_dir = '../data/fmnist/'

            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
    
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)
    
            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)

    # sample training data amongst users
    if args.iid:
        # Sample IID user data from Mnist
        user_groups = data_iid(train_dataset, args.num_users)
    else:
        # Sample Non-IID user data from Mnist
        if args.unequal:
            # Chose uneuqal splits for every user
            user_groups = data_noniid_unbalanced(train_dataset, args.num_users) #raise NotImplementedError()
        else:
            # Chose euqal splits for every user
            user_groups = data_noniid_split5(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def get_data(args):
    if args.dataset == 'sl_a':
        # Import Dataset A1/2/3
        #df1 = pd.read_table('../data/A/dataset_A1_RMA.txt')
        # df1 = df1.T
        # info1 = pd.read_table('../data/A/dataset_A1_annotation.txt')

        df2 = pd.read_table('../data/A/dataset_A2_RMA.txt')
        df2 = df2.T
        info2 = pd.read_table('../data/A/dataset_A2_annotation.txt')

        # df3 = pd.read_table('../data/A/dataset_A3_DESeq2.txt')
        # df3 = df3.T
        # info3 = pd.read_table('../data/A/dataset_A3_annotation.txt')

        # Combine the RMA data with the annotation
        # dt1 = df1.join(info1)
        # dt1 = dt1.drop(columns=['Dataset', 'GSE', 'Disease', 'Tissue', 'FAB', 'Filename', 'FAB_all'])
        # dt1.Condition = dt1.Condition.map(dict(CASE = 1, CONTROL = 0))
        # dt1 = dt1.dropna()
        # print(dt1)
        # dt1.head()

        dt2 = df2.join(info2)
        dt2 = dt2.drop(columns=['Dataset', 'GSE', 'Disease', 'Tissue', 'FAB', 'Filename', 'FAB_all'])
        dt2.Condition = dt2.Condition.map(dict(CASE = 1, CONTROL = 0))
        dt2 = dt2.dropna()
        # print(dt2)
        # dt2.head()

        # dt3 = df3.join(info3)
        # dt3 = dt3.drop(columns=['Dataset', 'GSE', 'Disease', 'Tissue', 'FAB', 'Filename'])
        # dt3.Condition = dt3.Condition.map(dict(CASE = 1, CONTROL = 0))
        # dt3 = dt3.dropna()
        # print(dt3)
        # dt3.head()

        # Combine all the dataset
        # dt = dt1.append(dt2)
        dt = dt2
        # dt = dt.append(dt3)
        # print(dt)
        # dt.head()


    if args.dataset == 'sl_b':
        # Import Dataset B
        df = pd.read_table('../data/B/dataset_B_DESeq2_ranktransformed_FG.txt')
        df = df.T
        info = pd.read_table('../data/B/dataset_B_annotation_FG.txt')
        info = pd.DataFrame(info)
        info.Condition = info.Condition.map(dict(CASE=1, CONTROL=0))
        info = info.drop(columns=['ID'])
        # print(info)
        # info.head()

        df = df.reset_index(drop=True)
        dt = df.join(info)
        # print(dt)
        # dt.head()


    if args.dataset == 'sl_d':
        # Import Dataset D
        df = pd.read_table('../data/D/dataset_D_DESeq2_ranktransformed_FG.txt')
        df = df.T
        info = pd.read_table('../data/D/dataset_D_annotation_FG.txt')
        info = pd.DataFrame(info)
        info.Condition = info.Condition.map(dict(CASE=1, CONTROL=0))
        info = info.drop(columns=['ID'])
        # print(info)
        # info.head()

        df = df.reset_index(drop=True)
        dt = df.join(info)
        # print(dt)
        # dt.head()

    if args.dataset == 'sl_e':
        # Import Dataset E
        df = pd.read_table('../data/E/dataset_E_DESeq2_ranktransformed_FG.txt')
        df = df.T
        info = pd.read_table('../data/E/dataset_E_annotation_FG.txt')
        info = pd.DataFrame(info)
        info.Condition = info.Condition.map(dict(CASE=1, CONTROL=0))
        info = info.drop(columns=['ID'])
        info = info.reset_index(drop=True)
        # print(info)
        # info.head()

        df = df.reset_index(drop=True)
        dt = df.join(info)
        # print(dt)
        # dt.head()

    # Split the data randomly according to percentages

    # balance 0-1
    data_p = dt[dt['Condition'] == 1]  # 2588
    data_n = dt[dt['Condition'] == 0]  # 5760
    if data_n.shape[0] > data_p.shape[0]:
        data_n = data_n.sample(n=data_p.shape[0], random_state=10)
    else:
        data_p = data_p.sample(n=data_n.shape[0], random_state=10)
    dt = pd.concat([data_p, data_n])

    # dt = dt.sample(frac=1)
    X = dt.drop(['Condition'], axis=1)
    Y = dt['Condition']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=69)

    X_train=torch.from_numpy(X_train.values.astype(np.float32))
    y_train=torch.from_numpy(y_train.values.astype(np.float32))
    X_test=torch.from_numpy(X_test.values.astype(np.float32))
    y_test=torch.from_numpy(y_test.values.astype(np.float32))

    batch_size = 512
    train_dataset = Data.TensorDataset(X_train, y_train)
    # train_dataset = Data.DataLoader(dataset=train_data, batch_size=batch_size)

    test_dataset = Data.TensorDataset(X_test, y_test)
    # test_dataset = Data.DataLoader(dataset=test_data, batch_size=batch_size)


    # sample training data amongst users
    if args.iid:
        # Sample IID user data from Mnist
        user_groups = data_iid(train_dataset, args.num_users)
    else:
        # Sample Non-IID user data from Mnist
        if args.unequal:
            # Chose uneuqal splits for every user
            user_groups = data_noniid_unbalanced(train_dataset, args.num_users) #raise NotImplementedError()
        else:
            # Chose euqal splits for every user
            user_groups = data_noniid_split5(train_dataset, args.num_users)
            
    return train_dataset, test_dataset, user_groups

## Old method
def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.true_divide(w_avg[key], len(w)) #true_divide
    return w_avg


def average_weights_sl(w, args):
    """
    Returns the average of the weights.
    """
    node_sel = np.random.choice(args.num_users, args.num_users-1, replace=False)
    w_avg = copy.deepcopy(w[node_sel[0]])
    for key in w_avg.keys():
        for i in node_sel[1:]:
            w_avg[key] += w[i][key]
        w_avg[key] = torch.true_divide(w_avg[key], args.num_users-1) #true_divide
    return w_avg


def average_weights_w(w,w_weight):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    
    for key in w_avg.keys():
        for i in range(len(w)):
            w_avg[key] = w_avg[key].float()
            w_avg[key] += w[i][key]*w_weight[i]
        w_avg[key] = w_avg[key] - w[0][key]
        #w_avg[key] = torch.true_divide(w_avg[key], len(w)) #true_divide
    return w_avg

## New method
def average_sgrads(grad,grad_old,weight,w_weight,idx):
    s_avg_i = copy.deepcopy(grad[idx])
    for key_idx, key in enumerate(s_avg_i.keys()):

        for i in range(len(grad)):
            s_avg_i[key] += weight[i][key]*w_weight[i]
        
        s_avg_i[key] = s_avg_i[key] - grad_old[idx][key]
        #w_avg[key] = torch.div(w_avg[key], len(w))
    return s_avg_i


## New method
def average_sgrads_new(grad,grad_old,s_avg,w_weight,idx):
    s_avg_i = copy.deepcopy(grad[idx])
    for key_idx, key in enumerate(s_avg_i.keys()):

        for i in range(len(grad)):
            s_avg_i[key] += s_avg[i][key]*w_weight[i]
        
        s_avg_i[key] = s_avg_i[key] - grad_old[idx][key]
        #w_avg[key] = torch.div(w_avg[key], len(w))
    return s_avg_i


def average_weights_new(w,s_avg,w_weight,idx,mu):
    """
    Returns the average of the weights.
    """
    s_avg_keys = list(s_avg[idx].keys())
    #print(s_avg_keys)
    
    w_avg = copy.deepcopy(w[idx])
    #print(w_avg.keys())
    
    for key_idx, key in enumerate(w_avg.keys()):

        for i in range(len(w)):
            w_avg[key] = w_avg[key].float()
            w_avg[key] += w[i][key]*w_weight[i]
        
        w_avg[key] = w_avg[key] - w[idx][key]
    
    for key_idx, key in enumerate(s_avg[idx].keys()):
        w_avg[key] = w_avg[key] - mu*s_avg[idx][key]
        #w_avg[key] = torch.div(w_avg[key], len(w))
    
    return w_avg


def erdos_renyi(n,p):
    while True:
        A = np.random.random((n,n));
        A[A<p] = 1;
        A[A<1] = 0;
        #symmetrize A, get adjacency matrix
        A = np.triu(A,1); A = A + A.T;
        #get laplacian
        L = -A;
        for k in range(n):
            L[k,k] = sum(A[k,:]);

        eig_L = np.linalg.eig(L)[0];
        pos_eig_0 = np.where(np.abs(eig_L) <1e-5)[0];
        if len(pos_eig_0)==1:
            break;
    
    degrees = np.diag(L);
    D = np.diag(degrees);
    max_degree = np.max(degrees);
    W = np.eye(n) - 1/(max_degree+1)* (D-A);
    
    return W

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


def get_d(args):
    X_train = torch.tensor([1.0])
    y_train = torch.tensor([1.0])
    # X_train = torch.from_numpy(X_train.values.astype(np.float32))
    # y_train = torch.from_numpy(y_train.values.astype(np.float32))
    train_dataset = Data.TensorDataset(X_train, y_train)
    test_dataest = Data.TensorDataset(X_train, y_train)
    user_groups = {}
    for i in range(args.num_users):
        user_groups[i] = [0]

    return train_dataset, test_dataest, user_groups


# define the initial function to init the layer's parameters for the network
def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        # m.weight.data.normal_(0,0.01)
        m.weight.data.zero_()
        m.bias.data.zero_()
