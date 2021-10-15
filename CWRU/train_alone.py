#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from options import args_parser
from update import LocalUpdate, test_inference, DatasetSplit
from models import Linear, MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, VGGNet, ResNet, DNNModel, Logistic
from utils import get_dataset, get_data, average_weights, erdos_renyi, exp_details, get_data


def create_model(args):

    # BUILD MODEL
    if args.model == 'linear':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = Linear(dim_in=len_in, dim_hidden=64,
                              dim_out=args.num_classes)

    elif args.model == 'logistic':
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = Logistic(dim_in=len_in, dim_hidden=64,
                                dim_out=args.num_classes)

    elif args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            # global_model = CNNCifar(args=args)
            # global_model = VGGNet(args=args)
            global_model = ResNet(args=args, depth=32, block_name='BasicBlock')

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)

    elif args.model == 'dnn':
        len_in = len(train_dataset[0][0])
        global_model = DNNModel(dim_in=len_in, dim_out=args.num_classes)

    else:
        raise Exception('Error: unrecognized model')

    return global_model


def train_one_epoch(model, trainloader, optimizer, criterion, args, device):
    model.train()

    batch_loss = 0.0
    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)

        model.zero_grad()
        log_probs = model(images)
        loss = criterion(log_probs, labels.long())
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)

        optimizer.step()


if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)

    # if args.gpu:
    # torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # set random seed
    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)  # sets the seed for generating random numbers.   　　
        torch.cuda.manual_seed(args.seed)  # Sets the seed for generating random numbers for the current GPU.
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

        print(f'Set seed {args.seed}\n')

    # load datasets
    if args.dataset[:2] == 'sl':
        train_dataset, test_dataset, user_groups = get_data(args)
    else:
        train_dataset, test_dataset, user_groups = get_dataset(args)

    train_acc_dict, train_loss_dict = defaultdict(list), defaultdict(list)
    test_acc_dict,  test_loss_dict  = defaultdict(list), defaultdict(list)
    log_path = f'../save/node{args.num_users}'
    for idx in range(args.num_users):
        model = create_model(args)
        model.to(device)
        model.train()

        trainloader = DataLoader(DatasetSplit(train_dataset, list(user_groups[idx])), batch_size=args.local_bs, shuffle=True)

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4)  # momentum=0.5
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        else:
            raise Exception()

        step_size = args.step_size
        StepLR_optimizer = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.2)
        criterion = nn.NLLLoss().to(device)

        for epoch in range(args.epochs * args.local_ep):
            print('\n')
            train_one_epoch(model, trainloader, optimizer, criterion, args, device)
            StepLR_optimizer.step()

            if (epoch+1) % args.local_ep == 0:
                train_acc, train_loss = test_inference(args, model, trainloader.dataset)
                test_acc,  test_loss  = test_inference(args, model, test_dataset)

                train_acc_dict[idx].append(train_acc)
                train_loss_dict[idx].append(train_loss)
                test_acc_dict[idx].append(test_acc)
                test_loss_dict[idx].append(test_loss)

                print(f'node {idx + 1} | round {epoch} | train set | acc: {train_acc} | loss: {train_loss}\n')
                print(f'node {idx + 1} | round {epoch} | test set | acc: {test_acc} | loss: {test_loss}\n')
                print('================================================')

        os.makedirs(os.path.join(log_path, 'model'), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(log_path, f'model/alone_model_{args.dataset}_{args.model}_{args.num_users}_{args.epochs}_iid[{args.iid}]_node{idx}.pt'))

    os.makedirs(os.path.join(log_path, 'objects'), exist_ok=True)
    with open(os.path.join(log_path, f'objects/alone_train_{args.dataset}_{args.model}_{args.num_users}_{args.epochs}_iid[{args.iid}].pkl'), 'wb') as f:
        pickle.dump([train_acc_dict, train_loss_dict], f)

    with open(os.path.join(log_path, f'objects/alone_test_{args.dataset}_{args.model}_{args.num_users}_{args.epochs}_iid[{args.iid}].pkl'), 'wb') as f:
        pickle.dump([test_acc_dict, test_loss_dict], f)

