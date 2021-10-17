#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
# from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset

from options import args_parser
from update import LocalUpdate, test_inference, DatasetSplit
from models import Linear, MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, VGGNet, ResNet, DNNModel, Logistic
from utils import get_dataset, get_data, average_weights, erdos_renyi, exp_details, get_data, get_d

from swarm import SwarmCallback

def create_model(args):

    # BUILD MODEL
    if args.model == 'linear':
        global_model = Linear(dim_in=1, dim_hidden=64, dim_out=1)

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

def local_train_step(model, trainloader, args, device):
    model.train()
    epoch_loss = []
    # Set optimizer for the local updates
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)  # momentum=0.5
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    step_size = 20
    StepLR_optimizer = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.2)
    # if args.model == 'logistic':
    #     criterion = nn.BCELoss().to(device)
    # elif args.model == 'dnn':
    # criterion = nn.NLLLoss().to(device)
    criterion = nn.MSELoss().to(device) # 一致性用mse损失函数

    for iter in range(args.local_ep):
        batch_loss = 0.0
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            model.zero_grad()
            log_probs = model(images)
            # if args.model == 'logistic':
            #     loss = criterion(log_probs, labels.float())
            # elif args.model == 'dnn':
            loss = criterion(log_probs, labels)
            # loss = self.criterion(log_probs, labels.float().view(-1, 1))
            loss.backward()

            # for name,param in model.named_parameters():
            #    param.grad = grad_avg.grad
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)

            optimizer.step()

            batch_loss += loss.item()
            # batch_loss.append(loss.item())
        epoch_loss.append(batch_loss / (batch_idx + 1))

        StepLR_optimizer.step()

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    # logger = SummaryWriter('../logs')

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
    dataDir = os.getenv('DATA_DIR', '../app-data')
    # modelDir = os.getenv('DATA_DIR', '../app-data')
    # if args.dataset[:2] == 'sl':
    #     train_dataset, test_dataset, user_groups = get_data(args, dataDir)
    # else:
    #     train_dataset, test_dataset, user_groups = get_dataset(args, dataDir)

    # 8-12 一致性检验，设置每个节点数据为（1，1）
    train_dataset, test_dataset, user_groups = get_d(args)

    model = create_model(args)

    # Set the model to train and send it to device.
    model.to(device)
    model.train()
    # Create Swarm callback

    swarmCallback = None
    swarmCallback = SwarmCallback(sync_interval=1,
                                  min_peers=args.num_users,
                                  val_data=1,
                                  val_batch_size=1,
                                  model_name='my_sl',
                                  model=model)

    # 节点编号
    with open(os.path.join('node_id.txt'), 'r') as f:
        idx = f.readline()
        idx = int(idx) - 1

    # log路径
    log_path = 'log/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    output_info = f'{args}\n'
    with open(os.path.join(log_path, 'log.txt'), 'w') as f:
        f.write(output_info)

    output_info = f'This is node {idx + 1}\n'
    print(output_info)
    with open(os.path.join(log_path, 'log.txt'), 'a') as f:
        f.write(output_info)

    trainloader = DataLoader(DatasetSplit(train_dataset, list(user_groups[idx])),
                             batch_size=args.local_bs, shuffle=True)

    swarmCallback.on_train_begin()
    swarmCallback.on_batch_end()    # 不添加这个，模型在第一个round未聚合

    train_acc_dict, train_loss_dict = {}, {}
    test_acc_dict, test_loss_dict = {}, {}
    model_state = []

    for epoch in tqdm(range(args.epochs)):
        print('\n')
        local_train_step(model, trainloader, args, device)

        # train_acc, train_loss = test_inference(args, model, trainloader.dataset)
        # test_acc, test_loss = test_inference(args, model, test_dataset)
        # print(f'before: train set | acc:{train_acc} | loss:{train_loss}')
        # print(f'before: test set | acc:{test_acc} | loss:{test_loss}')

        swarmCallback.on_batch_end()

        # merge后的模型在本地训练集、测试集上测试
        train_acc, train_loss = test_inference(args, model, trainloader.dataset)
        # test_acc, test_loss = test_inference(args, model, test_dataset)

        model_state.append(copy.deepcopy(model.state_dict()))

        train_acc_dict[epoch], train_loss_dict[epoch] = (train_acc, train_loss)
        # test_acc_dict[epoch], test_loss_dict[epoch] = (test_acc, test_loss)

        output_info = f'node {idx + 1} | round {epoch} | train set | acc: {train_acc} | loss: {train_loss}\n'
        # output_info += f'node {idx + 1} | round {epoch} | test set | acc: {test_acc} | loss: {test_loss}\n'
        output_info += '================================================'
        print(output_info)
        with open(os.path.join(log_path, 'log.txt'), 'a') as f:
            f.write(output_info)

        swarmCallback.on_epoch_end(epoch)

    swarmCallback.on_train_end()

    # with open(os.path.join(log_path, f'swarm_train_{args.dataset}_{args.model}_{args.num_users}_{args.epochs}_iid[{args.iid}]_node{idx}.pkl'), 'wb') as f:
    #     pickle.dump([train_acc_dict, train_loss_dict], f)
    #
    # with open(os.path.join(log_path, f'swarm_test_{args.dataset}_{args.model}_{args.num_users}_{args.epochs}_iid[{args.iid}]_node{idx}.pkl'), 'wb') as f:
    #     pickle.dump([test_acc_dict, test_loss_dict], f)

    with open(os.path.join(log_path, 'sl_final_state.pkl'), 'wb') as f:
        pickle.dump(model_state, f)

    torch.save(model.state_dict(), os.path.join(log_path, 'model.pt'))
