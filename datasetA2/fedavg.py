#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import random

import torch
# from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import Linear, MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, VGGNet, ResNet, DNNModel, Logistic
from utils import get_dataset,get_data,average_weights, erdos_renyi, exp_details, get_data


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    # logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    #if args.gpu:
        #torch.cuda.set_device(args.gpu)
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
            #global_model = CNNCifar(args=args)
            #global_model = VGGNet(args=args)
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
        global_model = DNNModel(dim_in=len_in)                       
                           
    else:
        raise Exception('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    # test_loss_each, test_accuracy_each = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    # for i in range(args.num_users):
    #     test_loss_each.append([])
    #     test_accuracy_each.append([])

    m = max(int(args.frac * args.num_users), 1)

    idxs_users = np.random.choice(range(args.num_users), m, replace=False)

    groups_num = [len(user_groups[idx]) for idx in idxs_users]
    print(groups_num)

    test_acc, test_loss = {}, {}
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                              idxs=user_groups[idx]) #, logger=logger
            model_i, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            w = model_i.state_dict()
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            
            # acc_t, loss_t = local_model.inference(model=copy.deepcopy(global_model))

            # # recording test_acc and test_loss of each node
            # test_acc_i, test_loss_i = test_inference(args, copy.deepcopy(model_i), test_dataset)
            # test_accuracy_each[idx].append(test_acc_i)
            # test_loss_each[idx].append(test_loss_i)

        # update global weights
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        # loss_avg = sum(local_losses) / len(local_losses)
        # train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c]) #, logger=logger
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))
        train_loss.append(sum(list_loss)/len(list_loss))

        # print global training loss after every 'i' rounds
        # if (epoch+1) % print_every == 0:
        print(f' \nAvg Training Stats after {epoch+1} global rounds:')
        print(f'Training Loss : {np.mean(np.array(train_loss))}')
        print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

        test_acc[epoch], test_loss[epoch] = test_inference(args, global_model, test_dataset)
        print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc[epoch]))

    torch.save(global_model.state_dict(), '../save/node{}/model/fedavg_global_model_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pt'.
               format(args.num_users, args.dataset, args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs))
    # Test inference after completion of training
    # test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    # print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # file_name = '../save/node{}/objects/fed_test_each_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_M[{}]_p[{}].pkl'. \
    #     format(args.num_users, args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs, args.method, args.p)
    # with open(file_name, 'wb') as f:
    #     pickle.dump([test_accuracy_each, test_loss_each], f)

    file_name = '../save/node{}/objects/fed_test_acc-vs-round_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
        format(args.num_users, args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)
    with open(file_name, 'wb') as f:
        pickle.dump([test_acc, test_loss], f)

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/node{}/objects/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.num_users, args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))


    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/node{}/figure/fed2_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
                format(args.num_users, args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))
    
    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/node{}/figure/fed2_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
                format(args.num_users, args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))
