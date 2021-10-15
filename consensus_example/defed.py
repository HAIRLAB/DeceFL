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
# from tensorboardX import SummaryWriter
from collections import OrderedDict
from options import args_parser
from update import LocalUpdate, test_inference
from models import Linear, MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, VGGNet, ResNet, DNNModel, Logistic
from utils import get_dataset, get_data, average_weights_new, average_weights, erdos_renyi, exp_details, average_sgrads, get_data, get_d, weigth_init

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

    # 8-12 一致性检验，设置每个节点数据为（1，1）
    train_dataset, test_dataset, user_groups = get_d(args)

    # Training
    model_ini = []
    train_loss, train_accuracy = [], []
    test_loss, test_accuracy = [], []
    train_mean_loss, train_mean_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    test_loss_each, test_accuracy_each = [], []
    model_state = []

    Num_model = int(args.frac * args.num_users)
    
    global_model_i = []
    for i in range(Num_model):

        global_model = Linear(dim_in=1, dim_hidden=64, dim_out=args.num_classes)
        model_ini.append(copy.deepcopy(global_model.state_dict()))

        global_model_i.append(copy.deepcopy(global_model))
        global_model_i[i].to(device)
        global_model_i[i].train()
        train_loss.append([])
        train_accuracy.append([])
        test_loss.append([])
        test_accuracy.append([])
        test_loss_each.append([])
        test_accuracy_each.append([])
        model_state.append([])     # 记录每个round后的state_dict

    with open(f'./save/node{args.num_users}/objects/defed_ini_state_p{args.p}.pkl', 'wb') as f:
        pickle.dump(model_ini, f)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    p = args.p

    if args.method == 'er':
        W = erdos_renyi(Num_model,p)

    if args.method == 'diag':
        W = np.eye(Num_model)
        for i in range(1,int((Num_model*p-1)/2)+1):
            W = W + np.diag(np.ones(Num_model-i),i) + np.diag(np.ones(Num_model-i),-i)

            W[i-1,Num_model-(int((Num_model*p-1)/2)+1-i):] = 1
            W[Num_model-i,0:(int((Num_model*p-1)/2)+1-i)] = 1

        if p==1:
            W = np.ones((Num_model,Num_model))
    
        W = W/(Num_model*p)
    
    print('W:\n',W)
    
    m = max(int(args.frac * args.num_users), 1)

    # if args.seed:
    #     np.random.seed(args.seed*200)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)    

    groups_num = [len(user_groups[idx]) for idx in idxs_users]
    print('\nGroups size:\n',groups_num)
    
    local_weights_grad_old = []
    test_acc_global, test_loss_global = {}, {}
    for epoch in tqdm(range(args.epochs)):
        local_weights = []
        local_weights_grad = []

        print(f'\n | Global Training Round : {epoch+1} |\n')
        
        for i in range(Num_model):
            global_model_i[i].train()
        
        global_model.train()


        for ind,idx in enumerate(idxs_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx]) #, logger=logger
            model_i_update, loss = local_model.update_weights(
                model=copy.deepcopy(global_model_i[ind]), global_round=epoch)
            
            #.state_dict(), model.grad.state_dict()
            w_update = model_i_update.state_dict()
            w_i = global_model_i[ind].state_dict()

            grad_i = copy.deepcopy(w_update)

            for key_idx, key in enumerate(grad_i.keys()):
                grad_i[key] = (w_i[key] - grad_i[key]).float()
            
            local_weights.append(copy.deepcopy(w_i))
            local_weights_grad.append(copy.deepcopy(grad_i))
            train_loss[ind].append(copy.deepcopy(loss))

            # # recording test_acc and test_loss of each node
            # test_acc_i, test_loss_i = test_inference(args, copy.deepcopy(model_i_update), test_dataset)
            # test_accuracy_each[idx].append(test_acc_i)
            # test_loss_each[idx].append(test_loss_i)
        

        for ind in range(len(idxs_users)):
            # update global weights
            w_weights = W[:,ind]
            mu = 0.1
            global_weights = average_weights_new(local_weights,local_weights_grad,w_weights,ind,mu)
            # update global weights
            global_model_i[ind].load_state_dict(global_weights)

        local_weights_grad_old = local_weights_grad
        
        # update global weights
        #w_weights = np.ones(len(idxs_users))/len(idxs_users)
        #global_weights = average_weights(local_weights,w_weights)

        # update global weights
        #global_model.load_state_dict(global_weights)
        #loss_avg = sum(local_losses) / len(local_losses)
        #train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        
        # update global weights
        global_weights_mean = average_weights(local_weights)
        
        # update global weights
        global_model.load_state_dict(global_weights_mean)
        global_model.eval()
        
        
        # for ind in range(len(idxs_users)):
        #     list_acc, list_loss = [], []
        #     global_model_i[ind].eval()
        #     for c in range(args.num_users):
        #         local_model = LocalUpdate(args=args, dataset=train_dataset,
        #                                   idxs=user_groups[c]) #, logger=logger
        #         acc, loss = local_model.inference(model=global_model_i[ind])
        #         list_acc.append(acc)
        #         list_loss.append(loss)
        #
        #     train_accuracy[ind].append(sum(list_acc)/len(list_acc))
        #
        #     # print global training loss after every 'i' rounds
        #     if (epoch+1) % print_every == 0:
        #         print(f' \nAvg Training Stats after {epoch+1} global rounds:')
        #         print(f'Training Loss : {np.mean(np.array(train_loss[ind]))}')
        #         print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[ind][-1]))
        
        # list_mean_acc, list_mean_loss = [], []
        # for c in range(args.num_users):
        #     local_model = LocalUpdate(args=args, dataset=train_dataset,
        #                               idxs=user_groups[c]) #, logger=logger
        #     acc_mean, loss_mean = local_model.inference(model=global_model)
        #     list_mean_acc.append(acc_mean)
        #     list_mean_loss.append(loss_mean)
        #
        # train_mean_accuracy.append(sum(list_mean_acc)/len(list_mean_acc))
        # train_mean_loss.append(sum(list_mean_loss)/len(list_mean_loss))
        #
        for ind in range(len(idxs_users)):
        #     # Test inference after completion of training
        #     test_acc_i, test_loss_i = test_inference(args, global_model_i[ind], test_dataset)
        #     test_accuracy[ind].append(copy.deepcopy(test_acc_i))
        #     test_loss[ind].append(copy.deepcopy(test_loss_i))
        #
            model_state[ind].append(copy.deepcopy(global_model_i[ind].state_dict()))  # 记录每个round后的state
        #
        #     print(f' \n Local Model:{ind} Results after {args.epochs} global rounds of training:')
        #     print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[ind][-1]))
        #     print("|---- Test Accuracy: {:.2f}%".format(100*test_acc_i))
        #
        # test_acc_global[epoch], test_loss_global[epoch] = test_inference(args, global_model, test_dataset)
        # print("|---- Test Accuracy Global: {:.2f}%".format(100 * test_acc_global[epoch]))


    with open(f'./save/node{args.num_users}/objects/defed_final_state_p{args.p}.pkl', 'wb') as f:
        pickle.dump(model_state, f)


    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))


