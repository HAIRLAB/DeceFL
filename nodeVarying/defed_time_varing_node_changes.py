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
from collections import OrderedDict
from options import args_parser
from update import LocalUpdate, test_inference
from models import Linear, MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, VGGNet, ResNet, DNNModel, Logistic
from utils import get_dataset, get_data, average_weights_new, average_weights, erdos_renyi, exp_details, \
    average_sgrads, get_data, average_weights_w, unsel_weights_new

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

    # load dataset and user groups
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
            # global_model = CNNCifar(args=args)
            # global_model = VGGNet(args=args)
            global_model = ResNet(args=args, depth=32, block_name='BasicBlock')

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
                               
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
    # global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    test_loss, test_accuracy = [], []
    train_mean_loss, train_mean_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    Num_model = int(args.frac * args.num_users)
    
    global_model_i = []
    for i in range(Num_model):
        global_model_i.append(copy.deepcopy(global_model))
        global_model_i[i].to(device)
        global_model_i[i].train()
        train_loss.append([])
        train_accuracy.append([])
        test_loss.append([])
        test_accuracy.append([])

    p = args.p
    if p > 0:
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
    

    Num_model_sel_1 = Num_model - 2
    if args.method == 'er':
        W1 = erdos_renyi(Num_model_sel_1, p)
        print('W1:\n', W1)
    user_sel_1 = random.sample(range(len(idxs_users)), Num_model_sel_1)
    user_else_1 = set(range(len(idxs_users))) - set(user_sel_1)


    Num_model_sel_2 = Num_model
    if args.method == 'er':
        W2 = erdos_renyi(Num_model_sel_2, p)
        print('W2:\n', W2)
    user_sel_2 = random.sample(range(len(idxs_users)), Num_model_sel_2)
    user_else_2 = set(range(len(idxs_users))) - set(user_sel_2)


    Num_model_sel_3 = Num_model - 2
    if args.method == 'er':
        W3 = erdos_renyi(Num_model_sel_3, p)
        print('W3:\n', W3)
    user_sel_3 = random.sample(range(len(idxs_users)), Num_model_sel_3)
    user_else_3 = set(range(len(idxs_users))) - set(user_sel_3)
    
    lr_start = args.lr

    for epoch in tqdm(range(args.epochs)):
        local_weights = [[] for _ in range(len(idxs_users))]
        local_weights_grad = [[] for _ in range(len(idxs_users))]
        # local_weights_new = []

        print(f'\n | Global Training Round : {epoch+1} |\n')

        for i in range(Num_model):
            global_model_i[i].train()
        
        global_model.train()

        # args.lr = lr_start*(0.95**epoch)

        for ind,idx in enumerate(idxs_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx]) #, logger=logger

            # # Set optimizer for the local updates
            # if args.optimizer == 'sgd':
            #     optimizer = torch.optim.SGD(global_model_i[idx].parameters(), lr=lr_now, weight_decay=1e-4) #momentum=0.5
            # elif args.optimizer == 'adam':
            #     optimizer = torch.optim.Adam(global_model_i[idx].parameters(), lr=lr_now, weight_decay=1e-4)
            
            # step_size = 20
            # StepLR_optimizer = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.2)

            # model_i_update, loss = local_model.update_weights(
            #     model=copy.deepcopy(global_model_i[idx]), global_round=epoch, optimizer=optimizer, StepLR_optimizer=StepLR_optimizer)

            model_i_update, loss = local_model.update_weights(
                model=copy.deepcopy(global_model_i[idx]), global_round=epoch)
            
            #.state_dict(), model.grad.state_dict()
            w_update = model_i_update.state_dict()
            w_i = global_model_i[idx].state_dict()
            
            # local_weights_new.append(copy.deepcopy(w_update))    # 保存本地更新完后的模型

            grad_i = copy.deepcopy(w_update)

            for key_idx, key in enumerate(grad_i.keys()):
                grad_i[key] = (w_i[key] - grad_i[key]).float()
            
            local_weights[idx] = copy.deepcopy(w_i)
            local_weights_grad[idx] = copy.deepcopy(grad_i)
            # train_loss[ind].append(copy.deepcopy(loss))

        if args.varying == 1:    # 设置时变，每轮只连接一半的节点

            if epoch <= args.epochs//3:
                print('Step 1:\n')
                user_sel = user_sel_1
                user_else = user_else_1

                print(f'user_sel:{user_sel}')
                local_weights_sel = [local_weights[i] for i in user_sel]
                local_weights_grad_sel = [local_weights_grad[i] for i in user_sel]
                
                for ind, idx in enumerate(user_sel):
                    w_weights = W1[:, ind]  
                    mu = 1 #*(0.95**epoch)
                    global_weights = average_weights_new(local_weights_sel, local_weights_grad_sel, w_weights, ind, mu)
                    global_model_i[idx].load_state_dict(global_weights)
                for idx in user_else:  # 未连接的节点也按步长更新
                    mu = 1 #*(0.95**epoch)
                    global_weights = unsel_weights_new(local_weights[idx], local_weights_grad[idx], mu)
                    global_model_i[idx].load_state_dict(global_weights)

            if epoch > args.epochs//3 and epoch <= args.epochs//3*2:
                print('Step 2:\n')
                user_sel = user_sel_2
                user_else = user_else_2

                print(f'user_sel:{user_sel}')
                local_weights_sel = [local_weights[i] for i in user_sel]
                local_weights_grad_sel = [local_weights_grad[i] for i in user_sel]
                
                for ind, idx in enumerate(user_sel):
                    w_weights = W2[:, ind]
                    mu = 1 #*(0.95**epoch)
                    global_weights = average_weights_new(local_weights_sel, local_weights_grad_sel, w_weights, ind, mu)
                    global_model_i[idx].load_state_dict(global_weights)
                for idx in user_else:  # 未连接的节点也按步长更新
                    mu = 1 #*(0.95**epoch)
                    global_weights = unsel_weights_new(local_weights[idx], local_weights_grad[idx], mu)
                    global_model_i[idx].load_state_dict(global_weights)

            elif epoch > args.epochs//3*2:
                print('Step 3:\n')
                user_sel = user_sel_3
                user_else = user_else_3
                
                print(f'user_sel:{user_sel}')
                local_weights_sel = [local_weights[i] for i in user_sel]
                local_weights_grad_sel = [local_weights_grad[i] for i in user_sel]
                
                for ind, idx in enumerate(user_sel):
                    w_weights = W3[:, ind]
                    mu = 1 #*(0.95**epoch)
                    global_weights = average_weights_new(local_weights_sel, local_weights_grad_sel, w_weights, ind, mu)
                    global_model_i[idx].load_state_dict(global_weights)
                for idx in user_else:  # 未连接的节点也按步长更新
                    mu = 1 #*(0.95**epoch)
                    global_weights = unsel_weights_new(local_weights[idx], local_weights_grad[idx], mu)
                    global_model_i[idx].load_state_dict(global_weights)
        
        else:
            for ind,idx in enumerate(idxs_users):
                # update global weights
                w_weights = W[:,ind]
                mu = 1 #*(0.95**epoch)
                global_weights = average_weights_new(local_weights,local_weights_grad,w_weights,ind,mu)
                # global_weights = average_weights_w(local_weights_new, w_weights)
                # update global weights
                global_model_i[idx].load_state_dict(global_weights)

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
        
        
        for ind,idx in enumerate(idxs_users):
            list_acc, list_loss = [], []
            global_model_i[idx].eval()

            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx]) #, logger=logger
            train_accuracy_temp,train_loss_temp = local_model.inference(model=global_model_i[idx])
            
            train_accuracy[idx].append(train_accuracy_temp)
            train_loss[idx].append(train_loss_temp)
            
            # print global training loss after every 'i' rounds
            if (epoch+1) % print_every == 0:
                print(f' \nAvg Training Stats after {epoch+1} global rounds:')
                print(f'Training Loss : {np.mean(np.array(train_loss[ind]))}')
                print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[idx][-1]))
        
        list_mean_acc, list_mean_loss = [], []
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c]) #, logger=logger
            acc_mean, loss_mean = local_model.inference(model=global_model)
            list_mean_acc.append(acc_mean)
            list_mean_loss.append(loss_mean)

        train_mean_accuracy.append(sum(list_mean_acc)/len(list_mean_acc))
        train_mean_loss.append(sum(list_mean_loss)/len(list_mean_loss))

        for ind in range(len(idxs_users)):
            # Test inference after completion of training
            test_acc_i, test_loss_i = test_inference(args, global_model_i[ind], test_dataset)
            test_accuracy[ind].append(copy.deepcopy(test_acc_i))
            test_loss[ind].append(copy.deepcopy(test_loss_i))

            print(f' \n Local Model:{ind} Results after {args.epochs} global rounds of training:')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[ind][-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc_i))

        test_acc_global[epoch], test_loss_global[epoch] = test_inference(args, global_model, test_dataset)
        print("|---- Test Accuracy Global: {:.2f}%".format(100 * test_acc_global[epoch]))

    # test_acc_global, test_loss_global = test_inference(args, global_model, test_dataset)
    # print("|---- Test Accuracy Global: {:.2f}%".format(100*test_acc_global))

    torch.save(global_model.state_dict(), '../save/node{}/model/defed_global_model_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_M[{}]_p[{}].pt'.
               format(args.num_users, args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, args.method, args.p))
    for ind in range(len(idxs_users)):
        torch.save(global_model_i[ind].state_dict(), '../save/node{}/model/defed_model{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_M[{}]_p[{}].pt'.
                   format(args.num_users, ind, args.dataset, args.model, args.epochs, args.frac, args.iid,
                          args.local_ep, args.local_bs, args.method, args.p))

    # test_acc test_loss
    file_name = '../save/node{}/objects/New_fed_test_each_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_M[{}]_p[{}].pkl'. \
        format(args.num_users, args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, args.method, args.p)
    with open(file_name, 'wb') as f:
        pickle.dump([test_accuracy, test_loss], f)

    file_name = '../save/node{}/objects/New_fed_test_acc-vs-round_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_M[{}]_p[{}].pkl'. \
        format(args.num_users, args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, args.method, args.p)
    with open(file_name, 'wb') as f:
        pickle.dump([test_acc_global, test_loss_global], f)

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/node{}/objects/New_fed_w_only_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_M[{}]_p[{}].pkl'.\
        format(args.num_users, args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, args.method, args.p)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/node{}/objects/New_fed_w_only_mean_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_M[{}]_p[{}].pkl'.\
        format(args.num_users, args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, args.method, args.p)

    with open(file_name, 'wb') as f:
        pickle.dump([train_mean_loss, train_mean_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    for ind in range(len(idxs_users)):
        plt.plot(range(len(train_loss[ind])), train_loss[ind], color='r',label='Local Model:'+str(ind))
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.legend()
    plt.savefig('../save/node{}/figure/New_fed_w_only_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_M[{}]_p[{}]_loss.png'.
                format(args.num_users, args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.method, args.p))
    
    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    for ind in range(len(idxs_users)):
        plt.plot(range(len(train_accuracy[ind])), train_accuracy[ind], color='k',label='Local Model:'+str(ind))
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.legend()
    plt.savefig('../save/node{}/figure/New_fed_w_only_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_M[{}]_p[{}]_acc.png'.
                format(args.num_users, args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.method, args.p))

    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_mean_loss)), train_mean_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/node{}/figure/New_Avg_fed_w_only_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_M[{}]_p[{}]_loss.png'.
                format(args.num_users, args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.method, args.p))
    
    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_mean_accuracy)), train_mean_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/node{}/figure/New_Avg_fed_w_only_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_M[{}]_p[{}]_acc.png'.
                format(args.num_users, args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.method, args.p))
