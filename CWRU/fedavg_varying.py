# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jindou time: 2021/9/3
# !/usr/bin/env python
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

    m = max(int(args.frac * args.num_users), 1)

    idxs_users = np.random.choice(range(args.num_users), m, replace=False)

    groups_num = [len(user_groups[idx]) for idx in idxs_users]
    print('\nGroups size:\n', groups_num)

    local_weights_grad_old = []
    test_acc_global, test_loss_global = {}, {}
    for epoch in tqdm(range(args.epochs)):
        local_weights = []
        local_weights_new = []

        print(f'\n | Global Training Round : {epoch + 1} |\n')

        for i in range(Num_model):
            global_model_i[i].train()

        global_model.train()

        for ind, idx in enumerate(idxs_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx])  # , logger=logger
            model_i_update, loss = local_model.update_weights(
                model=copy.deepcopy(global_model_i[ind]), global_round=epoch)

            # .state_dict(), model.grad.state_dict()
            w_update = model_i_update.state_dict()
            w_i = global_model_i[ind].state_dict()

            local_weights.append(copy.deepcopy(w_i))           # 保存本地更新完前的模型
            local_weights_new.append(copy.deepcopy(w_update))  # 保存本地更新完后的模型
            train_loss[ind].append(copy.deepcopy(loss))

        if args.varying == 1:  # 设置时变，每轮只连接一半的节点
            Num_model_sel = int(Num_model / 2)
            user_sel = random.sample(range(len(idxs_users)), Num_model_sel)
            user_else = set(range(len(idxs_users))) - set(user_sel)
            print(f'user_sel:{user_sel}')
            local_weights_sel = [local_weights_new[i] for i in user_sel]

            global_weights = average_weights(local_weights_sel)
            for idx in user_sel:
                global_model_i[idx].load_state_dict(global_weights)

            for idx in user_else:
                global_model_i[idx].load_state_dict(local_weights_new[idx])

        else:
            global_weights = average_weights(local_weights_new)
            for ind in range(len(idxs_users)):
                # update global weights
                global_model_i[ind].load_state_dict(global_weights)


        # Calculate avg training accuracy over all users at every epoch

        # update global weights
        global_weights_mean = average_weights(local_weights_new)   # fedavg对比defed在此处的模型有修正

        # update global weights
        global_model.load_state_dict(global_weights_mean)
        global_model.eval()

        for ind in range(len(idxs_users)):
            list_acc, list_loss = [], []
            global_model_i[ind].eval()
            for c in range(args.num_users):
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[c])  # , logger=logger
                acc, loss = local_model.inference(model=global_model_i[ind])
                list_acc.append(acc)
                list_loss.append(loss)

            train_accuracy[ind].append(sum(list_acc) / len(list_acc))

            # print global training loss after every 'i' rounds
            if (epoch + 1) % print_every == 0:
                print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
                print(f'Training Loss : {np.mean(np.array(train_loss[ind]))}')
                print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[ind][-1]))

        list_mean_acc, list_mean_loss = [], []
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c])  # , logger=logger
            acc_mean, loss_mean = local_model.inference(model=global_model)
            list_mean_acc.append(acc_mean)
            list_mean_loss.append(loss_mean)

        train_mean_accuracy.append(sum(list_mean_acc) / len(list_mean_acc))
        train_mean_loss.append(sum(list_mean_loss) / len(list_mean_loss))

        for ind in range(len(idxs_users)):
            # Test inference after completion of training
            test_acc_i, test_loss_i = test_inference(args, global_model_i[ind], test_dataset)
            test_accuracy[ind].append(copy.deepcopy(test_acc_i))
            test_loss[ind].append(copy.deepcopy(test_loss_i))

            print(f' \n Local Model:{ind} Results after {args.epochs} global rounds of training:')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[ind][-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc_i))

        test_acc_global[epoch], test_loss_global[epoch] = test_inference(args, global_model, test_dataset)
        print("|---- Test Accuracy Global: {:.2f}%".format(100 * test_acc_global[epoch]))

    # test_acc_global, test_loss_global = test_inference(args, global_model, test_dataset)
    # print("|---- Test Accuracy Global: {:.2f}%".format(100*test_acc_global))

    torch.save(global_model.state_dict(),
               '../save/node{}/model/fedavg_global_model_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_M[{}]_p[{}].pt'.
               format(args.num_users, args.dataset, args.model, args.epochs, args.frac, args.iid,
                      args.local_ep, args.local_bs, args.method, args.p))
    for ind in range(len(idxs_users)):
        torch.save(global_model_i[ind].state_dict(),
                   '../save/node{}/model/fedavg_model{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_M[{}]_p[{}].pt'.
                   format(args.num_users, ind, args.dataset, args.model, args.epochs, args.frac, args.iid,
                          args.local_ep, args.local_bs, args.method, args.p))

    # test_acc test_loss
    file_name = '../save/node{}/objects/fed_test_each_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
        format(args.num_users, args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)
    with open(file_name, 'wb') as f:
        pickle.dump([test_accuracy, test_loss], f)

    file_name = '../save/node{}/objects/fed_test_acc-vs-round_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
        format(args.num_users, args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)
    with open(file_name, 'wb') as f:
        pickle.dump([test_acc_global, test_loss_global], f)

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/node{}/objects/fed_train_each_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
        format(args.num_users, args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/node{}/objects/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
        format(args.num_users, args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)
    with open(file_name, 'wb') as f:
        pickle.dump([train_mean_loss, train_mean_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    # # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    #
    # matplotlib.use('Agg')
    #
    # # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # for ind in range(len(idxs_users)):
    #     plt.plot(range(len(train_loss[ind])), train_loss[ind], color='r', label='Local Model:' + str(ind))
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.legend()
    # plt.savefig('../save/node{}/figure/fedavg_w_only_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.num_users, args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # for ind in range(len(idxs_users)):
    #     plt.plot(range(len(train_accuracy[ind])), train_accuracy[ind], color='k', label='Local Model:' + str(ind))
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.legend()
    # plt.savefig('../save/node{}/figure/fedavg_w_only_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.num_users, args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_mean_loss)), train_mean_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/node{}/figure/fedavg_w_only_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.num_users, args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_mean_accuracy)), train_mean_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/node{}/figure/fedavg_w_only_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.num_users, args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
