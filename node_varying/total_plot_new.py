# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jindou time: 2021/8/2
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pickle
import numpy as np

frac = 1
# local_ep = 30
local_bs = 64
method = 'er'
seed = 1

y_x_lim_dict = {'sl_a_logistic_iid_node4': [[95, 99.2, 1], [95, 100.2, 1], None, [-6, 200, 50]],
                
                'sl_a_logistic_iid_node8': [[90, 99.2, 2], [90, 100.2, 2], None, [-12, 700, 100]],
                
                'sl_a_logistic_iid_node16': [[95, 99.2, 1], [95, 100.2, 1], None, [-24, 800, 200]],
                'sl_a_dnn_iid_node4': [[47, 100, 10], [47, 102, 10], [-0.03, 0.7, 0.1], [0, 50, 10]],
                'sl_a_dnn_iid_node8': [[47, 100, 10], [47, 102, 10], [-0.03, 0.7, 0.1], [-2, 60, 10]],
                'sl_a_dnn_iid_node16': [[47, 100, 10], [47, 102, 10], [-0.03, 0.7, 0.1], [-5, 150, 30]],
                'cwru_dnn_iid_node4': [None, None, None, [-3, 100, 20]],
                'cwru_dnn_iid_node8': [None, None, None, [-10, 200, 50]],
                'cwru_dnn_iid_node16': [None, None, None, [-10, 300, 50]],
                'cwru_logistic_iid_node4': [None, None, None, [0, 50, 10]],
                'cwru_logistic_iid_node8': [None, None, None, [0, 50, 10]],
                'cwru_logistic_iid_node16': [None, None, None, [-3, 100, 20]],
                'sl_a_logistic_noniid_node4': [[95, 99.3, 1], [95, 100.2, 1], None, [-9, 300, 60]],
                
                'sl_a_logistic_noniid_node8': [[90, 99.2, 2], [90, 100.2, 2], None, [-18, 700, 100]],
                
                'sl_a_logistic_noniid_node16': [[95, 99.3, 1], [95, 100.2, 1], None, [-27, 900, 180]],
                'sl_a_dnn_noniid_node4': [[47, 100, 10], [60, 101, 10], [-0.03, 0.7, 0.1], [0, 50, 10]],
                'sl_a_dnn_noniid_node8': [[47, 100, 10], [57, 101, 10], [-0.03, 0.7, 0.1], [-2, 80, 20]],
                'sl_a_dnn_noniid_node16': [[47, 100, 10], [57, 101, 10], [-0.03, 0.7, 0.1], [-5, 150, 30]],
                'cwru_dnn_noniid_node4': [None, None, None, [-3, 100, 20]],
                'cwru_dnn_noniid_node8': [None, None, None, [-5, 150, 30]],
                'cwru_dnn_noniid_node16': [None, None, None, [-6, 200, 50]],
                'cwru_logistic_noniid_node4': [None, None, None, [-5, 150, 30]],
                'cwru_logistic_noniid_node8': [None, None, None, [0, 50, 10]],
                'cwru_logistic_noniid_node16': [None, None, None, [-10, 300, 50]],
                
                'sl_a_logistic_iid_node10': [[90, 100.1, 1], [90, 100.2, 1], None, [-9, 600, 60]],
                'sl_a_dnn_iid_node10': [[90, 100.1, 1], [90, 100.2, 1], None, [-9, 600, 60]],
                
                'sl_a_logistic_noniid_node10': [[90, 100.1, 1], [90, 100.2, 1], None, [-9, 600, 60]],
                'sl_a_dnn_noniid_node10': [[90, 100.1, 1], [90, 100.2, 1], None, [-9, 600, 60]],                

                'cwru_logistic_iid_node10': [None, None, None, [-10, 300, 50]],

                'cwru_logistic_noniid_node10': [None, None, None, [-10, 300, 50]], 
                }


comb = [['sl_a', 'logistic', 1, 10, 1500, f'../results/A2_logistic_round1500_seed{seed}/'],
        ['sl_a', 'logistic', 0, 10, 1500, f'../results/A2_logistic_round1500_seed{seed}/'],
        ['sl_a', 'dnn', 1, 30, 300, f'../results/A2_dnn_seed{seed}/'],
        ['sl_a', 'dnn', 0, 30, 300, f'../results/A2_dnn_seed{seed}/'],
        ['cwru', 'dnn', 1, 30, 300, f'../results/CWRU_dnn_seed{seed}/'],
        ['cwru', 'dnn', 0, 30, 300, f'../results/CWRU_dnn_seed{seed}/'],
        ['cwru', 'mlp', 1, 30, 300, f'../results/CWRU_mlp_seed{seed}/'],
        ['cwru', 'mlp', 0, 30, 300, f'../results/CWRU_mlp_seed{seed}/'],
        # ['sl_a', 'logistic', 1, 10, 1500, f'../results/varying_A2_logistic_iid_round1500_seed{seed}/'],
        # ['sl_a', 'logistic', 0, 10, 1500, f'../results/varying_A2_logistic_noniid_round1500_seed{seed}/'],
        # ['cwru', 'logistic', 1, 10, 300, f'../results/varying_CWRU_logistic_round300_seed{seed}/'],
        # ['cwru', 'logistic', 0, 10, 300, f'../results/varying_CWRU_logistic_round300_seed{seed}/'],
        ['sl_a', 'logistic', 1, 10, 1500, f'../results/lr_decay_0.1/'],
        
        ['sl_a', 'logistic', 1,10, 900, f'../results/A2_varying2_seed{seed}/'],
        ['sl_a', 'dnn', 1,30, 300, f'../results/A2_varying2_seed{seed}/'],
        
        ['sl_a', 'logistic', 0,10, 900, f'../results/A2_varying2_seed{seed}/'],
        ['sl_a', 'dnn', 0,30, 300, f'../results/A2_varying2_seed{seed}/'],

        ['cwru', 'logistic', 1,10, 300, f'../results/cwru_varying2_seed{seed}/'],

        ['cwru', 'logistic', 0,10, 300, f'../results/cwru_varying2_seed{seed}/'],
        
        ]

dataset, model, iid, local_ep, epochs, path_data = comb[-1]     # 选择需要绘制的实验组数
# dataset, model, iid, epochs, path_data = ['cwru', 'dnn', 1, 300, '../']

p_list = [0.9, 0.7, 0.5, 0.3]      # 选择需要绘制的p
# p_list = [0.5]
# num_users_list = [4, 8, 16]        # 选择需要绘制的节点数
num_users_list = [10]

save_state = 1      # 保存图片设为1
ylim_state = 1      # 需要对坐标范围进行限制设为1

dataset_dict = {'sl_a': 'A2', 'sl_b': 'B', 'sl_e': 'E', 'cwru': 'CWRU'}
iid_state = 'iid' if iid else 'noniid'

if len(p_list) == 1:
    set_p = f'p={p_list[0]}'
    color = ['coral', 'g']
else:
    set_p = 'total_p'
    color = ['orangered', 'orange', 'y', 'g', 'deepskyblue', 'royalblue', 'plum']

# 图片保存路径
# path_save = f'../results/figure_gather/{dataset_dict[dataset]}_{model}_{iid_state}/'
path_save = path_data
print(dataset, model, set_p, iid_state)


# 图片保存路径
if path_data == f'../results/A2_varying2_seed{seed}/':
    path_save = f'../figure_gather/A2_varying2/'
elif path_data == f'../results/cwru_varying2_seed{seed}/':
    path_save = f'../figure_gather/CWRU_varying2/'     
else:
    path_save = f'../figure_gather/{dataset_dict[dataset]}_{model}_{iid_state}/'

print(path_save)

def single_plot(**kwargs):
    plt.figure(figsize=(5, 3), dpi=500)
    # plt.title(kwargs['title'], pad=20)

    # plt.plot(kwargs['x'], np.array(kwargs['y_sl'])*kwargs['m'], color=color[-1], label='SL')
    #plt.plot(kwargs['x'], np.array(kwargs['y_fedavg'])*kwargs['m'], color=color[-2], label='FedAvg')
    plt.plot(range(1, len(kwargs['y_fedavg'])+1), np.array(kwargs['y_fedavg'])*kwargs['m'], color=color[-2], label='FedAvg')
    
    for i, p in enumerate(p_list):
        plt.plot(kwargs['x'], np.array(kwargs['y_defed'][p])*kwargs['m'], color=color[i], label=f'DeceFL: p={p}')

    plt.ylabel(kwargs['ylabel'], fontsize=15)
    plt.xlabel(kwargs['xlabel'], fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # plt.legend(loc='best', fontsize=10)
    plt.legend(loc=kwargs['lc'], fontsize=14)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.set_xscale('log')
    if (ylim_state == 1) and (kwargs['ylim'] is not None):
        plt.ylim(kwargs['ylim'][0], kwargs['ylim'][1])
        ax.yaxis.set_major_locator(MultipleLocator(kwargs['ylim'][2]))
    if (ylim_state == 1) and (kwargs['xlim'] is not None):
        plt.xlim(kwargs['xlim'][0], kwargs['xlim'][1])
        ax.xaxis.set_major_locator(MultipleLocator(kwargs['xlim'][2]))
    ax.tick_params("both", which='major', length=5, width=0.5, colors='k', direction='in')  # "y", 'x', 'both'
    # ax.tick_params(which='minor', length=5, width=1.0, labelsize=10, labelcolor='0.6', direction='in')
    if save_state:
        print(kwargs['name_save'])
        plt.savefig(kwargs['name_save']+'.png', bbox_inches='tight')
    plt.show()


for num_users in num_users_list:
    test_mean_accuracy, test_mean_loss = {}, {}
    train_mean_loss, train_mean_accuracy = {}, {}

    # record of defed
    for p in p_list:
        file_name = '{}save/node{}/objects/New_fed_test_each_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_M[{}]_p[{}].pkl'. \
            format(path_data, num_users, dataset, model, epochs, frac, iid, local_ep, local_bs, method, p)
        with open(file_name, 'rb') as f:
            a, b = pickle.load(f)  # defed
            test_mean_accuracy[p], test_mean_loss[p] = np.mean(a, axis=0), np.mean(b, axis=0)
        file_name = '{}save/node{}/objects/New_fed_w_only_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_M[{}]_p[{}].pkl'. \
            format(path_data, num_users, dataset, model, epochs, frac, iid, local_ep, local_bs, method, p)
        with open(file_name, 'rb') as f:
            a, b = pickle.load(f)
            train_mean_loss[p], train_mean_accuracy[p] = np.mean(a, axis=0), np.mean(b, axis=0)

    # path_data = f'../results/A2_logistic_round1500_seed{seed}/'
    # record of fedavg
    file_name = '{}save/node{}/objects/fed_test_acc-vs-round_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
        format(path_data, num_users, dataset, model, epochs, frac, iid, local_ep, local_bs)
    with open(file_name, 'rb') as f:
        a, b = pickle.load(f)  # fedavg
        test_accuracy_fedavg, test_loss_fedavg = list(a.values()), list(b.values())
    file_name = '{}save/node{}/objects/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
        format(path_data, num_users, dataset, model, epochs, frac, iid, local_ep, local_bs)
    with open(file_name, 'rb') as f:
        train_loss_fedavg, train_accuracy_fedavg = pickle.load(f)

    # # record of time-varying fedavg
    # file_name = '{}save/node{}/objects/fed_test_each_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
    #     format(path_data, num_users, dataset, model, epochs, frac, iid, local_ep, local_bs)
    # with open(file_name, 'rb') as f:
    #     a, b = pickle.load(f)  # fedavg
    #     test_accuracy_fedavg, test_loss_fedavg = np.mean(a, axis=0), np.mean(b, axis=0)
    #
    # file_name = '{}save/node{}/objects/fed_train_each_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
    #     format(path_data, num_users, dataset, model, epochs, frac, iid, local_ep, local_bs)
    # with open(file_name, 'rb') as f:
    #     a, b = pickle.load(f)
    #     train_loss_fedavg, train_accuracy_fedavg = np.mean(a, axis=0), np.mean(b, axis=0)

    # # record of sl
    # test_loss_sl_all, test_accuracy_sl_all = [], []
    # train_loss_sl_all, train_accuracy_sl_all = [], []
    # path_data_sl = f'../sl/{dataset_dict[dataset]}_{model}_{iid_state}_node{num_users}-my_sl/'
    # for node in range(1, num_users+1):
    #     file_name = f'{path_data_sl}node{node}/model/log/' \
    #                 f'swarm_test_{dataset}_{model}_{num_users}_{epochs}_iid[{iid}]_node{node-1}.pkl'
    #     with open(file_name, 'rb') as f:
    #         test_accuracy_sl_dict, test_loss_sl_dict = pickle.load(f)
    #         test_accuracy_sl_each, test_loss_sl_each =\
    #             list(test_accuracy_sl_dict.values()), list(test_loss_sl_dict.values())
    #         test_accuracy_sl_all.append(test_accuracy_sl_each)
    #         test_loss_sl_all.append(test_loss_sl_each)
    #
    #     file_name = f'{path_data_sl}node{node}/model/log/' \
    #                 f'swarm_train_{dataset}_{model}_{num_users}_{epochs}_iid[{iid}]_node{node - 1}.pkl'
    #     with open(file_name, 'rb') as f:
    #         train_accuracy_sl_dict, train_loss_sl_dict = pickle.load(f)
    #         train_accuracy_sl_each, train_loss_sl_each = \
    #             list(train_accuracy_sl_dict.values()), list(train_loss_sl_dict.values())
    #         train_accuracy_sl_all.append(train_accuracy_sl_each)
    #         train_loss_sl_all.append(train_loss_sl_each)
    # test_accuracy_sl = np.mean(test_accuracy_sl_all, axis=0)
    # test_loss_sl = np.mean(test_loss_sl_all, axis=0)
    # train_accuracy_sl = np.mean(train_accuracy_sl_all, axis=0)
    # train_loss_sl = np.mean(train_loss_sl_all, axis=0)

    x = range(1, epochs+1)
    # y_x_lim_sel = [None, None, None, None]
    y_x_lim_sel = y_x_lim_dict[f'{dataset}_{model}_{iid_state}_node{num_users}']
    print(y_x_lim_sel)

    single_plot(x=x,
                y_fedavg=test_accuracy_fedavg, y_defed=test_mean_accuracy,
                # y_sl=test_accuracy_sl,
                title=f'Dataset {dataset_dict[dataset]}: Test Accuracy vs Communication rounds',
                ylabel='Test Accuracy %', xlabel='Communication Rounds',
                ylim=y_x_lim_sel[0],
                xlim=y_x_lim_sel[3],
                name_save=f'{path_save}node{num_users}-{iid_state}-{model}-test-accu',
                m=100, lc='lower right')
    single_plot(x=x,
                y_fedavg=train_accuracy_fedavg, y_defed=train_mean_accuracy,
                # y_sl=train_accuracy_sl,
                title=f'Dataset {dataset_dict[dataset]}: Train Accuracy vs Communication rounds',
                ylabel='Train Accuracy %', xlabel='Communication Rounds',
                ylim=y_x_lim_sel[1],
                xlim=y_x_lim_sel[3],
                name_save=f'{path_save}node{num_users}-{iid_state}-{model}-train-accu',
                m=100, lc='lower right')
   
    #for p in p_list:
    #    train_mean_loss[p] = train_mean_loss[p][1::2]
    
    single_plot(x=x,
                y_fedavg=train_loss_fedavg, y_defed=train_mean_loss,
                # y_sl=train_loss_sl,
                title=f'Dataset {dataset_dict[dataset]}: Training Loss vs Communication rounds',
                ylabel='Train Loss', xlabel='Communication Rounds',
                ylim=y_x_lim_sel[2],
                xlim=y_x_lim_sel[3],
                name_save=f'{path_save}node{num_users}-{iid_state}-{model}-train-loss',
                m=1, lc='upper right')


