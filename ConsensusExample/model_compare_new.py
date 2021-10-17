# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jindou time: 2021/8/11
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
num_users = 8
epoch = 1000
y_list = [1, 1, 1, 1, 1, 1, 1, 1]
color = ['b', 'peru', 'c', 'r', 'm', 'y', 'k', 'g']
p = 0.3


def get_score_single(state_dict, ymean):
    weight = state_dict['layer_input.weight'].cpu().numpy().reshape(-1)
    bias = state_dict['layer_input.bias'].cpu().numpy()
    output = np.linalg.norm(weight+bias - ymean)
    print(f'weight:{weight} | bias:{bias} | output:{output}')
    return output


if __name__ == '__main__':

    with open(f'../save/node{num_users}/objects/defed_final_state_p{p}.pkl', 'rb') as f:
        defed_final_state = pickle.load(f)
    # with open(f'./save/node{num_users}/objects/defed_ini_state_p{p}.pkl', 'rb') as f:
    #     defed_ini_state = pickle.load(f)
    # with open(f'./save/node{num_users}/objects/fedavg_ini_state.pkl', 'rb') as f:
    #     fedavg_ini_state = pickle.load(f)
    with open(f'../save/node{num_users}/objects/fedavg_final_state.pkl', 'rb') as f:
        fedavg_final_state = pickle.load(f)

    with open(f'../save/node{num_users}/sl_final_state.pkl', 'rb') as f:
        sl_final_state = pickle.load(f)

    defed_score = []
    fedavg_score = []
    sl_score = []

    y_mean = np.mean(y_list)

    for i in range(num_users):
        defed_score.append([])

    for e in tqdm(range(epoch)):
        fedavg_s = get_score_single(fedavg_final_state[e], y_mean)
        fedavg_score.append(fedavg_s)

        sl_s = get_score_single(sl_final_state[e], y_mean)
        sl_score.append(sl_s)

        for users in range(num_users):
            defed_s = get_score_single(defed_final_state[users][e], y_mean)
            defed_score[users].append(defed_s)

    plt.figure(figsize=(5, 3), dpi=500)
    # plt.title(f'compare: p={p}')
    plt.ylabel('$\Vert w-w^* \Vert$', fontsize=12)
    plt.xlabel('Communication Rounds', fontsize=12)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params("both", which='major', length=5, width=0.5, colors='k', direction='in')
    plt.plot(sl_score, color='plum', label=f'SL')
    plt.plot(fedavg_score, color='coral', label=f'FedAvg')
    for i in range(num_users):
        plt.plot(defed_score[i], color=color[i], label=f'DeceFL {i}: p={p}')
    plt.legend(fontsize=9)
    # plt.savefig(f'./save/compare2_node{num_users}_p{p}.png')
    plt.savefig(f'../save/consensus_p{int(p*10)}.pdf', bbox_inches='tight')
    plt.show()

