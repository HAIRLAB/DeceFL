# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jindou time: 2021/8/2
import pathlib
import pickle
import numpy as np
from functools import partial

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


class LineChart:
    def __init__(self, n_class, num_users):
        self.n_class   = n_class
        self.seed      = 1
        self.method    = 'er'
        self.frac      = 1
        self.local_bs  = 64
        self.num_users = num_users
        self.p_list    = [0.9, 0.7, 0.5, 0.3]  # 选择需要绘制的p
        self.save_bool = True                  # 是否保存图片
        self.ylim_bool = False                  # 是否对坐标范围进行限制
        self.dataset_dict = {'sl_a': 'A2', 'sl_b': 'B', 'sl_e': 'E', 'cwru': 'CWRU'}

        if len(self.p_list) == 1:
            self.set_p = f'p={self.p_list[0]}'
            self.color = ['coral', 'g']
        else:
            self.set_p = 'total_p'
            self.color = ['orangered', 'orange', 'y', 'g', 'deepskyblue', 'royalblue', 'plum']

        self.y_x_lim_dict = {
            'sl_a_logistic_iid_node4':     [[95, 99.2, 1], [95, 100.2, 1], None, [-6, 200, 50]],
            'sl_a_logistic_iid_node8':     [[95, 99.2, 1], [95, 100.2, 1], None, [-12, 400, 100]],
            'sl_a_logistic_iid_node16':    [[95, 99.2, 1], [95, 100.2, 1], None, [-24, 800, 200]],
            'sl_a_dnn_iid_node4':          [[47, 100, 10], [47, 102, 10], [-0.03, 0.7, 0.1], [0, 50, 10]],
            # 'sl_a_dnn_iid_node8':          [[65, 100, 10], [65, 101, 10], [-0.03, 0.7, 0.1], [-5, 150, 30]],
            'sl_a_dnn_iid_node8':          [[47, 100, 10], [47, 102, 10], [-0.03, 0.7, 0.1], [-2, 60, 10]],
            'sl_a_dnn_iid_node16':         [[47, 100, 10], [47, 102, 10], [-0.03, 0.7, 0.1], [-5, 150, 30]],
            'sl_b_logistic_iid_node4':     [[80, 90, 2], None, [0.3, 0.55, 0.05], None],
            'sl_b_logistic_iid_node8':     [[80, 90, 2], None, [0.3, 0.55, 0.05], None],
            'sl_b_logistic_iid_node16':    [[80, 90, 2], None, [0.3, 0.55, 0.05], None],
            'sl_b_dnn_iid_node4':          [None, None, None, [-5, 100, 20]],
            'sl_b_dnn_iid_node8':          [None, None, None, [-10, 200, 50]],
            'sl_b_dnn_iid_node16':         [None, None, None, [-10, 300, 50]],
            'sl_e_logistic_iid_node4':     [None, None, None, None],
            'sl_e_logistic_iid_node8':     [None, None, None, None],
            'sl_e_logistic_iid_node16':    [None, None, None, None],
            'sl_e_dnn_iid_node4':          [None, None, None, [-5, 100, 20]],
            'sl_e_dnn_iid_node8':          [None, None, None, [-5, 100, 20]],
            'sl_e_dnn_iid_node16':         [None, None, None, [-10, 300, 50]],
            'cwru_dnn_iid_node4':          [None, None, None, [-1, 50, 10]],
            'cwru_dnn_iid_node8':          [None, None, None, [-1, 50, 10]],
            'cwru_dnn_iid_node16':         [None, None, None, [-2, 100, 20]],
            'cwru_logistic_iid_node4':     [None, None, None, [-1, 50, 10]],
            'cwru_logistic_iid_node8':     [None, None, None, [-1, 50, 10]],
            'cwru_logistic_iid_node16':    [None, None, None, [-2, 100, 20]],
            'sl_a_logistic_noniid_node4':  [[95, 99.1, 1], [95, 100.2, 1], None, [-9, 300, 60]],
            'sl_a_logistic_noniid_node8':  [[95, 99.1, 1], [95, 100.2, 1], None, [-18, 600, 120]],
            'sl_a_logistic_noniid_node16': [[95, 99.1, 1], [95, 100.2, 1], None, [-27, 900, 180]],
            'sl_a_dnn_noniid_node4':       [[47, 100, 10], [60, 101, 10], [-0.03, 0.7, 0.1], [-3, 100, 20]],
            'sl_a_dnn_noniid_node8':       [[47, 100, 10], [57, 101, 10], [-0.03, 0.7, 0.1], [-5, 150, 30]],
            'sl_a_dnn_noniid_node16':      [[47, 100, 10], [57, 101, 10], [-0.03, 0.7, 0.1], [-6, 200, 50]],
            'cwru_dnn_noniid_node4':       [None, None, None, [-1, 50, 10]],
            'cwru_dnn_noniid_node8':       [None, None, None, [-1, 50, 10]],
            'cwru_dnn_noniid_node16':      [None, None, None, [-2, 100, 20]],
            'cwru_logistic_noniid_node4':  [None, None, None, [-1, 50, 10]],
            'cwru_logistic_noniid_node8':  [None, None, None, [-1, 50, 10]],
            'cwru_logistic_noniid_node16': [None, None, None, [-2, 100, 20]]
        }

    def plot(self, dataset, model, iid, local_ep, epochs):
        iid_state = 'iid' if iid else 'noniid'
        x = range(1, epochs + 1)
        y_x_lim_sel = self.y_x_lim_dict[f'{dataset}_{model}_{iid_state}_node{self.num_users}']
        print(y_x_lim_sel)

        root_dir = pathlib.Path(f'../results/{self.n_class}分类/{self.dataset_dict[dataset]}')
        read_dir = root_dir / pathlib.Path(f'node{self.num_users}/objects')
        assert read_dir.exists(), '路径不存在'

        suffix_decefl = lambda p: f'{dataset}_{model}_{epochs}_C[{self.frac}]_iid[{iid}]_E[{local_ep}]_B[{self.local_bs}]_M[{self.method}]_p[{p}].pkl'
        suffix_fedavg =           f'{dataset}_{model}_{epochs}_C[{self.frac}]_iid[{iid}]_E[{local_ep}]_B[{self.local_bs}].pkl'
        epochs_sl     = epochs
        suffix_sl     = lambda node: f'{dataset}_{model}_{self.num_users}_{epochs_sl}_iid[{iid}]_node{node}.pkl'
        prefix_save   = f'node{self.num_users}-{iid_state}-{model}'

        d = {}
        # DeceFL
        d_tmp = self.load_decefl_data(
            test_path  = lambda p: read_dir / f'New_fed_test_each_{suffix_decefl(p)}',
            train_path = lambda p: read_dir / f'New_fed_w_only_{suffix_decefl(p)}',
        )
        d.update(d_tmp)

        # FedAvg
        d_tmp = self.load_fedavg_data(
            test_path  = read_dir / f'fed_test_acc-vs-round_{suffix_fedavg}',
            train_path = read_dir / f'fed_{suffix_fedavg}',
        )
        d.update(d_tmp)

        # SL
        d_tmp = self.load_sl_data(
            test_path      = lambda node: read_dir / f'swarm_test_{suffix_sl(node)}',
            train_path     = lambda node: read_dir / f'swarm_train_{suffix_sl(node)}'
        )
        d.update(d_tmp)

        self.plot_single_figure(x=x,
                                y_fedavg=d['fedavg_acc_test'], y_defed=d['decefl_acc_test'],
                                y_sl=d['sl_acc_test'],
                                title=f'Dataset {self.dataset_dict[dataset]}: Test Accuracy vs Communication rounds',
                                ylabel='Test Accuracy %', xlabel='Communication Rounds',
                                ylim=y_x_lim_sel[0],
                                xlim=y_x_lim_sel[3],
                                save_path=root_dir / f'{prefix_save}-test-accu',
                                m=100, lc='lower right')
        self.plot_single_figure(x=x,
                                y_fedavg=d['fedavg_acc_train'], y_defed=d['decefl_acc_train'],
                                y_sl=d['sl_acc_train'],
                                title=f'Dataset {self.dataset_dict[dataset]}: Train Accuracy vs Communication rounds',
                                ylabel='Train Accuracy %', xlabel='Communication Rounds',
                                ylim=y_x_lim_sel[1],
                                xlim=y_x_lim_sel[3],
                                save_path=root_dir / f'{prefix_save}-train-accu',
                                m=100, lc='lower right')
        self.plot_single_figure(x=x,
                                y_fedavg=d['fedavg_loss_train'], y_defed=d['decefl_loss_train'],
                                y_sl=d['sl_loss_train'],
                                title=f'Dataset {self.dataset_dict[dataset]}: Training Loss vs Communication rounds',
                                ylabel='Train Loss', xlabel='Communication Rounds',
                                ylim=y_x_lim_sel[2],
                                xlim=y_x_lim_sel[3],
                                save_path=root_dir / f'{prefix_save}-train-loss',
                                m=1, lc='upper right')

    def load_decefl_data(self,
                         test_path,
                         train_path) -> dict:
        loss_test,  acc_test  = dict(), dict()
        loss_train, acc_train = dict(), dict()

        for p in self.p_list:
            with open(test_path(p), 'rb') as f:
                acc, loss = pickle.load(f)  # defed
                loss_test[p],  acc_test[p]  = np.mean(loss, axis=0), np.mean(acc, axis=0)
            with open(train_path(p), 'rb') as f:
                loss, acc = pickle.load(f)
                loss_train[p], acc_train[p] = np.mean(loss, axis=0), np.mean(acc, axis=0)

        res = {
            'decefl_loss_test':  loss_test,
            'decefl_loss_train': loss_train,
            'decefl_acc_test':   acc_test,
            'decefl_acc_train':  acc_train,
        }
        return res

    def load_fedavg_data(self,
                         test_path,
                         train_path, ) -> dict:
        with open(test_path, 'rb') as f:
            acc, loss = pickle.load(f)
            loss_test,  acc_test  = list(loss.values()), list(acc.values())
        with open(train_path, 'rb') as f:
            loss_train, acc_train = pickle.load(f)

        # # record of time-varying fedavg
        # file_name = '{}save/node{}/objects/fed_test_each_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
        #     format(path_data, num_users, dataset, model, epochs, frac, iid, local_ep, local_bs)
        # with open(file_name, 'rb') as f:
        #     a, b = pickle.load(f)  # fedavg
        #     fedavg_acc_test, fedavg_loss_test = np.mean(a, axis=0), np.mean(b, axis=0)
        #
        # file_name = '{}save/node{}/objects/fed_train_each_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
        #     format(path_data, num_users, dataset, model, epochs, frac, iid, local_ep, local_bs)
        # with open(file_name, 'rb') as f:
        #     a, b = pickle.load(f)
        #     fedavg_loss_train, fedavg_acc_train = np.mean(a, axis=0), np.mean(b, axis=0)

        res = {
            'fedavg_loss_test':  loss_test,
            'fedavg_loss_train': loss_train,
            'fedavg_acc_test':   acc_test,
            'fedavg_acc_train':  acc_train,
        }
        return res

    def load_sl_data(self,
                     test_path,
                     train_path) -> dict:
        loss_test,  acc_test  = [], []
        loss_train, acc_train = [], []

        for node in range(self.num_users):
            with open(test_path(node), 'rb') as f:
                test_acc_dict, test_loss_dict = pickle.load(f)
                test_acc_each, test_loss_each = list(test_acc_dict.values()), list(test_loss_dict.values())
                acc_test.append(test_acc_each)
                loss_test.append(test_loss_each)

            with open(train_path(node), 'rb') as f:
                train_acc_dict, train_loss_dict = pickle.load(f)
                train_acc_each, train_loss_each = list(train_acc_dict.values()), list(train_loss_dict.values())
                acc_train.append(train_acc_each)
                loss_train.append(train_loss_each)

        acc_test   = np.mean(acc_test,   axis=0)
        loss_test  = np.mean(loss_test,  axis=0)
        acc_train  = np.mean(acc_train,  axis=0)
        loss_train = np.mean(loss_train, axis=0)

        res = {
            'sl_loss_test':  loss_test,
            'sl_loss_train': loss_train,
            'sl_acc_test':   acc_test,
            'sl_acc_train':  acc_train,
        }
        return res

    def plot_single_figure(self,
                           x,
                           y_fedavg=None,
                           y_defed=None,
                           y_sl=None,
                           save_path=None,
                           title=None,
                           xlabel=None,
                           ylabel=None,
                           lc=None,
                           m=None,
                           xlim=None,
                           ylim=None):
        plt.figure(figsize=(5, 3))

        if y_sl is not None:
            x_sl = range(1, len(y_sl)+1)
            plt.plot(x_sl, np.array(y_sl) * m, color=self.color[-1], label='SL')

        if y_fedavg is not None:
            plt.plot(x, np.array(y_fedavg) * m, color=self.color[-2], label='FedAvg')

        if y_defed is not None:
            for i, p in enumerate(self.p_list):
                plt.plot(x, np.array(y_defed[p]) * m, color=self.color[i], label=f'DeceFL: p={p}')

        # plt.title(kwargs['title'], pad=20)
        plt.ylabel(ylabel, fontsize=15)
        plt.xlabel(xlabel, fontsize=15)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        # plt.legend(loc='best', fontsize=10)
        plt.legend(loc=lc, fontsize=14)
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left']  .set_linewidth(0.5)
        ax.spines['right'] .set_visible(False)
        ax.spines['top']   .set_visible(False)
        # ax.set_xscale('log')

        if self.ylim_bool:
            if xlim:
                plt.xlim(xlim[0], xlim[1])
                ax.xaxis.set_major_locator(MultipleLocator(xlim[2]))
            if ylim:
                plt.ylim(ylim[0], ylim[1])
                ax.yaxis.set_major_locator(MultipleLocator(ylim[2]))

        ax.tick_params("both", which='major', length=5, width=0.5, colors='k', direction='in')  # "y", 'x', 'both'
        # ax.tick_params(which='minor', length=5, width=1.0, labelsize=10, labelcolor='0.6', direction='in')

        if self.save_bool:
            plt.savefig(save_path.parent / (save_path.name + '.svg'), bbox_inches='tight')
            plt.savefig(save_path.parent / (save_path.name + '.pdf'), bbox_inches='tight')
        else:
            plt.show()


if __name__ == '__main__':
    comb = [
        # ['sl_a', 'logistic', 1, 10, 1500],
        # ['sl_a', 'logistic', 0, 10, 1500],
        # ['sl_a', 'dnn',      1, 30, 300],
        # ['sl_a', 'dnn',      0, 30, 300],
        # ['sl_b', 'logistic', 1, 30, 300],
        # ['sl_b', 'dnn',      1, 30, 300],
        # ['sl_e', 'logistic', 1, 30, 300],
        # ['sl_e', 'dnn',      1, 30, 300],
        ['cwru', 'dnn',      1, 30, 300],
        ['cwru', 'dnn',      0, 30, 300],
        ['cwru', 'logistic', 1, 10, 300],
        ['cwru', 'logistic', 0, 10, 300]
    ]

    for c in comb:
        dataset, model, iid, local_ep, epochs = c
        p = LineChart(n_class=10, num_users=4)
        p.plot(dataset, model, iid, local_ep, epochs)
