import os
import pickle
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


class DataDistribution:
    def __init__(self, num_classes, num_nodes, category_names):
        self.num_classes    = num_classes
        self.num_nodes      = num_nodes
        self.category_names = category_names
        self.dataset        = 'CWRU'

    def plot(self, results_iid, results_noniid):
        self.plot_single_figure(results_iid,    f'../results/{self.num_classes}分类/{self.dataset}/data_distribution_node{self.num_nodes}_iid')
        self.plot_single_figure(results_noniid, f'../results/{self.num_classes}分类/{self.dataset}/data_distribution_node{self.num_nodes}_noniid')

    def plot_single_figure(self, results, save_path):
        labels          = list(results.keys())
        data            = np.array(list(results.values()))
        data_cum        = data.cumsum(axis=1)
        category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, data.shape[1]))

        fig, ax = plt.subplots(figsize=(5, 2))
        ax.invert_yaxis()
        ax.set_xlim(0, np.sum(data, axis=1).max())

        for i, (colname, color) in enumerate(zip(self.category_names, category_colors)):
            widths = data[:, i]
            starts = data_cum[:, i] - widths
            rects  = ax.barh(labels, widths, left=starts, height=0.5, label=colname, color=color)

        ncol = min(4, len(self.category_names))
        ax.legend(ncol=ncol, bbox_to_anchor=(0, 1), loc='lower left', fontsize='small')
        ax.set_xlabel('Sample number')
        ax.spines['top']  .set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.savefig(save_path + '.svg', bbox_inches='tight')
        plt.savefig(save_path + '.pdf', bbox_inches='tight')


class Accuracy:
    def __init__(self, x, p, num_classes, num_nodes):
        self.x           = x
        self.p           = p
        self.dataset     = 'CWRU'
        self.num_classes = num_classes
        self.num_nodes   = num_nodes
        self.alone_fp    = lambda model, iid, round_:   f'../results/{self.num_classes}分类/{self.dataset}/node{self.num_nodes}/objects/alone_test_cwru_{model}_{self.num_nodes}_{round_}_iid[{iid}].pkl'
        self.defed_fp    = lambda model, iid, local_ep: f'../results/{self.num_classes}分类/{self.dataset}/node{self.num_nodes}/objects/New_fed_test_each_cwru_{model}_300_C[1]_iid[{iid}]_E[{local_ep}]_B[64]_M[er]_p[{self.p}].pkl'
        self.save_path   = lambda model, iid:           f'../results/{self.num_classes}分类/{self.dataset}/test_acc_node{self.num_nodes}_{model}_iid[{iid}]_p[{self.p}]'

    def plot(self, logistic_round, dnn_round):
        self.plot_single_figure('logistic', iid=1, alone_round=logistic_round, local_ep=10)
        self.plot_single_figure('logistic', iid=0, alone_round=logistic_round, local_ep=10)
        self.plot_single_figure('dnn',      iid=1, alone_round=dnn_round,      local_ep=30)
        self.plot_single_figure('dnn',      iid=0, alone_round=dnn_round,      local_ep=30)

    def plot_single_figure(self, model, iid, alone_round, local_ep):
        # 加载数据
        with open(self.alone_fp(model, iid, alone_round), 'rb') as f:
            acc_dict, loss_dict = pickle.load(f)
        accs_alone = [100 * acc_dict[i][-1] for i in range(len(acc_dict))]

        with open(self.defed_fp(model, iid, local_ep), 'rb') as f:
            acc_list, loss_list = pickle.load(f)
        acc_defed = [100 * np.mean(acc_list, axis=0)[-1]]

        accs = accs_alone + acc_defed

        # 画图
        fig, ax = plt.subplots(figsize=(4, 2))
        rects   = ax.bar(self.x, accs, width=0.5, alpha=0.9)

        ax.set_ylabel('Test Accuracy %')
        # ax.set_ylim(75, 100)

        ax.tick_params("y", which='major', length=5, width=0.5, colors='k', direction='in')  # "y", 'x', 'both'
        ax.spines['top']  .set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.savefig(self.save_path(model, iid) + '.svg', bbox_inches='tight')
        plt.savefig(self.save_path(model, iid) + '.pdf', bbox_inches='tight')


if __name__ == '__main__':
    # p = DataDistribution(num_classes=4, num_nodes=4, category_names=['Normal', 'B007', 'IR007', 'OR007'])
    # p.plot(
    #     results_iid    = {'C1': [306, 284, 307, 303], 'C2': [305, 300, 285, 310], 'C3': [313, 308, 291, 288], 'C4': [276, 308, 317, 299]},
    #     results_noniid = {'C1': [348, 228, 233, 138], 'C2': [600, 144, 317, 592], 'C3': [0, 290, 247,  18],   'C4': [252, 538, 403, 452]},
    # )
    #
    # p = DataDistribution(num_classes=10, num_nodes=4, category_names=['Normal', 'B007', 'B014', 'B021', 'IR007', 'IR014', 'IR021', 'OR007', 'OR014', 'OR021'])
    # p.plot(
    #     results_iid    = {'C1': (332, 297, 291, 336, 299, 275, 273, 303, 295, 299), 'C2': (292, 281, 323, 310, 271, 331, 340, 284, 287, 281), 'C3': (281, 311, 303, 265, 321, 300, 257, 331, 317, 314), 'C4': (295, 311, 283, 289, 309, 294, 330, 282, 301, 306)},
    #     results_noniid = {'C1': [348, 228, 233, 138, 381, 346, 555, 130, 460, 360], 'C2': [600, 144, 317, 592, 510, 419, 566, 672, 256, 437], 'C3': [  0, 290, 247,  18, 128, 136,  54,  76, 332,  10], 'C4': [252, 538, 403, 452, 181, 299,  25, 322, 152, 393]},
    # )

    p = Accuracy(x=['C1', 'C2', 'C3', 'C4', 'DeceFL'], p=0.9, num_classes=4, num_nodes=4)
    p.plot(logistic_round=5, dnn_round=50)

    p = Accuracy(x=['C1', 'C2', 'C3', 'C4', 'DeceFL'], p=0.9, num_classes=10, num_nodes=4)
    p.plot(logistic_round=5, dnn_round=50)


