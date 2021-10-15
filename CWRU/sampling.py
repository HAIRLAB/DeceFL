#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random

import numpy as np
from collections import defaultdict, Counter
from torchvision import datasets, transforms


def data_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    # np.random.seed(0)
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    # 输出各个节点的数据信息
    idx2label = dict()
    for i, (x, y) in enumerate(dataset):
        idx2label[i] = int(y)

    num_node_class = dict()
    for i in range(num_users):
        c = Counter([idx2label[j] for j in dict_users[i]])
        num_node_class[f'C{i+1}'] = list(zip(*sorted(c.items())))[1]
    print("每个节点的数据信息：\n", num_node_class)

    return dict_users


def data_noniid_split(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_items = int(len(dataset)/(num_users*2))
    dict_users_doub, dict_users, all_idxs = {}, {}, [i for i in range(len(dataset))]
    labels_list = []
    for i in range(len(dataset)):
        labels_list.append(dataset.__getitem__(i)[1])

    labels_set = set(labels_list)

    zipped = zip(labels_list,all_idxs)
    sort_zipped = sorted(zipped,key=lambda x: x[0])
    result = zip(*sort_zipped)
    labels_list_sort, all_idxs_sort = [list(x) for x in result]

    #dict_users = [all_idxs_sort[i:i + num_items] for i in range(0, len(all_idxs_sort), num_items)]
    for idx, i in enumerate(range(0, len(all_idxs_sort), num_items)):
        dict_users_doub[idx] = all_idxs_sort[i:i + num_items]
    # for i in range(num_users):
    #     idx_u = set(np.random.choice(labels_set, 2, replace=False))
    #     for j in idx_u:
    #         dict_users[i] = set(np.random.choice(np.where(labels_array == j), num_items,
    #                                              replace=False))
    #         all_idxs = list(set(all_idxs) - dict_users[i])

    idxs_users = np.random.choice(range(num_users*2), num_users*2, replace=False)
    for idx,i in enumerate(range(0, len(idxs_users), 2)):
        dict_users[idx] = dict_users_doub[idxs_users[i]]+dict_users_doub[idxs_users[i+1]]

    return dict_users

def data_noniid_split5(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_items = int(len(dataset)/(num_users*5))
    dict_users_doub, dict_users, all_idxs = {}, {}, [i for i in range(len(dataset))]
    labels_list = []
    for i in range(len(dataset)):
        labels_list.append(dataset.__getitem__(i)[1])

    labels_set = set(labels_list)

    zipped = zip(labels_list,all_idxs)
    sort_zipped = sorted(zipped,key=lambda x: x[0])
    result = zip(*sort_zipped)
    labels_list_sort, all_idxs_sort = [list(x) for x in result]

    #dict_users = [all_idxs_sort[i:i + num_items] for i in range(0, len(all_idxs_sort), num_items)]
    for idx, i in enumerate(range(0, len(all_idxs_sort), num_items)):
        dict_users_doub[idx] = all_idxs_sort[i:i + num_items]
    # for i in range(num_users):
    #     idx_u = set(np.random.choice(labels_set, 2, replace=False))
    #     for j in idx_u:
    #         dict_users[i] = set(np.random.choice(np.where(labels_array == j), num_items,
    #                                              replace=False))
    #         all_idxs = list(set(all_idxs) - dict_users[i])

    idxs_users = np.random.choice(range(num_users*5), num_users*5, replace=False)
    for idx,i in enumerate(range(0, len(idxs_users), 5)):
        dict_users[idx] = dict_users_doub[idxs_users[i]]+dict_users_doub[idxs_users[i+1]]+dict_users_doub[idxs_users[i+2]]+dict_users_doub[idxs_users[i+3]]+dict_users_doub[idxs_users[i+4]]

    return dict_users


def data_noniid_split_new(dataset, num_users):
    dict_users_doub, dict_users, all_idxs = {}, {}, [i for i in range(len(dataset))]
    labels_list = []
    for i in range(len(dataset)):
        labels_list.append(dataset.__getitem__(i)[1])

    zipped = zip(labels_list, all_idxs)
    sort_zipped = sorted(zipped, key=lambda x: x[0])
    result = zip(*sort_zipped)
    labels_list_sort, all_idxs_sort = [list(x) for x in result]

    end_idx = int(divmod(len(all_idxs_sort), num_users)[0]*num_users)

    for i in range(num_users):
        # sel = range(i, len(all_idxs_sort), num_users)
        dict_users[i] = all_idxs_sort[i:end_idx:num_users]

    return dict_users


def data_noniid_unbalanced(dataset, num_users):
    """
    Sample non-I.I.D unbalanced client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # num_items = int(len(dataset)/num_users)
    num_lower_bound = int(len(dataset)/(num_users*5))
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users-1):
        #print(num_lower_bound,int(len(all_idxs)/num_users))
        num_items_i = np.random.choice(range(num_lower_bound,int(2*len(all_idxs)/num_users)), 1)
        #print(num_items_i)
        dict_users[i] = set(np.random.choice(all_idxs, num_items_i,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    
    dict_users[num_users-1] = all_idxs
    return dict_users


def data_noniid_example1(dataset, num_users, n_class):
    # if num_users == 4:
    #     pNode = [0.125, 0.125, 0.25, 0.5]
    #     pEach = [[0.01, 0.99], [0.99, 0.01], [0.3, 0.7], [0.6, 0.4]]
    # if num_users == 8:
    #     # num_n = [104, 104, 307, 307, 311, 311, 313, 313]
    #     # num_p = [104, 104, 3, 3, 207, 207, 721, 721]
    #     pNode = [0.05, 0.05, 0.075, 0.075, 0.125, 0.125, 0.25, 0.25]
    #     pEach = [[0.5, 0.5], [0.5, 0.5], [0.01, 0.99], [0.01, 0.99], [0.4, 0.6], [0.4, 0.6], [0.7, 0.3], [0.7, 0.3]]
    # elif num_users == 16:
    #     # num_n = [52, 52, 52, 52, 153, 153, 153, 153, 156, 156, 156, 156, 156, 156, 156, 156]
    #     # num_p = [52, 52, 52, 52, 2, 2, 2, 2, 103, 103, 103, 103, 361, 361, 361, 361]
    #     pNode = [0.025, 0.025, 0.025, 0.025, 0.0375, 0.0375, 0.0375, 0.0375,
    #              0.0625, 0.0625, 0.0625, 0.0625, 0.125, 0.125, 0.125, 0.125]
    #     pEach = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.01, 0.99], [0.01, 0.99], [0.01, 0.99], [0.01, 0.99],
    #              [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.7, 0.3], [0.7, 0.3], [0.7, 0.3], [0.7, 0.3]]
    # elif num_users == 10:
    #     # num_n = [104, 104, 307, 307, 3, 3, 291, 291, 330, 330]
    #     # num_p = [104, 104, 3, 3, 307, 307, 125, 125, 496, 496]
    #     pNode = [0.05, 0.05, 0.075, 0.075, 0.075, 0.075, 0.1, 0.1, 0.2, 0.2]
    #     pEach = [[0.5, 0.5], [0.5, 0.5], [0.01, 0.99],  [0.01, 0.99], [0.99, 0.01], [0.99, 0.01],
    #              [0.3, 0.7], [0.3, 0.7], [0.6, 0.4], [0.6, 0.4]]

    # num_total = len(dataset)
    # num_p, num_n, num_total = [], [], len(dataset)
    # for p1, p2 in zip(pNode, pEach):
    #     num_p.append(int(p1*p2[0]*num_total))
    #     num_n.append(int(p1*p2[1]*num_total))
    #
    # dict_users  = {}
    # all_idxs    = list(range(len(dataset)))
    # labels_list = [y for x, y in dataset]
    #
    # zipped = zip(labels_list, all_idxs)
    # sort_zipped = sorted(zipped, key=lambda x: x[0])
    # result = zip(*sort_zipped)
    # labels_list_sort, all_idxs_sort = [list(x) for x in result]
    #
    # for i in range(num_users):
    #     dict_users[i] = all_idxs_sort[sum(num_n[:i]):sum(num_n[:i+1])] + \
    #                     all_idxs_sort[int(num_total/2)+sum(num_p[:i]):int(num_total/2)+sum(num_p[:i+1])]

    # labels_dict: {class: [0, 1,2,3...], }， 每类样本索引
    labels_dict = defaultdict(list)
    for i, (x, y) in enumerate(dataset):
        labels_dict[int(y)].append(i)

    # labels_num: 各类样本数量
    labels_num = np.array([len(labels_dict[i]) for i in range(n_class)])
    labels_num = labels_num[:, np.newaxis]

    # sample_idx: 随机 end 索引, shape = (类别数, 节点数)
    np.random.seed(1)
    rate  = np.random.uniform(0, 1, (n_class, num_users))
    rate /= rate.sum(axis=1, keepdims=True)
    rate  = rate.cumsum(axis=1)
    sample_idx = (rate * labels_num).round().astype(int)

    sample_num = np.diff(sample_idx, axis=1, prepend=0)
    print('节点样本数：\n', sample_num.sum(axis=0))
    print('节点样本比例：\n', (sample_num.sum(axis=0) / sample_num.sum()).round(2))
    print('节点类目数：\n', sample_num)
    print('节点类目比例：\n', (sample_num / sample_num.sum(axis=0, keepdims=True)).round(2))

    # 给每个节点分配样本
    dict_users = defaultdict(list)
    for u in range(num_users):
        for c in range(n_class):
            l = sample_idx[c][u-1] if u != 0 else None
            r = sample_idx[c][u]
            dict_users[u].extend(labels_dict[c][l:r])

    assert set(sum([s for c, s in dict_users.items()], [])) == set(range(len(dataset))), '落下了某些样本'

    return dict_users



# def mnist_iid(dataset, num_users):
#     """
#     Sample I.I.D. client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     num_items = int(len(dataset)/num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items,
#                                              replace=False))
#     return dict_users


# def mnist_noniid(dataset, num_users):
#     """
#     Sample non-I.I.D client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     num_items = int(len(dataset)/num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items,
#                                              replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users


# def mnist_noniid_split(dataset, num_users):
#     """
#     Sample non-I.I.D client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     num_items = int(len(dataset)/(num_users*2))
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     labels_list = []
#     for i in range(len(dataset)):
#         labels_list.append(dataset.__getitem__(i)[1])

#     labels_set = set(labels_list)

#     zipped = zip(labels_list,all_idxs)
#     sort_zipped = sorted(zipped,key=lambda x: x[0])
#     result = zip(*sort_zipped)
#     labels_list_sort, all_idxs_sort = [list(x) for x in result]

#     #dict_users = [all_idxs_sort[i:i + num_items] for i in range(0, len(all_idxs_sort), num_items)]
#     for idx, i in enumerate(range(0, len(all_idxs_sort), num_items)):
#         dict_users[idx] = all_idxs_sort[i:i + num_items]
#     # for i in range(num_users):
#     #     idx_u = set(np.random.choice(labels_set, 2, replace=False))
#     #     for j in idx_u:
#     #         dict_users[i] = set(np.random.choice(np.where(labels_array == j), num_items,
#     #                                              replace=False))
#     #         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users

# def mnist_noniid_unbalanced(dataset, num_users):
#     """
#     Sample non-I.I.D unbalanced client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     # num_items = int(len(dataset)/num_users)
#     num_lower_bound = int(len(dataset)/(num_users*5))
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users-1):
#         #print(num_lower_bound,int(len(all_idxs)/num_users))
#         num_items_i = np.random.choice(range(num_lower_bound,int(2*len(all_idxs)/num_users)), 1)
#         #print(num_items_i)
#         dict_users[i] = set(np.random.choice(all_idxs, num_items_i,
#                                              replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
    
#     dict_users[num_users-1] = all_idxs
#     return dict_users

# def cifar_iid(dataset, num_users):
#     """
#     Sample I.I.D. client data from CIFAR10 dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     num_items = int(len(dataset)/num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items,
#                                              replace=False))
#     return dict_users


# def cifar_noniid(dataset, num_users):
#     """
#     Sample non-I.I.D client data from CIFAR10 dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     num_items = int(len(dataset)/num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items,
#                                              replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users

# def cifar_noniid_unbalanced(dataset, num_users):
#     """
#     Sample non-I.I.D unbalanced client data from CIFAR10 dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     # num_items = int(len(dataset)/num_users)
#     num_lower_bound = int(len(dataset)/(num_users*5))
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users-1):
#         num_items_i = np.random.choice(range(num_lower_bound,int(2*len(all_idxs)/num_users)), 1)
#         dict_users[i] = set(np.random.choice(all_idxs, num_items_i,
#                                              replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     dict_users[num_users-1] = all_idxs
#     return dict_users

if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = data_noniid_unbalanced(dataset_train, num)
