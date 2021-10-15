"""
上传文件：scp -r -P 10001 H:\袁烨团队\发表论文\蕊娟-联邦学习\DeFed_Lab\data\CWRU\train.npz study@120.195.221.5:/home/study/fl_sc/DeFed_Lab/data/CWRU/
下载文件：scp -r -P 10001 study@120.195.221.5:/home/study/fl_sc/DeFed_Lab/result H:\
"""

import os
import random
from os import listdir
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils import data as Data


def load_data(path, label, load_num):
    totalFileList = [item for item in listdir(path) if os.path.splitext(item)[0].endswith(str(load_num))]      # 筛出 path 下对应负载序号的文件名
    X, Y = [], []

    for i, item in enumerate(totalFileList):            # 目前就一个文件
        data_dict = sio.loadmat(path+item)

        DE_names = [n for n in data_dict.keys() if n.endswith('DE_time')]       # 找出 key
        FE_names = [n for n in data_dict.keys() if n.endswith('FE_time')]
        if len(DE_names) == 2:
            DE_names = [n for n in DE_names if '99' in n]
            FE_names = [n for n in FE_names if '99' in n]
        assert len(DE_names) == 1 and len(FE_names) == 1
        DE_name, FE_name = DE_names[0], FE_names[0]

        data_single_DE, data_single_FE = data_dict[DE_name], data_dict[FE_name]

        data_single_DE = data_single_DE[:-(len(data_single_DE) % 6000)].reshape(-1, 300)
        data_single_FE = data_single_FE[:-(len(data_single_FE) % 6000)].reshape(-1, 300)

        # FFT
        l = data_single_DE.shape[1]
        data_single_DE = np.abs(np.fft.fft(data_single_DE))[:, :l//2] / l * 2
        l = data_single_FE.shape[1]
        data_single_FE = np.abs(np.fft.fft(data_single_FE))[:, :l//2] / l * 2

        data_single = np.concatenate((data_single_DE, data_single_FE), axis=1)

        X.append(data_single)
        Y.append(label * np.ones(data_single.shape[0]))

    X, Y = np.concatenate(X, axis=0), np.concatenate(Y)

    return X, Y


def main():
    cats = [
        'normal',
         'B007', # 'B014',  'B021',
        'IR007', #'IR014', 'IR021',
        'OR007', #'OR014', 'OR021'
    ]
    labels = list(range(len(cats)))

    # Train
    X_train, Y_train = [], []
    for p, l in zip(cats, labels):
        for i in range(3):
            X, Y = load_data(path=f'../data/CWRU/{p}/', label=l, load_num=i)
            X_train.append(X[:400])
            Y_train.append(Y[:400])

    # Test
    X_test, Y_test = [], []
    for p, l in zip(cats, labels):
        X, Y = load_data(path=f'../data/CWRU/{p}/', label=l, load_num=3)
        X_test.append(X[:400])
        Y_test.append(Y[:400])

    X_train = np.concatenate(X_train, axis=0)
    Y_train = np.concatenate(Y_train, axis=0)
    X_test  = np.concatenate(X_test,  axis=0)
    Y_test  = np.concatenate(Y_test,  axis=0)

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test  = ss.transform(X_test)

    np.savez('../data/CWRU/train', X_train=X_train, Y_train=Y_train)
    np.savez('../data/CWRU/test',  X_test=X_test,   Y_test=Y_test)


if __name__ == '__main__':
    main()
    print('success')