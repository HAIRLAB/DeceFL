# Time-varying Experiments with Node Changes

## Introduction

本部分实验是实现对比Fedavg和DeceFL在增删节点情况下的实验情况，实验对比数
据集为A2数据集，基础实验的节点变化为 6-8-6，网络在初始情况下基于选择的6
个节点进行训练迭代，在迭代一定基础上加入新加入的两个节点，在8节点的情况
下继续训练，在进一步训练一段时长后，再随机删去其中的两个节点，进行后续训
练。

## Dependencies

OS: Linux

Python packages:

- `numpy == 1.19.5`
- `sklearn == 0.24.1`
- `torch == 1.9.0`


## File Organization

Folder structure:

```
├─data
│  ├─A
│  └─CWRU
│
├─figure_gather
│
├─result
│  └─node8
│ 
├─save
│  └─node8
│      ├─figure
│      ├─model
│      └─objects
│
├─results
│  └─A2_varying2_seed1
│     └─save
│
└─src
   │  defed_time_varing_node_changes.py
   │  fedavg_time_varing_node_changes.py
   │  models.py
   │  options.py
   │  sampling.py
   │  update.py
   └─ utils.py
```


File description:

- `options.py` 中设置了代码中默认的参数系数
- `sampling.py` 中定义了文件的样本采样方式
- `models.py` 中定义了所有的模型网络
- `update.py` 中定义了局部更新函数
- `utils.py` 中定义了数据分配方式和随机链接矩阵生成等函数
- `fedavg_time_varing_node_changes.py` 代表着基于Fedavg实现的增删节点实验主代码
- `defed_time_varing_node_changes.py` 代表着基于DeceFL实现的增删节点的实验主代码


### Experiment Setup

每个节点单个round跑了10epochs，batch-size设置为64，优化方式选择为SGD，设
定的SGD的权重衰减（weight-decay）系数为1e-4（实现L2正则化），初始学习率
为0.01，每5 个epoch乘一次学习率衰减系数为0.2，外部迭代的模型聚合时候
(DeceFL:梯度更新系数为0.1，Fedavg没有更新梯度），round数目根据实验数据的
收敛情况，从而设置不同的数值。

## DeceFL

通过以下 bash 命令可以运行不同设置下的 **DeceFL** 模型：

### IID Setup

```bash
CUDA_VISIBLE_DEVICES=1 nohup python defed_time_varing_node_changes.py --dataset sl_a --gpu 1 --iid 1 --unequal 0 --num_channels 1 --model logistic --epochs 600 --local_ep 10 --lr 0.01 --local_bs 64 --num_users 10 --p 0.3 --num_classes 1 --seed 1 --varying 1 > ../result/node8/defed_varying2_logistic_iid_r600_p0.3_seed1.txt 2>&1 &
```

### Non-IID Setup

```bash
CUDA_VISIBLE_DEVICES=1 nohup python fedavg_time_varing_node_changes.py --dataset sl_a --gpu 1 --iid 1 --unequal 0 --num_channels 1 --model logistic --epochs 600 --local_ep 10 --lr 0.01 --local_bs 64 --num_users 10 --p 0.9 --num_classes 1 --seed 1 --varying 1 > ../result/node8/fedavg_varying2_logistic_iid_r600_seed1.txt 2>&1 &
```

## FedAvg

通过以下 bash 命令可以运行不同设置下的 **Fedavg** 模型：

### IID Setup

```bash
CUDA_VISIBLE_DEVICES=1 nohup python fedavg_time_varing_node_changes.py --dataset sl_a --gpu 1 --iid 1 --unequal 0 --num_channels 1 --model logistic --epochs 600 --local_ep 10 --lr 0.01 --local_bs 64 --num_users 10 --p 0.9 --num_classes 1 --seed 1 --varying 1 > ../result/node8/fedavg_varying2_logistic_iid_r600_seed1.txt 2>&1 &
```

### Non-IID Setup

```bash
CUDA_VISIBLE_DEVICES=1 nohup python fedavg_time_varing_node_changes.py --dataset sl_a --gpu 1 --iid 0 --unequal 1 --num_channels 1 --model logistic --epochs 600 --local_ep 10 --lr 0.01 --local_bs 64 --num_users 10 --p 0.9 --num_classes 1 --seed 1 --varying 1 > ../result/node8/fedavg_varying2_logistic_iid_r600_seed1.txt 2>&1 &
```

Options:

- `model`: 控制模型类型参数，参数值为：{dnn, logistic}，分别选模型为 dnn 和 logistic
- `num_users`: 控制节点数量，参数值任意
- `num_classes`: 告诉模型类别数量，参数值由数据类别数量所觉得
- `iid,unequal`: 一起设置数据分别类型参数，当iid为1、unequal为0时候为无
  放回的随机抽取，当iid为0、unequal为1时候为按照给定的非均衡数据样本分布
  抽取。
- `p`: 设置随机连通图的参数，p值越大，连通度越高
- `local_ep`: 设置局部模型的迭代epoch数
- `lr`: 设置局部模型的初始学习率
- `varying`: 参数为设置为时变的参数


## Figure Plotting

在运行完相关实验并将数据存储到对应文件夹后，Loss 和 Accuracy 的曲线图通可过运行 `total_plot_new.py` 文件画出来。
