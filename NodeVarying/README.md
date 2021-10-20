# Time-varying Experiments with Node Changes

## Introduction

This experiment benchmarks the performance of DeceFL using graphs that delete or add nodes over time, using FedAvg as the reference performance. It uses dataset A2 in both IID and Non-IID setups. The chosen graph starts with 6 nodes, then 8 nodes by adding two such that the whole graph keeps connected, and finally 6 nodes by randomly deleting two without violating graph connected-ness. 

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

- `options.py`: sets default arguments and coefficients
- `sampling.py`: defines sampling methods for data preparation
- `models.py`: defines graphs used in the federated learning 中定义了所有的模型网络
- `update.py`: defines the local update functions
- `utils.py`: includes data preparation strategies, and functions on generating random adjacency matrices.
- `fedavg_time_varing_node_changes.py`: the main script to apply FedAvg in this experiment
- `defed_time_varing_node_changes.py`: the main script to apply DeceFL in this experiment


### Experiment Setup

Every node runs 10 epochs in each round, with batch-size 64. It uses the
SGD optimizer, with weight decay coefficient $10^{-4}$ for the
realization of $l_2$ regularization. The initial learning rate (for deep
learning framework) is 0.01, which is later decayed by multiplying 0.2
every 5 epochs. It uses *sigmoid* as the activation function in the
output layer for binary classification (dataset A2). At aggregation, the
gradient update coefficient for DeceFL is $0.1$ (FedAvg does not use
this variable). The total number of running rounds is selected by
visualization effects of convergence for all methods in comparison.


## DeceFL

Run the experiment of DeceFL using logistic regression on IID or Non-IID dataset by the following bash commands.

### IID Setup

```bash
CUDA_VISIBLE_DEVICES=1 nohup python defed_time_varing_node_changes.py --dataset sl_a --gpu 1 --iid 1 --unequal 0 --num_channels 1 --model logistic --epochs 600 --local_ep 10 --lr 0.01 --local_bs 64 --num_users 10 --p 0.3 --num_classes 1 --seed 1 --varying 1 > ../result/node8/defed_varying2_logistic_iid_r600_p0.3_seed1.txt 2>&1 &
```

### Non-IID Setup

```bash
CUDA_VISIBLE_DEVICES=1 nohup python fedavg_time_varing_node_changes.py --dataset sl_a --gpu 1 --iid 1 --unequal 0 --num_channels 1 --model logistic --epochs 600 --local_ep 10 --lr 0.01 --local_bs 64 --num_users 10 --p 0.9 --num_classes 1 --seed 1 --varying 1 > ../result/node8/fedavg_varying2_logistic_iid_r600_seed1.txt 2>&1 &
```

## FedAvg

Run the experiment of FedAvg using logistic regression on IID or Non-IID dataset by
the following bash commands.

### IID Setup

```bash
CUDA_VISIBLE_DEVICES=1 nohup python fedavg_time_varing_node_changes.py --dataset sl_a --gpu 1 --iid 1 --unequal 0 --num_channels 1 --model logistic --epochs 600 --local_ep 10 --lr 0.01 --local_bs 64 --num_users 10 --p 0.9 --num_classes 1 --seed 1 --varying 1 > ../result/node8/fedavg_varying2_logistic_iid_r600_seed1.txt 2>&1 &
```

### Non-IID Setup

```bash
CUDA_VISIBLE_DEVICES=1 nohup python fedavg_time_varing_node_changes.py --dataset sl_a --gpu 1 --iid 0 --unequal 1 --num_channels 1 --model logistic --epochs 600 --local_ep 10 --lr 0.01 --local_bs 64 --num_users 10 --p 0.9 --num_classes 1 --seed 1 --varying 1 > ../result/node8/fedavg_varying2_logistic_iid_r600_seed1.txt 2>&1 &
```

Options:

- `model`：choosing training model, it can be `dnn` and `logistic`.
- `num_users`: the number of graph nodes/clients, e.g., 4, 8, 16 in use.
- `num_classes`: the number of classes for classification problems; dataset A2 is a binary classification problem.
- `iid,unequal`: together control how data is split and prepared as local datasets. When `iid == 1, unequal == 0` , perform random sampling without put samples back to the set; when `iid == 1, unequal == 1`, use the designed Non-IID setup, which specifies the sample size and the unbalanced ratio between positive and negative samples.
- `p`: the connectivity probability of a graph used in DeceFL; our experiment uses 0.9, 0.7, 0.5, 0.3.
- `local_ep`：the number of training epochs between two adjacent aggregation.
- `lr`: initial learning rate in deep learning frameworks for local models.
- `varying`: controls whether it is a time-varying experiment.


## Figure Plotting

After running these experiments, results will placed in the corresponding folders. The figures on `loss` and `accuracy` can be obtained by `total_plot_new.py`.


Last modified on 20 Oct 2021

By oracleyue
