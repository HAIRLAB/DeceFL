# CWRU 凯斯西储数据集实验

## Dependencies

OS: Linux

Python packages:

- `torch == 1.9.0`
- `numpy == 1.21.0`


## Data and Pre-procesing

Download CWRU datset from website [Case Western Reserve University -
Bearing Data
Center](https://engineering.case.edu/bearingdatacenter/download-data-file).

Then use the following script to do data pre-processing：

```bash
python dataset_cwru.py
```

> :bulb: **Tips**: Use option `cats` to control the number of classes.


## FedAvg

### IID Setup

To run FedAvg on IID dataset, use the bash command:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python fedavg.py --dataset cwru --gpu 1 --iid 1 --unequal 0 --num_channels 1 --model dnn --epochs 300 --local_ep 30 --lr 0.1 --step_size 20 --local_bs 64 --num_users 4 --p 0.9 --num_classes 4 --seed 1 > ../result/node4/fedavg_cwru_dnn_iid_r300_seed1.txt 2>&1 &
```

Options:

- `model`: chooses `dnn` or `logistic`
- `num_users`: - `num_users`: the number of graph nodes/clients
- `num_classes`: the number of classes for classification problems; CWRU has 4-way and 10-way classification problems


### Non-IID Setup

To run FedAvg on Non-IID dataset, use the bash command:

```bash
CUDA_VISIBLE_DEVICES=4 nohup python fedavg.py --dataset cwru --gpu 1 --iid 0 --unequal 1 --num_channels 1 --model dnn --epochs 300 --local_ep 30 --lr 0.1 --step_size 20 --local_bs 64 --num_users 4 --p 0.9 --num_classes 4 --seed 1 > ../result/node4/fedavg_cwru_dnn_noniid_r300_seed1.txt 2>&1 &
```


## DeceFL

### IID Setup

To run DeceFL on IID dataset, use the bash command:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python defed.py --dataset cwru --gpu 1 --iid 1 --unequal 0 --num_channels 1 --model dnn --epochs 300 --local_ep 30 --lr 0.1 --step_size 20 --local_bs 64 --num_users 4 --p 0.9 --num_classes 4 --seed 1 > ../result/node4/defed_cwru_dnn_iid_r300_p0.9_seed1.txt 2>&1 &
```

Options:

- `p`: the connectivity probability of a graph used in DeceFL; our experiment uses 0.9, 0.7, 0.5, 0.3
- others, refer to previous *Options* description


### Non-IID Setup

To run DeceFL on Non-IID dataset, use the bash command:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python defed.py --dataset cwru --gpu 1 --iid 0 --unequal 1 --num_channels 1 --model dnn --epochs 300 --local_ep 30 --lr 0.1 --step_size 20 --local_bs 64 --num_users 4 --p 0.9 --num_classes 4 --seed 1 > ../result/node4/defed_cwru_dnn_noniid_r300_p0.9_seed1.txt 2>&1 &
```


## Independent Training at Each Node

### IID Setup

```bash
CUDA_VISIBLE_DEVICES=0 nohup python train_alone.py --dataset cwru --gpu 1 --iid 1 --unequal 0 --num_channels 1 --model dnn --epochs 50 --local_ep 30 --lr 0.1 --step_size 300 --local_bs 64 --num_users 4 --p 0.9 --num_classes 4 --seed 1 > ../result/node4/alone_cwru_dnn_iid_r300_p0.9_seed1.txt 2>&1 &
```

### NonIID Setup

```bash
CUDA_VISIBLE_DEVICES=0 nohup python train_alone.py --dataset cwru --gpu 1 --iid 0 --unequal 1 --num_channels 1 --model dnn --epochs 50 --local_ep 30 --lr 0.1 --step_size 300 --local_bs 64 --num_users 4 --p 0.9 --num_classes 4 --seed 1 > ../result/node4/alone_cwru_dnn_noniid_r300_p0.9_seed1.txt 2>&1 &
```


## Figure Plotting

Run `total_plot_new.py` to plot performance figures of *loss* (train) and *accuracy* (train/test).

Run `cwru_plot.py` to plot accessory figures, like data distribution figures, accuracy plot of independent training at each local node.


*Last modified on 20 Oct 2021*
