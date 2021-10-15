# CWRU 凯斯西储数据集实验

## requirement

- torch: 1.9.0
- numpy: 1.21.0

- 运行环境为Linux



## 数据预处理

先从[凯斯西储轴承数据网站](https://engineering.case.edu/bearingdatacenter/download-data-file)下载数据集。

然后运行下面命令进行数据预处理：

```bash
python dataset_cwru.py
```

可通过控制 `cats` 变量来调整类别数量



## FedAvg 训练

### IID

通过以下 bash 命令可以运行 IID 情况下的 FedAvg 模型：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python fedavg.py --dataset cwru --gpu 1 --iid 1 --unequal 0 --num_channels 1 --model dnn --epochs 300 --local_ep 30 --lr 0.1 --step_size 20 --local_bs 64 --num_users 4 --p 0.9 --num_classes 4 --seed 1 > ../result/node4/fedavg_cwru_dnn_iid_r300_seed1.txt 2>&1 &
```

model: 可选模型为 dnn 和 logistic

num_users: 控制节点数量

num_classes: 告诉模型类别数量



### NonIID

通过以下 bash 命令可以运行 NonIID 情况下的 FedAvg 模型：

```bash
CUDA_VISIBLE_DEVICES=4 nohup python fedavg.py --dataset cwru --gpu 1 --iid 0 --unequal 1 --num_channels 1 --model dnn --epochs 300 --local_ep 30 --lr 0.1 --step_size 20 --local_bs 64 --num_users 4 --p 0.9 --num_classes 4 --seed 1 > ../result/node4/fedavg_cwru_dnn_noniid_r300_seed1.txt 2>&1 &
```



## DeceFL 训练

### IID

通过以下 bash 命令可以运行 IID 情况下的 DeceFL 模型

```bash
CUDA_VISIBLE_DEVICES=0 nohup python defed.py --dataset cwru --gpu 1 --iid 1 --unequal 0 --num_channels 1 --model dnn --epochs 300 --local_ep 30 --lr 0.1 --step_size 20 --local_bs 64 --num_users 4 --p 0.9 --num_classes 4 --seed 1 > ../result/node4/defed_cwru_dnn_iid_r300_p0.9_seed1.txt 2>&1 &
```

p: 控制节点联通矩阵的稀疏程度



### NonIID

通过以下 bash 命令可以运行 NonIID 情况下的 DeceFL 模型

```bash
CUDA_VISIBLE_DEVICES=0 nohup python defed.py --dataset cwru --gpu 1 --iid 0 --unequal 1 --num_channels 1 --model dnn --epochs 300 --local_ep 30 --lr 0.1 --step_size 20 --local_bs 64 --num_users 4 --p 0.9 --num_classes 4 --seed 1 > ../result/node4/defed_cwru_dnn_noniid_r300_p0.9_seed1.txt 2>&1 &
```



## 节点独立训练

### IID

```bash
CUDA_VISIBLE_DEVICES=0 nohup python train_alone.py --dataset cwru --gpu 1 --iid 1 --unequal 0 --num_channels 1 --model dnn --epochs 50 --local_ep 30 --lr 0.1 --step_size 300 --local_bs 64 --num_users 4 --p 0.9 --num_classes 4 --seed 1 > ../result/node4/alone_cwru_dnn_iid_r300_p0.9_seed1.txt 2>&1 &
```



### NonIID

```bash
CUDA_VISIBLE_DEVICES=0 nohup python train_alone.py --dataset cwru --gpu 1 --iid 0 --unequal 1 --num_channels 1 --model dnn --epochs 50 --local_ep 30 --lr 0.1 --step_size 300 --local_bs 64 --num_users 4 --p 0.9 --num_classes 4 --seed 1 > ../result/node4/alone_cwru_dnn_noniid_r300_p0.9_seed1.txt 2>&1 &
```



## 作图

Loss 和 Accuracy 的曲线图通过 `total_plot_new.py` 文件画出来。

数据分布和节点单独训练准确率通过 `cwru_plot.py` 文件画出来。
