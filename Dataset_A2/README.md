# Experiments on Dataset A2

## Dependencies

OS: Linux

Python packages:

- `torch == 1.9.0`
- `numpy == 1.21.0`
- `sklearn == 0.24.2`
- `pandas == 1.3.1`
- `tqdm == 4.46.0`
- `matplotlib == 3.4.2`


## File Organization

Folder structure:

```
├─data
│  ├─A
│  └─CWRU
│
├─result
│  └─node8
│ 
├─save
│  └─node8
│      ├─figure 
│      ├─model
│      └─objects
|
└─src
   │  defed.py
   │  fedavg.py
   │  models.py
   │  options.py
   │  ReadMe.md
   │  sampling.py
   │  update.py
   └─ utils.py 
```

File description:

- `defed.py` 代表着DeceFL实验主代码
- `fedavg.py` 代表着Fedavg实验主代码
- `models.py` 中定义了所有的模型网络
- `options.py` 中设置了代码中默认的参数系数
- `sampling.py** 中定义了文件的样本采样方式
- `update.py** 中定义了局部更新函数
- `utils.py** 中定义了数据分配方式和随机链接矩阵生成等函数


## 实验设置

**Logistic Regression***：
每个节点单个round跑了10epochs，batch-size设置为64，优化方式选择为SGD，设定的SGD的权重衰减（weight-decay）系数为1e-4（实现L2正则化），初始学习率为0.01，每5 个epoch乘一次学习率衰减系数为0.2，外部迭代的模型聚合时候((DeceFL:梯度更新系数为0.1，Fedavg没有更新梯度），round数目根据实验数据的收敛情况，从而设置不同的数值。

**DNN**:
每个节点单个round跑了30epochs，batch-size设置为64，优化方式选择为SGD，设定的SGD的权重衰减（weight-decay）系数为1e-4（实现L2正则化），初始学习率为0.1，每20个epoch乘一次学习率衰减系数为0.2，外部迭代的模型聚合时候((DeceFL:梯度更新系数为0.1，Fedavg没有更新梯度），round数目根据实验数据的收敛情况，从而设置不同的数值。


## DeceFL

### Logistic Regression

#### IID Setup

通过以下 bash 命令可以运行Logistic, IID 情况下的 DeceFL 模型

```bash
CUDA_VISIBLE_DEVICES=0 nohup python defed.py --dataset sl_a --gpu 1 --iid 1 --unequal 0 --num_channels 1 --model logistic --epochs 1500 --local_ep 10 --lr 0.01 --local_bs 64 --num_users 8 --p 0.9 --num_classes 2 --seed 1 > ../result/node8/defed_sla_logistic_iid_r1500_p0.9_seed1.txt 2>&1 &
```

- `dataset`：选择数据集，sl_a代表A2数据
- `model`：选择训练模型
- `epochs`：聚合次数
- `local_ep`：两次聚合之间的训练epcoh数目
- `num_users`: 控制节点数量，建议选择4/8/16
- `num_classes`: 数据类别数量，A2数据为2分类问题
- `p`: 控制节点联通矩阵的稀疏程度，文中实验选择了0.9/0.7/0.5/0.3

####  Non-IID Setup

只需要将对应指令中的 `--iid 1 --unequal 0` 修改为 `--iid 0 --unequal 1` 即可，如:

通过以下 bash 命令可以运行 logistic regression, NONIID 情况下的 DeceFL 模型

```bash
CUDA_VISIBLE_DEVICES=0 nohup python defed.py --dataset sl_a --gpu 1 --iid 0 --unequal 1 --num_channels 1 --model logistic --epochs 1500 --local_ep 10 --lr 0.01 --local_bs 64 --num_users 8 --p 0.9 --num_classes 2 --seed 1 > ../result/node8/defed_sla_logistic_noniid_r1500_p0.9_seed1.txt 2>&1 &
```


### DNN

#### IID Setup

通过以下 bash 命令可以运行DNN, IID 情况下的 DeceFL 模型

```bash
CUDA_VISIBLE_DEVICES=0 nohup python defed.py --dataset sl_a --gpu 1 --iid 1 --unequal 0 --num_channels 1 --model dnn --epochs 300 --local_ep 30 --lr 0.1 --local_bs 64 --num_users 8 --p 0.9 --num_classes 2 --seed 1 --optimizer sgd > ../result/node8/defed_sla_dnn_iid_r300_p0.9_seed1.txt 2>&1 &
```

NONIID实验只需要将对应指令中的 `--iid 1 --unequal 0` 修改为 `--iid 0 --unequal 1` 即可.


### Time-varying Experiments with Edge Changes

只需要在对应指令中添加 `--varying 1` 即可，如：

通过以下 bash 命令可以运行logistic regression，IID时变情况下的 DeceFL 模型

```bash
CUDA_VISIBLE_DEVICES=0 nohup python defed.py --dataset sl_a --gpu 1 --iid 1 --unequal 0 --num_channels 1 --model logistic --epochs 1500 --local_ep 10 --lr 0.01 --local_bs 64 --num_users 8 --p 0.9 --num_classes 2 --seed 1 --varying 1 > ../result/node8/defed_varying_sla_logistic_iid_r1500_p0.9_seed1.txt 2>&1 &
```


## FedAvg

### Logistic Regression

#### IID Setup

通过以下 bash 命令可以运行 logistic regression, IID 情况下的 FedAvg 模型：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python fedavg.py --dataset sl_a --gpu 1 --iid 1 --unequal 0 --num_channels 1 --model logistic --epochs 1500 --local_ep 10 --lr 0.01 --local_bs 64 --num_users 8 --p 0.9 --num_classes 2 --seed 1 > ../result/node8/fedavg_sla_logistic_iid_r1500_seed1.txt 2>&1 &
```

Non-IID实验只需要将对应指令中的 `--iid 1 --unequal 0` 修改为 `--iid 0 --unequal 1` 即可.


### DNN

#### IID Setup

通过以下 bash 命令可以运行DNN, IID 情况下的 FedAvg 模型：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python fedavg.py --dataset sl_a --gpu 1 --iid 1 --unequal 0 --num_channels 1 --model dnn --epochs 300 --local_ep 30 --lr 0.1 --local_bs 64 --num_users 8 --p 0.9 --num_classes 2 --seed 1 --optimizer sgd > ../result/node8/fedavg_sl_dnn_iid_r300_seed1.txt 2>&1 &
```

Non-IID实验只需要将对应指令中的 `--iid 1 --unequal 0` 修改为 `--iid 0 --unequal 1` 即可.
