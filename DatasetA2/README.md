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

- `defed.py`: the main codes that realize DeceFL
- `fedavg.py`: the main codes that realize FedAvg
- `models.py`: defines all graph models
- `options.py`: specific the default options and parameter values
- `sampling.py`: defines the sampling methods for data preparation
- `update.py`: defines the local update functions
- `utils.py`: includes data preparation strategies, and functions on generating random adjacency matrices.


## Experiment Setup

**Logistic Regression**：Every node runs 10 epochs in each round, with
batch-size 64. It uses the SGD optimizer, with weight decay
coefficient $10^{-4}$ for the realization of $l_2$ regularization. The
initial learning rate (for deep learning framework) is 0.01, which is
later decayed by multiplying 0.2 every 5 epochs.

**DNN**: Every node runs 30 epochs in each round, with batch-size 64.
It uses the SGD optimizer, with weight decay coefficient $10^{-4}$. The
initial learning rate (for deep learning framework) is 0.1, which is
decayed by multiplying 0.2 every $20$ epochs. This DNN has 8 hidden
layers, whose dimensions are 256, 512, 512, 256, 256, 128, 128, 64,
respectively. The dropout rate is set to 0.3.

Both methods use *sigmoid* as the activation function in the output
layer for binary classification (dataset A2) and *softmax* for
multiclass classification (CWRU dataset). At aggregation, the gradient
update coefficient for DeceFL is $0.1$ (FedAvg does not use this variable).
The total number of running rounds is selected by visualization effects
of convergence for all methods in comparison.


## DeceFL

### Logistic Regression

#### IID Setup

Run the experiment of DeceFL using logistic regression on IID dataset by the following bash commands:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python defed.py --dataset sl_a --gpu 1 --iid 1 --unequal 0 --num_channels 1 --model logistic --epochs 1500 --local_ep 10 --lr 0.01 --local_bs 64 --num_users 8 --p 0.9 --num_classes 2 --seed 1 > ../result/node8/defed_sla_logistic_iid_r1500_p0.9_seed1.txt 2>&1 &
```

- `dataset`：choosing dataset, `sl_a` denotes dataset A2
- `model`：choosing training model
- `epochs`：the number of aggregation
- `local_ep`：the number of training epochs between two adjacent aggregation
- `num_users`: the number of graph nodes/clients, e.g., 4, 8, 16 in use
- `num_classes`: the number of classes for classification problems; dataset A2 is a binary classification problem
- `p`: the connectivity probability of a graph used in DeceFL; our experiment uses 0.9, 0.7, 0.5, 0.3

####  Non-IID Setup

To use the Non-IID setup of training data, one only needs to set the options to `--iid 0 --unequal 1`.

Run the experiment of DeceFL using logistic regression on Non-IID dataset by the following bash commands:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python defed.py --dataset sl_a --gpu 1 --iid 0 --unequal 1 --num_channels 1 --model logistic --epochs 1500 --local_ep 10 --lr 0.01 --local_bs 64 --num_users 8 --p 0.9 --num_classes 2 --seed 1 > ../result/node8/defed_sla_logistic_noniid_r1500_p0.9_seed1.txt 2>&1 &
```

### DNN

#### IID Setup

Run the experiment of DeceFL using DNN on IID dataset by the following bash commands:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python defed.py --dataset sl_a --gpu 1 --iid 1 --unequal 0 --num_channels 1 --model dnn --epochs 300 --local_ep 30 --lr 0.1 --local_bs 64 --num_users 8 --p 0.9 --num_classes 2 --seed 1 --optimizer sgd > ../result/node8/defed_sla_dnn_iid_r300_p0.9_seed1.txt 2>&1 &
```

To run experiments on Non-IID dataset, one just needs to replace the
options `--iid 1 --unequal 0` with `--iid 0 --unequal 1`.


### Time-varying Experiments with Edge Changes

One needs to specify the option `--varying 1`. For example, use the bash command to run DeceFL using logistic regression on IID data:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python defed.py --dataset sl_a --gpu 1 --iid 1 --unequal 0 --num_channels 1 --model logistic --epochs 1500 --local_ep 10 --lr 0.01 --local_bs 64 --num_users 8 --p 0.9 --num_classes 2 --seed 1 --varying 1 > ../result/node8/defed_varying_sla_logistic_iid_r1500_p0.9_seed1.txt 2>&1 &
```


## FedAvg

### Logistic Regression

#### IID Setup

To run FedAvg using logistic regression on IID dataset, use the bash command:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python fedavg.py --dataset sl_a --gpu 1 --iid 1 --unequal 0 --num_channels 1 --model logistic --epochs 1500 --local_ep 10 --lr 0.01 --local_bs 64 --num_users 8 --p 0.9 --num_classes 2 --seed 1 > ../result/node8/fedavg_sla_logistic_iid_r1500_seed1.txt 2>&1 &
```

To run experiments on Non-IID dataset, one just needs to replace the
options `--iid 1 --unequal 0` with `--iid 0 --unequal 1`.

### DNN

#### IID Setup

To run FedAvg using DNN on IID dataset, use the bash command:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python fedavg.py --dataset sl_a --gpu 1 --iid 1 --unequal 0 --num_channels 1 --model dnn --epochs 300 --local_ep 30 --lr 0.1 --local_bs 64 --num_users 8 --p 0.9 --num_classes 2 --seed 1 --optimizer sgd > ../result/node8/fedavg_sl_dnn_iid_r300_seed1.txt 2>&1 &
```

To run experiments on Non-IID dataset, one just needs to replace the
options `--iid 1 --unequal 0` with `--iid 0 --unequal 1`.


*Last modified on 19 Oct 2021*
