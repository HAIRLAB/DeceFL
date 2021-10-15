### Introduction:

A2数据集基本实验和时变实验

### Library:

-torch == 1.9.0
-numpy == 1.21.0
-sklearn == 0.24.2

- 运行环境为Linux

### Folder structure:

├─data
│  ├─A
│  │      
│  └─CWRU
│      
│
├─result
│  └─node8
│ 
├─save
│  └─node8
│      ├─figure
│      │      
│      ├─model
│      │ 
│      └─objects
│          
│          
└─src
   │  defed.py
   │  fedavg.py
   │  models.py
   │  options.py
   │  ReadMe.md
   │  sampling.py
   │  update.py
   └─ utils.py 

### Code:

options.py 中设置了代码中默认的参数系数
sampling.py 中定义了文件的样本采样方式
models.py 中定义了所有的模型网络
update.py 中定义了局部更新函数
utils.py 中定义了数据分配方式和随机链接矩阵生成等函数
fedavg.py 代表着Fedavg实验主代码
defed.py 代表着DeceFL实验主代码

### 实验设置:

选用logistic模型时：每个节点单个round跑了10epochs，batch-size设置为64，优化方式选择为SGD，设定的SGD的权重衰减（weight-decay）系数为1e-4（实现L2正则化），初始学习率为0.01，每5 个epoch乘一次学习率衰减系数为0.2，外部迭代的模型聚合时候((DeceFL:梯度更新系数为0.1，Fedavg没有更新梯度），round数目根据实验数据的收敛情况，从而设置不同的数值。
选用dnn模型时：每个节点单个round跑了30epochs，batch-size设置为64，优化方式选择为SGD，设定的SGD的权重衰减（weight-decay）系数为1e-4（实现L2正则化），初始学习率为0.1，每20个epoch乘一次学习率衰减系数为0.2，外部迭代的模型聚合时候((DeceFL:梯度更新系数为0.1，Fedavg没有更新梯度），round数目根据实验数据的收敛情况，从而设置不同的数值。

### Example:

**DeceFL**
```cmd
CUDA_VISIBLE_DEVICES=0 nohup python defed.py --dataset sl_a --gpu 1 --iid 1 --unequal 0 --num_channels 1 --model logistic --epochs 1500 --local_ep 10 --lr 0.01 --local_bs 64 --num_users 4 --p 0.9 --num_classes 2 --seed 1 > ../result/node4/defed_sl_a_logistic_iid_r1500_p0.9_seed1.txt 2>&1 &
```

**fedavg**
```cmd
CUDA_VISIBLE_DEVICES=2 nohup python fedavg.py --dataset sl_a --gpu 1 --iid 1 --unequal 0 --num_channels 1 --model logistic --epochs 1500 --local_ep 10 --lr 0.01 --local_bs 64 --num_users 4 --p 0.9 --num_classes 2 --seed 1 > ../result/node4/fedavg_sl_a_logistic_iid_r1500_seed1.txt 2>&1 &
```

Tips:varying参数为设置为时变的参数