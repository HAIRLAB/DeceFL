# A2数据集实验



## requirement:

-torch == 1.9.0
-numpy == 1.19.2
-sklearn == 0.23.2
-pandas == 1.1.3
-tqdm == 4.61.1
-matplotlib == 3.3.2

- 运行环境为windows 10



## Folder structure:

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
   ├─ defed.py
   ├─ fedavg.py
   ├─model_compare_loss.py
   ├─model_compare_new.py
   ├─models.py
   ├─options.py
   ├─README_consensus.md
   ├─sampling.py
   ├─update.py
   └─ utils.py 



## Code:

options.py 中设置了代码中默认的参数系数
sampling.py 中定义了文件的样本采样方式
models.py 中定义了所有的模型网络
update.py 中定义了局部更新函数
utils.py 中定义了数据分配方式和随机链接矩阵生成等函数
model_compare_loss.py 为框架loss的可视化代码
model_compare_new.py 为模型参数收敛情况的可视化代码
fedavg.py 代表着Fedavg实验主代码
defed.py 代表着DeceFL实验主代码



## 实验设置:
每个节点只包含一个样本（1，1），即x=1，y=1；采用linear模型；local_epoch=1；lr=0.01；epochs=1000；优化方式选择为SGD,不设置权重衰减；损失采用MSE损失。




## DeceFL 训练
直接运行defed.py即可



## FedAvg 训练
直接运行fedavg.py即可
