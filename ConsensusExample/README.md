# A2数据集实验



## Dependencies

Python libraries:

- `torch == 1.9.0`
- `numpy == 1.19.2`
- `sklearn == 0.23.2`
- `pandas == 1.1.3`
- `tqdm == 4.61.1`
- `matplotlib == 3.3.2`

Our experiment runs on Windows 10.


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
```

File description:

- `defed.py`: the main codes that realize DeceFL
- `fedavg.py`: the main codes that realize FedAvg
- `models.py`: defines all graph models
- `options.py`: specific the default options and parameter values
- `sampling.py`: defines the sampling methods for data preparation
- `update.py`: defines the local update functions
- `utils.py`: includes data preparation strategies, and functions on generating random adjacency matrices
- `model_compare_loss.py`: script to visualize the performance of training loss
- `model_compare_new.py`: script to visualize the convergence of model parameters


## Experiment Setup

Every node has only one sample $(1,1)$, that is $x = 1, y = 1$. It uses a linear model,
and in implementation chooses `local_epoch=1`, `lr=0.01`, `epochs=1000`, SGD optimizer, no weight decay, and MSE loss.


To do training in DeceFL, use `defed.py`.

To do training in FedAvg, use `fedavg.py`.


*Last modified on 20 Oct 2021*
