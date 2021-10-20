# SL Algorithm for Experiments on CWRU Dataset

## Basic library

Refer to the github repo https://github.com/HewlettPackard/swarm-learning to deploy and use SL [1].

## Updates

There is a few updates required for the SL library in order to perform
our experiments. Listed as follows.

### Item 1

Copy `my_sl` folder to `./swarm-learning/example/`.

Folder structure of `my_sl`:
```
├─my_sl
│  ├─app-data
│  │   ├─A
│  │   ├─CWRU_c10
│  │   └─CWRU_c4
│  │
│  ├─bin
│  └─model
```

Descriptions:

- Folder `app-data` keeps data for experiments, which should be added. 
- Folder `bin` contains executable scripts, which has been updated. 
- Folder `model` includes machine learning algorithms, which perform multiple experiments by setting arguments in `options`.

### Item 2

Replace the commands in Step 1 in https://github.com/HewlettPackard/swarm-learning/tree/master/examples/mnist-keras with

```bash
export APLS_IP=<License Host Server IP>
export EXAMPLE=my_sl
export WORKSPACE_DIR=$PWD
export WS_DIR_PREFIX=CWRU_c10_logistic_iid_node4-

./my_sl/bin/init-workspace -e $EXAMPLE -i $APLS_IP -d $WORKSPACE_DIR -w $WS_DIR_PREFIX -n 4
```
where `-n` specifies the number of nodes.

### Item 3

Replace the commands in Step 2 in https://github.com/HewlettPackard/swarm-learning/tree/master/examples/mnist-keras with

```bash
export APLS_IP=<License Host Server IP>
export EXAMPLE=my_sl
export WORKSPACE_DIR=$PWD
export WS_DIR_PREFIX=CWRU_c10_logistic_iid_node4-
export TRAINING_NODE=node1

../swarm-learning/bin/run-sl --name $TRAINING_NODE-sl --network $EXAMPLE-net --host-ip $TRAINING_NODE-sl --sn-ip node-sn --apls-ip $APLS_IP --serverAddress node-spire -genJoinToken --data-dir $WORKSPACE_DIR/$EXAMPLE/app-data --model-dir $WORKSPACE_DIR/$WS_DIR_PREFIX$EXAMPLE/$TRAINING_NODE/model --model-program demo1.py --sl-platform PYT
```

where `WS_DIR_PREFIX` is the name of working directory, which can be modified by users.

## References

[1] Warnat-Herresthal, S. et al. *Swarm learning for decentralized and
confidential clinical machine learning*. Nature 594, 265–270 (2021). URL
https://doi.org/10.1038/s41586-021-03583-3.


*Last modified on 20 Oct 2021*
