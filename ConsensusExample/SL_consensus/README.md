# A2数据集SL相关实验

## 基本框架
请参照SL论文中提供的github链接 https://github.com/HewlettPackard/swarm-learning 进行部署

## 需要改动的内容
### 1
将 my_sl 文件夹拷贝到./swarm-learning/example/目录下。
my_sl 文件夹的基本结构如下：
├─my_sl
│  ├─app-data
│  │   ├─A
│  │   ├─CWRU_c10
│  │   └─CWRU_c4
│  │
│  ├─bin
│  └─model

其中 app-data目录下为实验数据；bin目录下为我们修改后的可执行脚本；model目录下为机器学习算法。

### 2
将 https://github.com/HewlettPackard/swarm-learning/tree/master/examples/mnist-keras 中步骤1中的指令替换为
```bash
export APLS_IP=<License Host Server IP>
export EXAMPLE=my_sl
export WORKSPACE_DIR=$PWD
export WS_DIR_PREFIX=consensus-

./my_sl/bin/init-workspace -e $EXAMPLE -i $APLS_IP -d $WORKSPACE_DIR -w $WS_DIR_PREFIX -n 8
```
其中 -n 表示节点个数

### 3
将 https://github.com/HewlettPackard/swarm-learning/tree/master/examples/mnist-keras 中步骤2中的指令替换为
```bash
export APLS_IP=<License Host Server IP>
export EXAMPLE=my_sl
export WORKSPACE_DIR=$PWD
export WS_DIR_PREFIX=consensus-
export TRAINING_NODE=node1

../swarm-learning/bin/run-sl --name $TRAINING_NODE-sl --network $EXAMPLE-net --host-ip $TRAINING_NODE-sl --sn-ip node-sn --apls-ip $APLS_IP --serverAddress node-spire -genJoinToken --data-dir $WORKSPACE_DIR/$EXAMPLE/app-data --model-dir $WORKSPACE_DIR/$WS_DIR_PREFIX$EXAMPLE/$TRAINING_NODE/model --model-program demo1.py --sl-platform PYT
```
其中 WS_DIR_PREFIX 为工作目录名称，可自行修改。
