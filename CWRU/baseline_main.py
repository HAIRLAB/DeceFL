#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from tqdm import tqdm
import time
import torch
import pickle
from torch.utils.data import DataLoader

from utils import get_dataset,get_data
from options import args_parser
from update import test_inference
from models import Linear, MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, VGGNet, ResNet, DNNModel, Logistic


if __name__ == '__main__':
    start_time = time.time()
    
    args = args_parser()
    #if args.gpu:
    #    torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load datasets
    if args.dataset[:2] == 'sl':
        train_dataset, test_dataset, _ = get_data(args)
    else:
        train_dataset, test_dataset, _ = get_dataset(args)
    
    # BUILD MODEL
    if args.model == 'linear':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = Linear(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)

    elif args.model == 'logistic':
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = Logistic(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)

    elif args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            #global_model = CNNCifar(args=args)
            global_model = ResNet(args=args, depth=32, block_name='BasicBlock')

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    
    elif args.model == 'dnn':
        len_in = len(train_dataset[0][0])
        global_model = DNNModel(dim_in=len_in, dim_out=args.num_classes)
                           
    else:
        raise Exception('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)
    
    step_size = 20
    StepLR_optimizer = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.2)
    
    trainloader = DataLoader(train_dataset, batch_size=args.local_bs, shuffle=True)
    criterion = torch.nn.NLLLoss().to(device)
    epoch_loss = []
    train_accuracy = []
    for epoch in tqdm(range(args.epochs)):
        batch_loss = []
        global_model.train()
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            global_model.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            # if batch_idx % 50 == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch+1, batch_idx * len(images), len(trainloader.dataset),
            #         100. * batch_idx / len(trainloader), loss.item()))
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        epoch_loss.append(loss_avg)
        
        
        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        train_acc, train_loss = test_inference(args, global_model, train_dataset)
        train_accuracy.append(train_acc)
        
        StepLR_optimizer.step()
    
    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/baseline_{}_{}_{}.pkl'.\
        format(args.dataset, args.model, args.epochs)

    with open(file_name, 'wb') as f:
        pickle.dump([epoch_loss, train_accuracy], f)
    
    
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # testing
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    print('Test on', len(test_dataset), 'samples')

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')


    # Plot loss
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    plt.savefig('../save/figure/nn_loss_{}_{}_{}.png'.format(args.dataset, args.model,
                                                 args.epochs))

    plt.figure()
    plt.plot(range(len(train_accuracy)), train_accuracy)
    plt.xlabel('epochs')
    plt.ylabel('Train acc')
    plt.savefig('../save/figure/nn_acc_{}_{}_{}.png'.format(args.dataset, args.model,
                                                 args.epochs))


