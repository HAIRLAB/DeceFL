#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):  # , logger
        self.args = args
        # self.logger = logger
        self.trainloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 20)
        idxs_train = idxs[:]  # int(0.8*len(idxs))
        idxs_test = idxs[:]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test) / 10), shuffle=False)
        return trainloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, weight_decay=1e-4)  # momentum=0.5
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        step_size = 20
        StepLR_optimizer = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.2)

        for iter in range(self.args.local_ep):
            batch_loss = 0.0
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels.long())
                # loss = self.criterion(log_probs, labels.float().view(-1, 1))
                loss.backward()

                # for name,param in model.named_parameters():
                #    param.grad = grad_avg.grad
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)

                optimizer.step()

                if self.args.verbose and (batch_idx % 5 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter + 1, (batch_idx + 1) * len(images),
                        len(self.trainloader.dataset),
                                      100. * (batch_idx + 1) / len(self.trainloader), loss.item()))
                # self.logger.add_scalar('loss', loss.item())
                batch_loss += loss.item()
                # batch_loss.append(loss.item())
            epoch_loss.append(batch_loss / (batch_idx + 1))

            StepLR_optimizer.step()

        return model, epoch_loss[-1]  # sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels.long())
            # batch_loss = self.criterion(outputs, labels.float().view(-1, 1))
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        loss = loss / (batch_idx + 1)
        accuracy = correct / total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    # criterion = nn.NLLLoss().to(device)
    criterion = nn.MSELoss().to(device)  # 8-13日改
    testloader = DataLoader(test_dataset, batch_size=1,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        # batch_loss = criterion(outputs, labels.long())
        batch_loss = criterion(outputs, labels)  # 8-13日改
        loss += batch_loss.item()

        # # Prediction
        # _, pred_labels = torch.max(outputs, 1)
        # pred_labels = pred_labels.view(-1)
        # correct += torch.sum(torch.eq(pred_labels, labels)).item()
        # total += len(labels)

    loss = loss / (batch_idx + 1)
    accuracy = 0
    return accuracy, loss
