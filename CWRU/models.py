#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torch.nn
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

# Defining the dnn model
class DNNModel(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super(DNNModel, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(dim_in, 256), nn.ReLU(), nn.Dropout(0.3))
        self.layer3 = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.3))
        self.layer4 = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.3))
        self.layer5 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3))
        self.layer6 = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3))
        self.layer7 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3))
        self.layer8 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3))
        self.layer9 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3))
        self.layer10 = nn.Sequential(nn.Linear(64, dim_out))

    def forward(self, x):
        fc = self.layer1(x)
        fc = self.layer3(fc)
        fc = self.layer4(fc)
        fc = self.layer5(fc)
        fc = self.layer6(fc)
        fc = self.layer7(fc)
        fc = self.layer8(fc)
        fc = self.layer9(fc)
        fc = self.layer10(fc)
        return F.log_softmax(fc, dim=1)


# class DNNModel(torch.nn.Module):
#     def __init__(self, dim_in, dim_out):
#         super(DNNModel, self).__init__()
#         self.layer1 = nn.Sequential(nn.Linear(dim_in, 256))
#         self.layer2 = nn.Sequential(nn.ReLU(), nn.Linear(256, 256), nn.Dropout(0.3))
#         self.layer3 = nn.Sequential(nn.ReLU(), nn.Linear(256, 256), nn.Dropout(0.3))
#         self.layer4 = nn.Sequential(nn.ReLU(), nn.Linear(256, 256), nn.Dropout(0.3))
#         self.layer5 = nn.Sequential(nn.ReLU(), nn.Linear(256, 256), nn.Dropout(0.3))
#         self.layer6 = nn.Sequential(nn.ReLU(), nn.Linear(256, 256), nn.Dropout(0.3))
#         self.layer7 = nn.Sequential(nn.ReLU(), nn.Linear(256, 256), nn.Dropout(0.3))
#         self.layer8 = nn.Sequential(nn.ReLU(), nn.Linear(256, 256), nn.Dropout(0.3))
#         self.layer9 = nn.Sequential(nn.ReLU(), nn.Linear(256, 256), nn.Dropout(0.3))
#         self.layer10 = nn.Sequential(nn.Linear(256, dim_out))
#
#     def forward(self, x):
#         fc = self.layer1(x)
#
#         res = fc
#         fc = self.layer2(fc)
#         fc = self.layer3(fc)
#         fc += res
#
#         res = fc
#         fc = self.layer4(fc)
#         fc = self.layer5(fc)
#         fc += res
#
#         res = fc
#         fc = self.layer6(fc)
#         fc = self.layer7(fc)
#         fc += res
#
#         res = fc
#         fc = self.layer8(fc)
#         fc = self.layer9(fc)
#         fc += res
#
#         fc = self.layer10(fc)
#         return F.log_softmax(fc, dim=1)


class Linear(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(Linear, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        # x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        return F.log_softmax(x, dim=1) #self.softmax(x)

class Logistic(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(Logistic, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = self.layer_input(x)
        # x = F.sigmoid(x)
        return F.log_softmax(x, dim=1) #self.softmax(x)
        # return x


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden*2)
        self.layer_hidden = nn.Linear(dim_hidden*2, dim_hidden)
        self.layer_output = nn.Linear(dim_hidden, dim_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_output(x)
        return F.log_softmax(x, dim=1) #self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, args.num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class modelC(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, n_classes, 1)


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out

class VGGNet(nn.Module):

    def __init__(self, args):
        super(VGGNet,self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,padding=1)
        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3,padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv5 = nn.Conv2d(128,128, 3,padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3,padding=1)
        self.conv7 = nn.Conv2d(128, 128, 1,padding=1)
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.conv8 = nn.Conv2d(128, 256, 3,padding=1)
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 256, 1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        self.conv11 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 1, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()

        self.fc14 = nn.Linear(512*4*4,1024)
        self.drop1 = nn.Dropout2d()
        self.fc15 = nn.Linear(1024,512)
        self.drop2 = nn.Dropout2d()
        self.fc16 = nn.Linear(512,args.num_classes)


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)


        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = x.view(-1,512*4*4)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        x = self.fc16(x)

        return F.log_softmax(x, dim=1)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv_1 = conv3x3(inplanes, planes, stride)
        self.bn_1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_2 = conv3x3(planes, planes)
        self.bn_2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu(out)

        out = self.conv_2(out)
        out = self.bn_2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv_1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(planes)
        self.conv_2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(planes)
        self.conv_3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn_3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu(out)

        out = self.conv_2(out)
        out = self.bn_2(out)
        out = self.relu(out)

        out = self.conv_3(out)
        out = self.bn_3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, args, depth=32, block_name='BasicBlock'):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name == 'BasicBlock':
            assert (
                depth - 2) % 6 == 0, 'depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name == 'Bottleneck':
            assert (
                depth - 2) % 9 == 0, 'depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.inplanes = 16
        self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                                bias=False)
        self.bn_1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.stage_1 = self._make_layer(block, 16, n)
        self.stage_2 = self._make_layer(block, 32, n, stride=2)
        self.stage_3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, args.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_normal(m.weight.data)
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)     # 32x32

        x = self.stage_1(x)  # 32x32
        x = self.stage_2(x)  # 16x16
        x = self.stage_3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)


def resnet20(num_classes):
    return ResNet(depth=20, num_classes=num_classes)


def resnet32(num_classes):
    return ResNet(depth=32, num_classes=num_classes)


def resnet44(num_classes):
    return ResNet(depth=44, num_classes=num_classes)


def resnet56(num_classes):
    return ResNet(depth=56, num_classes=num_classes)


def resnet110(num_classes):
    return ResNet(depth=110, num_classes=num_classes)


def resnet1202(num_classes):
    return ResNet(depth=1202, num_classes=num_classes)

