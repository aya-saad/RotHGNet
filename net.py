import torch
import torch.nn as nn
from layers_2D import *
from layers import *


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.main = nn.Sequential(
            # in_channels, out_channels, kernel_size, stride=1,
            #                  padding=0, dilation=1, n_angles = 8, mode=1)
            RotConv(3, 6, [9, 9], 1, 9 // 2, n_angles=17, mode=1),
            VectorMaxPool(2),
            VectorBatchNorm(6),

            RotConv(6, 16, [9, 9], 1, 9 // 2, n_angles=17, mode=2),
            VectorMaxPool(2),
            VectorBatchNorm(16),

            RotConv(16, 32, [9, 9], 1, 1, n_angles=17, mode=2),
            Vector2Magnitude(),

            nn.Conv2d(32, 128, 1),  # FC1
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.7),
            nn.Conv2d(128, 10, 1),  # FC2

        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size()[0], x.size()[1])

        return x

class SimpleConvNet(nn.Module):

    def __init__(self,inp_dim, oup_dim, nstack=1, bn=False, increase=0, **kwargs):
        super(SimpleConvNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.pre = nn.Sequential(
            Conv(3, 28, 7, 2, bn=True, relu=True),
            Residual(28, 56),
            Pool(2, 2),
            Residual(56, 56),
            Residual(56, inp_dim)
        )

    def forward(self, x):
        x = self.pre(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Unit(nn.Module):
    def __init__(self, in_channels, out_channels, max_pool=False, **kwargs):
        super(Unit, self).__init__()
        self.max_pool = max_pool

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=out_channels, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        if (self.max_pool):
            output = self.maxpool(output)

        return output


class SimpleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleNet, self).__init__()

        # Create 14 layers of the unit with max pooling in between
        self.unit1 = Unit(in_channels=3, out_channels=32)
        self.unit2 = Unit(in_channels=32, out_channels=32)
        self.unit3 = Unit(in_channels=32, out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(in_channels=32, out_channels=64)
        self.unit5 = Unit(in_channels=64, out_channels=64)
        self.unit6 = Unit(in_channels=64, out_channels=64)
        self.unit7 = Unit(in_channels=64, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit8 = Unit(in_channels=64, out_channels=128)
        self.unit9 = Unit(in_channels=128, out_channels=128)
        self.unit10 = Unit(in_channels=128, out_channels=128)
        self.unit11 = Unit(in_channels=128, out_channels=128)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.unit12 = Unit(in_channels=128, out_channels=128)
        self.unit13 = Unit(in_channels=128, out_channels=128)
        self.unit14 = Unit(in_channels=128, out_channels=128)

        self.avgpool = nn.AvgPool2d(kernel_size=4)

        # Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5, self.unit6,
                                 self.unit7, self.pool2, self.unit8, self.unit9, self.unit10, self.unit11, self.pool3,
                                 self.unit12, self.unit13, self.unit14, self.avgpool)# 64: 32-16-8-2

        #Pytorch output shapes from Conv2D
        # Hout = (Hin +2xpadding(0)-dilation(0)x(kernel_size(0)-1)-1)/stride(0) + 1
        # Wout = (Win +2xpadding(1)-dilation(1)x(kernel_size(1)-1)-1)/stride(1) + 1

        # MaxPool2D
        # Hout = (Hin +2xpadding(0)-dilation(0)x(kernel_size(0)-1)-1)/stride(0) + 1
        # Wout = (Win + 2xpadding(1) - dilation(1)x(kernel_size(1) - 1) - 1) / stride(1) + 1
        # -> default padding is 0, default stride = kernel_size dilation=1

        self.fc = nn.Linear(in_features=2*2*128, out_features=num_classes)

    def forward(self, input):
        output = self.net(input)
        #print('output.shape before flatening', output.shape)
        #output = output.view(output.size(0), -1)
        output = output.view(-1, 128*2*2)
        #print('output.shape after flatening ', output.shape)
        output = self.fc(output)
        output = nn.Softmax(dim=1)(output)
        return output

'''
COAP net structure as described by Bjarne
network = input_data(shape=[None, 64, 64, 3], data_preprocessing=img_prep))
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = fully_connected(network, 256, activation='relu')
network = fully_connected(network, 256, activation='relu')
network = fully_connected(network, OUTPUTS, activation='softmax'
'''
class COAPNet(nn.Module):

    def __init__(self,num_classes=6, **kwargs):
        super(COAPNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        # (conv): Conv2d(3, 28, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), #  64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),        # 32


            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),        # 16

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # 16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 8

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # 8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 4
        )
        #self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(4*4*512, 512),
            nn.ReLU(inplace=True),
            #nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        #x = x.view(-1, self.num_flat_features(x))
        x = self.features(x)
        #x = torch.flatten(x, 1)
        #output = output.view(-1, 16 * 16 * 24)
        x = x.view(-1, 4*4*512)
        x = self.classifier(x)
        x = nn.Softmax(dim=1)(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

'''
COAP net structure as described by Bjarne
network = input_data(shape=[None, 64, 64, 3], data_preprocessing=img_prep))
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = fully_connected(network, 256, activation='relu')
network = fully_connected(network, 256, activation='relu')
network = fully_connected(network, OUTPUTS, activation='softmax'
'''

class COAPModNet(nn.Module):
    def __init__(self, num_classes=10):
        super(COAPModNet, self).__init__()

        # Create 14 layers of the unit with max pooling in between
        self.unit1 = Unit(in_channels=3, out_channels=64, max_pool=True)
        #self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.unit2 = Unit(in_channels=64, out_channels=128, max_pool=True)
        #self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.unit3 = Unit(in_channels=128, out_channels=256, max_pool=True)
        #self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.unit4 = Unit(in_channels=256, out_channels=512, max_pool=True)
        #self.pool4 = nn.MaxPool2d(kernel_size=2)

        # Add all the units into the Sequential layer in exact order
        self.features = nn.Sequential(self.unit1, #self.pool1,
                                      self.unit2, #self.pool2,
                                      self.unit3, #self.pool3,
                                      self.unit4) #self.pool4)# 64: 32-16-8-4

        #Pytorch output shapes from Conv2D
        # Hout = (Hin +2xpadding(0)-dilation(0)x(kernel_size(0)-1)-1)/stride(0) + 1
        # Wout = (Win +2xpadding(1)-dilation(1)x(kernel_size(1)-1)-1)/stride(1) + 1

        # MaxPool2D
        # Hout = (Hin +2xpadding(0)-dilation(0)x(kernel_size(0)-1)-1)/stride(0) + 1
        # Wout = (Win + 2xpadding(1) - dilation(1)x(kernel_size(1) - 1) - 1) / stride(1) + 1
        # -> default padding is 0, default stride = kernel_size dilation=1
        self.classifier = nn.Sequential(
            nn.Linear(in_features=4*4*512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_classes)
        )

    def forward(self, input):
        output = self.features(input)
        #print('output.shape before flatening', output.shape)
        #output = output.view(output.size(0), -1)
        output = output.view(-1, 4*4*512)
        #print('output.shape after flatening ', output.shape)
        output = self.classifier(output)
        output = nn.Softmax(dim=1)(output)
        return output

class HGNet(nn.Module):
    def __init__(self, inp_dim, oup_dim, nstack=1, bn=False, increase=0, **kwargs):
        super(HGNet, self).__init__()

        self.nstack = nstack
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim)
        )

        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(inp_dim, inp_dim),
                Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
            ) for i in range(nstack)])

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, inp_dim, bn, increase),
            ) for i in range(nstack)])
        self.outs = nn.ModuleList([Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)])

    def forward(self, x):
        print('----------- HOURGLASS NET -----------')
        print('----------- pre() ')
        #x = self.pre(x)
        x = self.hgs[0](x)
        print(x)
        #print(x)
        print('----------- STACK() ')
        #for i in range(self.nstack):
        #    print('----------- HGS() ')
        #    hg = self.hgs[i](x)
            #print(hg)
        #    print('----------- FEATURE() ')
        #    feature = self.features[i](hg)
            #print(feature)
        #    print('----------- PREDS() ')
            #preds = self.outs[i](feature)
        #    x = self.outs[i](feature)
            #print(preds)
        # x = x.view(x.size()[0], x.size()[1])
        #x = x.view(preds.size()[0], preds.size()[1])
        #x = x.view(x.size()[0], x.size()[1])
        #print(x)

        return x


