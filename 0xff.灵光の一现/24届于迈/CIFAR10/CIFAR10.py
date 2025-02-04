# 一个CIFAR10的分类网络

import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class CIFAR10(nn.Module):
    def __init__(self):
        super(CIFAR10, self).__init__()
        self.model1 = Sequential(Conv2d(3, 32, 5, 1, 2),
                                 MaxPool2d(2),
                                 Conv2d(32, 32, 5, 1, 2),
                                 MaxPool2d(2),
                                 Conv2d(32, 64, 5, 1, 2),
                                 MaxPool2d(2),
                                 Flatten(),
                                 Linear(1024, 64),
                                 Linear(64, 10))

    # 卷积层的stride和padding可以用公式算;线性层10个类别,out_features是10.

    def forward(self, x):
        x = self.model1(x)
        return x


classification = CIFAR10()
print(classification)
input = torch.ones((64, 3, 32, 32))  # 30-32行：测试
output = classification(input)
print(output.shape)

writer = SummaryWriter("logs_CIFAR10")  # 可以用tensorboard查看网络结构
writer.add_graph(classification, input)
writer.close()
