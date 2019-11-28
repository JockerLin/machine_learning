#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lin CunQin'
__version__ = '1.0'
__date__ = '2019.10.16'
__copyright__ = 'Copyright 2019, PI'


# from example.cifar10.cifar_10 import testloader, classes, Net, module_name
import torch as t
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
show = ToPILImage() # 可以把Tensor转成Image，方便可视化
from torch import optim

module_name = 'cifar_10_model.pkl'

# 定义对数据的预处理
transform = transforms.Compose([
    transforms.ToTensor(),# 转为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),# 归一化
])

# 测试集
testset = tv.datasets.CIFAR10(
                    './data',
                    train=False,
                    download=True,
                    transform=transform)

testloader = t.utils.data.DataLoader(
                    testset,
                    batch_size=4,
                    shuffle=False,
                    num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    print('begin restore')
    net = Net()
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    net.cuda()
    net.to(device)
    # 模型数据恢复
    net.load_state_dict(t.load(module_name))

    dataiter = iter(testloader)
    for index, data in enumerate(dataiter):
        image, label = data
        print("image:", image)
        print("label:", label)
        if index == 12:
            break
    images, labels = dataiter.next() # 一个batch返回4张图片
    print('实际的label: ', ' '.join('%08s'%classes[labels[j]] for j in range(4)))
    show(tv.utils.make_grid(images / 2 - 0.5)).resize((400, 100)).show()

    # 计算图片在每个类别上的分数
    images = images.to(device)
    labels = labels.to(device)
    outputs = net(images)
    # 得分最高的那个类
    _, predicted = t.max(outputs.data, 1)

    print('预测结果: ', ' '.join('%5s'% classes[predicted[j]] for j in range(4)))

    correct = 0 # 预测正确的图片数
    total = 0 # 总共的图片数

    # 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
    with t.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = t.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

    print('10000张测试集中的准确率为: %d %%' % (100 * correct / total))