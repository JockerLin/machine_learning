#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lin CunQin'
__version__ = '1.0'
__date__ = '2019.10.16'
__copyright__ = 'Copyright 2019, PI'


import PIL
import torch as t
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
show = ToPILImage() # 可以把Tensor转成Image，方便可视化
from torch import optim

# 第一次运行程序torchvision会自动下载CIFAR-10数据集，
# 大约100M，需花费一定的时间，
# 如果已经下载有CIFAR-10，可通过root参数指定

module_name = 'cifar_10_model.pkl'
data_path = "data"

# 定义对数据的预处理
transform = transforms.Compose([
    transforms.ToTensor(),# 转为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),# 归一化
])


# 训练集
trainset = tv.datasets.CIFAR10(
    root=data_path,
    train=True,
    download=True,
    transform=transform,
)

# window下DataLoader的num_workers改为0，否则报错BrokenPipeError: [Errno 32] Broken pipe 多线程的问题

trainloader = t.utils.data.DataLoader(
                    trainset,
                    batch_size=4,
                    shuffle=True,
                    num_workers=0)

# 测试集
testset = tv.datasets.CIFAR10(
                    data_path,
                    train=False,
                    download=True,
                    transform=transform)

testloader = t.utils.data.DataLoader(
                    testset,
                    batch_size=4,
                    shuffle=False,
                    num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print("train data is {}, test set is {}".format(len(trainloader), len(testloader)))

# Dataset对象是一个数据集，可以按下标访问，返回形如(data, label)的数据
(data, label) = trainset[100]
print(classes[label])

# (data + 1) / 2是为了还原被归一化的数据
show((data + 1) / 2).resize((100, 100))

dataiter = iter(trainloader)
images, labels = dataiter.next() # 返回4张图片及标签
print(' '.join('%11s'%classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid((images+1)/2)).resize((400, 100))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层 '3'表示输入图片为3通道, '6'表示输出通道数，'5'表示卷积核为5*5
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


net = Net()
print(net)

# 定义损失函数和优化器(loss和optimizer)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# SGD方法的一个缺点是，其更新方向完全依赖于当前的batch，因而其更新十分不稳定。解决这一问题的一个简单的做法便是引入momentum。

# momentum即动量，它模拟的是物体运动时的惯性，即更新的时候在一定程度上保留之前更新的方向，
# 同时利用当前batch的梯度微调最终的更新方向。这样一来，可以在一定程度上增加稳定性，
# 从而学习地更快，并且还有一定摆脱局部最优的能力

print("cuda use:", t.cuda.is_available())
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
# 设置默认的数据类型
# t.set_default_tensor_type('torch.cuda.FloatTensor')
# net.cuda()
net.cpu()
net.to(device)
# images.to(device)
# labels.to(device)

t.set_num_threads(8)
# 设置PyTorch进行CPU多线程并行计算时候所占用的线程数，这个可以用来限制PyTorch所占用的CPU数目
print("\nbegin train")
for epoch in range(10):
    print("epoch step is :", epoch)
    # 轮5遍训练集
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        # 输入数据
        inputs, labels = data

        # 梯度清零
        optimizer.zero_grad()

        # forward + backward
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # 更新参数
        optimizer.step()

        # 打印log信息
        # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
        running_loss += loss.item()
        if i % 250 == 0 and i != 0:  # 每2000个batch打印一下训练状态
            print('[%d, %5d] loss: %.5f' \
                  % (epoch + 1, i, running_loss / 250))
            running_loss = 0.0
print('Finished Training')
# 保存模型
t.save(net.state_dict(), module_name)

dataiter = iter(testloader)
images, labels = dataiter.next() # 一个batch返回4张图片
print('实际的label: ', ' '.join('%08s'%classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid(images / 2 - 0.5)).resize((400,100)).show()

# 计算图片在每个类别上的分数
images = images.to(device)
outputs = net(images)
# 得分最高的那个类
_, predicted = t.max(outputs.data, 1)

print('预测结果: ', ' '.join('%5s'% classes[predicted[j]] for j in range(4)))

correct = 0 # 预测正确的图片数
total = 0 # 总共的图片数


# 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
# 计算测试集的正确率
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


"""question--------------------------------------------------------------------------------------------------------------
lr增大后，loss下降的速率变的更慢?

"""