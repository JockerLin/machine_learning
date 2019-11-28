#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lin CunQin'
__version__ = '1.0'
__date__ = '2019.10.15'
__copyright__ = 'Copyright 2019, PI'
__all__ = []


import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        # 下式等价于nn.Module.__init__(self)
        super(Net, self).__init__()

        # 卷积层 '1'表示输入图片为单通道, '6'表示输出通道数，'5'表示卷积核为5*5
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 卷积层
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 仿射层/全连接层，y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积 -> 激活 -> 池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # reshape，‘-1’表示自适应
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


restore_module_flag = True
module_name = 'model.pkl'

net = Net()

if restore_module_flag:
    net.load_state_dict(t.load(module_name))

"自定义net的网络结构"
print(net)
params = list(net.parameters())
for name, parameters in net.named_parameters():
    print(name, ":", parameters.size())
"conv1.weight : torch.Size([6, 1, 5, 5]) 四个数分别代表输出通道、输入通道、卷积核尺寸"

input = t.randn(1, 1, 32, 32)
out = net(input)
print(out.size())


target = t.arange(0, 10).view(1, 10).float() # 假设的target
criterion = nn.MSELoss()
# 手动优化过程-----------------------------------------------------------------
# net.zero_grad() # 所有参数的梯度清零
# out.backward(t.ones(1, 10)) # 反向传播
# output = net(input)
# print(target, type(t.arange(0, 10)), type(t.arange(0, 10).view(1, 10)))
# loss = criterion(output, target)
# # loss是个scalar
# # 如果对loss进行反向传播溯源(使用gradfn属性)，可看到它的计算图如下：
# #
# # input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
# #       -> view -> linear -> relu -> linear -> relu -> linear
# #       -> MSELoss
# #       -> loss
#
# net.zero_grad() # 把可学习参数的梯度清零
# print('反向传播之前 conv1.bias的梯度')
# print(net.conv1.bias.grad)
# loss.backward()
# print('反向传播之后 conv1.bias的梯度')
# print(net.conv1.bias.grad)
#
# learning_rate = 0.01
# for f in net.parameters():
#     f.data.sub_(f.grad.data*learning_rate)


# 自动优化过程-----------------------------------------------------------------
#新建一个优化器，指定要调整的参数和学习率
optimizer = optim.SGD(net.parameters(), lr=0.01)
# 在训练过程中
# 先梯度清零(与net.zero_grad()效果一样)
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
#反向传播
loss.backward()
#更新参数
optimizer.step()

t.save(net.state_dict(), module_name)

