#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lin CunQin'
__version__ = '1.0'
__date__ = '2019.10.18'
__copyright__ = 'Copyright 2019, PI'
__all__ = []


import torch as t
from matplotlib import pylab as plt
from IPython import display

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
t.manual_seed(1000)


def get_fake_data(batchSize=8):
    x = t.rand(batchSize, 1, device=device)*5
    y = x*2+3+t.randn(batchSize, 1, device=device)
    return x, y


x, y = get_fake_data(batchSize=16)
plt.scatter(x.squeeze().cpu().numpy(), y.squeeze().cpu().numpy())
# plt.show()

w = t.randn(1, 1).to(device)
b = t.randn(1, 1).to(device)

lr = 0.02

print('begin')
for ii in range(500):
    x, y = get_fake_data(batchSize=4)

    # forward：计算loss
    # print(x.dtype, w.dtype)
    y_pred = x.mm(w) + b.expand_as(y) # x@W等价于x.mm(w);for python3 only
    loss = 0.5 * (y_pred - y) ** 2# 均方误差
    loss = loss.mean()
    
    # backward：手动计算梯度
    dloss = 1
    dy_pred = dloss * (y_pred - y) #

    dw = x.t().mm(dy_pred)
    db = dy_pred.sum()

    # 更新参数
    w.sub_(lr * dw)
    b.sub_(lr * db)

    print("training %d"%ii)

    if ii%50 ==0:

        # 画图
        display.clear_output(wait=True)
        x = t.arange(0, 6, device=device, dtype=t.float32).view(-1, 1)

        # print(x.dtype, w.dtype)
        y = x.mm(w) + b.expand_as(x)
        plt.plot(x.cpu().numpy(), y.cpu().numpy()) # predicted line

        x2, y2 = get_fake_data(batchSize=32)
        plt.scatter(x2.cpu().numpy(), y2.cpu().numpy()) # true data

        plt.xlim(0, 5)
        plt.ylim(0, 13)
        # plt.show()
        plt.pause(0.5)

print('end')
print('w: ', w.item(), 'b: ', b.item())