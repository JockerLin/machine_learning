#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lin CunQin'
__version__ = '1.0'
__date__ = '2020.01.10'
__copyright__ = 'Copyright 2020, PI'
__all__ = []

import torch
from visdom import Visdom
import numpy as np

# run this in terminal first
# python -m visdom.server

# 新建名为'demo'的环境
viz = Visdom(env='demo')

arr = np.random.rand(10)

# Numpy Array
viz.line(Y=arr)
# Python List
viz.line(Y=list(arr))
# PyTorch tensor
viz.line(Y=torch.Tensor(arr))

#单张
viz.image(
    np.random.rand(3, 512, 256),
    opts=dict(title='Random!', caption='How random.'),
)
#多张
viz.images(
    np.random.randn(20, 3, 64, 64),
    opts=dict(title='Random images', caption='How random.')
)
