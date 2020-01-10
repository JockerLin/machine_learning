#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lin CunQin'
__version__ = '1.0'
__date__ = '2020.01.10'
__copyright__ = 'Copyright 2020, PI'
__all__ = []


import torch
import torchvision
import cv2
import numpy as np


def Tensor2Image(tensor):
    return np.transpose(tensor.numpy(), (1, 2, 0))


image = cv2.imread('./dog.jpg')
image = torch.from_numpy(np.transpose(image, (2, 0, 1)))
print("img size", image.size())  # torch.Size([3, 244, 206])
img = image.contiguous().view(4, 3, 61, 206)
# img = image.contiguous().view(3, 4, 61, 206).permute(1, 0, 2, 3)
print('end')
img0 = Tensor2Image(img[0])
img1 = Tensor2Image(img[1])
img2 = Tensor2Image(img[2])
img3 = Tensor2Image(img[3])

cv2.imshow('0', np.vstack((img0, img1, img2, img3)))
cv2.waitKey()
# def showImage(tensor):
#     image =

