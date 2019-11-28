#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lin CunQin'
__version__ = '1.0'
__date__ = '2019.11.15'
__copyright__ = 'Copyright 2019, PI'
__all__ = []


import os
import cv2
import datetime
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.python.keras import layers, datasets, models
from tensorflow.python.keras.applications import resnet, vgg16


DATA_SET_PATH = 'data_set_tf2/mnist.npz'
SAVE_MODEL_PATH = 'ckpt/cp-{epoch:04d}.ckpt'

if tf.test.is_gpu_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = tf.test.gpu_device_name()


class CNNModel(models.Sequential):
    def __init__(self, layer=None, name=None):
        super(CNNModel, self).__init__(layers=layer, name=name)
        self.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.add(layers.MaxPooling2D((2, 2)))
        # 第2层卷积，卷积核大小为3*3，64个
        self.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.add(layers.MaxPooling2D((2, 2), strides=1))
        self.add(layers.MaxPooling2D((2, 2)))
        # 第3层卷积，卷积核大小为3*3，64个
        self.add(layers.Conv2D(64, (3, 3), activation='relu'))

        # 将输出展平
        self.add(layers.Flatten())
        # 全连接层
        self.add(layers.Dense(64, activation='relu'))
        self.add(layers.Dense(10, activation='softmax'))

        # 查看结构
        self.summary()


class AlexNet(models.Sequential):
    def __init__(self, layer=None, name=None):
        super(AlexNet, self).__init__(layers=layer, name=name)
        self.add(layers.Conv2D(3, (11, 11), (4, 4), activation='relu', input_shape=(227, 227, 3)))
        # add more


class DataSource(object):
    def __init__(self):
        # mnist数据集存储的位置，不存在将自动下载
        data_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), DATA_SET_PATH)
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data(path=data_path)
        # 6万张训练图片，1万张测试图片
        # cv2.imshow('sss', train_images[30])
        # cv2.waitKey()
        train_images = train_images.reshape((60000, 28, 28, 1))
        test_images = test_images.reshape((10000, 28, 28, 1))
        # 像素值映射到 0 - 1 之间
        train_images, test_images = train_images / 255.0, test_images / 255.0

        self.train_images, self.train_labels = train_images, train_labels
        self.test_images, self.test_labels = test_images, test_labels


class Train(object):
    def __init__(self):
        self.cnn = CNNModel()
        self.data = DataSource()

    def train(self):
        check_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), SAVE_MODEL_PATH)
        # period 每隔5 epoch保存一次
        save_model_cb = keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True, verbose=1, period=5)
        self.cnn.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
        log_dir = "./log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                              histogram_freq=1,
                                                              write_graph=True,
                                                              write_grads=False,
                                                              write_images=True,
                                                              update_freq=500)  # 定义TensorBoard对象

        history = self.cnn.fit(x=self.data.train_images,
                               y=self.data.train_labels,
                               epochs=5,
                               callbacks=[save_model_cb, tensorboard_callback])

        # plt.plot(history.history['accuracy'])
        # plt.plot(history.history['loss'])
        # plt.legend(['training', 'valivation'], loc='upper left')
        # plt.show()

        test_loss, test_acc = self.cnn.evaluate(self.data.test_images, self.data.test_labels)
        print("准确率: %.6f，共测试了%d张图片 " % (test_acc, len(self.data.test_labels)))

    def methods(self):
        return (list(filter(lambda m: not m.startswith("__") and not m.endswith("__") and callable(getattr(self, m)),
                            dir(self))))


class Predict(object):
    def __init__(self):
        latest = tf.train.latest_checkpoint('./ckpt')
        self.cnn = CNNModel()
        # 恢复网络权重
        self.cnn.load_weights(latest)

    def predict(self, image_path):
        # 以黑白方式读取图片
        img = Image.open(image_path).convert('L')
        flatten_img = np.reshape(img, (28, 28, 1))
        x = np.array([1 - flatten_img])

        # API refer: https://keras.io/models/model/
        y = self.cnn.predict(x)

        # 因为x只传入了一张图片，取y[0]即可
        # np.argmax()取得最大值的下标，即代表的数字
        print(image_path)
        print(y[0])
        print('        -> Predict digit', np.argmax(y[0]))


def training():
    app = Train()
    app.train()


def predict():
    app = Predict()
    app.predict('./test_images/0.png')
    app.predict('./test_images/1.png')
    app.predict('./test_images/4.png')


if __name__ == "__main__":
    # app = Train()
    # print(dir(app))
    training()
    # predict()
    # try:
    #     exec("abc()")
    # except NameError:
    #     print('no name')
    # except Exception as e:
    #     print(str(e), type(e))
