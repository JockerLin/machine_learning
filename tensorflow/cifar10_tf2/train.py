#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lin CunQin'
__version__ = '1.0'
__date__ = '2019.11.22'
__copyright__ = 'Copyright 2019, PI'
__all__ = []

import time
import os
import cv2
import datetime
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.python.keras import layers, datasets, models, regularizers
from tensorflow.python.keras.applications import resnet, vgg16
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

"精简版vgg net 训练cifar-10"
"当前最好效果 Accuracy:训练集95.84%  测试集91.46% Loss:训练集0.3722 测试集0.5887 训练时间9h"

TRAIN_TRUE_LOAD_FALSE = True
SAVE_MODEL_PATH = 'ckpt/cp-{epoch:04d}.ckpt'
DATA_SET_PATH = 'data_set/cifar-10-python.tar.gz'
CLASS_TYPE = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class VGGModel(models.Sequential):
    def __init__(self, layer=None, name=None):
        super(VGGModel, self).__init__(layers=layer, name=name)

        weight_decay = 0.0005

        self.add(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                               input_shape=(32, 32, 3)))
        self.add(layers.Activation('relu'))
        self.add(layers.BatchNormalization())
        self.add(layers.Dropout(0.3))

        self.add(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(layers.Activation('relu'))
        self.add(layers.BatchNormalization())

        self.add(layers.MaxPooling2D((2, 2), strides=2))

        self.add(layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(layers.Activation('relu'))
        self.add(layers.BatchNormalization())
        self.add(layers.Dropout(0.4))

        self.add(layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(layers.Activation('relu'))
        self.add(layers.BatchNormalization())

        self.add(layers.MaxPooling2D((2, 2), strides=2))

        self.add(layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(layers.Activation('relu'))
        self.add(layers.BatchNormalization())
        self.add(layers.Dropout(0.4))

        self.add(layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(layers.Activation('relu'))
        self.add(layers.BatchNormalization())
        self.add(layers.Dropout(0.4))

        self.add(layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(layers.Activation('relu'))
        self.add(layers.BatchNormalization())

        self.add(layers.MaxPooling2D((2, 2), strides=2, padding='valid'))

        self.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(layers.Activation('relu'))
        self.add(layers.BatchNormalization())
        self.add(layers.Dropout(0.4))

        self.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(layers.Activation('relu'))
        self.add(layers.BatchNormalization())
        self.add(layers.Dropout(0.4))

        self.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(layers.Activation('relu'))
        self.add(layers.BatchNormalization())

        self.add(layers.MaxPooling2D((2, 2), strides=2, padding='valid'))

        self.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(layers.Activation('relu'))
        self.add(layers.BatchNormalization())
        self.add(layers.Dropout(0.4))

        self.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(layers.Activation('relu'))
        self.add(layers.BatchNormalization())
        self.add(layers.Dropout(0.4))

        self.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(layers.Activation('relu'))
        self.add(layers.BatchNormalization())

        self.add(layers.MaxPooling2D((2, 2), strides=2, padding='valid'))
        self.add(layers.Dropout(0.5))

        # 将输出展平
        self.add(layers.Flatten())

        self.add(layers.Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(layers.Activation('relu'))
        self.add(layers.BatchNormalization())
        self.add(layers.Dropout(0.5))

        self.add(layers.Dense(10))
        self.add(layers.Activation('softmax'))

        self.summary()


class RestNetModel(models.Sequential):
    def __init__(self, layer=None, name=None):
        super(RestNetModel, self).__init__(layers=layer, name=name)

        self.add(layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
        self.add(layers.MaxPooling2D((2, 2), strides=2, padding='valid'))

        self.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.add(layers.MaxPooling2D((2, 2), strides=2, padding='valid'))

        self.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.add(layers.MaxPooling2D((2, 2), strides=2, padding='valid'))

        self.add(layers.Flatten())
        self.add(layers.Dense(512, activation='relu', kernel_initializer='he_normal'))
        self.add(layers.Dense(128, activation='relu', kernel_initializer='he_normal'))
        self.add(layers.Dense(10, activation='relu', kernel_initializer='he_normal'))

        self.summary()


class TransferLearningVGGModel(models.Sequential):
    def __init__(self, layer=None, name=None):
        super(TransferLearningVGGModel, self).__init__(layers=layer, name=name)

        # 添加对图像缩放的处理层
        self.add(layers.Lambda(lambda img: keras.backend.resize_images(img, 7, 7, data_format='channels_last'),
                               input_shape=(32, 32, 3)))

        model_VGG_class1000 = vgg16.VGG16(weights='imagenet', include_top=False)
        model_VGG_class1000.summary()
        model_VGG_class1000.trainable = False

        self.add(model_VGG_class1000)
        self.add(layers.Flatten())
        self.add(layers.Dense(10, activation='softmax'))

        self.summary()


class DataSource(object):
    def __init__(self):
        print("Loading Data ......")
        # data_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), DATA_SET_PATH)
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
        # 50000 训练 图片与label  10000 测试
        # cv2.imshow('sss', train_images[30])
        # cv2.waitKey()
        # train_images = train_images.reshape((60000, 28, 28, 1))
        # test_images = test_images.reshape((10000, 28, 28, 1))
        # # 像素值映射到 0 - 1 之间
        train_images, test_images = train_images / 255.0, test_images / 255.0

        train_labels = keras.utils.to_categorical(train_labels, 10)
        test_labels = keras.utils.to_categorical(test_labels, 10)

        self.train_images, self.train_labels = train_images, train_labels
        self.test_images, self.test_labels = test_images, test_labels

        print("Loading Data Finish......")


def train(model, batchSize=128, epoch=250, data=None):
    """
    训练模型

    Parameters
    ----------
    model: models.Sequential

    batchSize: int

    epoch: int

    data: DataSource

    Returns
    -------
    history

    """
    start = time.time()
    lr_drop = 20 # 每隔20个epoch调整一次学习率
    lr_decay = 1e-6 # 学习率衰减
    learning_rate = 0.1 # 初始学习率

    # 此处的optimizer的参数用str参数模型迭代到200，测试集的正确率只有40%
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(learning_rate=learning_rate, decay=lr_decay),
                  metrics=['accuracy'])

    if TRAIN_TRUE_LOAD_FALSE:
        log_dir = datetime.datetime.now().strftime('model_%Y%m%d_%H%M')
        os.mkdir(log_dir)

        # 图像预处理模块，尝试将训练、验证图片resize
        dategen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                                     samplewise_center=False,  # set each sample mean to 0
                                     featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                     samplewise_std_normalization=False,  # divide each input by its std
                                     zca_whitening=False,  # apply ZCA whitening
                                     rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
                                     width_shift_range=0.1,
                                     # randomly shift images horizontally (fraction of total width)
                                     height_shift_range=0.1,
                                     # randomly shift images vertically (fraction of total height)
                                     horizontal_flip=True,  # randomly flip images
                                     vertical_flip=False)  # randomly flip images
        dategen.fit(data.train_images)
        # 20 个epoch内val_accuracy变化小于0.1则停止训练
        early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, min_delta=0.1)

        def lrScheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))

        reduce_lr = keras.callbacks.LearningRateScheduler(lrScheduler)

        # verbose = 0 为不在标准输出流输出日志信息
        # verbose = 1 为输出进度条记录
        save_model_cb = keras.callbacks.ModelCheckpoint('ckpt/CIFAR10-EP{epoch:02d}.h5',
                                                        monitor='val_accuracy',
                                                        save_weights_only=True,
                                                        verbose=1,
                                                        period=10)
        tensor_board = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=0)

        # 可视化训练进程 可以在terminal 输入 tensorboard --logdir "path/to/logdir"

        # history = model.fit(x=data.train_images,
        #                     y=data.train_labels,
        #                     batch_size=batchSize,
        #                     epochs=epoch,
        #                     validation_data=(data.test_images, data.test_labels),
        #                     callbacks=[tensor_board, save_model_cb]
        #                     )

        # 利用Python的生成器，逐个生成数据的batch并进行训练。

        history = model.fit_generator(dategen.flow(data.train_images, data.train_labels, batch_size=batchSize),
                                      steps_per_epoch=data.train_images.shape[0] // batchSize,
                                      epochs=epoch,
                                      validation_data=(data.test_images, data.test_labels),
                                      callbacks=[tensor_board, save_model_cb, reduce_lr])

        print('\n@ Total Time Spent: %.2f seconds' % (time.time() - start))
        accuracy, val_acc = history.history['accuracy'], history.history['val_accuracy']
        max_acc, max_val_acc = np.argmax(accuracy), np.argmax(val_acc)

        print("@ Best Training Accuracy: %.2f %% achieved at EP #%d." % (accuracy[max_acc] * 100, max_acc + 1))
        print("@ Best Testing Accuracy: %.2f %% achieved at EP #%d." % (val_acc[max_val_acc] * 100, max_val_acc + 1))

        return history
    else:
        restorModel(data=data, model=model)

    test_loss, test_acc = model.evaluate(data.test_images, data.test_labels)
    print("准确率: %.6f，共测试了 EP #%d 张图片 " % (test_acc, len(data.test_labels)))


def restorModel(weightPath="./ckpt/model_20191126_1659\CIFAR10-EP250.h5",
                data=DataSource(),
                model=VGGModel()):
    """
    模型恢复

    Parameters
    ----------
    weightPath: str
        模型路径

    data: DataSource
        数据集

    Returns
    -------
    None

    """
    print(os.path.exists(os.path.abspath(weightPath)))
    if os.path.exists(os.path.abspath(weightPath)):
        # new_model = keras.models.load_model(os.path.abspath(weightPath))
        # new_model.summary()
        model.load_weights(weightPath)  # 仅仅恢复权重

        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow("img", np.hstack(data.test_images[5:10]))
        index = model.predict(np.array(data.test_images[5:10]))
        # print("index:", type(index))
        pre_type = [CLASS_TYPE[np.argmax(x)] for x in index]
        print("pre_type:", pre_type)
        # cv2.waitKey()


if __name__ == "__main__":
    # 并不需要判断，会自己调用gpu
    # if tf.test.is_gpu_available():
    #     os.environ["CUDA_VISIBLE_DEVICES"] = tf.test.gpu_device_name()

    model = TransferLearningVGGModel()
    data_sources = DataSource()
    history = train(model=model, data=data_sources, batchSize=64)
    # restorModel()

