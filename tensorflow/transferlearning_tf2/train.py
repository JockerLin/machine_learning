#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lin CunQin'
__version__ = '1.0'
__date__ = '2019.11.25'
__copyright__ = 'Copyright 2019, PI'
__all__ = []


import os
import datetime
import cv2
import numpy as np
from PIL import Image
import matplotlib.pylab as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from tensorflow.python.keras import layers, datasets, utils
from tensorflow.python.keras.applications import resnet, vgg16, vgg19, imagenet_utils
from tensorflow.python.keras import layers, datasets, models, regularizers


"基于imagenet训练好的模型进行cifar10分类"
"学习在训练好的模型上finetune"


def importRestNet50():
    # 利用ResNet50网络进行ImageNet分类
    model = resnet.ResNet50()
    print(model.summary())
    # utils.plot_model(model, to_file='RestNet50 construct.png')
    img = keras.preprocessing.image.load_img("ele.webp", target_size=(224, 224))
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = imagenet_utils.preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', imagenet_utils.decode_predictions(preds, top=3)[0])
    # >>> Predicted: [('n02504458', 'African_elephant', 0.8458869), ('n02504013', 'Indian_elephant', 0.092129566), ('n01871265', 'tusker', 0.061881535)]


def importVGG16():
    # 利用VGG16提取特征
    model = vgg16.VGG16(weights='imagenet', include_top=True)
    print(model.summary())
    img = keras.preprocessing.image.load_img("rb.png", target_size=(224, 224))
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = imagenet_utils.preprocess_input(x)

    preds = model.predict(x)
    # print('Predicted:', preds)
    print('Predicted:', imagenet_utils.decode_predictions(preds, top=3)[0])
    # >>> Predicted: [('n02504458', 'African_elephant', 0.70556164), ('n01871265', 'tusker', 0.20658919), ('n02504013', 'Indian_elephant', 0.087826274)]


def importVGG19():
    # 利用VGG19提取特征
    model = vgg19.VGG19(weights='imagenet', include_top=True)
    print(model.summary())
    img = keras.preprocessing.image.load_img("ele.webp", target_size=(224, 224))
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = imagenet_utils.preprocess_input(x)

    block4_pool_features = model.predict(x)
    print('Predicted:', block4_pool_features)


class TransferLearningVGGModel(models.Sequential):
    def __init__(self, layer=None, name=None):
        super(TransferLearningVGGModel, self).__init__(layers=layer, name=name)

        # 添加对图像缩放的处理层 vgg16模型输入224 图像尺寸32
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
        input_size = 224

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

        # 对输入数据的resize 会爆内存
        # resize_train_images = []
        # for img in train_images:
        #     resize_img = cv2.resize(img, (input_size, input_size))
        #     resize_train_images.append(resize_img)
        #
        # resize_test_images = []
        # for img in test_images:
        #     resize_img = cv2.resize(img, (input_size, input_size))
        #     resize_test_images.append(resize_img)

        # train_images = [cv2.resize(img, (input_size, input_size)) for img in train_images]
        # test_images = [cv2.resize(img, (input_size, input_size)) for img in test_images]

        train_labels = keras.utils.to_categorical(train_labels, 10)
        test_labels = keras.utils.to_categorical(test_labels, 10)

        self.train_images, self.train_labels = train_images, train_labels
        self.test_images, self.test_labels = test_images, test_labels

        print("Loading Data Finish......")


def train(batchSize=64, epoch=100):
    model = TransferLearningVGGModel()
    data = DataSource()

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(learning_rate=0.001),
                  metrics=['accuracy'])

    save_model_cb = keras.callbacks.ModelCheckpoint('ckpt/CIFAR10-EP{epoch:02d}.h5',
                                                    monitor='val_accuracy',
                                                    save_weights_only=True,
                                                    verbose=1,
                                                    period=10)

    log_dir = datetime.datetime.now().strftime('model_%Y%m%d_%H%M')
    os.mkdir(log_dir)
    tensor_board = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                  histogram_freq=0)

    history = model.fit(x=data.train_images,
                        y=data.train_labels,
                        batch_size=batchSize,
                        epochs=epoch,
                        validation_data=(data.test_images, data.test_labels),
                        callbacks=[save_model_cb, tensor_board]
                        )

    test_loss, test_acc = model.evaluate(data.test_images, data.test_labels)
    print("准确率: %.6f，共测试了 EP #%d 张图片 " % (test_acc, len(data.test_labels)))


if __name__ == "__main__":
    # importRestNet50()
    train()
    # importVGG19()
