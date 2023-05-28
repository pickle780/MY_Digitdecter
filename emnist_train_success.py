# -*- coding: utf-8 -*-
"""
Created on Tue May  2 18:54:11 2023

@author: 14620
"""
from keras.utils import to_categorical
from keras import models, layers, regularizers
from keras.optimizers import RMSprop
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1 # 占用100%的显存
session = tf.compat.v1.Session(config=config)

with np.load("emnist.npz") as data:
    train_images = data['images']
    train_labels = data['labels']

train_images = train_images.reshape((-1, 28,28)).astype('float')/255

times=0;
label_map = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
             10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
             19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
             28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a',
             37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j',
             46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's',
             55: 't', 56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y', 61: 'z'}
while(1):

    word=input("要继续吗?Y/N")
    if(word=='Y' or word=='y'):
        break
    elif(word=="N" or word=="n"):
        print("程序结束")
        exit()
    else:
        for(times) in range(20000):
            temp_image = train_images[times]
            plt.imshow(temp_image)
            plt.title(label_map[train_labels[times]])
            plt.show()

train_labels = to_categorical(train_labels)

tf.keras.backend.clear_session()

print(train_images.shape)
# 初始化序列模型   神经网络
network = models.Sequential()

# 第一层卷积，卷积核大小为3*3，输出通道数为32，输入尺寸为(28, 28, 1)
network.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
network.add(layers.BatchNormalization()) # Batch Normalization层
network.add(layers.MaxPooling2D((2, 2))) # 最大池化层

# 第二层卷积，卷积核大小为3*3，输出通道数为64
network.add(layers.Conv2D(64, (3, 3), activation='relu'))
network.add(layers.BatchNormalization()) # Batch Normalization层
network.add(layers.MaxPooling2D((2, 2))) # 最大池化层

# 第三层卷积，卷积核大小为3*3，输出通道数为128
network.add(layers.Conv2D(128, (3, 3), activation='relu'))
network.add(layers.BatchNormalization()) # Batch Normalization层
network.add(layers.MaxPooling2D((2, 2))) # 最大池化层

network.add(layers.Flatten()) # 展开层

# 全连接层，输出节点数为128
network.add(layers.Dense(128, activation='relu'))
network.add(layers.Dropout(0.2)) # Dropout层

# 输出层，输出节点数为62
network.add(layers.Dense(62, activation='softmax'))
network.summary()
# 编译步骤
network.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# 训练网络，用fit函数, epochs表示训练多少个回合， batch_size表示每次训练给多大的数据
network.fit(train_images, train_labels, epochs=25, batch_size=200, verbose=1)
# 来在测试集上测试一下模型的性能吧
y_pre = network.predict(train_images[290:300])
print(y_pre, train_labels[290:300])
test_loss, test_accuracy = network.evaluate(train_images, train_labels)
print("test_loss:", test_loss, "test_accuracy:", test_accuracy)
while(1):
    word=input("要保存吗?Y/N")
    if(word=='Y' or word=='y'):
        name=input("请输入保存的名字")
        network.save("{}.h5".format(name))
        print("保存成功")
        exit()
    else:
        print("程序结束")
        exit()
#保存网络