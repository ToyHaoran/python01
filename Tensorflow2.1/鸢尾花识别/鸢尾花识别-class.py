import tensorflow as tf
from keras import models, layers, Model
from keras.layers import Dense
from sklearn import datasets
import numpy as np

"""
用Sequential可以搭建出上层输出就是下层输入的顺序网络结构,
但是无法写出一些带有跳连的非顺序网络结构。这个时候我们可以选择用类class搭建神经网络结构。
"""
x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)


class IrisModel(Model):
    # 继承Model
    def __init__(self):
        super(IrisModel, self).__init__()
        # 定义网络结构块
        self.d1 = Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, x):
        y = self.d1(x)  # 调用网络结构块，实现前向传播
        return y

model = IrisModel()  # 就是把原来的层封装到了类中，与下面等价
# model = models.Sequential()
# model.add(layers.Dense(3, activation='softmax', kernel_regularizer=regularizers.l2()))

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)
model.summary()
