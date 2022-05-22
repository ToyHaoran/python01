import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, \
    Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras import Model

"""
cifar10数据集一共有6万张彩色图片，每张图片有32行32列像素点的红绿蓝三通道数据。
提供5万张32*32像素点的十分类彩色图片和标签，用于训练。
提供1万张32*32像素点的十分类彩色图片和标签，用于测试。
十个分类分别是：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船和卡车，分别对应标签0、1、2、3一直到9
"""

np.set_printoptions(threshold=np.inf)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

class Baseline(Model):
    def __init__(self):
        super(Baseline, self).__init__()
        # 一层卷积层
        """
        tf描述卷积层 layers.Conv2D (
            filters =卷积核个数，
            kernel_size =卷积核尺寸(核高h,核宽w)， 或正方形写核长整数 
            strides =滑动步长(纵向步长h，横向步长w)，或横纵向相同写步长整数，默认1
            padding = "same" or "valid", # 使用全零填充是"same",不使用是"valid" (默认)
            activation ="relu" or "sigmoid" or "tanh" or "softmax"等，# 如有BN此处不写
            input_shape = (高，宽，通道数)  # 输入特征图维度，可省略
            )
        """
        self.c1 = Conv2D(filters=6, kernel_size=(5, 5), padding='same')  # 卷积层
        self.b1 = BatchNormalization()  # BN层
        self.a1 = Activation('relu')  # 激活层
        """ 池化层
        tf.keras.layers.MaxPool2D, AveragePooling2D(
            pool_size=池化核尺寸, # 正方形写核长整数，或(核高h，核宽w)
            strides=池化步长, #步长整数，或(纵向步长h,横向步长w)，默认为pool_size
            padding='valid'or'same' #使用全零填充是"same"，不使用是"valid" (默认)
        )
        """
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d1 = Dropout(0.2)  # dropout层
        # 两层 全连接层
        self.flatten = Flatten()
        self.f1 = Dense(128, activation='relu')
        self.d2 = Dropout(0.2)
        self.f2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)
        y = self.f2(x)
        return y

model = Baseline()  # 基础CNN模型
# model = LeNet5()
# model = Inception10(num_blocks=2, num_classes=10)  # Block数是2，block_0和block_1; 网络10分类
# model = ResNet18([2, 2, 2, 2])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "Baseline.ckpt"
# checkpoint_save_path = "./checkpoint/LeNet5.ckpt"  # 这里不同模型需要修改一下
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32, epochs=5,
                    validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()

# print(model.trainable_variables)
"""
baseline/conv2d/kernel:0 (5, 5, 3, 6)记录了第层网络用的5*5*3的卷积核，一共6个，下边给出了这6个卷积核中的所有参数W；
baseline/conv2d/bias:0 (6,)这里记录了6个卷积核各自的偏置项b，每个卷积核一个 b，6个卷积核共有6个偏置6 ；
baseline/batch_normalization/gamma:0 (6,)，这里记录了BN操作中的缩放因子γ，每个卷积核一个γ，一个6个γ；
baseline/batch_normalization/beta:0 (6,)，里记录了BN操作中的偏移因子β，每个卷积核一个β，一个6个β；
baseline/dense/kernel:0 (1536, 128)，这里记录了第一层全链接网络，1536 行、128列的线上的权量w；
baseline/dense/bias:0 (128,)，这里记录了第一层全连接网络128个偏置b；
baseline/dense_1/kernel:0 (128, 10)，这里记录了第二层全链接网络，128行、10列的线上的权量w；
baseline/dense_1/bias:0 (10,)，这里记录了第二层全连接网络10个偏置b。
"""
file = open('weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
